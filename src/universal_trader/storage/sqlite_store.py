"""SQLiteStateStore in WAL mode for audit trail, HWM tracking, idempotency, and paper state."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from universal_trader.core.models import (
    StepOutcome,
    ensure_utc,
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_name TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    candle_timestamp TEXT NOT NULL,
    executed_at TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    UNIQUE(config_name, instrument_id, timeframe, candle_timestamp)
);

CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    config_name TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    candle_timestamp TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    spot_price REAL,
    quantiles_json TEXT NOT NULL,
    future_timestamps_json TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    config_name TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    side TEXT NOT NULL,
    decision TEXT NOT NULL,
    quantity REAL,
    estimated_price REAL,
    estimated_notional REAL,
    estimated_fee REAL,
    guard_name TEXT,
    reason TEXT,
    halted INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    order_id TEXT NOT NULL,
    config_name TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    fill_price REAL NOT NULL,
    notional REAL NOT NULL,
    fee REAL NOT NULL,
    slippage REAL NOT NULL,
    status TEXT NOT NULL,
    raw_response_json TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    config_name TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    quote_currency TEXT NOT NULL,
    cash REAL NOT NULL,
    position_qty REAL NOT NULL,
    market_price REAL NOT NULL,
    equity REAL NOT NULL,
    high_water_mark REAL NOT NULL,
    drawdown REAL NOT NULL,
    halted INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS paper_accounts (
    account_key TEXT PRIMARY KEY,
    quote_currency TEXT NOT NULL,
    cash REAL NOT NULL,
    high_water_mark REAL NOT NULL,
    halted INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_positions (
    account_key TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    quantity REAL NOT NULL,
    avg_entry_price REAL NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(account_key, instrument_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_key
    ON runs(config_name, instrument_id, timeframe, candle_timestamp);
CREATE INDEX IF NOT EXISTS idx_equity_config_ts
    ON equity_snapshots(config_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_forecasts_config_ts
    ON forecasts(config_name, candle_timestamp);
"""


class SQLiteStateStore:
    """WAL-mode SQLite persistence layer for runs, forecasts, orders, fills, equity, and paper accounts."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    @staticmethod
    def _iso_utc(dt: datetime) -> str:
        return ensure_utc(dt).isoformat()

    def has_processed(
        self,
        config_name: str,
        instrument_id: str,
        timeframe: str,
        candle_timestamp: datetime,
    ) -> bool:
        ts_str = self._iso_utc(candle_timestamp)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM runs
                WHERE config_name = ? AND instrument_id = ? AND timeframe = ? AND candle_timestamp = ?
                  AND status != 'ERROR'
                LIMIT 1
                """,
                (config_name, instrument_id, timeframe, ts_str),
            ).fetchone()
            return row is not None

    def get_high_water_mark(self, config_name: str, default: float = 0.0) -> float:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(high_water_mark) AS hwm FROM equity_snapshots WHERE config_name = ?",
                (config_name,),
            ).fetchone()
            if row is not None and row["hwm"] is not None:
                return max(float(row["hwm"]), default)
            return default

    def is_halted(self, config_name: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT halted FROM equity_snapshots
                WHERE config_name = ?
                ORDER BY id DESC LIMIT 1
                """,
                (config_name,),
            ).fetchone()
            return bool(row["halted"]) if row is not None else False

    def record_step(self, outcome: StepOutcome) -> int:
        ts_str = self._iso_utc(outcome.candle_timestamp)
        now_str = datetime.now(UTC).isoformat()

        if outcome.error:
            status = "ERROR"
        elif outcome.fill is not None:
            status = f"FILLED:{outcome.fill.status.value}"
        elif outcome.rejected_order is not None:
            status = f"REJECTED:{outcome.rejected_order.guard_name}"
        else:
            status = "HOLD"

        with self._connect() as conn:
            # Upsert cleanly if previous attempt was ERROR or --force re-run
            conn.execute(
                """
                INSERT INTO runs (config_name, instrument_id, timeframe, candle_timestamp, executed_at, status, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(config_name, instrument_id, timeframe, candle_timestamp)
                DO UPDATE SET executed_at = excluded.executed_at, status = excluded.status, error = excluded.error
                """,
                (
                    outcome.config_name,
                    outcome.instrument_id,
                    outcome.timeframe,
                    ts_str,
                    now_str,
                    status,
                    outcome.error,
                ),
            )
            run_row = conn.execute(
                """
                SELECT id FROM runs
                WHERE config_name = ? AND instrument_id = ? AND timeframe = ? AND candle_timestamp = ?
                """,
                (outcome.config_name, outcome.instrument_id, outcome.timeframe, ts_str),
            ).fetchone()
            assert run_row is not None
            run_id = int(run_row["id"])

            if outcome.forecast is not None:
                spot = outcome.intent.signal.current_price if outcome.intent is not None else None
                conn.execute(
                    """
                    INSERT INTO forecasts (
                        run_id, config_name, instrument_id, model_id, candle_timestamp,
                        horizon, spot_price, quantiles_json, future_timestamps_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        outcome.config_name,
                        outcome.instrument_id,
                        outcome.forecast.model_id,
                        ts_str,
                        outcome.forecast.horizon,
                        spot,
                        json.dumps({str(k): v.tolist() for k, v in outcome.forecast.quantiles.items()}),
                        json.dumps([t.isoformat() for t in outcome.forecast.future_timestamps]),
                    ),
                )

            if outcome.approved_order is not None:
                ao = outcome.approved_order
                conn.execute(
                    """
                    INSERT INTO orders (
                        run_id, config_name, instrument_id, timestamp, side, decision,
                        quantity, estimated_price, estimated_notional, estimated_fee,
                        guard_name, reason, halted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        outcome.config_name,
                        outcome.instrument_id,
                        self._iso_utc(ao.timestamp),
                        ao.side.value,
                        "APPROVED",
                        ao.quantity,
                        ao.estimated_price,
                        ao.estimated_notional,
                        ao.estimated_fee,
                        None,
                        ao.reason,
                        0,
                    ),
                )
            elif outcome.rejected_order is not None:
                ro = outcome.rejected_order
                conn.execute(
                    """
                    INSERT INTO orders (
                        run_id, config_name, instrument_id, timestamp, side, decision,
                        quantity, estimated_price, estimated_notional, estimated_fee,
                        guard_name, reason, halted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        outcome.config_name,
                        outcome.instrument_id,
                        self._iso_utc(ro.timestamp),
                        ro.side.value,
                        "REJECTED",
                        0.0,
                        None,
                        None,
                        None,
                        ro.guard_name,
                        ro.reason,
                        1 if ro.halted else 0,
                    ),
                )
            elif outcome.intent is not None:
                conn.execute(
                    """
                    INSERT INTO orders (
                        run_id, config_name, instrument_id, timestamp, side, decision,
                        quantity, estimated_price, estimated_notional, estimated_fee,
                        guard_name, reason, halted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        outcome.config_name,
                        outcome.instrument_id,
                        self._iso_utc(outcome.intent.timestamp),
                        outcome.intent.side.value,
                        "HOLD",
                        0.0,
                        outcome.intent.signal.current_price,
                        0.0,
                        0.0,
                        None,
                        outcome.intent.reason,
                        1 if (outcome.portfolio_after and outcome.portfolio_after.halted) else 0,
                    ),
                )

            if outcome.fill is not None:
                fl = outcome.fill
                conn.execute(
                    """
                    INSERT INTO fills (
                        run_id, order_id, config_name, instrument_id, timestamp, side,
                        quantity, fill_price, notional, fee, slippage, status, raw_response_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        fl.order_id,
                        outcome.config_name,
                        outcome.instrument_id,
                        self._iso_utc(fl.timestamp),
                        fl.side.value,
                        fl.quantity,
                        fl.fill_price,
                        fl.notional,
                        fl.fee,
                        fl.slippage,
                        fl.status.value,
                        json.dumps(fl.raw_response, default=str),
                    ),
                )

            pf = outcome.portfolio_after or outcome.portfolio_before
            if pf is not None:
                pos = pf.positions.get(outcome.instrument_id)
                pos_qty = pos.quantity if pos is not None else 0.0
                mkt_price = (
                    pos.market_price
                    if (pos is not None and pos.market_price > 0)
                    else (outcome.intent.signal.current_price if outcome.intent else 0.0)
                )
                conn.execute(
                    """
                    INSERT INTO equity_snapshots (
                        run_id, config_name, instrument_id, timestamp, quote_currency,
                        cash, position_qty, market_price, equity, high_water_mark, drawdown, halted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        outcome.config_name,
                        outcome.instrument_id,
                        self._iso_utc(pf.timestamp),
                        pf.quote_currency,
                        pf.cash,
                        pos_qty,
                        mkt_price,
                        pf.equity,
                        pf.high_water_mark,
                        pf.drawdown,
                        1 if pf.halted else 0,
                    ),
                )

            return run_id

    # ── LocalPaperBroker persistence helpers ───────────────────────────────────────

    def load_or_init_paper_account(
        self,
        account_key: str,
        quote_currency: str,
        initial_cash: float,
    ) -> dict[str, Any]:
        now_str = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM paper_accounts WHERE account_key = ?",
                (account_key,),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO paper_accounts (account_key, quote_currency, cash, high_water_mark, halted, updated_at)
                    VALUES (?, ?, ?, ?, 0, ?)
                    """,
                    (account_key, quote_currency, initial_cash, initial_cash, now_str),
                )
                cash = initial_cash
                hwm = initial_cash
                halted = False
            else:
                cash = float(row["cash"])
                hwm = float(row["high_water_mark"])
                halted = bool(row["halted"])
                quote_currency = str(row["quote_currency"])

            pos_rows = conn.execute(
                "SELECT instrument_id, quantity, avg_entry_price FROM paper_positions WHERE account_key = ?",
                (account_key,),
            ).fetchall()
            positions = {
                str(r["instrument_id"]): {
                    "quantity": float(r["quantity"]),
                    "avg_entry_price": float(r["avg_entry_price"]),
                }
                for r in pos_rows
                if float(r["quantity"]) > 0
            }
            return {
                "account_key": account_key,
                "quote_currency": quote_currency,
                "cash": cash,
                "high_water_mark": hwm,
                "halted": halted,
                "positions": positions,
            }

    def save_paper_account(
        self,
        *,
        account_key: str,
        quote_currency: str,
        cash: float,
        high_water_mark: float,
        halted: bool,
        positions: dict[str, dict[str, float]],
    ) -> None:
        now_str = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO paper_accounts (account_key, quote_currency, cash, high_water_mark, halted, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_key) DO UPDATE SET
                    cash = excluded.cash,
                    high_water_mark = excluded.high_water_mark,
                    halted = excluded.halted,
                    updated_at = excluded.updated_at
                """,
                (account_key, quote_currency, cash, high_water_mark, 1 if halted else 0, now_str),
            )
            conn.execute("DELETE FROM paper_positions WHERE account_key = ?", (account_key,))
            for inst_id, pos in positions.items():
                if pos["quantity"] > 0:
                    conn.execute(
                        """
                        INSERT INTO paper_positions (account_key, instrument_id, quantity, avg_entry_price, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (account_key, inst_id, pos["quantity"], pos["avg_entry_price"], now_str),
                    )

    # ── Reporting & Ex-Post Evaluation Queries ─────────────────────────────────────

    def get_equity_history(self, config_name: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM equity_snapshots"
        params: tuple[Any, ...] = ()
        if config_name:
            query += " WHERE config_name = ?"
            params = (config_name,)
        query += " ORDER BY timestamp ASC, id ASC"
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(query, params).fetchall()]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def get_fills(self, config_name: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM fills"
        params: tuple[Any, ...] = ()
        if config_name:
            query += " WHERE config_name = ?"
            params = (config_name,)
        query += " ORDER BY timestamp ASC, id ASC"
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(query, params).fetchall()]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def get_forecasts(self, config_name: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM forecasts"
        params: tuple[Any, ...] = ()
        if config_name:
            query += " WHERE config_name = ?"
            params = (config_name,)
        query += " ORDER BY candle_timestamp ASC, id ASC"
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(query, params).fetchall()]
        for r in rows:
            r["quantiles"] = {
                float(k): np.asarray(v, dtype=np.float64)
                for k, v in json.loads(r["quantiles_json"]).items()
            }
            r["future_timestamps"] = [
                ensure_utc(t) for t in json.loads(r["future_timestamps_json"])
            ]
        return rows

    def evaluate_stored_forecasts(
        self,
        config_name: str | None = None,
        realized_prices: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Compare past saved forecasts against realized prices (from snapshots or external series)."""
        forecasts = self.get_forecasts(config_name=config_name)
        if not forecasts:
            return {"evaluated_points": 0, "mae": None, "rmse": None, "mean_pinball_loss": None}

        if realized_prices is None:
            eq_df = self.get_equity_history(config_name=config_name)
            if eq_df.empty:
                return {"evaluated_points": 0, "mae": None, "rmse": None, "mean_pinball_loss": None}
            price_map = {
                ensure_utc(ts): float(px)
                for ts, px in zip(eq_df["timestamp"], eq_df["market_price"], strict=False)
                if px > 0
            }
        else:
            price_map = {
                ensure_utc(ts): float(px)
                for ts, px in realized_prices.dropna().items()
            }

        abs_errors: list[float] = []
        sq_errors: list[float] = []
        pinball_losses: list[float] = []
        coverage_hits: dict[float, list[float]] = {}

        for fc in forecasts:
            q_map: dict[float, np.ndarray] = fc["quantiles"]
            fut_ts: list[datetime] = fc["future_timestamps"]
            if not fut_ts or 0.5 not in q_map:
                continue
            for step_idx, target_ts in enumerate(fut_ts):
                actual = price_map.get(target_ts)
                if actual is None:
                    continue
                pred_med = float(q_map[0.5][step_idx])
                err = actual - pred_med
                abs_errors.append(abs(err))
                sq_errors.append(err * err)
                for q, arr in q_map.items():
                    q_pred = float(arr[step_idx])
                    diff = actual - q_pred
                    pb = max(q * diff, (q - 1.0) * diff)
                    pinball_losses.append(pb)
                    coverage_hits.setdefault(q, []).append(1.0 if actual <= q_pred else 0.0)

        if not abs_errors:
            return {"evaluated_points": 0, "mae": None, "rmse": None, "mean_pinball_loss": None}

        return {
            "evaluated_points": len(abs_errors),
            "mae": float(np.mean(abs_errors)),
            "rmse": float(np.sqrt(np.mean(sq_errors))),
            "mean_pinball_loss": float(np.mean(pinball_losses)) if pinball_losses else None,
            "quantile_coverage": {
                round(q, 3): float(np.mean(vals)) for q, vals in sorted(coverage_hits.items())
            },
        }
