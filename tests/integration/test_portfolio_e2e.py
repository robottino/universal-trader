"""Integration tests for multi-asset portfolio: Backtest/Live exact parity & CLI E2E."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from universal_trader import load_builtin_plugins
from universal_trader.brokers.simulated import SimulatedBroker
from universal_trader.cli.app import app
from universal_trader.core.calendar import MarketCalendar
from universal_trader.core.config import (
    AppConfig,
    BrokerConfig,
    DataConfig,
    FeeModelConfig,
    ForecasterConfig,
    PortfolioConfig,
    RiskConfig,
    StrategyConfig,
)
from universal_trader.core.models import AssetClass, Candle, Instrument
from universal_trader.engine.portfolio import (
    PortfolioAllocator,
    PortfolioBacktestRunner,
    PortfolioStepRunner,
)
from universal_trader.risk.manager import RiskManager


def _synthetic_basket(n: int = 320, end_dt: datetime | None = None) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    if end_dt is None:
        idx = pd.date_range(datetime(2025, 1, 1, tzinfo=UTC), periods=n, freq="1h", tz=UTC)
    else:
        idx = pd.date_range(end=end_dt, periods=n, freq="1h", tz=UTC)
    t = np.arange(n, dtype=np.float64)

    # Decorrelated waves so assets rotate in and out (testing SELL->BUY rotation!)
    waves = {
        "BTC/USD": 0.002 + 0.012 * np.sin(2.0 * np.pi * t / 120.0) + rng.normal(0.0, 0.006, size=n),
        "ETH/USD": 0.002 + 0.015 * np.sin(2.0 * np.pi * (t + 20.0) / 120.0) + rng.normal(0.0, 0.009, size=n),
        "XAUT/USD": 0.001 + 0.006 * np.cos(2.0 * np.pi * t / 140.0) + rng.normal(0.0, 0.002, size=n),
    }
    base_px = {"BTC/USD": 50000.0, "ETH/USD": 2000.0, "XAUT/USD": 3000.0}
    out: dict[str, pd.DataFrame] = {}
    for k, w in waves.items():
        close = base_px[k] * np.exp(np.cumsum(w))
        open_ = np.concatenate(([close[0]], close[:-1]))
        high = np.maximum(open_, close) * 1.002
        low = np.minimum(open_, close) * 0.998
        vol = rng.uniform(500.0, 2000.0, size=n)
        out[k] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol, "adj_close": close},
            index=idx,
        )
    return out


def test_portfolio_backtest_and_cold_start_single_step_exact_parity() -> None:
    """PortfolioBacktestRunner (incremental) MUST match cold-start bar-by-bar PortfolioStepRunner bit-for-bit."""
    load_builtin_plugins()
    insts = [
        Instrument(id="BTC/USD", asset_class=AssetClass.CRYPTO, base="BTC", quote_currency="USD", min_qty=0.00004, qty_step=0.00001),
        Instrument(id="ETH/USD", asset_class=AssetClass.CRYPTO, base="ETH", quote_currency="USD", min_qty=0.0008, qty_step=0.0001),
        Instrument(id="XAUT/USD", asset_class=AssetClass.CRYPTO, base="XAUT", quote_currency="USD", min_qty=0.002, qty_step=0.0001),
    ]
    cfg = AppConfig(
        name="portfolio_parity",
        instrument=insts[0],
        instruments=insts,
        data=DataConfig(provider="local", timeframe="1h"),
        forecaster=ForecasterConfig(type="noop"),
        strategy=StrategyConfig(
            type="ema_trend",
            trend_fast_window=10,
            trend_slow_window=30,
            trend_hysteresis_pct=0.015,
            macro_ema_window=80,
            macro_hysteresis_pct=0.02,
        ),
        portfolio=PortfolioConfig(
            allocation_method="inverse_volatility",
            max_weight_per_asset=0.60,
            max_total_exposure=0.98,
            vol_lookback_bars=30,
            vol_min_bars=12,
            vol_floor=0.10,
            rebalance_band=0.12,
            warmup_bars=40,
            min_trade_notional=25.0,
        ),
        risk=RiskConfig(
            enforce_market_hours=False,
            max_exposure_pct=0.98,
            max_drawdown_pct=0.45,
            fee_model=FeeModelConfig(pct=0.0, fixed_min=0.0),
            slippage_bps=5.0,
            spread_bps=2.0,
            enable_fee_guard=False,
        ),
        broker=BrokerConfig(type="simulated", initial_cash=10000.0, execution_price="next_open"),
    )

    candles = _synthetic_basket(n=260)
    idx = candles["BTC/USD"].index
    start_dt = idx[60].to_pydatetime()
    end_dt = idx[-1].to_pydatetime()
    cal = MarketCalendar("24/7")

    # 1. Run PortfolioBacktestRunner (uses O(1) incremental EMA + rolling vol cache)
    bt_res = PortfolioBacktestRunner(config=cfg, calendar=cal).run(
        candles,
        start=start_dt,
        end=end_dt,
        initial_cash=10000.0,
    )

    # 2. Manually loop cold-start PortfolioStepRunner with a BRAND NEW PortfolioAllocator on EVERY bar!
    live_broker = SimulatedBroker(
        initial_cash=10000.0,
        quote_currency="USD",
        fee_model=cfg.risk.fee_model,
        slippage_bps=cfg.risk.slippage_bps,
        spread_bps=cfg.risk.spread_bps,
        execution_price="next_open",
        max_drawdown_pct=cfg.risk.max_drawdown_pct,
    )
    risk_mgr = RiskManager(config=cfg.risk, calendar=cal)
    live_eqs: list[float] = []
    for i in range(60, len(idx)):
        ts = idx[i].to_pydatetime()
        bars_now = {}
        slices_now = {}
        for inst in insts:
            row = candles[inst.id].iloc[i]
            bars_now[inst.id] = Candle(
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                adj_close=float(row["close"]),
            )
            slices_now[inst.id] = candles[inst.id].iloc[: i + 1]
        live_broker.on_new_bars(bars_now, insts)
        # Fresh allocator every step = exact cold-start production live behavior!
        cold_runner = PortfolioStepRunner(
            config=cfg,
            allocator=PortfolioAllocator(cfg, cal),
            risk_manager=risk_mgr,
            broker=live_broker,
        )
        outcome = cold_runner.run_step(
            candles_by_id=slices_now,
            now=ts,
            check_staleness=False,
            enforce_idempotency=False,
            persist=False,
            notify=False,
        )
        assert outcome.portfolio_after is not None
        live_eqs.append(outcome.portfolio_after.equity)

    np.testing.assert_allclose(
        bt_res.equity_df["equity"].to_numpy(dtype=np.float64),
        np.asarray(live_eqs, dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    assert len(bt_res.fills) == len(live_broker.fills)
    assert len(bt_res.fills) >= 4  # Ensure multi-asset rotations genuinely executed


def test_portfolio_cli_e2e_doctor_backtest_trade_idempotency_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full CLI E2E test for multi-asset portfolio: doctor, backtest + 3-panel PNG, trade dry-run/paper/idempotency, report."""
    now_hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    data_dir = tmp_path / "basket_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Strong recent uptrend across all 3 assets so initial paper trade executes buys cleanly
    n = 160
    idx = pd.date_range(end=now_hour, periods=n, freq="1h", tz=UTC)
    t = np.arange(n, dtype=np.float64)
    rng = np.random.default_rng(11)
    for stem, base_p, vol_s in [("BTC_USD", 50000.0, 0.008), ("ETH_USD", 2000.0, 0.011), ("XAUT_USD", 3000.0, 0.002)]:
        close = base_p * np.exp(0.004 * t + rng.normal(0.0, vol_s, size=n).cumsum() * 0.1)
        df = pd.DataFrame(
            {
                "timestamp": idx,
                "open": close * 0.999,
                "high": close * 1.002,
                "low": close * 0.998,
                "close": close,
                "volume": np.full(n, 1500.0),
                "adj_close": close,
            }
        )
        df.to_parquet(data_dir / f"{stem}.parquet", index=False)

    db_file = tmp_path / "state" / "pf_trader.db"
    cfg_file = tmp_path / "pf_test.yaml"
    cfg_file.write_text(
        f"""
name: "cli_portfolio_test"
mode: paper
instruments:
  - id: "BTC/USD"
    asset_class: crypto
    base: BTC
    quote_currency: USD
    min_qty: 0.00004
    qty_step: 0.00001
    min_notional: 10.0
  - id: "ETH/USD"
    asset_class: crypto
    base: ETH
    quote_currency: USD
    min_qty: 0.0008
    qty_step: 0.0001
    min_notional: 10.0
  - id: "XAUT/USD"
    asset_class: crypto
    base: XAUT
    quote_currency: USD
    min_qty: 0.002
    qty_step: 0.0001
    min_notional: 10.0
portfolio:
  allocation_method: inverse_volatility
  max_weight_per_asset: 0.60
  max_total_exposure: 0.98
  vol_lookback_bars: 30
  vol_min_bars: 10
  vol_floor: 0.10
  rebalance_band: 0.12
  warmup_bars: 40
data:
  provider: local
  file_path: "{data_dir}"
  timeframe: "1h"
  lookback_bars: 160
  use_cache: false
forecaster:
  type: noop
strategy:
  type: ema_trend
  trend_fast_window: 10
  trend_slow_window: 30
  trend_hysteresis_pct: 0.01
  macro_ema_window: 60
  macro_hysteresis_pct: 0.01
risk:
  enforce_market_hours: false
  max_data_age_minutes: 180
  max_exposure_pct: 0.98
  max_drawdown_pct: 0.40
  fee_model: {{ pct: 0.0, fixed_min: 0.0 }}
  slippage_bps: 5.0
  spread_bps: 2.0
  enable_fee_guard: false
broker:
  type: local_paper
  initial_cash: 10000.0
storage:
  db_path: "{db_file}"
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("UTRADER_ROOT", str(tmp_path))
    monkeypatch.setenv("UTRADER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("UTRADER_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("UTRADER_DATA_DIR", str(tmp_path / "data_cache"))
    monkeypatch.setenv("UTRADER_FORECAST_CACHE_DIR", str(tmp_path / "forecast_cache"))

    runner = CliRunner()

    # 1. doctor validates portfolio config
    res_doc = runner.invoke(app, ["doctor", "--config", str(cfg_file)])
    assert res_doc.exit_code == 0, res_doc.output
    assert "Portfolio: BTC+ETH+XAUT" in res_doc.output

    # 2. backtest generates CSV + 3-panel PNG with all assets
    start_s = idx[50].strftime("%Y-%m-%d")
    end_s = idx[-1].strftime("%Y-%m-%d")
    res_bt = runner.invoke(
        app,
        ["backtest", "--config", str(cfg_file), "--start", start_s, "--end", end_s, "--output-dir", str(tmp_path / "reports")],
    )
    assert res_bt.exit_code == 0, res_bt.output
    eq_csv = tmp_path / "reports" / "backtest_cli_portfolio_test_equity.csv"
    chart_png = tmp_path / "reports" / "backtest_cli_portfolio_test.png"
    assert eq_csv.exists()
    assert chart_png.exists()
    eq_df = pd.read_csv(eq_csv)
    for col in ("bh_BTC/USD", "bh_ETH/USD", "bh_XAUT/USD", "w_BTC/USD", "w_ETH/USD", "w_XAUT/USD"):
        assert col in eq_df.columns

    # 3. trade --dry-run executes cleanly without mutating LocalPaper cash
    res_dry = runner.invoke(app, ["trade", "--config", str(cfg_file), "--dry-run"])
    assert res_dry.exit_code == 0, res_dry.output
    assert "DRY-RUN" in res_dry.output

    # 4. duplicate trade on same candle -> idempotent skip
    res_dup = runner.invoke(app, ["trade", "--config", str(cfg_file)])
    assert res_dup.exit_code == 0, res_dup.output
    assert "Idempotent skip" in res_dup.output

    # 5. trade --force executes real LocalPaper multi-asset allocation
    res_live = runner.invoke(app, ["trade", "--config", str(cfg_file), "--force"])
    assert res_live.exit_code == 0, res_live.output
    assert "FILLED" in res_live.output

    # 6. report reads portfolio snapshots and fills from SQLite
    res_rep = runner.invoke(app, ["report", "--db", str(db_file)])
    assert res_rep.exit_code == 0, res_rep.output
    assert "Operational Report" in res_rep.output
