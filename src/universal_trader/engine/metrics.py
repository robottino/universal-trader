"""Calendar-aware financial performance metrics and probabilistic quantile calibration."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from universal_trader.core.calendar import MarketCalendar
from universal_trader.core.models import (
    FillResult,
    ForecastResult,
    OrderSide,
    OrderStatus,
    ensure_utc,
)


def compute_round_trip_pnls(fills: list[FillResult]) -> list[float]:
    """Compute realized PnLs across closed round-trips tracked per instrument_id."""
    executed = [f for f in fills if f.status == OrderStatus.FILLED and f.quantity > 0]
    pnls: list[float] = []
    state: dict[str, tuple[float, float]] = {}

    for f in executed:
        pos_qty, avg_cost = state.get(f.instrument_id, (0.0, 0.0))
        if f.side == OrderSide.BUY:
            total_cost = pos_qty * avg_cost + f.notional + f.fee
            pos_qty += f.quantity
            avg_cost = total_cost / pos_qty if pos_qty > 0 else 0.0
            state[f.instrument_id] = (pos_qty, avg_cost)
        elif f.side == OrderSide.SELL and pos_qty > 0:
            sold = min(pos_qty, f.quantity)
            proceeds = (f.notional - f.fee) * (sold / f.quantity)
            cost_basis = sold * avg_cost
            pnls.append(proceeds - cost_basis)
            pos_qty = max(0.0, pos_qty - sold)
            if pos_qty < 1e-12:
                pos_qty = 0.0
                avg_cost = 0.0
            state[f.instrument_id] = (pos_qty, avg_cost)
    return pnls


def compute_performance_metrics(
    *,
    equity_df: pd.DataFrame,
    fills: list[FillResult],
    calendar: MarketCalendar,
    timeframe: str,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """Compute calendar-aware ROI, CAGR, Sharpe, Sortino, Calmar, MaxDD, WinRate, PF, Costs."""
    if equity_df.empty:
        return {
            "initial_equity": 0.0,
            "final_equity": 0.0,
            "total_roi": 0.0,
            "buy_and_hold_roi": 0.0,
            "outperformance": 0.0,
            "cagr": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "num_trades": 0,
            "num_round_trips": 0,
            "turnover": 0.0,
            "total_fees": 0.0,
            "total_slippage": 0.0,
            "total_costs": 0.0,
            "bars_per_year": calendar.bars_per_year(timeframe),
        }

    eq = equity_df["equity"].to_numpy(dtype=np.float64)
    bh = equity_df["buy_and_hold_equity"].to_numpy(dtype=np.float64)
    initial_eq = float(eq[0])
    final_eq = float(eq[-1])

    total_roi = (final_eq / initial_eq - 1.0) if initial_eq > 0 else 0.0
    bh_roi = (float(bh[-1]) / float(bh[0]) - 1.0) if len(bh) > 0 and bh[0] > 0 else 0.0
    outperformance = total_roi - bh_roi

    bars_per_year = float(calendar.bars_per_year(timeframe))
    n_periods = max(1, len(eq) - 1)
    years = n_periods / bars_per_year if bars_per_year > 0 else 1.0
    if initial_eq > 0 and final_eq > 0 and years > 0:
        cagr = float((final_eq / initial_eq) ** (1.0 / years) - 1.0)
    else:
        cagr = -1.0 if final_eq <= 0 else 0.0

    # Per-bar simple returns
    returns = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], np.nan)
    returns = returns[np.isfinite(returns)]
    rf_bar = risk_free_rate / bars_per_year if bars_per_year > 0 else 0.0
    excess = returns - rf_bar

    if len(excess) >= 2:
        std_r = float(np.std(excess, ddof=1))
        sharpe = float((np.mean(excess) / std_r) * math.sqrt(bars_per_year)) if std_r > 1e-12 else 0.0
        downside = excess[excess < 0.0]
        down_std = float(np.sqrt(np.mean(downside**2))) if len(downside) > 0 else 0.0
        sortino = float((np.mean(excess) / down_std) * math.sqrt(bars_per_year)) if down_std > 1e-12 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    running_max = np.maximum.accumulate(eq)
    dd_series = np.where(running_max > 0, 1.0 - (eq / running_max), 0.0)
    max_dd = float(np.max(dd_series)) if len(dd_series) > 0 else 0.0
    calmar = float(cagr / max_dd) if max_dd > 1e-9 else (float("inf") if cagr > 0 else 0.0)
    if not math.isfinite(calmar):
        calmar = 0.0

    executed_fills = [f for f in fills if f.status == OrderStatus.FILLED and f.quantity > 0]
    round_trips = compute_round_trip_pnls(executed_fills)
    wins = [p for p in round_trips if p > 0]
    losses = [abs(p) for p in round_trips if p < 0]
    win_rate = float(len(wins) / len(round_trips)) if round_trips else 0.0
    gross_profit = float(sum(wins))
    gross_loss = float(sum(losses))
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 1e-12 else (gross_profit if gross_profit > 0 else 0.0)

    total_notional = float(sum(f.notional for f in executed_fills))
    avg_equity = float(np.mean(eq)) if len(eq) > 0 else initial_eq
    turnover = float(total_notional / avg_equity) if avg_equity > 0 else 0.0
    total_fees = float(sum(f.fee for f in executed_fills))
    total_slippage = float(sum(f.slippage for f in executed_fills))

    return {
        "initial_equity": initial_eq,
        "final_equity": final_eq,
        "total_roi": float(total_roi),
        "buy_and_hold_roi": float(bh_roi),
        "outperformance": float(outperformance),
        "cagr": float(cagr),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "num_trades": len(executed_fills),
        "num_round_trips": len(round_trips),
        "turnover": float(turnover),
        "total_fees": total_fees,
        "total_slippage": total_slippage,
        "total_costs": total_fees + total_slippage,
        "bars_per_year": bars_per_year,
    }


def compute_forecast_calibration(
    forecasts: dict[datetime, ForecastResult],
    candles: pd.DataFrame,
    price_column: str = "close",
) -> dict[str, Any]:
    """Compute ex-post MAE, RMSE, Pinball Loss, and empirical quantile coverage."""
    if not forecasts or candles.empty:
        return {
            "evaluated_points": 0,
            "mae": None,
            "rmse": None,
            "mean_pinball_loss": None,
            "quantile_coverage": {},
            "interval_80_coverage": None,
            "interval_90_coverage": None,
        }

    col = price_column if price_column in candles.columns else "close"
    prices = candles[col].to_numpy(dtype=np.float64)
    ts_to_pos = {ensure_utc(ts): i for i, ts in enumerate(candles.index)}

    abs_errors: list[float] = []
    sq_errors: list[float] = []
    pinball_losses: list[float] = []
    q_hits: dict[float, list[float]] = {}
    in_80: list[float] = []
    in_90: list[float] = []

    for origin_ts, fc in forecasts.items():
        origin_pos = ts_to_pos.get(ensure_utc(origin_ts))
        if origin_pos is None:
            continue
        med = fc.median
        q05 = fc.quantile(0.05)
        q10 = fc.quantile(0.10)
        q90 = fc.quantile(0.90)
        q95 = fc.quantile(0.95)

        for h in range(fc.horizon):
            target_pos = origin_pos + h + 1
            if target_pos >= len(prices):
                break
            actual = float(prices[target_pos])
            if not math.isfinite(actual):
                continue
            err = actual - float(med[h])
            abs_errors.append(abs(err))
            sq_errors.append(err * err)

            in_80.append(1.0 if float(q10[h]) <= actual <= float(q90[h]) else 0.0)
            in_90.append(1.0 if float(q05[h]) <= actual <= float(q95[h]) else 0.0)

            for q, arr in fc.quantiles.items():
                pred_q = float(arr[h])
                diff = actual - pred_q
                pb = max(q * diff, (q - 1.0) * diff)
                pinball_losses.append(pb)
                q_hits.setdefault(q, []).append(1.0 if actual <= pred_q else 0.0)

    if not abs_errors:
        return {
            "evaluated_points": 0,
            "mae": None,
            "rmse": None,
            "mean_pinball_loss": None,
            "quantile_coverage": {},
            "interval_80_coverage": None,
            "interval_90_coverage": None,
        }

    return {
        "evaluated_points": len(abs_errors),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(sq_errors))),
        "mean_pinball_loss": float(np.mean(pinball_losses)) if pinball_losses else None,
        "quantile_coverage": {
            round(q, 3): float(np.mean(hits)) for q, hits in sorted(q_hits.items())
        },
        "interval_80_coverage": float(np.mean(in_80)) if in_80 else None,
        "interval_90_coverage": float(np.mean(in_90)) if in_90 else None,
    }
