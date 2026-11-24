"""Matplotlib multi-quantile fan charts and backtest equity/drawdown visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Headless-safe for containers, servers, and CLI

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from universal_trader.core.models import FillResult, ForecastResult, Instrument, OrderSide


def plot_fan_chart(
    candles: pd.DataFrame,
    forecast: ForecastResult,
    instrument: Instrument,
    output_path: str | Path,
    *,
    price_column: str = "close",
    history_bars: int = 96,
) -> Path:
    """Render a multi-quantile probabilistic fan chart (q05-q95, q10-q90, q25-q75, median)."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    col = price_column if price_column in candles.columns else "close"
    hist = candles[col].iloc[-history_bars:]
    hist_x = hist.index
    hist_y = hist.to_numpy(dtype=np.float64)

    if forecast.future_timestamps and len(forecast.future_timestamps) == forecast.horizon:
        fut_x = pd.DatetimeIndex(forecast.future_timestamps)
    else:
        step = (hist_x[-1] - hist_x[-2]) if len(hist_x) >= 2 else pd.Timedelta(hours=1)
        fut_x = pd.DatetimeIndex([hist_x[-1] + step * (i + 1) for i in range(forecast.horizon)])

    # Anchor fan chart smoothly at the last observed point
    anchor_x = pd.DatetimeIndex([hist_x[-1]]).append(fut_x)
    last_val = float(hist_y[-1])

    def _anchored(q: float) -> np.ndarray:
        return np.concatenate(([last_val], forecast.quantile(q)))

    q05, q95 = _anchored(0.05), _anchored(0.95)
    q10, q90 = _anchored(0.10), _anchored(0.90)
    q25, q75 = _anchored(0.25), _anchored(0.75)
    q50 = _anchored(0.50)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    ax.plot(hist_x, hist_y, color="#e2e8f0", linewidth=1.8, label=f"Historical ({col})")
    ax.fill_between(anchor_x, q05, q95, color="#38bdf8", alpha=0.14, label="90% interval (q05–q95)")
    ax.fill_between(anchor_x, q10, q90, color="#38bdf8", alpha=0.24, label="80% interval (q10–q90)")
    ax.fill_between(anchor_x, q25, q75, color="#0ea5e9", alpha=0.38, label="50% interval (q25–q75)")
    ax.plot(anchor_x, q50, color="#fbbf24", linewidth=2.2, linestyle="-", label="Median (q50)")
    ax.scatter([hist_x[-1]], [last_val], color="#f8fafc", s=36, zorder=5)

    exp_ret = forecast.expected_return(last_val, mode="last")
    ax.set_title(
        f"{instrument.id} ({instrument.quote_currency}) — {forecast.model_id} | "
        f"Horizon: {forecast.horizon} bars | Median Δ: {exp_ret:+.2%}",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel(f"Price ({instrument.quote_currency})", color="#cbd5e1")
    ax.set_xlabel("Timestamp (UTC)", color="#cbd5e1")
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle="--", alpha=0.2, color="#64748b")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    legend = ax.legend(loc="upper left", facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    legend.get_frame().set_alpha(0.9)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def plot_backtest_report(
    equity_df: pd.DataFrame,
    fills: list[FillResult],
    metrics: dict[str, Any],
    instrument: Instrument,
    output_path: str | Path,
) -> Path:
    """Render 2-panel equity vs Buy & Hold + drawdown chart with trade markers and KPIs."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        dpi=140,
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0]},
    )
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle="--", alpha=0.2, color="#64748b")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    ts = equity_df.index
    eq = equity_df["equity"].to_numpy(dtype=np.float64)
    bh = equity_df["buy_and_hold_equity"].to_numpy(dtype=np.float64)
    dd = equity_df["drawdown"].to_numpy(dtype=np.float64) * 100.0

    ax1.plot(ts, eq, color="#38bdf8", linewidth=2.0, label="Strategy Equity")
    ax1.plot(ts, bh, color="#94a3b8", linewidth=1.4, linestyle="--", label="Buy & Hold")

    # Overlay buy/sell markers mapped onto equity curve
    eq_series = equity_df["equity"]
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    for f in fills:
        if f.quantity <= 0:
            continue
        f_ts = pd.Timestamp(f.timestamp)
        idx = eq_series.index.get_indexer([f_ts], method="nearest")[0]
        if 0 <= idx < len(eq_series):
            if f.side == OrderSide.BUY:
                buy_x.append(eq_series.index[idx])
                buy_y.append(float(eq_series.iloc[idx]))
            elif f.side == OrderSide.SELL:
                sell_x.append(eq_series.index[idx])
                sell_y.append(float(eq_series.iloc[idx]))

    if buy_x:
        ax1.scatter(buy_x, buy_y, marker="^", color="#22c55e", s=48, zorder=5, label="BUY")
    if sell_x:
        ax1.scatter(sell_x, sell_y, marker="v", color="#ef4444", s=48, zorder=5, label="SELL")

    roi = metrics.get("total_roi", 0.0)
    bh_roi = metrics.get("buy_and_hold_roi", 0.0)
    sharpe = metrics.get("sharpe_ratio", 0.0)
    max_dd = metrics.get("max_drawdown", 0.0)
    trades = metrics.get("num_trades", 0)

    ax1.set_title(
        f"{instrument.id} Backtest — ROI: {roi:+.2%} (B&H: {bh_roi:+.2%}) | "
        f"Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.2%} | Trades: {trades}",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax1.set_ylabel(f"Equity ({instrument.quote_currency})", color="#cbd5e1")
    leg = ax1.legend(loc="upper left", facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    leg.get_frame().set_alpha(0.9)

    ax2.fill_between(ts, 0.0, -dd, color="#ef4444", alpha=0.35)
    ax2.plot(ts, -dd, color="#f87171", linewidth=1.2)
    ax2.set_ylabel("Drawdown (%)", color="#cbd5e1")
    ax2.set_xlabel("Timestamp (UTC)", color="#cbd5e1")

    locator = mdates.AutoDateLocator()
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


_ASSET_COLOR_HINTS: dict[str, str] = {
    "BTC": "#f59e0b",   # Bitcoin orange
    "ETH": "#a855f7",   # Ethereum purple
    "XAUT": "#eab308",  # Gold yellow
    "PAXG": "#eab308",
    "GOLD": "#eab308",
    "SOL": "#14b8a6",   # Solana teal
}
_FALLBACK_PALETTE: tuple[str, ...] = (
    "#fb7185",
    "#34d399",
    "#60a5fa",
    "#f472b6",
    "#c084fc",
    "#fbbf24",
)


def _color_for_instrument(inst: Instrument, idx: int) -> str:
    hint = _ASSET_COLOR_HINTS.get(inst.base.upper())
    if hint is not None:
        return hint
    return _FALLBACK_PALETTE[idx % len(_FALLBACK_PALETTE)]


def plot_portfolio_backtest_report(
    equity_df: pd.DataFrame,
    metrics: dict[str, Any],
    instruments: list[Instrument],
    benchmark_id: str,
    config_name: str,
    output_path: str | Path,
) -> Path:
    """Render 3-panel multi-asset portfolio chart: Equity + all per-asset B&H, Drawdown, Allocation stack."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resample daily for crisp visual rendering on multi-year intraday series
    plot_df = equity_df.resample("1D").last().dropna(how="all") if len(equity_df) > 4000 else equity_df.copy()
    plot_df = plot_df.ffill().dropna(subset=["equity"])

    ts = plot_df.index
    eq = plot_df["equity"].to_numpy(dtype=np.float64)
    dd_pct = (1.0 - eq / np.maximum.accumulate(eq)) * 100.0
    bh_col = f"bh_{benchmark_id}" if f"bh_{benchmark_id}" in plot_df.columns else "buy_and_hold_equity"
    bh_bench = plot_df[bh_col].to_numpy(dtype=np.float64)
    bh_dd_pct = (1.0 - bh_bench / np.maximum.accumulate(bh_bench)) * 100.0

    quote_ccy = instruments[0].quote_currency
    initial_eq = float(metrics.get("initial_equity", eq[0] if len(eq) > 0 else 10000.0))
    final_eq = float(metrics.get("final_equity", eq[-1] if len(eq) > 0 else initial_eq))
    roi = float(metrics.get("total_roi", 0.0))
    cagr = float(metrics.get("cagr", 0.0))
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    calmar = float(metrics.get("calmar_ratio", 0.0))
    max_dd = float(metrics.get("max_drawdown", 0.0))
    trades = int(metrics.get("num_trades", 0))

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(14, 10.5),
        dpi=140,
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.0, 1.0]},
    )
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle="--", alpha=0.2, color="#64748b")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Panel 1: Portfolio equity + every individual asset Buy & Hold normalized from initial_eq
    ax1.plot(
        ts,
        eq,
        color="#38bdf8",
        linewidth=2.4,
        zorder=5,
        label=f"Portfolio ({roi:+.1%}, {final_eq:,.0f} {quote_ccy})",
    )
    colors = [_color_for_instrument(inst, i) for i, inst in enumerate(instruments)]
    for inst, col_hex in zip(instruments, colors, strict=True):
        col_name = f"bh_{inst.id}"
        if col_name not in plot_df.columns:
            continue
        series = plot_df[col_name].to_numpy(dtype=np.float64)
        asset_roi = (series[-1] / series[0] - 1.0) if len(series) > 0 and series[0] > 0 else 0.0
        ax1.plot(
            ts,
            series,
            color=col_hex,
            linewidth=1.35,
            alpha=0.82,
            linestyle="--",
            label=f"{inst.base} Buy & Hold ({asset_roi:+.1%})",
        )

    basket_label = " + ".join(inst.base for inst in instruments)
    ax1.set_title(
        f"{config_name} ({basket_label})\n"
        f"ROI: {roi:+.1%} | CAGR: {cagr:+.1%} | Sharpe: {sharpe:.2f} | "
        f"Calmar: {calmar:.2f} | MaxDD: {max_dd:.1%} | Trades: {trades}",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax1.set_ylabel(f"Equity from {initial_eq:,.0f} ({quote_ccy})", color="#cbd5e1")
    leg1 = ax1.legend(
        loc="upper left",
        ncol=2,
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor="#e2e8f0",
    )
    leg1.get_frame().set_alpha(0.92)

    # Panel 2: Drawdown comparison
    bench_base = next((i.base for i in instruments if i.id == benchmark_id), benchmark_id)
    ax2.fill_between(ts, 0.0, -bh_dd_pct, color="#64748b", alpha=0.22, label=f"{bench_base} Buy & Hold DD")
    ax2.fill_between(ts, 0.0, -dd_pct, color="#ef4444", alpha=0.45, label="Portfolio DD")
    ax2.plot(ts, -dd_pct, color="#f87171", linewidth=1.1)
    ax2.set_ylabel("Drawdown (%)", color="#cbd5e1")
    leg2 = ax2.legend(loc="lower left", facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    leg2.get_frame().set_alpha(0.9)

    # Panel 3: Stacked allocation weights
    stack_vals = [
        (plot_df[f"w_{inst.id}"].to_numpy(dtype=np.float64) * 100.0)
        if f"w_{inst.id}" in plot_df.columns
        else np.zeros(len(ts), dtype=np.float64)
        for inst in instruments
    ]
    stack_labels = [inst.base for inst in instruments]
    ax3.stackplot(ts, *stack_vals, labels=stack_labels, colors=colors, alpha=0.84)
    ax3.set_ylim(0, 102)
    ax3.set_ylabel("Allocation (%)", color="#cbd5e1")
    ax3.set_xlabel(f"Date (UTC) — empty space = Cash {quote_ccy}", color="#cbd5e1")
    leg3 = ax3.legend(
        loc="upper left",
        ncol=len(instruments),
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor="#e2e8f0",
    )
    leg3.get_frame().set_alpha(0.9)

    locator = mdates.AutoDateLocator()
    ax3.xaxis.set_major_locator(locator)
    ax3.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path
