"""Multi-asset PortfolioAllocator, PortfolioStepRunner, and PortfolioBacktestRunner.

Shares the exact same causal trend + inverse-volatility allocation and risk-guarded
SELL-first / BUY-second execution pipeline across backtest, dry-run, paper, and live modes.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from universal_trader.brokers.local_paper import LocalPaperBroker
from universal_trader.brokers.simulated import SimulatedBroker
from universal_trader.core.calendar import MarketCalendar
from universal_trader.core.config import AppConfig
from universal_trader.core.logging import get_logger
from universal_trader.core.models import (
    AssetClass,
    Candle,
    FillResult,
    ForecastResult,
    Instrument,
    MarketSnapshot,
    OrderIntent,
    OrderSide,
    OrderStatus,
    PortfolioState,
    PortfolioStepOutcome,
    Quote,
    RejectedOrder,
    SignalMetrics,
    StepOutcome,
    ensure_utc,
)
from universal_trader.core.protocols import Broker, Notifier
from universal_trader.engine.metrics import compute_performance_metrics
from universal_trader.risk.manager import RiskManager
from universal_trader.storage.sqlite_store import SQLiteStateStore

logger = get_logger(__name__)

ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class AllocationDecision:
    """Strictly causal multi-asset signals, realized volatilities, and target weights."""

    active_signals: dict[str, bool]
    realized_vols: dict[str, float]
    target_weights: dict[str, float]
    current_weights: dict[str, float]
    rebalance_triggered: bool
    reasons: dict[str, str]


class _AssetSignalTracker:
    """Strictly causal EMA trend + macro regime + realized volatility tracker per instrument.

    Maintains bit-for-bit identical output between cold-start single-step evaluation (live/paper)
    and O(1) incremental sequential stepping (long historical backtests).
    """

    def __init__(
        self,
        *,
        fast_window: int,
        slow_window: int,
        trend_hysteresis_pct: float,
        macro_ema_window: int | None,
        macro_hysteresis_pct: float,
        vol_lookback_bars: int,
        vol_min_bars: int,
        bars_per_year: float,
        warmup_bars: int,
    ) -> None:
        self.fast_window = int(fast_window)
        self.slow_window = max(self.fast_window + 1, int(slow_window))
        self.trend_hysteresis_pct = float(trend_hysteresis_pct)
        self.macro_ema_window = int(macro_ema_window) if macro_ema_window is not None else None
        self.macro_hysteresis_pct = float(macro_hysteresis_pct)
        self.vol_lookback_bars = int(vol_lookback_bars)
        self.vol_min_bars = int(vol_min_bars)
        self.annualize_sqrt = math.sqrt(max(1.0, float(bars_per_year)))
        self.warmup_bars = max(self.slow_window, int(warmup_bars))

        self._alpha_fast = 2.0 / (self.fast_window + 1.0)
        self._alpha_slow = 2.0 / (self.slow_window + 1.0)
        self._alpha_macro = (
            2.0 / (self.macro_ema_window + 1.0) if self.macro_ema_window is not None else 0.0
        )

        # Incremental cache state
        self._initialized: bool = False
        self._raw_len: int = 0
        self._valid_count: int = 0
        self._last_ts: object | None = None
        self._last_price: float = 0.0
        self._ef: float = 0.0
        self._es: float = 0.0
        self._em: float = 0.0
        self._bull: bool = False
        self._macro_bull: bool = True
        self._lr_buf: np.ndarray = np.empty(0, dtype=np.float64)

    def _compute_vol_from_buf(self) -> float:
        if len(self._lr_buf) < self.vol_min_bars:
            return 0.5
        std_val = float(np.std(self._lr_buf, ddof=0))
        rv = std_val * self.annualize_sqrt
        return rv if math.isfinite(rv) else 0.5

    def _format_reason(self, active: bool, rv: float) -> str:
        macro_txt = ""
        if self.macro_ema_window is not None:
            macro_txt = f" | macroEMA({self.macro_ema_window})={'BULL' if self._macro_bull else 'BEAR'}"
        state_txt = "ACTIVE" if active else "FLAT"
        return (
            f"{state_txt} | EMA({self.fast_window}/{self.slow_window})="
            f"{self._ef:.2f}/{self._es:.2f}{macro_txt} | vol={rv:.1%}"
        )

    def evaluate(
        self,
        prices: np.ndarray,
        last_ts: object,
        raw_len: int,
    ) -> tuple[bool, float, str]:
        """Return (is_active_bull, annualized_realized_vol, diagnostic_reason)."""
        # Fast O(1) incremental path when advancing exactly 1 new valid bar sequentially
        if (
            self._initialized
            and raw_len == self._raw_len + 1
            and last_ts != self._last_ts
            and len(prices) > 0
        ):
            p0 = float(prices[-1])
            if math.isfinite(p0) and p0 > 0:
                return self._step_one(p0, last_ts=last_ts, raw_len=raw_len)

        # Cold-start full strictly-causal pass over all finite positive bars in history
        finite_mask = np.isfinite(prices) & (prices > 0.0)
        valid = prices[finite_mask]
        n = len(valid)
        if n == 0:
            self._initialized = False
            return False, 0.5, "No valid price bars"

        ef = float(valid[0])
        es = float(valid[0])
        em = float(valid[0])
        bull = False
        macro_bull = True
        hyst = self.trend_hysteresis_pct
        m_hyst = self.macro_hysteresis_pct
        af, as_, am = self._alpha_fast, self._alpha_slow, self._alpha_macro
        has_macro = self.macro_ema_window is not None
        slow_w = self.slow_window

        for j in range(1, n):
            pj = float(valid[j])
            ef = af * pj + (1.0 - af) * ef
            es = as_ * pj + (1.0 - as_) * es
            if has_macro:
                em = am * pj + (1.0 - am) * em
            if j < slow_w:
                continue
            if has_macro:
                if not macro_bull:
                    if pj > em * (1.0 + m_hyst):
                        macro_bull = True
                elif pj < em * (1.0 - m_hyst):
                    macro_bull = False
            if not bull:
                if pj > es * (1.0 + hyst) and ef > es:
                    bull = True
            elif pj < es * (1.0 - hyst) and ef < es:
                bull = False

        # Strictly causal log returns matching pd.Series(valid).pct/log with lr[0]=0.0
        lr = np.zeros(n, dtype=np.float64)
        if n > 1:
            lr[1:] = np.log(valid[1:] / valid[:-1])
        tail_lr = lr[-self.vol_lookback_bars :] if n > self.vol_lookback_bars else lr

        self._initialized = True
        self._raw_len = raw_len
        self._valid_count = n
        self._last_ts = last_ts
        self._last_price = float(valid[-1])
        self._ef = ef
        self._es = es
        self._em = em
        self._bull = bull
        self._macro_bull = macro_bull
        self._lr_buf = tail_lr.copy()

        rv = self._compute_vol_from_buf()
        active = (n - 1 >= self.warmup_bars) and bull and (macro_bull if has_macro else True)
        return active, rv, self._format_reason(active, rv)

    def _step_one(self, p0: float, *, last_ts: object, raw_len: int) -> tuple[bool, float, str]:
        j = self._valid_count  # 0-based index of this new valid bar
        self._ef = self._alpha_fast * p0 + (1.0 - self._alpha_fast) * self._ef
        self._es = self._alpha_slow * p0 + (1.0 - self._alpha_slow) * self._es
        has_macro = self.macro_ema_window is not None
        if has_macro:
            self._em = self._alpha_macro * p0 + (1.0 - self._alpha_macro) * self._em

        if j >= self.slow_window:
            if has_macro:
                if not self._macro_bull:
                    if p0 > self._em * (1.0 + self.macro_hysteresis_pct):
                        self._macro_bull = True
                elif p0 < self._em * (1.0 - self.macro_hysteresis_pct):
                    self._macro_bull = False
            if not self._bull:
                if p0 > self._es * (1.0 + self.trend_hysteresis_pct) and self._ef > self._es:
                    self._bull = True
            elif p0 < self._es * (1.0 - self.trend_hysteresis_pct) and self._ef < self._es:
                self._bull = False

        new_lr = math.log(p0 / self._last_price) if self._last_price > 0 else 0.0
        if len(self._lr_buf) < self.vol_lookback_bars:
            self._lr_buf = np.append(self._lr_buf, new_lr)
        else:
            # Shift window left by 1 (720 floats -> ~1 microsecond, zero drift vs cold-start)
            self._lr_buf[:-1] = self._lr_buf[1:]
            self._lr_buf[-1] = new_lr

        self._valid_count += 1
        self._raw_len = raw_len
        self._last_ts = last_ts
        self._last_price = p0

        rv = self._compute_vol_from_buf()
        active = (
            (j >= self.warmup_bars)
            and self._bull
            and (self._macro_bull if has_macro else True)
        )
        return active, rv, self._format_reason(active, rv)


class PortfolioAllocator:
    """Active multi-asset trend-filtered Inverse-Volatility Risk Parity allocator."""

    def __init__(self, config: AppConfig, calendar: MarketCalendar) -> None:
        self.config = config
        self.calendar = calendar
        bars_per_year = calendar.bars_per_year(config.data.timeframe)
        warmup = (
            config.portfolio.warmup_bars
            if config.portfolio.warmup_bars is not None
            else config.strategy.trend_slow_window
        )
        self._trackers: dict[str, _AssetSignalTracker] = {
            inst.id: _AssetSignalTracker(
                fast_window=config.strategy.trend_fast_window,
                slow_window=config.strategy.trend_slow_window,
                trend_hysteresis_pct=config.strategy.trend_hysteresis_pct,
                macro_ema_window=config.strategy.macro_ema_window,
                macro_hysteresis_pct=config.strategy.macro_hysteresis_pct,
                vol_lookback_bars=config.portfolio.vol_lookback_bars,
                vol_min_bars=config.portfolio.vol_min_bars,
                bars_per_year=bars_per_year,
                warmup_bars=warmup,
            )
            for inst in config.instruments
        }

    def _price_series_array(self, inst: Instrument, candles: pd.DataFrame) -> np.ndarray:
        use_adj = (
            inst.asset_class in {AssetClass.EQUITY, AssetClass.ETF}
            and "adj_close" in candles.columns
            and self.config.data.price_column == "adj_close"
        )
        col = "adj_close" if use_adj else (
            self.config.data.price_column if self.config.data.price_column in candles.columns else "close"
        )
        return np.asarray(candles[col].to_numpy(dtype=np.float64), dtype=np.float64)

    def evaluate(
        self,
        candles_by_id: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        quotes: dict[str, Quote],
    ) -> AllocationDecision:
        """Compute active signals, realized volatilities, target weights, and rebalance flag."""
        pf_cfg = self.config.portfolio
        max_cap = min(pf_cfg.max_weight_per_asset, self.config.risk.max_exposure_pct)
        max_total = min(pf_cfg.max_total_exposure, self.config.risk.max_exposure_pct)

        active_signals: dict[str, bool] = {}
        realized_vols: dict[str, float] = {}
        reasons: dict[str, str] = {}
        current_weights: dict[str, float] = {}

        equity = portfolio.equity
        for inst in self.config.instruments:
            df = candles_by_id[inst.id]
            prices = self._price_series_array(inst, df)
            last_ts = df.index[-1] if len(df) > 0 else None
            is_active, rv, reason = self._trackers[inst.id].evaluate(
                prices=prices,
                last_ts=last_ts,
                raw_len=len(df),
            )
            active_signals[inst.id] = is_active
            realized_vols[inst.id] = rv
            reasons[inst.id] = reason

            qty = portfolio.position_qty(inst.id)
            px = quotes[inst.id].last if inst.id in quotes else 0.0
            current_weights[inst.id] = (qty * px / equity) if equity > 0 else 0.0

        active_ids = [inst.id for inst in self.config.instruments if active_signals[inst.id]]
        target_weights: dict[str, float] = {inst.id: 0.0 for inst in self.config.instruments}

        if active_ids:
            if len(self.config.instruments) == 1:
                target_weights[active_ids[0]] = max_cap
            elif pf_cfg.allocation_method == "equal_weight":
                w_eq = min(max_cap, max_total / len(active_ids))
                for inst_id in active_ids:
                    target_weights[inst_id] = w_eq
            else:
                inv = {
                    inst_id: 1.0 / max(realized_vols[inst_id], pf_cfg.vol_floor)
                    for inst_id in active_ids
                    if math.isfinite(realized_vols[inst_id])
                }
                inv_sum = sum(inv.values())
                if inv_sum > 0:
                    for inst_id, inv_v in inv.items():
                        target_weights[inst_id] = min(max_cap, max_total * inv_v / inv_sum)

            # Enforce total portfolio exposure ceiling defensively
            tot_w = sum(target_weights.values())
            if tot_w > max_total and tot_w > 0:
                scale = max_total / tot_w
                target_weights = {k: v * scale for k, v in target_weights.items()}

        rebalance_triggered = any(
            (target_weights[inst.id] <= 1e-6 and current_weights[inst.id] > pf_cfg.exit_weight_threshold)
            or (
                current_weights[inst.id] <= pf_cfg.exit_weight_threshold
                and target_weights[inst.id] > pf_cfg.entry_min_weight
            )
            or abs(target_weights[inst.id] - current_weights[inst.id]) >= pf_cfg.rebalance_band
            for inst in self.config.instruments
        )

        return AllocationDecision(
            active_signals=active_signals,
            realized_vols=realized_vols,
            target_weights=target_weights,
            current_weights=current_weights,
            rebalance_triggered=rebalance_triggered,
            reasons=reasons,
        )


def _dummy_forecast(inst: Instrument, ts: datetime, spot: float) -> ForecastResult:
    flat = np.array([spot], dtype=np.float64)
    return ForecastResult(
        instrument_id=inst.id,
        model_id="portfolio/ema_inv_vol",
        timestamp=ts,
        horizon=1,
        quantiles={0.05: flat, 0.10: flat, 0.50: flat, 0.90: flat, 0.95: flat},
        metadata={"portfolio_rule": True},
    )


def _dummy_signal(spot: float) -> SignalMetrics:
    return SignalMetrics(
        signal_mode="portfolio_ema_inv_vol",
        signal_value=0.0,
        expected_return=0.0,
        prob_positive=0.5,
        quantile_asymmetry=1.0,
        uncertainty_spread=0.0,
        downside_tail_return=0.0,
        current_price=spot,
        target_price_median=spot,
    )


class PortfolioStepRunner:
    """Executes one atomic multi-asset rebalance cycle identically in backtest, paper, dry-run, and live."""

    def __init__(
        self,
        *,
        config: AppConfig,
        allocator: PortfolioAllocator,
        risk_manager: RiskManager,
        broker: Broker,
        store: SQLiteStateStore | None = None,
        notifier: Notifier | None = None,
    ) -> None:
        self.config = config
        self.allocator = allocator
        self.risk_manager = risk_manager
        self.broker = broker
        self.store = store
        self.notifier = notifier

    def _quote_from_candles(self, inst: Instrument, candles: pd.DataFrame) -> Quote:
        col = self.config.data.price_column if self.config.data.price_column in candles.columns else "close"
        px = float(candles[col].iloc[-1])
        ts = ensure_utc(candles.index[-1])
        half_spread = px * (self.config.risk.spread_bps / 20_000.0)
        return Quote(
            instrument_id=inst.id,
            timestamp=ts,
            bid=max(1e-9, px - half_spread),
            ask=px + half_spread,
            last=px,
        )

    def _get_quotes(self, candles_by_id: dict[str, pd.DataFrame]) -> dict[str, Quote]:
        inner_broker = getattr(self.broker, "_inner", self.broker)
        if isinstance(inner_broker, LocalPaperBroker):
            seeded = {
                inst.id: self._quote_from_candles(inst, candles_by_id[inst.id])
                for inst in self.config.instruments
            }
            inner_broker.set_quotes(seeded)

        # Use batch ticker query if broker exposes get_quotes (e.g., BitfinexBroker)
        batch_fn = getattr(inner_broker, "get_quotes", None)
        if callable(batch_fn):
            quotes_map: dict[str, Quote] = batch_fn(self.config.instruments)
            return quotes_map

        return {inst.id: self.broker.get_quote(inst) for inst in self.config.instruments}

    def _get_multi_portfolio(self, quotes: dict[str, Quote]) -> PortfolioState:
        fn = getattr(self.broker, "get_multi_portfolio", None)
        if callable(fn):
            pf: PortfolioState = fn(self.config.instruments, quotes=quotes)
            return pf
        first = self.config.instruments[0]
        return self.broker.get_portfolio(first, quote=quotes.get(first.id))

    def _merge_persisted_risk_state(self, portfolio: PortfolioState) -> PortfolioState:
        if self.store is None:
            return portfolio
        persisted_hwm = self.store.get_high_water_mark(
            self.config.name,
            default=portfolio.high_water_mark,
        )
        persisted_halted = self.store.is_halted(self.config.name)
        hwm = max(portfolio.high_water_mark, persisted_hwm, portfolio.equity)
        halted = portfolio.halted or persisted_halted
        if hwm == portfolio.high_water_mark and halted == portfolio.halted:
            return portfolio
        return portfolio.model_copy(update={"high_water_mark": hwm, "halted": halted})

    def _weights_from_portfolio(self, pf: PortfolioState, quotes: dict[str, Quote]) -> dict[str, float]:
        if pf.equity <= 0:
            return {inst.id: 0.0 for inst in self.config.instruments}
        out: dict[str, float] = {}
        for inst in self.config.instruments:
            pos = pf.positions.get(inst.id)
            val = (
                pos.quantity * quotes[inst.id].last
                if (pos is not None and inst.id in quotes)
                else (pos.market_value if pos is not None else 0.0)
            )
            out[inst.id] = val / pf.equity
        return out

    def run_step(
        self,
        candles_by_id: dict[str, pd.DataFrame],
        *,
        now: datetime | None = None,
        check_staleness: bool = True,
        enforce_idempotency: bool = True,
        force: bool = False,
        persist: bool = True,
        notify: bool = True,
    ) -> PortfolioStepOutcome:
        """Run one atomic multi-asset cycle: signals -> Inv-Vol weights -> risk -> SELLs -> BUYs -> audit."""
        for inst in self.config.instruments:
            df = candles_by_id.get(inst.id)
            if df is None or df.empty:
                raise ValueError(f"PortfolioStepRunner missing candles for '{inst.id}'")

        # Canonical step timestamp is the oldest latest-bar timestamp across the basket (no lookahead)
        candle_ts = min(ensure_utc(candles_by_id[inst.id].index[-1]) for inst in self.config.instruments)
        eval_now = ensure_utc(now) if now is not None else datetime.now(UTC)

        # 1. Atomic portfolio idempotency guard
        if (
            enforce_idempotency
            and not force
            and self.store is not None
            and self.store.has_processed_portfolio(
                self.config.name,
                self.config.data.timeframe,
                candle_ts,
            )
        ):
            logger.info(
                "Idempotent portfolio skip: %s (%s @ %s) already processed.",
                self.config.name,
                self.config.data.timeframe,
                candle_ts.isoformat(),
            )
            return PortfolioStepOutcome(
                config_name=self.config.name,
                timeframe=self.config.data.timeframe,
                candle_timestamp=candle_ts,
                skipped_idempotent=True,
            )

        # 2. Fetch live/simulated quotes and full multi-asset portfolio valuation
        quotes = self._get_quotes(candles_by_id)
        portfolio_before = self._merge_persisted_risk_state(self._get_multi_portfolio(quotes))

        # 3. Evaluate causal trend signals + Inverse-Volatility Risk Parity target weights
        decision = self.allocator.evaluate(
            candles_by_id=candles_by_id,
            portfolio=portfolio_before,
            quotes=quotes,
        )

        # Evaluate drawdown circuit breaker early
        hwm_now = max(portfolio_before.high_water_mark, portfolio_before.equity)
        dd_now = max(0.0, 1.0 - (portfolio_before.equity / hwm_now)) if hwm_now > 0 else 0.0
        is_halted = portfolio_before.halted or (dd_now >= self.config.risk.max_drawdown_pct)

        per_asset_outcomes: dict[str, StepOutcome] = {}
        executed_fills: list[FillResult] = []
        rejected_orders: list[RejectedOrder] = []
        min_notional_rebal = self.config.portfolio.min_trade_notional

        if decision.rebalance_triggered:
            working_pf = portfolio_before.model_copy(update={"high_water_mark": hwm_now, "halted": is_halted})
            projected_cash = working_pf.cash

            # Phase 1: Execute all SELL / reduction legs FIRST to free quote cash cleanly
            for inst in self.config.instruments:
                df = candles_by_id[inst.id]
                q = quotes[inst.id]
                cur_qty = working_pf.position_qty(inst.id)
                cur_val = cur_qty * q.last
                tgt_w = decision.target_weights[inst.id]
                tgt_val = tgt_w * working_pf.equity

                if cur_qty <= 0 or not (tgt_w <= 1e-6 or tgt_val < cur_val - min_notional_rebal):
                    continue

                sig = _dummy_signal(q.last)
                fc = _dummy_forecast(inst, candle_ts, q.last)
                snap = MarketSnapshot(instrument=inst, timestamp=candle_ts, candles=df, quote=q)
                intent = OrderIntent(
                    instrument_id=inst.id,
                    timestamp=candle_ts,
                    side=OrderSide.SELL,
                    target_weight=tgt_w,
                    target_fraction=1.0 if tgt_w <= 1e-6 else max(0.0, min(1.0, (cur_val - tgt_val) / max(cur_val, 1e-9))),
                    signal=sig,
                    reason=f"Portfolio SELL -> {tgt_w:.1%} ({decision.reasons[inst.id]})",
                )
                approved, rejected, leg_halted = self.risk_manager.evaluate(
                    instrument=inst,
                    snapshot=snap,
                    portfolio=working_pf,
                    forecast=fc,
                    intent=intent,
                    now=eval_now,
                    check_staleness=check_staleness,
                )
                is_halted = is_halted or leg_halted
                fill: FillResult | None = None
                if approved is not None:
                    fill = self.broker.submit_order(approved, inst)
                    executed_fills.append(fill)
                    if fill.status == OrderStatus.FILLED:
                        working_pf = self._merge_persisted_risk_state(self._get_multi_portfolio(quotes))
                        working_pf = working_pf.model_copy(update={"halted": working_pf.halted or is_halted})
                        projected_cash = working_pf.cash
                    elif fill.status in {OrderStatus.PENDING, OrderStatus.DRY_RUN}:
                        projected_cash += max(0.0, approved.estimated_notional - approved.estimated_fee)
                elif rejected is not None:
                    rejected_orders.append(rejected)

                per_asset_outcomes[inst.id] = StepOutcome(
                    config_name=self.config.name,
                    instrument_id=inst.id,
                    timeframe=self.config.data.timeframe,
                    candle_timestamp=candle_ts,
                    intent=intent,
                    approved_order=approved,
                    rejected_order=rejected,
                    fill=fill,
                )

            # Phase 2: Execute all BUY / increase legs using available/projected cash
            remaining_cash = projected_cash
            for inst in self.config.instruments:
                if inst.id in per_asset_outcomes:
                    continue
                df = candles_by_id[inst.id]
                q = quotes[inst.id]
                cur_qty = working_pf.position_qty(inst.id)
                cur_val = cur_qty * q.last
                tgt_w = decision.target_weights[inst.id]
                tgt_val = tgt_w * working_pf.equity

                if tgt_val <= cur_val + min_notional_rebal:
                    continue

                sig = _dummy_signal(q.last)
                fc = _dummy_forecast(inst, candle_ts, q.last)
                snap = MarketSnapshot(instrument=inst, timestamp=candle_ts, candles=df, quote=q)
                intent = OrderIntent(
                    instrument_id=inst.id,
                    timestamp=candle_ts,
                    side=OrderSide.BUY,
                    target_weight=tgt_w,
                    target_fraction=tgt_w,
                    signal=sig,
                    reason=f"Portfolio BUY -> {tgt_w:.1%} ({decision.reasons[inst.id]})",
                )
                buy_pf = working_pf.model_copy(
                    update={"cash": max(0.0, remaining_cash), "halted": is_halted}
                )
                approved, rejected, leg_halted = self.risk_manager.evaluate(
                    instrument=inst,
                    snapshot=snap,
                    portfolio=buy_pf,
                    forecast=fc,
                    intent=intent,
                    now=eval_now,
                    check_staleness=check_staleness,
                )
                is_halted = is_halted or leg_halted
                fill = None
                if approved is not None:
                    fill = self.broker.submit_order(approved, inst)
                    executed_fills.append(fill)
                    if fill.status == OrderStatus.FILLED:
                        working_pf = self._merge_persisted_risk_state(self._get_multi_portfolio(quotes))
                        working_pf = working_pf.model_copy(update={"halted": working_pf.halted or is_halted})
                        remaining_cash = working_pf.cash
                    elif fill.status in {OrderStatus.PENDING, OrderStatus.DRY_RUN}:
                        remaining_cash = max(
                            0.0,
                            remaining_cash - (approved.estimated_notional + approved.estimated_fee),
                        )
                elif rejected is not None:
                    rejected_orders.append(rejected)

                per_asset_outcomes[inst.id] = StepOutcome(
                    config_name=self.config.name,
                    instrument_id=inst.id,
                    timeframe=self.config.data.timeframe,
                    candle_timestamp=candle_ts,
                    intent=intent,
                    approved_order=approved,
                    rejected_order=rejected,
                    fill=fill,
                )

        # Record HOLD intents for assets without orders so audit trail is complete
        for inst in self.config.instruments:
            if inst.id not in per_asset_outcomes:
                q = quotes[inst.id]
                hold_intent = OrderIntent(
                    instrument_id=inst.id,
                    timestamp=candle_ts,
                    side=OrderSide.HOLD,
                    target_weight=decision.target_weights[inst.id],
                    target_fraction=0.0,
                    signal=_dummy_signal(q.last),
                    reason=f"HOLD @ {decision.current_weights[inst.id]:.1%} ({decision.reasons[inst.id]})",
                )
                per_asset_outcomes[inst.id] = StepOutcome(
                    config_name=self.config.name,
                    instrument_id=inst.id,
                    timeframe=self.config.data.timeframe,
                    candle_timestamp=candle_ts,
                    intent=hold_intent,
                )

        # 4. Refresh post-step portfolio snapshot and update HWM / halted state
        portfolio_after = self._get_multi_portfolio(quotes)
        new_hwm = max(portfolio_before.high_water_mark, portfolio_after.high_water_mark, portfolio_after.equity)
        portfolio_after = portfolio_after.model_copy(
            update={
                "high_water_mark": new_hwm,
                "halted": portfolio_after.halted or is_halted,
            }
        )
        weights_after = self._weights_from_portfolio(portfolio_after, quotes)

        outcome = PortfolioStepOutcome(
            config_name=self.config.name,
            timeframe=self.config.data.timeframe,
            candle_timestamp=candle_ts,
            skipped_idempotent=False,
            active_signals=decision.active_signals,
            realized_vols=decision.realized_vols,
            target_weights=decision.target_weights,
            current_weights_before=decision.current_weights,
            current_weights_after=weights_after,
            rebalance_triggered=decision.rebalance_triggered,
            outcomes=per_asset_outcomes,
            fills=executed_fills,
            rejected_orders=rejected_orders,
            portfolio_before=portfolio_before,
            portfolio_after=portfolio_after,
        )

        # 5. SQLite persistence
        if persist and self.store is not None:
            self.store.record_portfolio_step(outcome)

        # 6. Operational notifications
        if notify and self.notifier is not None:
            notify_pf_fn = getattr(self.notifier, "notify_portfolio_step", None)
            if callable(notify_pf_fn):
                notify_pf_fn(outcome, self.config.instruments)
            if is_halted and not portfolio_before.halted:
                self.notifier.notify_alert(
                    title="PORTFOLIO DRAWDOWN CIRCUIT BREAKER HALTED",
                    message=(
                        f"Portfolio drawdown reached {portfolio_after.drawdown:.2%} "
                        f"(limit {self.config.risk.max_drawdown_pct:.2%}). "
                        f"Equity: {portfolio_after.equity:,.2f} {portfolio_after.quote_currency}."
                    ),
                    instrument=None,
                )

        return outcome


@dataclass
class PortfolioBacktestResult:
    """Complete multi-asset backtest outputs: equity, per-asset B&H, weights, fills, and metrics."""

    config_name: str
    instruments: list[Instrument]
    benchmark_id: str
    timeframe: str
    equity_df: pd.DataFrame
    fills: list[FillResult]
    metrics: dict[str, Any]
    asset_bh_rois: dict[str, float]
    yearly_returns: dict[int, float]


def align_multi_asset_candles(
    candles_by_id: dict[str, pd.DataFrame],
    instruments: list[Instrument],
    benchmark_id: str,
) -> dict[str, pd.DataFrame]:
    """Align multi-asset OHLCV DataFrames onto benchmark UTC timeline strictly causally (ffill only)."""
    base_idx = candles_by_id[benchmark_id].index
    aligned: dict[str, pd.DataFrame] = {}
    for inst in instruments:
        df = candles_by_id[inst.id].reindex(base_idx).ffill()
        aligned[inst.id] = df
    return aligned


class PortfolioBacktestRunner:
    """Historical multi-asset backtest harness using the EXACT shared PortfolioStepRunner."""

    def __init__(
        self,
        *,
        config: AppConfig,
        calendar: MarketCalendar,
    ) -> None:
        self.config = config
        self.calendar = calendar

    def run(
        self,
        candles_by_id: dict[str, pd.DataFrame],
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        initial_cash: float | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> PortfolioBacktestResult:
        """Simulate bar-by-bar over [start, end] using SimulatedBroker + PortfolioStepRunner."""
        instruments = self.config.instruments
        benchmark = self.config.benchmark_instrument
        aligned = align_multi_asset_candles(candles_by_id, instruments, benchmark.id)
        common_idx = aligned[benchmark.id].index

        if len(common_idx) == 0:
            raise ValueError("Cannot run portfolio backtest on empty candles")

        start_ts = pd.Timestamp(ensure_utc(start)) if start is not None else common_idx[0]
        end_ts = pd.Timestamp(ensure_utc(end)) if end is not None else common_idx[-1]
        eval_indices = [i for i, ts in enumerate(common_idx) if start_ts <= ts <= end_ts]
        if not eval_indices:
            raise ValueError(
                f"No evaluable bars in [{start_ts}, {end_ts}] "
                f"(available: {common_idx.min()} .. {common_idx.max()}, rows={len(common_idx)})"
            )

        cash0 = float(initial_cash if initial_cash is not None else self.config.broker.initial_cash)
        broker = SimulatedBroker(
            initial_cash=cash0,
            quote_currency=benchmark.quote_currency,
            fee_model=self.config.risk.fee_model,
            slippage_bps=self.config.risk.slippage_bps,
            spread_bps=self.config.risk.spread_bps,
            execution_price=self.config.broker.execution_price,
            max_drawdown_pct=self.config.risk.max_drawdown_pct,
        )
        allocator = PortfolioAllocator(self.config, self.calendar)
        risk_manager = RiskManager(config=self.config.risk, calendar=self.calendar)
        step_runner = PortfolioStepRunner(
            config=self.config,
            allocator=allocator,
            risk_manager=risk_manager,
            broker=broker,
            store=None,
            notifier=None,
        )

        # Pre-extract numpy arrays for fast bar construction (still slicing causal DataFrames for step_runner)
        price_col = self.config.data.price_column
        col_map: dict[str, dict[str, np.ndarray]] = {}
        bh_start: dict[str, float] = {}
        first_idx = eval_indices[0]
        for inst in instruments:
            df = aligned[inst.id]
            pcol = price_col if price_col in df.columns else "close"
            c_arr = df[pcol].to_numpy(dtype=np.float64)
            col_map[inst.id] = {
                "open": df["open"].to_numpy(dtype=np.float64),
                "high": df["high"].to_numpy(dtype=np.float64),
                "low": df["low"].to_numpy(dtype=np.float64),
                "close": c_arr,
                "volume": df["volume"].to_numpy(dtype=np.float64),
            }
            start_px = float(c_arr[first_idx])
            bh_start[inst.id] = start_px if (math.isfinite(start_px) and start_px > 0) else 1.0

        total_bars = len(eval_indices)
        if progress_callback is not None:
            progress_callback("simulate", 0, total_bars)

        records: list[dict[str, Any]] = []
        for step_num, idx in enumerate(eval_indices, start=1):
            ts = ensure_utc(common_idx[idx])
            bars_now: dict[str, Candle] = {}
            slices_now: dict[str, pd.DataFrame] = {}
            for inst in instruments:
                arrs = col_map[inst.id]
                c_val = float(arrs["close"][idx])
                o_val = float(arrs["open"][idx])
                h_val = float(arrs["high"][idx])
                l_val = float(arrs["low"][idx])
                v_val = float(arrs["volume"][idx])
                if not math.isfinite(c_val) or c_val <= 0:
                    c_val = bh_start[inst.id]
                if not math.isfinite(o_val) or o_val <= 0:
                    o_val = c_val
                bars_now[inst.id] = Candle(
                    timestamp=ts,
                    open=o_val,
                    high=h_val if math.isfinite(h_val) else max(o_val, c_val),
                    low=l_val if math.isfinite(l_val) else min(o_val, c_val),
                    close=c_val,
                    volume=max(0.0, v_val) if math.isfinite(v_val) else 0.0,
                    adj_close=c_val,
                )
                slices_now[inst.id] = aligned[inst.id].iloc[: idx + 1]

            # Execute pending next_open orders (SELLs before BUYs) at open[t]
            broker.on_new_bars(bars_now, instruments)

            # Run the EXACT shared PortfolioStepRunner on causal slices [:idx+1]
            outcome = step_runner.run_step(
                candles_by_id=slices_now,
                now=ts,
                check_staleness=False,
                enforce_idempotency=False,
                persist=False,
                notify=False,
            )

            pf = outcome.portfolio_after or broker.get_multi_portfolio(instruments)
            rec: dict[str, Any] = {
                "timestamp": ts,
                "equity": pf.equity,
                "cash": pf.cash,
                "high_water_mark": pf.high_water_mark,
                "drawdown": pf.drawdown,
                "w_cash": (pf.cash / pf.equity) if pf.equity > 0 else 1.0,
                "halted": pf.halted,
            }
            for inst in instruments:
                px = bars_now[inst.id].close
                qty = pf.position_qty(inst.id)
                bh_eq = cash0 * (px / bh_start[inst.id])
                rec[f"price_{inst.id}"] = px
                rec[f"qty_{inst.id}"] = qty
                rec[f"w_{inst.id}"] = (qty * px / pf.equity) if pf.equity > 0 else 0.0
                rec[f"bh_{inst.id}"] = bh_eq
            rec["buy_and_hold_equity"] = rec[f"bh_{benchmark.id}"]
            records.append(rec)

            if progress_callback is not None and (step_num % 100 == 0 or step_num == total_bars):
                progress_callback("simulate", step_num, total_bars)

        equity_df = pd.DataFrame(records).set_index("timestamp")
        equity_df.index = pd.to_datetime(equity_df.index, utc=True)

        metrics = compute_performance_metrics(
            equity_df=equity_df,
            fills=broker.fills,
            calendar=self.calendar,
            timeframe=self.config.data.timeframe,
        )

        # Per-asset Buy & Hold ROI
        asset_bh_rois: dict[str, float] = {}
        for inst in instruments:
            col = f"bh_{inst.id}"
            s = equity_df[col]
            asset_bh_rois[inst.id] = float(s.iloc[-1] / s.iloc[0] - 1.0) if len(s) > 0 and s.iloc[0] > 0 else 0.0

        # Yearly compounded returns breakdown
        yearly_returns: dict[int, float] = {}
        eq_series = equity_df["equity"]
        for yr, grp in eq_series.groupby(eq_series.index.year):
            if len(grp) >= 2 and float(grp.iloc[0]) > 0:
                yearly_returns[int(yr)] = float(grp.iloc[-1] / grp.iloc[0] - 1.0)

        return PortfolioBacktestResult(
            config_name=self.config.name,
            instruments=list(instruments),
            benchmark_id=benchmark.id,
            timeframe=self.config.data.timeframe,
            equity_df=equity_df,
            fills=list(broker.fills),
            metrics=metrics,
            asset_bh_rois=asset_bh_rois,
            yearly_returns=yearly_returns,
        )
