"""Modular position sizers and distribution-first QuantileStrategy implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from universal_trader.core.config import StrategyConfig
from universal_trader.core.models import (
    AssetClass,
    ForecastResult,
    MarketSnapshot,
    OrderIntent,
    OrderSide,
    PortfolioState,
    SignalMetrics,
)
from universal_trader.core.registry import register_strategy
from universal_trader.strategies.signals import SignalExtractor


class QuantileStrategy(ABC):
    """Base strategy turning a ForecastResult + MarketSnapshot + PortfolioState into OrderIntent."""

    name: str = "base"

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.extractor = SignalExtractor(
            signal_mode=config.signal,
            tail_quantile=config.tail_quantile,
            theta_buy=config.theta_buy,
        )

    @abstractmethod
    def compute_target_weight(
        self,
        signal: SignalMetrics,
        current_weight: float,
    ) -> tuple[OrderSide, float, float, str]:
        """Return (side, target_weight, target_fraction, reason)."""

    def evaluate(
        self,
        forecast: ForecastResult,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> OrderIntent:
        price = snapshot.quote.last
        signal = self.extractor.extract(forecast, current_price=price)
        current_val = portfolio.position_value(snapshot.instrument.id)
        current_weight = (current_val / portfolio.equity) if portfolio.equity > 0 else 0.0

        side, target_weight, target_fraction, reason = self.compute_target_weight(
            signal=signal,
            current_weight=current_weight,
        )
        target_weight = float(min(1.0, max(0.0, target_weight)))
        target_fraction = float(min(1.0, max(0.0, target_fraction)))

        return OrderIntent(
            instrument_id=snapshot.instrument.id,
            timestamp=snapshot.timestamp,
            side=side,
            target_weight=target_weight,
            target_fraction=target_fraction,
            signal=signal,
            reason=reason,
        )


@register_strategy("binary")
class BinarySizer(QuantileStrategy):
    """100% allocation on buy signal, 100% liquidation on sell signal."""

    name = "binary"

    def compute_target_weight(
        self,
        signal: SignalMetrics,
        current_weight: float,
    ) -> tuple[OrderSide, float, float, str]:
        if signal.signal_value >= self.config.theta_buy:
            if current_weight >= 0.99:
                return OrderSide.HOLD, current_weight, 0.0, "Already fully allocated"
            return (
                OrderSide.BUY,
                1.0,
                1.0,
                f"Binary BUY: signal {signal.signal_value:.4%} >= {self.config.theta_buy:.4%}",
            )
        if signal.signal_value <= self.config.theta_sell:
            if current_weight <= 1e-6:
                return OrderSide.HOLD, 0.0, 0.0, "No position to liquidate"
            return (
                OrderSide.SELL,
                0.0,
                1.0,
                f"Binary SELL: signal {signal.signal_value:.4%} <= {self.config.theta_sell:.4%}",
            )
        return OrderSide.HOLD, current_weight, 0.0, "Signal inside deadband"


@register_strategy("fractional")
class FractionalSizer(QuantileStrategy):
    """Fixed fraction incremental accumulation / distribution (e.g., 50% or 25%)."""

    name = "fractional"

    def compute_target_weight(
        self,
        signal: SignalMetrics,
        current_weight: float,
    ) -> tuple[OrderSide, float, float, str]:
        frac = self.config.fraction
        if signal.signal_value >= self.config.theta_buy:
            target_w = min(1.0, current_weight + (1.0 - current_weight) * frac)
            if target_w - current_weight <= 1e-4:
                return OrderSide.HOLD, current_weight, 0.0, "Already at full allocation"
            return (
                OrderSide.BUY,
                target_w,
                frac,
                f"Fractional BUY ({frac:.0%}): signal {signal.signal_value:.4%}",
            )
        if signal.signal_value <= self.config.theta_sell:
            if current_weight <= 1e-6:
                return OrderSide.HOLD, 0.0, 0.0, "No position to reduce"
            target_w = max(0.0, current_weight * (1.0 - frac))
            return (
                OrderSide.SELL,
                target_w,
                frac,
                f"Fractional SELL ({frac:.0%}): signal {signal.signal_value:.4%}",
            )
        return OrderSide.HOLD, current_weight, 0.0, "Signal inside deadband"


@register_strategy("proportional")
class ProportionalSizer(QuantileStrategy):
    """Target allocation proportional to expected return strength."""

    name = "proportional"

    def compute_target_weight(
        self,
        signal: SignalMetrics,
        current_weight: float,
    ) -> tuple[OrderSide, float, float, str]:
        if signal.signal_value >= self.config.theta_buy:
            desired = min(1.0, max(0.0, signal.signal_value / self.config.proportional_scale))
            if desired > current_weight + 1e-3:
                return (
                    OrderSide.BUY,
                    desired,
                    desired,
                    f"Proportional BUY to {desired:.1%} (signal {signal.signal_value:.4%})",
                )
            return OrderSide.HOLD, current_weight, 0.0, "Position already >= proportional target"
        if signal.signal_value <= self.config.theta_sell:
            if current_weight <= 1e-6:
                return OrderSide.HOLD, 0.0, 0.0, "No position to reduce"
            reduction = min(1.0, abs(signal.signal_value) / self.config.proportional_scale)
            desired = max(0.0, current_weight * (1.0 - reduction))
            return (
                OrderSide.SELL,
                desired,
                reduction,
                f"Proportional SELL to {desired:.1%} (signal {signal.signal_value:.4%})",
            )
        return OrderSide.HOLD, current_weight, 0.0, "Signal inside deadband"


@register_strategy("volatility_adjusted")
class VolatilityAdjustedSizer(QuantileStrategy):
    """Sizes exposure inversely to forecast uncertainty spread (q90 - q10) / price."""

    name = "volatility_adjusted"

    def compute_target_weight(
        self,
        signal: SignalMetrics,
        current_weight: float,
    ) -> tuple[OrderSide, float, float, str]:
        spread = max(1e-4, signal.uncertainty_spread)
        if signal.signal_value >= self.config.theta_buy:
            # Inverse uncertainty scaling modulated by k_risk and target_vol
            vol_scale = min(1.0, (self.config.target_vol / spread) * self.config.k_risk)
            desired = float(min(1.0, max(0.0, vol_scale)))
            if desired > current_weight + 1e-3:
                return (
                    OrderSide.BUY,
                    desired,
                    desired,
                    f"VolAdjusted BUY to {desired:.1%} (spread={spread:.2%}, sig={signal.signal_value:.2%})",
                )
            return OrderSide.HOLD, current_weight, 0.0, "Current weight meets vol-adjusted target"
        if signal.signal_value <= self.config.theta_sell:
            if current_weight <= 1e-6:
                return OrderSide.HOLD, 0.0, 0.0, "No position to sell"
            return (
                OrderSide.SELL,
                0.0,
                1.0,
                f"VolAdjusted SELL (sig={signal.signal_value:.2%} <= {self.config.theta_sell:.2%})",
            )
        return OrderSide.HOLD, current_weight, 0.0, "Signal inside deadband"


@register_strategy("dynamic_kelly")
class DynamicKellySizer(QuantileStrategy):
    """Fractional Kelly criterion using expected return and quantile-implied volatility."""

    name = "dynamic_kelly"

    def compute_target_weight(
        self,
        signal: SignalMetrics,
        current_weight: float,
    ) -> tuple[OrderSide, float, float, str]:
        sigma = max(1e-4, signal.raw_metrics.get("sigma_est", signal.uncertainty_spread / 2.5631))
        if signal.signal_value >= self.config.theta_buy and signal.expected_return > 0:
            kelly_raw = (signal.expected_return / (sigma * sigma)) * self.config.k_risk
            desired = float(min(1.0, max(0.0, kelly_raw)))
            if desired > current_weight + 1e-3:
                return (
                    OrderSide.BUY,
                    desired,
                    desired,
                    f"DynamicKelly BUY to {desired:.1%} (E[r]={signal.expected_return:.2%}, sigma={sigma:.2%})",
                )
            return OrderSide.HOLD, current_weight, 0.0, "Position already >= Kelly target"
        if signal.signal_value <= self.config.theta_sell:
            if current_weight <= 1e-6:
                return OrderSide.HOLD, 0.0, 0.0, "No position to exit"
            return (
                OrderSide.SELL,
                0.0,
                1.0,
                f"DynamicKelly SELL (sig={signal.signal_value:.2%})",
            )
        return OrderSide.HOLD, current_weight, 0.0, "Signal inside deadband"


@register_strategy("tail_risk_kelly")
class TailRiskKellySizer(QuantileStrategy):
    """Timestral 21-quantile sizer penalizing Kelly sizing by left-tail CVaR / q05 risk."""

    name = "tail_risk_kelly"

    def compute_target_weight(
        self,
        signal: SignalMetrics,
        current_weight: float,
    ) -> tuple[OrderSide, float, float, str]:
        sigma = max(1e-4, signal.raw_metrics.get("sigma_est", signal.uncertainty_spread / 2.5631))
        cvar_ret = signal.raw_metrics.get("cvar_return", signal.downside_tail_return)
        left_tail_loss = max(0.0, -min(cvar_ret, signal.downside_tail_return))

        if signal.signal_value >= self.config.theta_buy and signal.expected_return > 0:
            kelly_weight = (signal.expected_return / (sigma * sigma)) * self.config.k_risk
            # Penalize exposure when left tail loss exceeds tail_risk_budget
            tail_cap = (
                self.config.tail_risk_budget / left_tail_loss if left_tail_loss > 1e-6 else 1.0
            )
            # Also modulate mildly by quantile asymmetry when skew is heavily negative
            skew_factor = min(1.0, max(0.25, signal.quantile_asymmetry))
            desired = float(min(1.0, max(0.0, min(kelly_weight, tail_cap) * skew_factor)))

            if desired > current_weight + 1e-3:
                return (
                    OrderSide.BUY,
                    desired,
                    desired,
                    (
                        f"TailRiskKelly BUY to {desired:.1%} "
                        f"(kelly={kelly_weight:.2f}, tail_cap={tail_cap:.2f}, "
                        f"cvar={cvar_ret:.2%}, skew={signal.quantile_asymmetry:.2f})"
                    ),
                )
            if current_weight - desired > 0.10 and left_tail_loss > self.config.tail_risk_budget:
                return (
                    OrderSide.SELL,
                    desired,
                    (current_weight - desired) / max(current_weight, 1e-9),
                    f"TailRiskKelly de-risk to {desired:.1%} (elevated left tail {cvar_ret:.2%})",
                )
            return OrderSide.HOLD, current_weight, 0.0, "Position within TailRiskKelly target band"

        if signal.signal_value <= self.config.theta_sell:
            if current_weight <= 1e-6:
                return OrderSide.HOLD, 0.0, 0.0, "No position to exit"
            return (
                OrderSide.SELL,
                0.0,
                1.0,
                f"TailRiskKelly SELL (sig={signal.signal_value:.2%} <= {self.config.theta_sell:.2%})",
            )
        return OrderSide.HOLD, current_weight, 0.0, "Signal inside deadband"


@register_strategy("trend_quantile_risk")
class TrendQuantileRiskSizer(QuantileStrategy):
    """Hybrid strategy: causal EMA trend regime for direction + Timestral 21-quantile risk sizing.

    Uses causal fast/slow EMAs on observed candles to determine market regime (long vs cash),
    while delegating position sizing and tail-risk throttling to Timestral's calibrated
    quantile distribution (q10-q90 spread and q05 / CVaR left-tail risk).
    """

    name = "trend_quantile_risk"

    def compute_target_weight(
        self,
        signal: SignalMetrics,
        current_weight: float,
    ) -> tuple[OrderSide, float, float, str]:
        return OrderSide.HOLD, current_weight, 0.0, "Evaluated via snapshot-aware evaluate()"

    def evaluate(
        self,
        forecast: ForecastResult,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> OrderIntent:
        price = snapshot.quote.last
        signal = self.extractor.extract(forecast, current_price=price)
        current_val = portfolio.position_value(snapshot.instrument.id)
        current_weight = (current_val / portfolio.equity) if portfolio.equity > 0 else 0.0

        candles = snapshot.candles
        use_adj = (
            snapshot.instrument.asset_class in {AssetClass.EQUITY, AssetClass.ETF}
            and "adj_close" in candles.columns
        )
        price_col = "adj_close" if use_adj else "close"
        series = candles[price_col].dropna()

        fast_w = self.config.trend_fast_window
        slow_w = max(fast_w + 1, self.config.trend_slow_window)

        if len(series) < slow_w:
            return OrderIntent(
                instrument_id=snapshot.instrument.id,
                timestamp=snapshot.timestamp,
                side=OrderSide.HOLD,
                target_weight=current_weight,
                target_fraction=0.0,
                signal=signal,
                reason=f"Warmup: need {slow_w} bars for trend EMAs (have {len(series)})",
            )

        ema_fast = float(series.ewm(span=fast_w, adjust=False).mean().iloc[-1])
        ema_slow = float(series.ewm(span=slow_w, adjust=False).mean().iloc[-1])
        spot_bar = float(series.iloc[-1])
        uptrend = (ema_fast > ema_slow) and (spot_bar > ema_slow)

        if not uptrend:
            if current_weight <= 1e-6:
                return OrderIntent(
                    instrument_id=snapshot.instrument.id,
                    timestamp=snapshot.timestamp,
                    side=OrderSide.HOLD,
                    target_weight=0.0,
                    target_fraction=0.0,
                    signal=signal,
                    reason=f"Trend OFF (EMA{fast_w}={ema_fast:.2f} <= EMA{slow_w}={ema_slow:.2f})",
                )
            return OrderIntent(
                instrument_id=snapshot.instrument.id,
                timestamp=snapshot.timestamp,
                side=OrderSide.SELL,
                target_weight=0.0,
                target_fraction=1.0,
                signal=signal,
                reason=f"Trend EXIT (EMA{fast_w}={ema_fast:.2f}, EMA{slow_w}={ema_slow:.2f}, px={spot_bar:.2f})",
            )

        # Uptrend active -> size exposure using Timestral's calibrated volatility & left tail
        spread = max(1e-4, signal.uncertainty_spread)
        cvar_ret = signal.raw_metrics.get("cvar_return", signal.downside_tail_return)
        left_tail_loss = max(0.0, -min(cvar_ret, signal.downside_tail_return))

        vol_weight = min(1.0, (self.config.target_vol / spread) * self.config.k_risk)
        tail_weight = (
            min(1.0, self.config.tail_risk_budget / left_tail_loss)
            if left_tail_loss > 1e-6
            else 1.0
        )
        desired = float(min(1.0, max(0.15, min(vol_weight, tail_weight))))

        band = self.config.rebalance_band
        if current_weight <= 1e-6 or (desired - current_weight) >= band:
            return OrderIntent(
                instrument_id=snapshot.instrument.id,
                timestamp=snapshot.timestamp,
                side=OrderSide.BUY,
                target_weight=desired,
                target_fraction=desired,
                signal=signal,
                reason=(
                    f"TrendQuantileRisk BUY to {desired:.1%} "
                    f"(EMA{fast_w}>{ema_slow:.2f}, spread={spread:.2%}, tail={cvar_ret:.2%})"
                ),
            )
        if (current_weight - desired) >= band:
            red_frac = (current_weight - desired) / max(current_weight, 1e-9)
            return OrderIntent(
                instrument_id=snapshot.instrument.id,
                timestamp=snapshot.timestamp,
                side=OrderSide.SELL,
                target_weight=desired,
                target_fraction=float(min(1.0, max(0.0, red_frac))),
                signal=signal,
                reason=(
                    f"TrendQuantileRisk DE-RISK to {desired:.1%} "
                    f"(spread={spread:.2%}, tail={cvar_ret:.2%})"
                ),
            )
        return OrderIntent(
            instrument_id=snapshot.instrument.id,
            timestamp=snapshot.timestamp,
            side=OrderSide.HOLD,
            target_weight=current_weight,
            target_fraction=0.0,
            signal=signal,
            reason=f"Trend ON: weight {current_weight:.1%} within band of target {desired:.1%}",
        )


@register_strategy("ema_trend")
class EmaTrendStrategy(QuantileStrategy):
    """Causal EMA trend-following strategy with hysteresis (single pair or multi-pair ensemble).

    Works standalone (with `forecaster.type: noop`) and maintains exact parity between
    historical backtests and single-step live/paper runs.
    """

    name = "ema_trend"

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        if config.ema_pairs:
            self.pairs: list[tuple[int, int]] = [
                (int(fw), max(int(fw) + 1, int(sw))) for fw, sw in config.ema_pairs
            ]
        else:
            fw = config.trend_fast_window
            sw = max(fw + 1, config.trend_slow_window)
            self.pairs = [(fw, sw)]
        self._max_slow: int = max(sw for _, sw in self.pairs)
        self._macro_span: int | None = config.macro_ema_window
        # Incremental O(1) cache for fast sequential backtests (strictly causal)
        self._cache_len: int = 0
        self._cache_last_ts: object | None = None
        self._ema_cache: dict[int, float] = {}
        self._pair_states: list[bool] = [False] * len(self.pairs)
        self._macro_bull: bool = True

    def compute_target_weight(
        self,
        signal: SignalMetrics,
        current_weight: float,
    ) -> tuple[OrderSide, float, float, str]:
        return OrderSide.HOLD, current_weight, 0.0, "Evaluated via snapshot-aware evaluate()"

    def _unique_spans(self) -> list[int]:
        spans = {w for pair in self.pairs for w in pair}
        if self._macro_span is not None:
            spans.add(self._macro_span)
        return sorted(spans)

    def _step_single_bar(
        self,
        p0: float,
        raw_len: int,
        last_ts: object,
    ) -> tuple[dict[int, float], list[bool], bool]:
        hyst = self.config.trend_hysteresis_pct
        m_hyst = self.config.macro_hysteresis_pct
        for span in self._unique_spans():
            alpha = 2.0 / (span + 1.0)
            self._ema_cache[span] = alpha * p0 + (1.0 - alpha) * self._ema_cache[span]
        if self._macro_span is not None:
            em = self._ema_cache[self._macro_span]
            if not self._macro_bull:
                if p0 > em * (1.0 + m_hyst):
                    self._macro_bull = True
            else:
                if p0 < em * (1.0 - m_hyst):
                    self._macro_bull = False
        for k, (fw, sw) in enumerate(self.pairs):
            ef, es = self._ema_cache[fw], self._ema_cache[sw]
            if not self._pair_states[k]:
                if p0 > es * (1.0 + hyst) and ef > es:
                    self._pair_states[k] = True
            else:
                if p0 < es * (1.0 - hyst) and ef < es:
                    self._pair_states[k] = False
        self._cache_len = raw_len
        self._cache_last_ts = last_ts
        return self._ema_cache, list(self._pair_states), self._macro_bull

    def _compute_emas_and_states(
        self,
        prices: Any,
        last_ts: object,
        current_weight: float,
        raw_len: int,
    ) -> tuple[dict[int, float], list[bool], bool]:
        """Initialize EMAs and hysteresis states strictly causally from history."""
        import numpy as np

        n = len(prices)
        hyst = self.config.trend_hysteresis_pct
        m_hyst = self.config.macro_hysteresis_pct
        unique_spans = self._unique_spans()

        # Full causal initialization from scratch (used on cold start / live single-step)
        arr = np.asarray(prices, dtype=np.float64)
        # Keep sufficient causal tail for accurate EMA convergence
        macro_tail = (self._macro_span * 3) if self._macro_span is not None else 0
        tail_len = min(n, max(self._max_slow * 6, macro_tail, 128))
        arr = arr[-tail_len:]

        ema_series: dict[int, Any] = {}
        for span in unique_spans:
            alpha = 2.0 / (span + 1.0)
            out = np.empty(len(arr), dtype=np.float64)
            out[0] = arr[0]
            for j in range(1, len(arr)):
                out[j] = alpha * arr[j] + (1.0 - alpha) * out[j - 1]
            ema_series[span] = out
            self._ema_cache[span] = float(out[-1])

        macro_bull = True
        if self._macro_span is not None:
            em_arr = ema_series[self._macro_span]
            macro_bull = float(arr[0]) >= float(em_arr[0])
            start_m = min(self._max_slow, len(arr) - 1)
            for j in range(start_m, len(arr)):
                pj = float(arr[j])
                emj = float(em_arr[j])
                if not macro_bull:
                    if pj > emj * (1.0 + m_hyst):
                        macro_bull = True
                else:
                    if pj < emj * (1.0 - m_hyst):
                        macro_bull = False

        states: list[bool] = [False] * len(self.pairs)
        if len(self.pairs) == 1 and current_weight > 0.01:
            # On single-step live/paper resume, honor live portfolio position for hysteresis
            fw, sw = self.pairs[0]
            p0 = float(arr[-1])
            ef, es = self._ema_cache[fw], self._ema_cache[sw]
            states[0] = not (p0 < es * (1.0 - hyst) and ef < es)
        else:
            for k, (fw, sw) in enumerate(self.pairs):
                ef_arr = ema_series[fw]
                es_arr = ema_series[sw]
                st = False
                start_scan = min(sw, len(arr) - 1)
                for j in range(start_scan, len(arr)):
                    pj = float(arr[j])
                    if not st:
                        if pj > float(es_arr[j]) * (1.0 + hyst) and float(ef_arr[j]) > float(es_arr[j]):
                            st = True
                    else:
                        if pj < float(es_arr[j]) * (1.0 - hyst) and float(ef_arr[j]) < float(es_arr[j]):
                            st = False
                states[k] = st

        self._macro_bull = macro_bull
        self._pair_states = states
        self._cache_len = raw_len
        self._cache_last_ts = last_ts
        return self._ema_cache, list(states), macro_bull

    def evaluate(
        self,
        forecast: ForecastResult,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> OrderIntent:
        price = snapshot.quote.last
        signal = self.extractor.extract(forecast, current_price=price)
        current_val = portfolio.position_value(snapshot.instrument.id)
        current_weight = (current_val / portfolio.equity) if portfolio.equity > 0 else 0.0

        candles = snapshot.candles
        use_adj = (
            snapshot.instrument.asset_class in {AssetClass.EQUITY, AssetClass.ETF}
            and "adj_close" in candles.columns
        )
        price_col = "adj_close" if use_adj else "close"
        n_rows = len(candles)

        if n_rows < self._max_slow:
            return OrderIntent(
                instrument_id=snapshot.instrument.id,
                timestamp=snapshot.timestamp,
                side=OrderSide.HOLD,
                target_weight=current_weight,
                target_fraction=0.0,
                signal=signal,
                reason=f"Warmup: need {self._max_slow} bars for EMA (have {n_rows})",
            )

        last_ts = candles.index[-1]
        if (
            self._cache_len > 0
            and n_rows == self._cache_len + 1
            and self._cache_last_ts is not None
            and last_ts != self._cache_last_ts
        ):
            # O(1) strictly causal sequential update (avoids copying 50k+ rows in long 1h backtests)
            last_px = float(candles[price_col].iloc[-1])
            emas, states, macro_bull = self._step_single_bar(last_px, n_rows, last_ts)
        else:
            prices = candles[price_col].dropna().to_numpy()
            if len(prices) < self._max_slow:
                return OrderIntent(
                    instrument_id=snapshot.instrument.id,
                    timestamp=snapshot.timestamp,
                    side=OrderSide.HOLD,
                    target_weight=current_weight,
                    target_fraction=0.0,
                    signal=signal,
                    reason=f"Warmup: need {self._max_slow} valid bars for EMA (have {len(prices)})",
                )
            emas, states, macro_bull = self._compute_emas_and_states(
                prices=prices,
                last_ts=last_ts,
                current_weight=current_weight,
                raw_len=n_rows,
            )
        active = sum(states)
        total = len(self.pairs)
        base_target = self.config.target_weight_bull * (active / total)
        if self._macro_span is not None and not macro_bull:
            target_w = min(base_target, self.config.macro_bear_weight * (active / total))
        else:
            target_w = base_target
        diff = target_w - current_weight

        fw0, sw0 = self.pairs[0]
        macro_note = (
            f" | macroEMA({self._macro_span})={'BULL' if macro_bull else 'BEAR'}"
            if self._macro_span is not None
            else ""
        )
        summary = f"EMA({fw0}/{sw0})={emas[fw0]:.2f}/{emas[sw0]:.2f} | sleeves={active}/{total}{macro_note}"

        if target_w <= 1e-6:
            if current_weight <= 0.01:
                return OrderIntent(
                    instrument_id=snapshot.instrument.id,
                    timestamp=snapshot.timestamp,
                    side=OrderSide.HOLD,
                    target_weight=0.0,
                    target_fraction=0.0,
                    signal=signal,
                    reason=f"Bear regime — staying in cash ({summary})",
                )
            return OrderIntent(
                instrument_id=snapshot.instrument.id,
                timestamp=snapshot.timestamp,
                side=OrderSide.SELL,
                target_weight=0.0,
                target_fraction=1.0,
                signal=signal,
                reason=f"Trend EXIT to cash ({summary})",
            )

        band = self.config.rebalance_band
        if current_weight <= 0.01 or diff >= band:
            return OrderIntent(
                instrument_id=snapshot.instrument.id,
                timestamp=snapshot.timestamp,
                side=OrderSide.BUY,
                target_weight=target_w,
                target_fraction=target_w,
                signal=signal,
                reason=f"Trend BUY to {target_w:.1%} ({summary})",
            )
        if (-diff) >= band:
            red_frac = (-diff) / max(current_weight, 1e-9)
            return OrderIntent(
                instrument_id=snapshot.instrument.id,
                timestamp=snapshot.timestamp,
                side=OrderSide.SELL,
                target_weight=target_w,
                target_fraction=float(min(1.0, max(0.0, red_frac))),
                signal=signal,
                reason=f"Trend REDUCE to {target_w:.1%} ({summary})",
            )
        return OrderIntent(
            instrument_id=snapshot.instrument.id,
            timestamp=snapshot.timestamp,
            side=OrderSide.HOLD,
            target_weight=current_weight,
            target_fraction=0.0,
            signal=signal,
            reason=f"Bull HOLD at {current_weight:.1%} ({summary})",
        )


def build_strategy(config: StrategyConfig) -> QuantileStrategy:
    """Instantiate a registered Strategy from StrategyConfig."""
    from universal_trader.core.registry import registry

    cls: Any = registry.get("strategy", config.type)
    strat: QuantileStrategy = cls(config)
    return strat
