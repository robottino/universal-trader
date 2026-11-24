"""RiskManager implementing an ordered chain of 7 independent RiskGuards."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from universal_trader.core.calendar import MarketCalendar
from universal_trader.core.config import RiskConfig
from universal_trader.core.models import (
    ApprovedOrder,
    ForecastResult,
    Instrument,
    MarketSnapshot,
    OrderIntent,
    OrderSide,
    PortfolioState,
    RejectedOrder,
    ensure_utc,
)
from universal_trader.core.protocols import RiskGuard


@dataclass
class RiskContext:
    """Mutable working context passed sequentially through the RiskGuard chain."""

    instrument: Instrument
    snapshot: MarketSnapshot
    portfolio: PortfolioState
    forecast: ForecastResult
    intent: OrderIntent
    config: RiskConfig
    calendar: MarketCalendar
    now: datetime
    check_staleness: bool = True
    target_weight: float = 0.0
    quantity: float = 0.0
    estimated_price: float = 0.0
    estimated_notional: float = 0.0
    estimated_fee: float = 0.0
    estimated_slippage: float = 0.0
    halted: bool = False


class MarketHoursGuard:
    """Guard 1: blocks orders outside exchange RTH when enforce_market_hours=True."""

    name = "MarketHoursGuard"

    def check(self, ctx: RiskContext) -> RejectedOrder | None:
        if not ctx.config.enforce_market_hours or ctx.calendar.is_24_7:
            return None
        if not ctx.calendar.is_market_open(ctx.now):
            return RejectedOrder(
                instrument_id=ctx.instrument.id,
                timestamp=ctx.now,
                side=ctx.intent.side,
                guard_name=self.name,
                reason=f"Market '{ctx.calendar.code}' is closed at {ctx.now.isoformat()}",
            )
        return None


class DataStalenessGuard:
    """Guard 2: rejects execution if the newest candle is older than max_data_age_minutes."""

    name = "DataStalenessGuard"

    def check(self, ctx: RiskContext) -> RejectedOrder | None:
        if not ctx.check_staleness:
            return None
        last_candle_ts = ensure_utc(ctx.snapshot.candles.index[-1])
        age_minutes = (ensure_utc(ctx.now) - last_candle_ts).total_seconds() / 60.0
        if age_minutes > ctx.config.max_data_age_minutes:
            return RejectedOrder(
                instrument_id=ctx.instrument.id,
                timestamp=ctx.now,
                side=ctx.intent.side,
                guard_name=self.name,
                reason=(
                    f"Market data is stale ({age_minutes:.1f}m old > "
                    f"limit {ctx.config.max_data_age_minutes}m)"
                ),
            )
        return None


class DrawdownCircuitBreaker:
    """Guard 6 (evaluated early): halts new risk-taking when drawdown exceeds max_drawdown_pct."""

    name = "DrawdownCircuitBreaker"

    def check(self, ctx: RiskContext) -> RejectedOrder | None:
        hwm = max(ctx.portfolio.high_water_mark, ctx.portfolio.equity)
        drawdown = max(0.0, 1.0 - (ctx.portfolio.equity / hwm)) if hwm > 0 else 0.0
        is_breached = ctx.portfolio.halted or (drawdown >= ctx.config.max_drawdown_pct)
        if is_breached:
            ctx.halted = True
            if ctx.intent.side == OrderSide.BUY:
                return RejectedOrder(
                    instrument_id=ctx.instrument.id,
                    timestamp=ctx.now,
                    side=ctx.intent.side,
                    guard_name=self.name,
                    reason=(
                        f"Circuit breaker HALTED: drawdown {drawdown:.2%} >= "
                        f"max {ctx.config.max_drawdown_pct:.2%}. Only closing orders permitted."
                    ),
                    halted=True,
                )
        return None


class ExposureLimitGuard:
    """Guard 5: ensures position weight never exceeds max_exposure_pct of portfolio equity."""

    name = "ExposureLimitGuard"

    def check(self, ctx: RiskContext) -> RejectedOrder | None:
        if ctx.intent.side != OrderSide.BUY:
            return None
        capped = min(ctx.target_weight, ctx.config.max_exposure_pct)
        current_weight = (
            ctx.portfolio.position_value(ctx.instrument.id) / ctx.portfolio.equity
            if ctx.portfolio.equity > 0
            else 0.0
        )
        if capped <= current_weight + 1e-6:
            return RejectedOrder(
                instrument_id=ctx.instrument.id,
                timestamp=ctx.now,
                side=ctx.intent.side,
                guard_name=self.name,
                reason=(
                    f"Exposure limit reached: current weight {current_weight:.2%} >= "
                    f"max_exposure_pct {ctx.config.max_exposure_pct:.2%}"
                ),
            )
        ctx.target_weight = capped
        return None


class InstrumentConstraintGuard:
    """Guard 4: translates target weight into exact rounded qty respecting cash reserve, qty_step, min_qty, min_notional."""

    name = "InstrumentConstraintGuard"

    @staticmethod
    def estimate_fee(notional: float, config: RiskConfig) -> float:
        if notional <= 0:
            return 0.0
        return max(notional * config.fee_model.pct, config.fee_model.fixed_min)

    def check(self, ctx: RiskContext) -> RejectedOrder | None:
        price = ctx.snapshot.quote.ask if ctx.intent.side == OrderSide.BUY else ctx.snapshot.quote.bid
        if price <= 0:
            price = ctx.snapshot.quote.last
        slippage_mult = ctx.config.slippage_bps / 10_000.0
        exec_price = (
            price * (1.0 + slippage_mult)
            if ctx.intent.side == OrderSide.BUY
            else price * max(1e-6, 1.0 - slippage_mult)
        )
        ctx.estimated_price = exec_price

        current_qty = ctx.portfolio.position_qty(ctx.instrument.id)
        equity = ctx.portfolio.equity

        if ctx.intent.side == OrderSide.BUY:
            target_value = ctx.target_weight * equity
            current_value = current_qty * price
            delta_value = max(0.0, target_value - current_value)

            # Respect available cash minus safety reserve and fees
            usable_cash = max(0.0, ctx.portfolio.cash * (1.0 - ctx.config.cash_reserve_pct))
            # Solve notional + max(notional*pct, fixed_min) <= usable_cash
            max_notional_pct = usable_cash / (1.0 + ctx.config.fee_model.pct)
            max_notional_fixed = max(0.0, usable_cash - ctx.config.fee_model.fixed_min)
            affordable_notional = min(max_notional_pct, max_notional_fixed)
            order_notional = min(delta_value, affordable_notional)

            raw_qty = order_notional / exec_price if exec_price > 0 else 0.0
            rounded_qty = ctx.instrument.round_qty(raw_qty)
        elif ctx.intent.side == OrderSide.SELL:
            if current_qty <= 0:
                return RejectedOrder(
                    instrument_id=ctx.instrument.id,
                    timestamp=ctx.now,
                    side=ctx.intent.side,
                    guard_name=self.name,
                    reason="No open position to sell",
                )
            if ctx.target_weight <= 1e-6:
                raw_qty = current_qty
            else:
                target_value = ctx.target_weight * equity
                current_value = current_qty * price
                delta_value = max(0.0, current_value - target_value)
                raw_qty = min(current_qty, delta_value / price)
            rounded_qty = min(current_qty, ctx.instrument.round_qty(raw_qty))
            # If full liquidation requested and rounding leaves tiny dust < min_qty, liquidate all
            if ctx.target_weight <= 1e-6 and ctx.instrument.fractional and (current_qty - rounded_qty) < ctx.instrument.min_qty:
                rounded_qty = current_qty
        else:
            return None

        if rounded_qty < ctx.instrument.min_qty:
            return RejectedOrder(
                instrument_id=ctx.instrument.id,
                timestamp=ctx.now,
                side=ctx.intent.side,
                guard_name=self.name,
                reason=(
                    f"Rounded quantity {rounded_qty} < min_qty {ctx.instrument.min_qty} "
                    f"(fractional={ctx.instrument.fractional}, step={ctx.instrument.qty_step})"
                ),
            )

        notional = rounded_qty * exec_price
        if notional < ctx.instrument.min_notional:
            return RejectedOrder(
                instrument_id=ctx.instrument.id,
                timestamp=ctx.now,
                side=ctx.intent.side,
                guard_name=self.name,
                reason=(
                    f"Order notional {notional:.2f} {ctx.instrument.quote_currency} < "
                    f"min_notional {ctx.instrument.min_notional:.2f}"
                ),
            )

        fee = self.estimate_fee(notional, ctx.config)
        # Final cash sufficiency check on BUY including exact fee
        if ctx.intent.side == OrderSide.BUY and (notional + fee) > ctx.portfolio.cash + 1e-9:
            return RejectedOrder(
                instrument_id=ctx.instrument.id,
                timestamp=ctx.now,
                side=ctx.intent.side,
                guard_name=self.name,
                reason=(
                    f"Insufficient cash ({ctx.portfolio.cash:.2f}) for notional+fee "
                    f"({notional + fee:.2f})"
                ),
            )

        ctx.quantity = rounded_qty
        ctx.estimated_notional = notional
        ctx.estimated_fee = fee
        ctx.estimated_slippage = abs(exec_price - price) * rounded_qty
        return None


class FeeAndSlippageGuard:
    """Guard 3: blocks BUY entries whose expected return does not cover round-trip fee + slippage."""

    name = "FeeAndSlippageGuard"

    def check(self, ctx: RiskContext) -> RejectedOrder | None:
        if not ctx.config.enable_fee_guard:
            return None
        # Never block protective exits or sells
        if ctx.intent.side != OrderSide.BUY or ctx.intent.is_stop_loss:
            return None
        if ctx.estimated_notional <= 0:
            return None

        one_way_fee = max(
            ctx.estimated_notional * ctx.config.fee_model.pct,
            ctx.config.fee_model.fixed_min,
        )
        one_way_slip = ctx.estimated_notional * (ctx.config.slippage_bps / 10_000.0)
        round_trip_cost = 2.0 * (one_way_fee + one_way_slip)
        round_trip_cost_pct = round_trip_cost / ctx.estimated_notional
        expected_gross_ret = ctx.intent.signal.expected_return

        if expected_gross_ret <= round_trip_cost_pct:
            return RejectedOrder(
                instrument_id=ctx.instrument.id,
                timestamp=ctx.now,
                side=ctx.intent.side,
                guard_name=self.name,
                reason=(
                    f"Expected return {expected_gross_ret:.3%} does not cover "
                    f"estimated round-trip cost {round_trip_cost_pct:.3%} "
                    f"({round_trip_cost:.2f} {ctx.instrument.quote_currency})"
                ),
            )
        return None


class QuantileStopLossGuard:
    """Guard 7: triggers protective SELL when price breaches lower-quantile or trailing stop."""

    name = "QuantileStopLossGuard"

    def evaluate_stop_loss(
        self,
        *,
        instrument: Instrument,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
        forecast: ForecastResult,
        intent: OrderIntent,
        config: RiskConfig,
    ) -> OrderIntent:
        if not config.enable_quantile_stop_loss:
            return intent
        pos = portfolio.positions.get(instrument.id)
        if pos is None or pos.quantity <= 0:
            return intent

        current_price = snapshot.quote.last
        # Check trailing/drawdown stop from entry price if configured
        if config.trailing_stop_pct is not None and pos.avg_entry_price > 0:
            stop_price = pos.avg_entry_price * (1.0 - config.trailing_stop_pct)
            if current_price <= stop_price:
                return OrderIntent(
                    instrument_id=instrument.id,
                    timestamp=snapshot.timestamp,
                    side=OrderSide.SELL,
                    target_weight=0.0,
                    target_fraction=1.0,
                    signal=intent.signal,
                    reason=(
                        f"QuantileStopLossGuard trailing stop triggered: "
                        f"price {current_price:.4f} <= {stop_price:.4f}"
                    ),
                    is_stop_loss=True,
                )

        # Dynamic quantile stop: if even the upper-median trajectory drops below entry * (1 + q_stop_ret)
        # or if current price has dropped below previous entry and the lower quantile signals severe tail loss
        q_level = config.stop_loss_quantile
        q_val = float(forecast.quantile(q_level)[-1])
        if pos.avg_entry_price > 0 and q_val < pos.avg_entry_price * (1.0 - config.max_drawdown_pct):
            return OrderIntent(
                instrument_id=instrument.id,
                timestamp=snapshot.timestamp,
                side=OrderSide.SELL,
                target_weight=0.0,
                target_fraction=1.0,
                signal=intent.signal,
                reason=(
                    f"QuantileStopLossGuard triggered: q{int(q_level * 100):02d}={q_val:.4f} "
                    f"breaches risk floor from entry {pos.avg_entry_price:.4f}"
                ),
                is_stop_loss=True,
            )

        return intent


class RiskManager:
    """Orchestrates all 7 composable RiskGuards into ApprovedOrder or RejectedOrder."""

    def __init__(self, config: RiskConfig, calendar: MarketCalendar) -> None:
        self.config = config
        self.calendar = calendar
        self.stop_loss_guard = QuantileStopLossGuard()
        self.guards: list[RiskGuard] = [
            MarketHoursGuard(),
            DataStalenessGuard(),
            DrawdownCircuitBreaker(),
            ExposureLimitGuard(),
            InstrumentConstraintGuard(),
            FeeAndSlippageGuard(),
        ]

    def evaluate(
        self,
        *,
        instrument: Instrument,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
        forecast: ForecastResult,
        intent: OrderIntent,
        now: datetime | None = None,
        check_staleness: bool = True,
    ) -> tuple[ApprovedOrder | None, RejectedOrder | None, bool]:
        """Evaluate risk chain. Returns (approved_order, rejected_order, is_halted)."""
        eval_time = ensure_utc(now) if now is not None else datetime.now(UTC)

        # Step 1: Check protective stop-loss override first
        effective_intent = self.stop_loss_guard.evaluate_stop_loss(
            instrument=instrument,
            snapshot=snapshot,
            portfolio=portfolio,
            forecast=forecast,
            intent=intent,
            config=self.config,
        )

        # Always compute circuit breaker state even on HOLD
        hwm = max(portfolio.high_water_mark, portfolio.equity)
        drawdown = max(0.0, 1.0 - (portfolio.equity / hwm)) if hwm > 0 else 0.0
        halted = portfolio.halted or (drawdown >= self.config.max_drawdown_pct)

        if effective_intent.side == OrderSide.HOLD:
            return None, None, halted

        ctx = RiskContext(
            instrument=instrument,
            snapshot=snapshot,
            portfolio=portfolio,
            forecast=forecast,
            intent=effective_intent,
            config=self.config,
            calendar=self.calendar,
            now=eval_time,
            check_staleness=check_staleness,
            target_weight=effective_intent.target_weight,
            halted=halted,
        )

        for guard in self.guards:
            rejection = guard.check(ctx)
            if rejection is not None:
                return None, rejection, ctx.halted

        approved = ApprovedOrder(
            instrument_id=instrument.id,
            timestamp=eval_time,
            side=effective_intent.side,
            quantity=ctx.quantity,
            estimated_price=ctx.estimated_price,
            estimated_notional=ctx.estimated_notional,
            estimated_fee=ctx.estimated_fee,
            estimated_slippage=ctx.estimated_slippage,
            reason=effective_intent.reason,
            is_stop_loss=effective_intent.is_stop_loss,
        )
        return approved, None, ctx.halted
