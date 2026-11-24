"""SimulatedBroker for realistic backtesting and walk-forward execution."""

from __future__ import annotations

from datetime import UTC, datetime

from universal_trader.core.config import FeeModelConfig
from universal_trader.core.models import (
    ApprovedOrder,
    Candle,
    FillResult,
    Instrument,
    OrderSide,
    OrderStatus,
    PortfolioState,
    Position,
    Quote,
    ensure_utc,
)
from universal_trader.core.registry import register_broker


@register_broker("simulated")
class SimulatedBroker:
    """In-memory execution simulator with spread, slippage, fixed+pct fees, and next_open fills."""

    name = "simulated"

    def __init__(
        self,
        *,
        initial_cash: float = 10000.0,
        quote_currency: str = "USD",
        fee_model: FeeModelConfig | None = None,
        slippage_bps: float = 5.0,
        spread_bps: float = 2.0,
        execution_price: str = "next_open",
        max_drawdown_pct: float = 0.15,
    ) -> None:
        self.initial_cash = float(initial_cash)
        self.quote_currency = quote_currency
        self.fee_model = fee_model or FeeModelConfig()
        self.slippage_bps = float(slippage_bps)
        self.spread_bps = float(spread_bps)
        self.execution_price = execution_price  # "next_open" | "close"
        self.max_drawdown_pct = float(max_drawdown_pct)

        self.cash: float = float(initial_cash)
        self.positions: dict[str, dict[str, float]] = {}
        self.high_water_mark: float = float(initial_cash)
        self.halted: bool = False
        self._current_bar: Candle | None = None
        self._pending_orders: list[tuple[ApprovedOrder, Instrument]] = []
        self.fills: list[FillResult] = []
        self._order_seq: int = 0

    def _compute_fee(self, notional: float) -> float:
        if notional <= 0:
            return 0.0
        return max(notional * self.fee_model.pct, self.fee_model.fixed_min)

    def _quote_from_price(self, instrument: Instrument, price: float, ts: datetime) -> Quote:
        half_spread = price * (self.spread_bps / 20_000.0)
        return Quote(
            instrument_id=instrument.id,
            timestamp=ts,
            bid=max(1e-9, price - half_spread),
            ask=price + half_spread,
            last=price,
        )

    def on_new_bar(self, bar: Candle, instrument: Instrument) -> list[FillResult]:
        """Advance simulation clock to `bar` and execute any queued next_open orders at `bar.open`."""
        self._current_bar = bar
        executed: list[FillResult] = []
        if self._pending_orders:
            queue = list(self._pending_orders)
            self._pending_orders.clear()
            for pending_order, inst in queue:
                fill = self._execute_at_price(
                    order=pending_order,
                    instrument=inst,
                    base_price=bar.open,
                    fill_ts=bar.timestamp,
                )
                executed.append(fill)
        # Update mark-to-market HWM using bar close
        pf = self.get_portfolio(instrument)
        if pf.equity > self.high_water_mark:
            self.high_water_mark = pf.equity
        if pf.drawdown >= self.max_drawdown_pct:
            self.halted = True
        return executed

    def set_current_bar(self, bar: Candle) -> None:
        self._current_bar = bar

    def get_quote(self, instrument: Instrument) -> Quote:
        if self._current_bar is None:
            raise RuntimeError("SimulatedBroker has no active bar; call on_new_bar or set_current_bar first")
        return self._quote_from_price(instrument, self._current_bar.close, self._current_bar.timestamp)

    def get_portfolio(self, instrument: Instrument, quote: Quote | None = None) -> PortfolioState:
        ts = (
            ensure_utc(quote.timestamp)
            if quote is not None
            else (self._current_bar.timestamp if self._current_bar is not None else datetime.now(UTC))
        )
        spot = (
            quote.last
            if quote is not None
            else (self._current_bar.close if self._current_bar is not None else 0.0)
        )
        pos_models: dict[str, Position] = {}
        pos_value = 0.0
        for inst_id, pdata in self.positions.items():
            qty = pdata["quantity"]
            if qty <= 0:
                continue
            mkt_px = spot if inst_id == instrument.id and spot > 0 else pdata["avg_entry_price"]
            pos_models[inst_id] = Position(
                instrument_id=inst_id,
                quantity=qty,
                avg_entry_price=pdata["avg_entry_price"],
                market_price=mkt_px,
            )
            pos_value += qty * mkt_px

        equity = max(0.0, self.cash + pos_value)
        hwm = max(self.high_water_mark, equity)
        self.high_water_mark = hwm
        return PortfolioState(
            timestamp=ts,
            quote_currency=self.quote_currency,
            cash=max(0.0, self.cash),
            positions=pos_models,
            equity=equity,
            high_water_mark=hwm,
            halted=self.halted,
        )

    def submit_order(self, order: ApprovedOrder, instrument: Instrument) -> FillResult:
        if self.execution_price == "next_open":
            self._order_seq += 1
            self._pending_orders.append((order, instrument))
            return FillResult(
                order_id=f"sim-pending-{self._order_seq}",
                instrument_id=instrument.id,
                timestamp=order.timestamp,
                side=order.side,
                quantity=order.quantity,
                fill_price=order.estimated_price,
                notional=order.estimated_notional,
                fee=0.0,
                slippage=0.0,
                status=OrderStatus.PENDING,
                raw_response={"execution_price": "next_open", "queued": True},
            )
        base_price = self._current_bar.close if self._current_bar is not None else order.estimated_price
        fill_ts = self._current_bar.timestamp if self._current_bar is not None else order.timestamp
        return self._execute_at_price(order, instrument, base_price=base_price, fill_ts=fill_ts)

    def _execute_at_price(
        self,
        order: ApprovedOrder,
        instrument: Instrument,
        base_price: float,
        fill_ts: datetime,
    ) -> FillResult:
        self._order_seq += 1
        order_id = f"sim-{self._order_seq}"
        half_spread = self.spread_bps / 20_000.0
        slip_rate = self.slippage_bps / 10_000.0

        if order.side == OrderSide.BUY:
            fill_price = base_price * (1.0 + half_spread + slip_rate)
            slip_cost_per_unit = max(0.0, fill_price - base_price)
            qty = instrument.round_qty(order.quantity)

            # Re-verify cash affordability at actual next_open price (gap-up protection!)
            notional = qty * fill_price
            fee = self._compute_fee(notional)
            if notional + fee > self.cash:
                max_notional_pct = self.cash / (1.0 + self.fee_model.pct)
                max_notional_fixed = max(0.0, self.cash - self.fee_model.fixed_min)
                affordable = min(max_notional_pct, max_notional_fixed)
                qty = instrument.round_qty(affordable / fill_price)
                notional = qty * fill_price
                fee = self._compute_fee(notional)

            if qty < instrument.min_qty or notional < instrument.min_notional or (notional + fee) > self.cash + 1e-9:
                return FillResult(
                    order_id=order_id,
                    instrument_id=instrument.id,
                    timestamp=fill_ts,
                    side=order.side,
                    quantity=0.0,
                    fill_price=fill_price,
                    notional=0.0,
                    fee=0.0,
                    slippage=0.0,
                    status=OrderStatus.REJECTED,
                    raw_response={"reason": "Insufficient cash or below min constraint at execution price"},
                )

            self.cash -= notional + fee
            prev = self.positions.get(instrument.id, {"quantity": 0.0, "avg_entry_price": 0.0})
            new_qty = prev["quantity"] + qty
            new_avg = (
                (prev["quantity"] * prev["avg_entry_price"] + qty * fill_price) / new_qty
                if new_qty > 0
                else fill_price
            )
            self.positions[instrument.id] = {"quantity": new_qty, "avg_entry_price": new_avg}
            fill = FillResult(
                order_id=order_id,
                instrument_id=instrument.id,
                timestamp=fill_ts,
                side=order.side,
                quantity=qty,
                fill_price=fill_price,
                notional=notional,
                fee=fee,
                slippage=slip_cost_per_unit * qty,
                status=OrderStatus.FILLED,
                raw_response={"base_price": base_price, "execution_mode": self.execution_price},
            )
            self.fills.append(fill)
            return fill

        # SELL
        fill_price = max(1e-9, base_price * (1.0 - half_spread - slip_rate))
        slip_cost_per_unit = max(0.0, base_price - fill_price)
        prev = self.positions.get(instrument.id, {"quantity": 0.0, "avg_entry_price": 0.0})
        avail_qty = prev["quantity"]
        qty = min(avail_qty, order.quantity)
        if not instrument.fractional:
            qty = min(avail_qty, instrument.round_qty(qty))

        if qty < instrument.min_qty or avail_qty <= 0:
            return FillResult(
                order_id=order_id,
                instrument_id=instrument.id,
                timestamp=fill_ts,
                side=order.side,
                quantity=0.0,
                fill_price=fill_price,
                notional=0.0,
                fee=0.0,
                slippage=0.0,
                status=OrderStatus.REJECTED,
                raw_response={"reason": "Insufficient position at execution"},
            )

        notional = qty * fill_price
        fee = self._compute_fee(notional)
        self.cash += max(0.0, notional - fee)
        rem_qty = max(0.0, avail_qty - qty)
        if rem_qty < 1e-12:
            self.positions.pop(instrument.id, None)
        else:
            self.positions[instrument.id] = {
                "quantity": rem_qty,
                "avg_entry_price": prev["avg_entry_price"],
            }

        fill = FillResult(
            order_id=order_id,
            instrument_id=instrument.id,
            timestamp=fill_ts,
            side=order.side,
            quantity=qty,
            fill_price=fill_price,
            notional=notional,
            fee=fee,
            slippage=slip_cost_per_unit * qty,
            status=OrderStatus.FILLED,
            raw_response={"base_price": base_price, "execution_mode": self.execution_price},
        )
        self.fills.append(fill)
        return fill
