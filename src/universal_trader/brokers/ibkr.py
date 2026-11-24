"""Extensible Interactive Brokers (IBKR) adapter via optional `ib_async` extra."""

from __future__ import annotations

from datetime import UTC, datetime

from universal_trader.brokers.bitfinex import BrokerError
from universal_trader.core.models import (
    ApprovedOrder,
    FillResult,
    Instrument,
    OrderSide,
    OrderStatus,
    PortfolioState,
    Position,
    Quote,
)
from universal_trader.core.registry import register_broker


@register_broker("ibkr")
class IBKRBroker:
    """Interactive Brokers execution adapter (requires `uv sync --extra ibkr`)."""

    name = "ibkr"

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        paper: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.paper = paper

    def _connect(self):  # type: ignore[no-untyped-def]
        try:
            from ib_async import IB
        except ImportError as exc:
            raise BrokerError(
                "IBKRBroker requires optional dependency 'ib-async'. "
                "Install it with: uv sync --extra ibkr"
            ) from exc
        ib = IB()
        ib.connect(self.host, self.port, clientId=self.client_id, readonly=False)
        return ib

    def _contract(self, instrument: Instrument):  # type: ignore[no-untyped-def]
        from ib_async import Stock

        sym = instrument.symbol_for("ibkr")
        return Stock(sym, "SMART", instrument.quote_currency)

    def get_quote(self, instrument: Instrument) -> Quote:
        ib = self._connect()
        try:
            contract = self._contract(instrument)
            ib.qualifyContracts(contract)
            ticker = ib.reqMktData(contract, "", True, False)
            ib.sleep(1.0)
            last = float(ticker.marketPrice())
            bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else last
            ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else last
            return Quote(
                instrument_id=instrument.id,
                timestamp=datetime.now(UTC),
                bid=bid,
                ask=ask,
                last=last,
            )
        finally:
            ib.disconnect()

    def get_portfolio(self, instrument: Instrument, quote: Quote | None = None) -> PortfolioState:
        ib = self._connect()
        try:
            summary = {s.tag: float(s.value) for s in ib.accountSummary() if s.tag in {"TotalCashValue", "NetLiquidation"}}
            cash = summary.get("TotalCashValue", 0.0)
            equity = summary.get("NetLiquidation", cash)
            positions: dict[str, Position] = {}
            sym = instrument.symbol_for("ibkr").upper()
            for p in ib.positions():
                if p.position <= 0:
                    continue
                key = instrument.id if p.contract.symbol.upper() == sym else p.contract.symbol
                mkt = quote.last if (quote is not None and key == instrument.id) else float(p.avgCost)
                positions[key] = Position(
                    instrument_id=key,
                    quantity=float(p.position),
                    avg_entry_price=float(p.avgCost),
                    market_price=mkt,
                )
            return PortfolioState(
                timestamp=datetime.now(UTC),
                quote_currency=instrument.quote_currency,
                cash=max(0.0, cash),
                positions=positions,
                equity=max(0.0, equity),
                high_water_mark=max(0.0, equity),
                halted=False,
            )
        finally:
            ib.disconnect()

    def submit_order(self, order: ApprovedOrder, instrument: Instrument) -> FillResult:
        from ib_async import MarketOrder

        ib = self._connect()
        try:
            contract = self._contract(instrument)
            ib.qualifyContracts(contract)
            action = "BUY" if order.side == OrderSide.BUY else "SELL"
            ib_order = MarketOrder(action, order.quantity)
            trade = ib.placeOrder(contract, ib_order)
            ib.sleep(1.0)
            avg_px = float(trade.orderStatus.avgFillPrice or order.estimated_price)
            return FillResult(
                order_id=str(trade.order.orderId),
                instrument_id=instrument.id,
                timestamp=datetime.now(UTC),
                side=order.side,
                quantity=order.quantity,
                fill_price=avg_px,
                notional=order.quantity * avg_px,
                fee=order.estimated_fee,
                slippage=order.estimated_slippage,
                status=OrderStatus.FILLED if trade.isDone() else OrderStatus.PENDING,
                raw_response={"ib_status": trade.orderStatus.status},
            )
        finally:
            ib.disconnect()
