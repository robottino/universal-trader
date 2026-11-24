"""DryRunBroker wrapper enforcing zero mutating calls to any underlying broker."""

from __future__ import annotations

import uuid

from universal_trader.core.logging import get_logger
from universal_trader.core.models import (
    ApprovedOrder,
    FillResult,
    Instrument,
    OrderStatus,
    PortfolioState,
    Quote,
)
from universal_trader.core.protocols import Broker

logger = get_logger(__name__)


class DryRunBroker:
    """Read-only wrapper that physically prevents submit_order from reaching the underlying broker."""

    def __init__(self, inner: Broker) -> None:
        self._inner = inner
        self.name = f"dry_run:{inner.name}"

    def get_quote(self, instrument: Instrument) -> Quote:
        return self._inner.get_quote(instrument)

    def get_portfolio(self, instrument: Instrument, quote: Quote | None = None) -> PortfolioState:
        return self._inner.get_portfolio(instrument, quote=quote)

    def submit_order(self, order: ApprovedOrder, instrument: Instrument) -> FillResult:
        """Never invokes self._inner.submit_order; returns a simulated DRY_RUN FillResult."""
        logger.info(
            "[DRY-RUN] Suppressed %s %.6f %s @ ~%.4f %s (est fee %.4f)",
            order.side.value,
            order.quantity,
            instrument.id,
            order.estimated_price,
            instrument.quote_currency,
            order.estimated_fee,
        )
        return FillResult(
            order_id=f"dryrun-{uuid.uuid4().hex[:12]}",
            instrument_id=instrument.id,
            timestamp=order.timestamp,
            side=order.side,
            quantity=order.quantity,
            fill_price=order.estimated_price,
            notional=order.estimated_notional,
            fee=order.estimated_fee,
            slippage=order.estimated_slippage,
            status=OrderStatus.DRY_RUN,
            raw_response={"dry_run": True, "underlying_broker": self._inner.name},
        )
