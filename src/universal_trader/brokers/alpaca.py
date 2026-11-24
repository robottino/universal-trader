"""Alpaca Markets REST v2 Broker adapter for US equities and ETFs (Paper & Live)."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from universal_trader.brokers.bitfinex import BrokerError
from universal_trader.core.logging import register_secret
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


@register_broker("alpaca")
class AlpacaBroker:
    """Alpaca Trading API v2 adapter supporting free Paper trading and Live execution."""

    name = "alpaca"

    def __init__(
        self,
        *,
        api_key: str = "",
        api_secret: str = "",
        paper: bool = True,
        timeout_seconds: float = 15.0,
        data_base_url: str = "https://data.alpaca.markets",
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        register_secret(self.api_key)
        register_secret(self.api_secret)
        self.paper = paper
        self.trading_url = (
            "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        )
        self.data_base_url = data_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
        if not self.api_key or not self.api_secret:
            raise BrokerError("AlpacaBroker requires api_key and api_secret")
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "content-type": "application/json",
        }

    def get_quote(self, instrument: Instrument) -> Quote:
        symbol = instrument.symbol_for("alpaca").upper()
        url = f"{self.data_base_url}/v2/stocks/{symbol}/quotes/latest"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.get(url, headers=self._headers())
        except httpx.HTTPError as exc:
            raise BrokerError(f"Alpaca quote request failed for '{symbol}': {exc}") from exc
        if resp.status_code != 200:
            raise BrokerError(f"Alpaca quote HTTP {resp.status_code}: {resp.text[:200]}")
        q = resp.json().get("quote", {})
        bid = float(q.get("bp", 0.0))
        ask = float(q.get("ap", 0.0))
        last = 0.5 * (bid + ask) if (bid > 0 and ask > 0) else max(bid, ask)
        if last <= 0:
            raise BrokerError(f"Alpaca returned invalid quote for '{symbol}': {q}")
        return Quote(
            instrument_id=instrument.id,
            timestamp=datetime.now(UTC),
            bid=bid if bid > 0 else last,
            ask=ask if ask > 0 else last,
            last=last,
        )

    def get_portfolio(self, instrument: Instrument, quote: Quote | None = None) -> PortfolioState:
        headers = self._headers()
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                acct_resp = client.get(f"{self.trading_url}/v2/account", headers=headers)
                pos_resp = client.get(f"{self.trading_url}/v2/positions", headers=headers)
        except httpx.HTTPError as exc:
            raise BrokerError(f"Alpaca account query failed: {exc}") from exc
        if acct_resp.status_code != 200:
            raise BrokerError(f"Alpaca account HTTP {acct_resp.status_code}: {acct_resp.text[:200]}")
        if pos_resp.status_code != 200:
            raise BrokerError(f"Alpaca positions HTTP {pos_resp.status_code}: {pos_resp.text[:200]}")

        acct = acct_resp.json()
        cash = float(acct.get("cash", 0.0))
        equity = float(acct.get("equity", cash))
        symbol = instrument.symbol_for("alpaca").upper()
        positions: dict[str, Position] = {}
        for p in pos_resp.json():
            sym = str(p.get("symbol", "")).upper()
            qty = float(p.get("qty", 0.0))
            if qty <= 0:
                continue
            inst_key = instrument.id if sym == symbol else sym
            positions[inst_key] = Position(
                instrument_id=inst_key,
                quantity=qty,
                avg_entry_price=float(p.get("avg_entry_price", 0.0)),
                market_price=float(p.get("current_price", 0.0)),
            )

        return PortfolioState(
            timestamp=datetime.now(UTC),
            quote_currency=str(acct.get("currency", instrument.quote_currency)),
            cash=max(0.0, cash),
            positions=positions,
            equity=max(0.0, equity),
            high_water_mark=max(0.0, equity),
            halted=False,
        )

    def submit_order(self, order: ApprovedOrder, instrument: Instrument) -> FillResult:
        symbol = instrument.symbol_for("alpaca").upper()
        qty_str = (
            str(int(round(order.quantity)))
            if not instrument.fractional
            else f"{order.quantity:.6f}".rstrip("0").rstrip(".")
        )
        payload = {
            "symbol": symbol,
            "qty": qty_str,
            "side": "buy" if order.side == OrderSide.BUY else "sell",
            "type": "market",
            "time_in_force": "day",
        }
        if order.client_order_id:
            payload["client_order_id"] = order.client_order_id

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(
                    f"{self.trading_url}/v2/orders",
                    headers=self._headers(),
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise BrokerError(f"Alpaca submit_order failed: {exc}") from exc
        if resp.status_code not in {200, 201}:
            raise BrokerError(f"Alpaca submit_order HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        fill_price = float(data.get("filled_avg_price") or order.estimated_price)
        filled_qty = float(data.get("filled_qty") or order.quantity)
        alpaca_status = str(data.get("status", "accepted")).lower()
        status = OrderStatus.FILLED if alpaca_status == "filled" else OrderStatus.PENDING
        return FillResult(
            order_id=str(data.get("id", "alpaca-unknown")),
            instrument_id=instrument.id,
            timestamp=datetime.now(UTC),
            side=order.side,
            quantity=filled_qty,
            fill_price=fill_price,
            notional=filled_qty * fill_price,
            fee=order.estimated_fee,
            slippage=order.estimated_slippage,
            status=status,
            raw_response=data,
        )
