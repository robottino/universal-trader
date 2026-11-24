"""Bitfinex REST v2 spot crypto broker adapter (HMAC-SHA384) for Paper and Live modes."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import UTC, datetime
from typing import Any

import httpx

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


class BrokerError(RuntimeError):
    """Explicit fail-fast error raised when a broker API call fails."""


@register_broker("bitfinex")
class BitfinexBroker:
    """Bitfinex REST v2 spot trading adapter supporting Paper (tTEST...) and Live accounts."""

    name = "bitfinex"

    def __init__(
        self,
        *,
        api_key: str = "",
        api_secret: str = "",
        paper: bool = True,
        timeout_seconds: float = 15.0,
        base_url: str = "https://api.bitfinex.com",
        pub_url: str = "https://api-pub.bitfinex.com",
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        register_secret(self.api_key)
        register_secret(self.api_secret)
        self.paper = paper
        self.timeout_seconds = timeout_seconds
        self.base_url = base_url.rstrip("/")
        self.pub_url = pub_url.rstrip("/")

    def _exchange_symbol(self, instrument: Instrument) -> str:
        explicit = instrument.provider_symbols.get("bitfinex")
        if explicit:
            return explicit
        base = instrument.base.upper()
        quote = instrument.quote_currency.upper()
        if self.paper:
            return f"tTEST{base}:TEST{quote}"
        if len(base) == 3 and len(quote) == 3:
            return f"t{base}{quote}"
        return f"t{base}:{quote}"

    def _wallet_currency(self, code: str) -> str:
        c = code.upper()
        return f"TEST{c}" if self.paper and not c.startswith("TEST") else c

    def _auth_headers(self, path: str, body_str: str) -> dict[str, str]:
        if not self.api_key or not self.api_secret:
            raise BrokerError("BitfinexBroker requires api_key and api_secret")
        nonce = str(int(time.time() * 10_000))
        payload = f"/api/{path}{nonce}{body_str}".encode()
        sig = hmac.new(self.api_secret.encode("utf-8"), payload, hashlib.sha384).hexdigest()
        return {
            "bfx-nonce": nonce,
            "bfx-apikey": self.api_key,
            "bfx-signature": sig,
            "content-type": "application/json",
        }

    def get_quote(self, instrument: Instrument) -> Quote:
        symbol = self._exchange_symbol(instrument)
        url = f"{self.pub_url}/v2/ticker/{symbol}"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.get(url)
        except httpx.HTTPError as exc:
            raise BrokerError(f"Bitfinex ticker request failed for '{symbol}': {exc}") from exc
        if resp.status_code != 200:
            raise BrokerError(f"Bitfinex ticker HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        if not isinstance(data, list) or len(data) < 7:
            raise BrokerError(f"Unexpected Bitfinex ticker payload for '{symbol}': {data}")
        # [BID, BID_SIZE, ASK, ASK_SIZE, DAILY_CHANGE, DAILY_CHANGE_PERC, LAST_PRICE, ...]
        bid = float(data[0])
        ask = float(data[2])
        last = float(data[6])
        return Quote(
            instrument_id=instrument.id,
            timestamp=datetime.now(UTC),
            bid=bid if bid > 0 else last,
            ask=ask if ask > 0 else last,
            last=last,
        )

    def get_portfolio(self, instrument: Instrument, quote: Quote | None = None) -> PortfolioState:
        path = "v2/auth/r/wallets"
        body_str = json.dumps({})
        headers = self._auth_headers(path, body_str)
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(f"{self.base_url}/{path}", headers=headers, content=body_str)
        except httpx.HTTPError as exc:
            raise BrokerError(f"Bitfinex wallets request failed: {exc}") from exc
        if resp.status_code != 200:
            raise BrokerError(f"Bitfinex wallets HTTP {resp.status_code}: {resp.text[:200]}")

        wallets = resp.json()
        base_code = self._wallet_currency(instrument.base)
        quote_code = self._wallet_currency(instrument.quote_currency)

        base_bal = 0.0
        quote_bal = 0.0
        for w in wallets:
            if not isinstance(w, list) or len(w) < 3:
                continue
            if w[0] != "exchange":
                continue
            curr = str(w[1]).upper()
            avail = float(w[4]) if (len(w) > 4 and w[4] is not None) else float(w[2])
            if curr == base_code:
                base_bal = max(0.0, avail)
            elif curr == quote_code:
                quote_bal = max(0.0, avail)

        q = quote or self.get_quote(instrument)
        positions: dict[str, Position] = {}
        if base_bal > 0:
            positions[instrument.id] = Position(
                instrument_id=instrument.id,
                quantity=base_bal,
                avg_entry_price=q.last,
                market_price=q.last,
            )
        equity = quote_bal + base_bal * q.last
        return PortfolioState(
            timestamp=datetime.now(UTC),
            quote_currency=instrument.quote_currency,
            cash=quote_bal,
            positions=positions,
            equity=equity,
            high_water_mark=equity,
            halted=False,
        )

    def submit_order(self, order: ApprovedOrder, instrument: Instrument) -> FillResult:
        symbol = self._exchange_symbol(instrument)
        path = "v2/auth/w/order/submit"
        signed_qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
        payload = {
            "type": "EXCHANGE MARKET",
            "symbol": symbol,
            "amount": f"{signed_qty:.8f}",
        }
        body_str = json.dumps(payload)
        headers = self._auth_headers(path, body_str)
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(f"{self.base_url}/{path}", headers=headers, content=body_str)
        except httpx.HTTPError as exc:
            raise BrokerError(f"Bitfinex submit_order network error: {exc}") from exc
        if resp.status_code != 200:
            raise BrokerError(f"Bitfinex submit_order HTTP {resp.status_code}: {resp.text[:300]}")

        res_data: Any = resp.json()
        order_id = "bitfinex-unknown"
        if isinstance(res_data, list) and len(res_data) > 6:
            status_str = str(res_data[6]).upper()
            message = str(res_data[7]) if len(res_data) > 7 else ""
            if status_str == "ERROR":
                raise BrokerError(f"Bitfinex rejected order: {message}")
            order_payload = res_data[4]
            if isinstance(order_payload, list) and order_payload:
                first = order_payload[0] if isinstance(order_payload[0], list) else order_payload
                if first:
                    order_id = str(first[0])

        return FillResult(
            order_id=order_id,
            instrument_id=instrument.id,
            timestamp=datetime.now(UTC),
            side=order.side,
            quantity=order.quantity,
            fill_price=order.estimated_price,
            notional=order.estimated_notional,
            fee=order.estimated_fee,
            slippage=order.estimated_slippage,
            status=OrderStatus.FILLED,
            raw_response={"response": res_data},
        )
