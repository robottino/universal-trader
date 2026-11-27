"""Bitfinex REST v2 spot crypto broker adapter (HMAC-SHA384) for Paper and Live modes."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import httpx

from universal_trader.core.logging import get_logger, register_secret
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

logger = get_logger(__name__)


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
        self._last_nonce: int = 0
        self._unsupported_paper_symbols: set[str] = set()

    @staticmethod
    def _live_exchange_symbol(instrument: Instrument) -> str:
        explicit = instrument.provider_symbols.get("bitfinex")
        if explicit and "TEST" not in explicit.upper():
            return explicit
        base = instrument.base.upper()
        quote = instrument.quote_currency.upper()
        if len(base) == 3 and len(quote) == 3:
            return f"t{base}{quote}"
        return f"t{base}:{quote}"

    def _exchange_symbol(self, instrument: Instrument) -> str:
        if self.paper:
            explicit_paper = instrument.provider_symbols.get("bitfinex_paper")
            if explicit_paper:
                return explicit_paper
            explicit_live = instrument.provider_symbols.get("bitfinex")
            if explicit_live and "TEST" in explicit_live.upper():
                return explicit_live
            base = instrument.base.upper()
            quote = instrument.quote_currency.upper()
            return f"tTEST{base}:TEST{quote}"

        return self._live_exchange_symbol(instrument)

    def _quote_symbol(self, instrument: Instrument) -> str:
        sym = self._exchange_symbol(instrument)
        if self.paper and sym in self._unsupported_paper_symbols:
            return self._live_exchange_symbol(instrument)
        return sym

    def _wallet_currency(self, code: str) -> str:
        c = code.upper()
        return f"TEST{c}" if self.paper and not c.startswith("TEST") else c

    def _auth_headers(self, path: str, body_str: str) -> dict[str, str]:
        if not self.api_key or not self.api_secret:
            raise BrokerError("BitfinexBroker requires api_key and api_secret")
        now_us = int(time.time() * 1_000_000)
        nonce_int = max(self._last_nonce + 1, now_us)
        self._last_nonce = nonce_int
        nonce = str(nonce_int)
        payload = f"/api/{path}{nonce}{body_str}".encode()
        sig = hmac.new(self.api_secret.encode("utf-8"), payload, hashlib.sha384).hexdigest()
        return {
            "bfx-nonce": nonce,
            "bfx-apikey": self.api_key,
            "bfx-signature": sig,
            "content-type": "application/json",
        }

    def _fetch_single_ticker(self, symbol: str) -> httpx.Response:
        url = f"{self.pub_url}/v2/ticker/{symbol}"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                return client.get(url)
        except httpx.HTTPError as exc:
            raise BrokerError(f"Bitfinex ticker request failed for '{symbol}': {exc}") from exc

    def get_quote(self, instrument: Instrument) -> Quote:
        symbol = self._quote_symbol(instrument)
        live_symbol = self._live_exchange_symbol(instrument)
        resp = self._fetch_single_ticker(symbol)
        if (
            resp.status_code != 200
            and self.paper
            and symbol != live_symbol
            and ("symbol: invalid" in resp.text.lower() or "10020" in resp.text)
        ):
            self._unsupported_paper_symbols.add(symbol)
            logger.warning(
                "Bitfinex Paper does not expose '%s'; falling back to live ticker '%s' for quotes.",
                symbol,
                live_symbol,
            )
            symbol = live_symbol
            resp = self._fetch_single_ticker(symbol)

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

    def get_quotes(self, instruments: Sequence[Instrument]) -> dict[str, Quote]:
        """Batch-fetch live quotes for multiple instruments in a single Bitfinex API call."""
        if not instruments:
            return {}
        sym_to_inst = {self._quote_symbol(inst): inst for inst in instruments}
        symbols_param = ",".join(sym_to_inst.keys())
        url = f"{self.pub_url}/v2/tickers"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.get(url, params={"symbols": symbols_param})
        except httpx.HTTPError as exc:
            raise BrokerError(f"Bitfinex batch tickers request failed: {exc}") from exc
        if resp.status_code != 200:
            raise BrokerError(f"Bitfinex batch tickers HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        if not isinstance(data, list):
            raise BrokerError(f"Unexpected Bitfinex batch tickers payload: {data}")
        now_utc = datetime.now(UTC)
        out: dict[str, Quote] = {}
        for row in data:
            # [SYMBOL, BID, BID_SIZE, ASK, ASK_SIZE, DAILY_CHANGE, DAILY_CHANGE_PERC, LAST_PRICE, ...]
            if not isinstance(row, list) or len(row) < 8:
                continue
            sym = str(row[0])
            inst = sym_to_inst.get(sym)
            if inst is None:
                continue
            bid = float(row[1])
            ask = float(row[3])
            last = float(row[7])
            out[inst.id] = Quote(
                instrument_id=inst.id,
                timestamp=now_utc,
                bid=bid if bid > 0 else last,
                ask=ask if ask > 0 else last,
                last=last,
            )
        # Fallback for any instrument missing from batch response (e.g. unsupported tTEST... paper book)
        for inst in instruments:
            if inst.id not in out:
                out[inst.id] = self.get_quote(inst)
        return out

    def _fetch_exchange_wallets(self) -> dict[str, float]:
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
        balances: dict[str, float] = {}
        for w in wallets:
            if not isinstance(w, list) or len(w) < 3:
                continue
            if w[0] != "exchange":
                continue
            curr = str(w[1]).upper()
            avail = float(w[4]) if (len(w) > 4 and w[4] is not None) else float(w[2])
            balances[curr] = max(0.0, avail)
        return balances

    def get_portfolio(self, instrument: Instrument, quote: Quote | None = None) -> PortfolioState:
        quotes = {instrument.id: quote} if quote is not None else None
        return self.get_multi_portfolio([instrument], quotes=quotes)

    def get_multi_portfolio(
        self,
        instruments: Sequence[Instrument],
        quotes: dict[str, Quote] | None = None,
    ) -> PortfolioState:
        """Query Bitfinex exchange wallet once and mark every basket asset + cash to market."""
        if not instruments:
            raise BrokerError("get_multi_portfolio requires at least one instrument")
        balances = self._fetch_exchange_wallets()
        quote_currency = instruments[0].quote_currency
        quote_code = self._wallet_currency(quote_currency)
        quote_bal = balances.get(quote_code, 0.0)

        resolved_quotes: dict[str, Quote] = dict(quotes or {})
        missing = [inst for inst in instruments if inst.id not in resolved_quotes]
        if missing:
            resolved_quotes.update(self.get_quotes(missing))

        positions: dict[str, Position] = {}
        pos_value = 0.0
        for inst in instruments:
            base_code = self._wallet_currency(inst.base)
            base_bal = balances.get(base_code, 0.0)
            q = resolved_quotes[inst.id]
            if base_bal > 0:
                positions[inst.id] = Position(
                    instrument_id=inst.id,
                    quantity=base_bal,
                    avg_entry_price=q.last,
                    market_price=q.last,
                )
                pos_value += base_bal * q.last

        equity = quote_bal + pos_value
        return PortfolioState(
            timestamp=datetime.now(UTC),
            quote_currency=quote_currency,
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
            if self.paper and ("symbol: invalid" in resp.text.lower() or "10020" in resp.text):
                self._unsupported_paper_symbols.add(symbol)
                logger.warning(
                    "Bitfinex Paper sandbox does not support order symbol '%s'; marking order REJECTED "
                    "(use broker.type='local_paper' for full 5-asset paper fills, or paper=false in live mode).",
                    symbol,
                )
                return FillResult(
                    order_id="bitfinex-paper-unsupported",
                    instrument_id=instrument.id,
                    timestamp=datetime.now(UTC),
                    side=order.side,
                    quantity=0.0,
                    fill_price=order.estimated_price,
                    notional=0.0,
                    fee=0.0,
                    slippage=0.0,
                    status=OrderStatus.REJECTED,
                    raw_response={"paper_symbol_unsupported": symbol, "error": resp.text[:300]},
                )
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
