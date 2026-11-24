"""Universal SQLite-persisted LocalPaperBroker supporting any stock, ETF, crypto, or FX."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from universal_trader.core.config import FeeModelConfig
from universal_trader.core.models import (
    ApprovedOrder,
    FillResult,
    Instrument,
    OrderSide,
    OrderStatus,
    PortfolioState,
    Position,
    Quote,
    ensure_utc,
)
from universal_trader.core.protocols import DataProvider
from universal_trader.core.registry import register_broker
from universal_trader.storage.sqlite_store import SQLiteStateStore


@register_broker("local_paper")
class LocalPaperBroker:
    """Paper-trading broker backed by SQLiteStateStore and live DataProvider quotes."""

    name = "local_paper"

    def __init__(
        self,
        *,
        store: SQLiteStateStore,
        account_key: str,
        quote_currency: str,
        initial_cash: float = 10000.0,
        fee_model: FeeModelConfig | None = None,
        slippage_bps: float = 5.0,
        spread_bps: float = 2.0,
        data_provider: DataProvider | None = None,
        timeframe: str = "1d",
        price_column: str = "close",
    ) -> None:
        self.store = store
        self.account_key = account_key
        self.quote_currency = quote_currency
        self.initial_cash = float(initial_cash)
        self.fee_model = fee_model or FeeModelConfig()
        self.slippage_bps = float(slippage_bps)
        self.spread_bps = float(spread_bps)
        self.data_provider = data_provider
        self.timeframe = timeframe
        self.price_column = price_column
        self._last_quote: Quote | None = None

        # Ensure account exists in SQLite
        self.store.load_or_init_paper_account(
            account_key=self.account_key,
            quote_currency=self.quote_currency,
            initial_cash=self.initial_cash,
        )

    def set_quote(self, quote: Quote) -> None:
        """Supply latest market quote (e.g. from already-fetched OHLCV snapshot)."""
        self._last_quote = quote

    def _compute_fee(self, notional: float) -> float:
        if notional <= 0:
            return 0.0
        return max(notional * self.fee_model.pct, self.fee_model.fixed_min)

    def get_quote(self, instrument: Instrument) -> Quote:
        if self._last_quote is not None and self._last_quote.instrument_id == instrument.id:
            return self._last_quote
        if self.data_provider is None:
            raise RuntimeError(
                "LocalPaperBroker needs either a DataProvider or a quote set via set_quote()"
            )
        df = self.data_provider.fetch_ohlcv(instrument=instrument, timeframe=self.timeframe, limit=2)
        col = self.price_column if self.price_column in df.columns else "close"
        last_price = float(df[col].iloc[-1])
        last_ts = ensure_utc(df.index[-1])
        half_spread = last_price * (self.spread_bps / 20_000.0)
        quote = Quote(
            instrument_id=instrument.id,
            timestamp=last_ts,
            bid=max(1e-9, last_price - half_spread),
            ask=last_price + half_spread,
            last=last_price,
        )
        self._last_quote = quote
        return quote

    def get_portfolio(self, instrument: Instrument, quote: Quote | None = None) -> PortfolioState:
        acct = self.store.load_or_init_paper_account(
            account_key=self.account_key,
            quote_currency=self.quote_currency,
            initial_cash=self.initial_cash,
        )
        q = quote or self._last_quote
        spot = q.last if q is not None else 0.0
        ts = q.timestamp if q is not None else datetime.now(UTC)

        pos_models: dict[str, Position] = {}
        pos_val = 0.0
        for inst_id, pdata in acct["positions"].items():
            mkt_px = spot if (inst_id == instrument.id and spot > 0) else pdata["avg_entry_price"]
            pos_models[inst_id] = Position(
                instrument_id=inst_id,
                quantity=pdata["quantity"],
                avg_entry_price=pdata["avg_entry_price"],
                market_price=mkt_px,
            )
            pos_val += pdata["quantity"] * mkt_px

        equity = max(0.0, float(acct["cash"]) + pos_val)
        hwm = max(float(acct["high_water_mark"]), equity)
        return PortfolioState(
            timestamp=ts,
            quote_currency=acct["quote_currency"],
            cash=float(acct["cash"]),
            positions=pos_models,
            equity=equity,
            high_water_mark=hwm,
            halted=bool(acct["halted"]),
        )

    def submit_order(self, order: ApprovedOrder, instrument: Instrument) -> FillResult:
        acct = self.store.load_or_init_paper_account(
            account_key=self.account_key,
            quote_currency=self.quote_currency,
            initial_cash=self.initial_cash,
        )
        q = self.get_quote(instrument)
        slip_rate = self.slippage_bps / 10_000.0
        order_id = f"paper-{uuid.uuid4().hex[:12]}"
        positions: dict[str, dict[str, float]] = dict(acct["positions"])
        cash = float(acct["cash"])

        if order.side == OrderSide.BUY:
            fill_price = q.ask * (1.0 + slip_rate)
            qty = instrument.round_qty(order.quantity)
            notional = qty * fill_price
            fee = self._compute_fee(notional)
            if notional + fee > cash + 1e-9:
                max_notional = min(cash / (1.0 + self.fee_model.pct), max(0.0, cash - self.fee_model.fixed_min))
                qty = instrument.round_qty(max_notional / fill_price)
                notional = qty * fill_price
                fee = self._compute_fee(notional)
            if qty < instrument.min_qty or notional < instrument.min_notional:
                return FillResult(
                    order_id=order_id,
                    instrument_id=instrument.id,
                    timestamp=order.timestamp,
                    side=order.side,
                    quantity=0.0,
                    fill_price=fill_price,
                    notional=0.0,
                    fee=0.0,
                    slippage=0.0,
                    status=OrderStatus.REJECTED,
                    raw_response={"reason": "Insufficient cash at paper fill price"},
                )
            cash -= notional + fee
            prev = positions.get(instrument.id, {"quantity": 0.0, "avg_entry_price": 0.0})
            new_qty = prev["quantity"] + qty
            new_avg = (prev["quantity"] * prev["avg_entry_price"] + qty * fill_price) / new_qty
            positions[instrument.id] = {"quantity": new_qty, "avg_entry_price": new_avg}
            slippage = max(0.0, fill_price - q.last) * qty
        else:
            fill_price = max(1e-9, q.bid * (1.0 - slip_rate))
            prev = positions.get(instrument.id, {"quantity": 0.0, "avg_entry_price": 0.0})
            avail = prev["quantity"]
            qty = min(avail, order.quantity)
            if not instrument.fractional:
                qty = min(avail, instrument.round_qty(qty))
            if qty < instrument.min_qty or avail <= 0:
                return FillResult(
                    order_id=order_id,
                    instrument_id=instrument.id,
                    timestamp=order.timestamp,
                    side=order.side,
                    quantity=0.0,
                    fill_price=fill_price,
                    notional=0.0,
                    fee=0.0,
                    slippage=0.0,
                    status=OrderStatus.REJECTED,
                    raw_response={"reason": "Insufficient position for paper sell"},
                )
            notional = qty * fill_price
            fee = self._compute_fee(notional)
            cash += max(0.0, notional - fee)
            rem = max(0.0, avail - qty)
            if rem < 1e-12:
                positions.pop(instrument.id, None)
            else:
                positions[instrument.id] = {"quantity": rem, "avg_entry_price": prev["avg_entry_price"]}
            slippage = max(0.0, q.last - fill_price) * qty

        new_pos_val = sum(
            p["quantity"] * (q.last if inst_id == instrument.id else p["avg_entry_price"])
            for inst_id, p in positions.items()
        )
        new_equity = cash + new_pos_val
        new_hwm = max(float(acct["high_water_mark"]), new_equity)

        self.store.save_paper_account(
            account_key=self.account_key,
            quote_currency=acct["quote_currency"],
            cash=cash,
            high_water_mark=new_hwm,
            halted=bool(acct["halted"]),
            positions=positions,
        )

        return FillResult(
            order_id=order_id,
            instrument_id=instrument.id,
            timestamp=order.timestamp,
            side=order.side,
            quantity=qty,
            fill_price=fill_price,
            notional=notional,
            fee=fee,
            slippage=slippage,
            status=OrderStatus.FILLED,
            raw_response={"broker": "local_paper", "account_key": self.account_key},
        )
