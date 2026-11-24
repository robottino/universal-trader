"""Yahoo Finance DataProvider (equities, ETFs, macro indices, and crypto) with adj_close support."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import yfinance as yf

from universal_trader.core.calendar import parse_timeframe
from universal_trader.core.logging import get_logger
from universal_trader.core.models import AssetClass, Instrument, ensure_utc
from universal_trader.core.registry import register_data_provider
from universal_trader.data.utils import (
    DataProviderError,
    normalize_ohlcv_dataframe,
    slice_date_range,
)

logger = get_logger(__name__)

_YF_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "1d": "1d",
    "1w": "1wk",
    "1wk": "1wk",
}

MACRO_TICKER_MAP: dict[str, str] = {
    "vix": "^VIX",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "gold": "GC=F",
    "dxy": "DX-Y.NYB",
    "bond10y": "^TNX",
}


@register_data_provider("yahoo")
class YahooDataProvider:
    """Fetches OHLCV (including Adj Close for corporate actions) via yfinance."""

    name = "yahoo"

    def fetch_ohlcv(
        self,
        instrument: Instrument,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        tf = timeframe.strip().lower()
        if tf not in _YF_INTERVAL_MAP:
            raise DataProviderError(
                f"YahooDataProvider does not support timeframe '{timeframe}'. "
                f"Supported: {sorted(_YF_INTERVAL_MAP)}"
            )
        yf_interval = _YF_INTERVAL_MAP[tf]
        symbol = instrument.symbol_for("yahoo")
        if symbol == instrument.id and "/" in symbol and instrument.asset_class == AssetClass.CRYPTO:
            symbol = symbol.replace("/", "-")

        now_utc = datetime.now(UTC)
        end_dt = ensure_utc(end) if end is not None else now_utc
        if start is not None:
            start_dt = ensure_utc(start)
        else:
            bars = limit or 512
            delta = parse_timeframe(tf)
            pad = 1.2 if instrument.calendar == "24/7" else 2.2
            start_dt = end_dt - timedelta(seconds=delta.total_seconds() * bars * pad)

        # Yahoo's end date is exclusive for daily bars, so add a small buffer.
        fetch_end = end_dt + timedelta(days=1) if tf in {"1d", "1w", "1wk"} else end_dt + timedelta(minutes=5)

        try:
            df = yf.download(
                tickers=symbol,
                start=start_dt.strftime("%Y-%m-%d") if tf in {"1d", "1w", "1wk"} else start_dt,
                end=fetch_end.strftime("%Y-%m-%d") if tf in {"1d", "1w", "1wk"} else fetch_end,
                interval=yf_interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            raise DataProviderError(
                f"YahooDataProvider failed downloading '{symbol}' ({timeframe}): {exc}"
            ) from exc

        if df is None or df.empty:
            raise DataProviderError(
                f"YahooDataProvider returned no data for '{symbol}' ({timeframe}) "
                f"between {start_dt.isoformat()} and {end_dt.isoformat()}"
            )

        normalized = normalize_ohlcv_dataframe(df)
        result = slice_date_range(normalized, start=start, end=end, limit=limit)
        if result.empty:
            raise DataProviderError(
                f"YahooDataProvider returned 0 bars in requested window for '{symbol}'"
            )
        return result

    def fetch_macro_series(
        self,
        macro_name: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.Series:
        """Fetch a macro close series (e.g. vix, sp500, nasdaq, gold, dxy, bond10y)."""
        key = macro_name.strip().lower()
        ticker = MACRO_TICKER_MAP.get(key)
        if ticker is None:
            raise DataProviderError(
                f"Unknown macro series '{macro_name}'. Known: {sorted(MACRO_TICKER_MAP)}"
            )
        macro_inst = Instrument(
            id=ticker,
            asset_class=AssetClass.ETF,
            base=ticker,
            quote_currency="USD",
            calendar="XNYS",
            fractional=True,
            min_qty=0.0001,
            qty_step=0.0001,
            min_notional=0.0,
        )
        # Yahoo intraday limits (<1d) cap at 730d; for macro on intraday fall back cleanly to 1d if range > 700d
        tf = timeframe.strip().lower()
        span_days = (ensure_utc(end) - ensure_utc(start)).days
        effective_tf = "1d" if (tf not in {"1d", "1w", "1wk"} and span_days > 600) else tf
        # Pad start backwards by 10 days so merge_asof always finds a prior macro close
        padded_start = ensure_utc(start) - timedelta(days=10)
        df = self.fetch_ohlcv(macro_inst, timeframe=effective_tf, start=padded_start, end=end)
        series = df["close"].rename(key)
        return series
