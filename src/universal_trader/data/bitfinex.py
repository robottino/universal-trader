"""Bitfinex public REST v2 OHLCV DataProvider with pagination and retry."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pandas as pd

from universal_trader.core.calendar import parse_timeframe
from universal_trader.core.logging import get_logger
from universal_trader.core.models import Instrument, ensure_utc
from universal_trader.core.registry import register_data_provider
from universal_trader.data.utils import (
    DataProviderError,
    normalize_ohlcv_dataframe,
    slice_date_range,
)

logger = get_logger(__name__)

_BITFINEX_TF_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "3h": "3h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1D",
    "1w": "1W",
    "1wk": "1W",
}


@register_data_provider("bitfinex")
class BitfinexDataProvider:
    """Fetches public spot OHLCV candles from Bitfinex v2 REST API (zero auth required)."""

    name = "bitfinex"

    def __init__(
        self,
        pub_url: str = "https://api-pub.bitfinex.com",
        timeout_seconds: float = 20.0,
        max_retries: int = 4,
    ) -> None:
        self.pub_url = pub_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    @staticmethod
    def _public_symbol(instrument: Instrument) -> str:
        explicit = instrument.provider_symbols.get("bitfinex")
        if explicit and "TEST" not in explicit.upper():
            return explicit
        base = instrument.base.upper()
        quote = instrument.quote_currency.upper()
        if len(base) == 3 and len(quote) == 3:
            return f"t{base}{quote}"
        return f"t{base}:{quote}"

    def _request_candles(self, tf_code: str, symbol: str, params: dict[str, int]) -> list[list[Any]]:
        url = f"{self.pub_url}/v2/candles/trade:{tf_code}:{symbol}/hist"
        last_error: Exception | None = None
        with httpx.Client(timeout=self.timeout_seconds) as client:
            for attempt in range(self.max_retries):
                try:
                    resp = client.get(url, params=params)
                    if resp.status_code == 200:
                        payload = resp.json()
                        if not isinstance(payload, list):
                            raise DataProviderError(f"Unexpected Bitfinex candles payload: {payload}")
                        return payload
                    if resp.status_code in {429, 500, 502, 503, 504}:
                        backoff = min(10.0, 1.0 * (2**attempt))
                        logger.warning("Bitfinex candles HTTP %s; retrying in %.1fs", resp.status_code, backoff)
                        time.sleep(backoff)
                        continue
                    raise DataProviderError(f"Bitfinex candles HTTP {resp.status_code}: {resp.text[:200]}")
                except httpx.HTTPError as exc:
                    last_error = exc
                    time.sleep(min(10.0, 1.0 * (2**attempt)))
        raise DataProviderError(f"BitfinexDataProvider failed after retries: {last_error}")

    def fetch_ohlcv(
        self,
        instrument: Instrument,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        tf_norm = timeframe.strip().lower()
        tf_code = _BITFINEX_TF_MAP.get(tf_norm)
        if tf_code is None:
            raise DataProviderError(
                f"BitfinexDataProvider does not support timeframe '{timeframe}'. "
                f"Supported: {sorted(_BITFINEX_TF_MAP)}"
            )

        symbol = self._public_symbol(instrument)
        delta = parse_timeframe(timeframe)
        step_ms = int(delta.total_seconds() * 1000)

        now_utc = datetime.now(UTC)
        end_dt = ensure_utc(end) if end is not None else now_utc
        if start is not None:
            start_dt = ensure_utc(start)
        else:
            bars = limit or 512
            start_dt = end_dt - timedelta(milliseconds=step_ms * (bars + 2))

        cursor_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        rows: list[dict[str, object]] = []
        max_pages = 200
        pages = 0
        while cursor_ms <= end_ms and pages < max_pages:
            pages += 1
            # sort=1 returns candles ascending by MTS; Bitfinex max limit is 10000
            batch = self._request_candles(
                tf_code,
                symbol,
                {"start": cursor_ms, "end": end_ms, "limit": 10000, "sort": 1},
            )
            if not batch:
                break
            for c in batch:
                # Bitfinex format: [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
                if not isinstance(c, list) or len(c) < 6:
                    continue
                open_ts = pd.Timestamp(int(c[0]), unit="ms", tz=UTC)
                rows.append(
                    {
                        "timestamp": open_ts,
                        "open": float(c[1]),
                        "high": float(c[3]),
                        "low": float(c[4]),
                        "close": float(c[2]),
                        "volume": max(0.0, float(c[5])),
                        "adj_close": float(c[2]),
                    }
                )
            last_mts = int(batch[-1][0])
            if last_mts < cursor_ms:
                break
            cursor_ms = last_mts + step_ms
            if len(batch) < 10000:
                break

        if not rows:
            raise DataProviderError(
                f"BitfinexDataProvider returned 0 bars for '{symbol}' ({timeframe}) "
                f"between {start_dt.isoformat()} and {end_dt.isoformat()}"
            )

        df = pd.DataFrame(rows).set_index("timestamp")
        normalized = normalize_ohlcv_dataframe(df)
        return slice_date_range(normalized, start=start, end=end, limit=limit)
