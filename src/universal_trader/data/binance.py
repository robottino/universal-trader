"""Binance spot OHLCV DataProvider with pagination and exponential backoff."""

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

_BINANCE_INTERVALS = {
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "1w",
}


@register_data_provider("binance")
class BinanceDataProvider:
    """Fetches spot OHLCV klines from Binance public REST API with robust pagination & retry."""

    name = "binance"

    def __init__(
        self,
        base_urls: tuple[str, ...] = (
            "https://data-api.binance.vision",
            "https://api.binance.com",
        ),
        timeout_seconds: float = 15.0,
        max_retries: int = 4,
    ) -> None:
        self.base_urls = base_urls
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def _request_klines(self, params: dict[str, str | int]) -> list[list[Any]]:
        last_error: Exception | None = None
        with httpx.Client(timeout=self.timeout_seconds) as client:
            for base_url in self.base_urls:
                url = f"{base_url.rstrip('/')}/api/v3/klines"
                for attempt in range(self.max_retries):
                    try:
                        resp = client.get(url, params=params)
                        if resp.status_code == 200:
                            payload = resp.json()
                            if not isinstance(payload, list):
                                raise DataProviderError(f"Unexpected Binance response: {payload}")
                            return payload
                        if resp.status_code in {418, 429, 500, 502, 503, 504}:
                            backoff = min(8.0, 0.5 * (2**attempt))
                            logger.warning(
                                "Binance HTTP %s on %s; retrying in %.1fs",
                                resp.status_code,
                                base_url,
                                backoff,
                            )
                            time.sleep(backoff)
                            continue
                        # 451 geo-block or 4xx: try next endpoint immediately
                        last_error = DataProviderError(
                            f"Binance HTTP {resp.status_code} ({base_url}): {resp.text[:200]}"
                        )
                        break
                    except httpx.HTTPError as exc:
                        last_error = exc
                        backoff = min(8.0, 0.5 * (2**attempt))
                        time.sleep(backoff)
        raise DataProviderError(f"BinanceDataProvider failed after retries: {last_error}")

    def fetch_ohlcv(
        self,
        instrument: Instrument,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        tf = timeframe.strip().lower()
        if tf == "1wk":
            tf = "1w"
        if tf not in _BINANCE_INTERVALS:
            raise DataProviderError(
                f"BinanceDataProvider does not support timeframe '{timeframe}'. "
                f"Supported: {sorted(_BINANCE_INTERVALS)}"
            )

        symbol = instrument.symbol_for("binance").replace("/", "").replace("-", "").upper()
        delta = parse_timeframe(tf)
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
        max_pages = 500
        pages = 0
        while cursor_ms <= end_ms and pages < max_pages:
            pages += 1
            batch = self._request_klines(
                {
                    "symbol": symbol,
                    "interval": tf,
                    "startTime": cursor_ms,
                    "endTime": end_ms,
                    "limit": 1000,
                }
            )
            if not batch:
                break
            for k in batch:
                open_ts = pd.Timestamp(int(k[0]), unit="ms", tz=UTC)
                rows.append(
                    {
                        "timestamp": open_ts,
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "adj_close": float(k[4]),
                    }
                )
            last_open_ms = int(batch[-1][0])
            if last_open_ms < cursor_ms:
                break
            cursor_ms = last_open_ms + step_ms
            if len(batch) < 1000:
                break

        if not rows:
            raise DataProviderError(
                f"BinanceDataProvider returned 0 bars for '{symbol}' ({tf}) "
                f"between {start_dt.isoformat()} and {end_dt.isoformat()}"
            )

        df = pd.DataFrame(rows).set_index("timestamp")
        normalized = normalize_ohlcv_dataframe(df)
        return slice_date_range(normalized, start=start, end=end, limit=limit)
