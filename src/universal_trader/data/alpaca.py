"""Alpaca Markets Data API v2 provider for regulated US equities and ETFs."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd

from universal_trader.core.calendar import parse_timeframe
from universal_trader.core.logging import register_secret
from universal_trader.core.models import Instrument, ensure_utc
from universal_trader.core.registry import register_data_provider
from universal_trader.data.utils import (
    DataProviderError,
    normalize_ohlcv_dataframe,
    slice_date_range,
)

_ALPACA_TIMEFRAMES: dict[str, str] = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h": "1Hour",
    "4h": "4Hour",
    "1d": "1Day",
    "1w": "1Week",
    "1wk": "1Week",
}


@register_data_provider("alpaca")
class AlpacaDataProvider:
    """Fetches US stock/ETF OHLCV bars from Alpaca Data API v2 (adjustment=all)."""

    name = "alpaca"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = "https://data.alpaca.markets",
        feed: str = "iex",
        timeout_seconds: float = 15.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET", "")
        register_secret(self.api_key)
        register_secret(self.api_secret)
        self.base_url = base_url.rstrip("/")
        self.feed = feed
        self.timeout_seconds = timeout_seconds

    def fetch_ohlcv(
        self,
        instrument: Instrument,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        if not self.api_key or not self.api_secret:
            raise DataProviderError(
                "AlpacaDataProvider requires ALPACA_API_KEY and ALPACA_API_SECRET"
            )
        tf = timeframe.strip().lower()
        if tf not in _ALPACA_TIMEFRAMES:
            raise DataProviderError(
                f"AlpacaDataProvider does not support timeframe '{timeframe}'. "
                f"Supported: {sorted(_ALPACA_TIMEFRAMES)}"
            )

        symbol = instrument.symbol_for("alpaca").upper()
        now_utc = datetime.now(UTC)
        end_dt = ensure_utc(end) if end is not None else now_utc
        if start is not None:
            start_dt = ensure_utc(start)
        else:
            bars = limit or 512
            delta = parse_timeframe(tf)
            start_dt = end_dt - timedelta(seconds=delta.total_seconds() * bars * 2.2)

        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }
        url = f"{self.base_url}/v2/stocks/{symbol}/bars"
        rows: list[dict[str, object]] = []
        page_token: str | None = None

        with httpx.Client(timeout=self.timeout_seconds) as client:
            while True:
                params: dict[str, str | int] = {
                    "timeframe": _ALPACA_TIMEFRAMES[tf],
                    "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "adjustment": "all",
                    "feed": self.feed,
                    "limit": 10000,
                }
                if page_token:
                    params["page_token"] = page_token
                try:
                    resp = client.get(url, headers=headers, params=params)
                except httpx.HTTPError as exc:
                    raise DataProviderError(f"AlpacaDataProvider network error: {exc}") from exc
                if resp.status_code != 200:
                    raise DataProviderError(
                        f"AlpacaDataProvider HTTP {resp.status_code}: {resp.text[:300]}"
                    )
                payload = resp.json()
                for bar in payload.get("bars") or []:
                    rows.append(
                        {
                            "timestamp": pd.Timestamp(bar["t"]).tz_convert(UTC),
                            "open": float(bar["o"]),
                            "high": float(bar["h"]),
                            "low": float(bar["l"]),
                            "close": float(bar["c"]),
                            "volume": float(bar["v"]),
                            "adj_close": float(bar["c"]),
                        }
                    )
                page_token = payload.get("next_page_token")
                if not page_token:
                    break

        if not rows:
            raise DataProviderError(
                f"AlpacaDataProvider returned 0 bars for '{symbol}' ({timeframe})"
            )

        df = pd.DataFrame(rows).set_index("timestamp")
        normalized = normalize_ohlcv_dataframe(df)
        return slice_date_range(normalized, start=start, end=end, limit=limit)
