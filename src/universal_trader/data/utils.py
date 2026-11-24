"""Shared OHLCV normalization and validation utilities."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from universal_trader.core.models import ensure_utc

REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
STANDARD_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume", "adj_close")


class DataProviderError(RuntimeError):
    """Explicit fail-fast error raised when a critical market data source fails."""


def normalize_ohlcv_dataframe(df: pd.DataFrame, *, allow_empty: bool = False) -> pd.DataFrame:
    """Normalize an OHLCV DataFrame to sorted, deduplicated, UTC-indexed float64 columns."""
    if df is None or df.empty:
        if allow_empty:
            empty = pd.DataFrame(columns=list(STANDARD_OHLCV_COLUMNS), dtype=float)
            empty.index = pd.DatetimeIndex([], tz=UTC, name="timestamp")
            return empty
        raise DataProviderError("Data provider returned empty OHLCV DataFrame")

    out = df.copy()

    # Flatten MultiIndex columns (e.g., from yfinance >= 0.2.50)
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[0]) for col in out.columns]

    rename_map: dict[str, str] = {}
    for col in out.columns:
        low = str(col).strip().lower().replace(" ", "_")
        if low in {"adj_close", "adjclose", "adjusted_close"}:
            rename_map[col] = "adj_close"
        elif low in REQUIRED_OHLCV_COLUMNS:
            rename_map[col] = low
        elif low in {"timestamp", "datetime", "date", "time"}:
            rename_map[col] = "timestamp"
    out = out.rename(columns=rename_map)

    if "timestamp" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        out = out.set_index("timestamp")

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    elif out.index.tz is None:
        out.index = out.index.tz_localize(UTC)
    else:
        out.index = out.index.tz_convert(UTC)

    out.index.name = "timestamp"

    missing = [c for c in REQUIRED_OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise DataProviderError(f"OHLCV DataFrame missing required columns: {missing}")

    if "adj_close" not in out.columns:
        out["adj_close"] = out["close"]

    out = out[list(STANDARD_OHLCV_COLUMNS)].copy()
    for col in STANDARD_OHLCV_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float64)

    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["open", "high", "low", "close"])

    if out.empty and not allow_empty:
        raise DataProviderError("OHLCV DataFrame became empty after cleaning NaN OHLC rows")
    return out


def slice_date_range(
    df: pd.DataFrame,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Filter UTC-indexed OHLCV DataFrame by [start, end] and optional trailing limit."""
    out = df
    if start is not None:
        out = out[out.index >= pd.Timestamp(ensure_utc(start))]
    if end is not None:
        out = out[out.index <= pd.Timestamp(ensure_utc(end))]
    if limit is not None and limit > 0 and len(out) > limit:
        out = out.iloc[-limit:]
    return out
