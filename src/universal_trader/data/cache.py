"""Incremental Apache Parquet cache decorator for any DataProvider."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from universal_trader.core.calendar import parse_timeframe
from universal_trader.core.logging import get_logger
from universal_trader.core.models import Instrument, ensure_utc
from universal_trader.core.protocols import DataProvider
from universal_trader.data.utils import (
    DataProviderError,
    normalize_ohlcv_dataframe,
    slice_date_range,
)

logger = get_logger(__name__)


def _sanitize_segment(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())


class ParquetCacheStore:
    """Decorator wrapping a DataProvider with incremental Parquet caching per (provider, symbol, timeframe)."""

    def __init__(self, provider: DataProvider, cache_dir: Path | str) -> None:
        self._provider = provider
        self.name = f"cached:{provider.name}"
        self.cache_dir = Path(cache_dir)

    def cache_path(self, instrument: Instrument, timeframe: str) -> Path:
        prov = _sanitize_segment(self._provider.name)
        sym = _sanitize_segment(instrument.id)
        tf = _sanitize_segment(timeframe.lower())
        return self.cache_dir / prov / sym / f"{tf}.parquet"

    def _read_cache(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            return normalize_ohlcv_dataframe(df, allow_empty=True)
        except Exception as exc:
            logger.warning("Failed to read Parquet cache '%s' (%s); refetching.", path, exc)
            return None

    def _write_cache(self, path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f".tmp.{os.getpid()}.parquet")
        df.to_parquet(tmp_path, engine="pyarrow", index=True)
        os.replace(tmp_path, path)

    def fetch_ohlcv(
        self,
        instrument: Instrument,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        path = self.cache_path(instrument, timeframe)
        delta = parse_timeframe(timeframe)
        now_utc = datetime.now(UTC)
        req_end = ensure_utc(end) if end is not None else now_utc
        req_start: datetime | None = ensure_utc(start) if start is not None else None
        if req_start is None and limit is not None:
            # Pad lookback generously for weekends/holidays on non-24/7 markets
            pad_factor = 1.15 if instrument.calendar == "24/7" else 2.2
            req_start = req_end - timedelta(seconds=delta.total_seconds() * limit * pad_factor)

        cached = self._read_cache(path)
        frames: list[pd.DataFrame] = []
        updated = False

        if cached is None or cached.empty:
            logger.info(
                "Cache miss for %s (%s, %s); fetching from %s",
                instrument.id,
                timeframe,
                self._provider.name,
                self._provider.name,
            )
            fresh = self._provider.fetch_ohlcv(
                instrument=instrument,
                timeframe=timeframe,
                start=req_start,
                end=req_end,
                limit=limit if req_start is None else None,
            )
            merged = normalize_ohlcv_dataframe(fresh)
            self._write_cache(path, merged)
            return slice_date_range(merged, start=start, end=end, limit=limit)

        frames.append(cached)
        cache_min = cached.index.min().to_pydatetime()
        cache_max = cached.index.max().to_pydatetime()

        # Left gap: requested earlier history than cache holds
        if req_start is not None and req_start < cache_min - delta:
            logger.info(
                "Incremental backfill [%s .. %s] for %s (%s)",
                req_start.isoformat(),
                cache_min.isoformat(),
                instrument.id,
                timeframe,
            )
            left_df = self._provider.fetch_ohlcv(
                instrument=instrument,
                timeframe=timeframe,
                start=req_start,
                end=cache_min,
                limit=None,
            )
            if not left_df.empty:
                frames.append(normalize_ohlcv_dataframe(left_df, allow_empty=True))
                updated = True

        # Right gap: requested newer bars than cache holds
        if req_end > cache_max + delta:
            fetch_from = cache_max - delta
            logger.info(
                "Incremental update [%s .. %s] for %s (%s)",
                fetch_from.isoformat(),
                req_end.isoformat(),
                instrument.id,
                timeframe,
            )
            try:
                right_df = self._provider.fetch_ohlcv(
                    instrument=instrument,
                    timeframe=timeframe,
                    start=fetch_from,
                    end=req_end,
                    limit=None,
                )
                if not right_df.empty:
                    frames.append(normalize_ohlcv_dataframe(right_df, allow_empty=True))
                    updated = True
            except DataProviderError:
                # Propagate provider failures explicitly (fail-fast: never silently pretend cache is fresh)
                raise

        merged = pd.concat(frames)
        merged = normalize_ohlcv_dataframe(merged)
        if updated:
            self._write_cache(path, merged)

        result = slice_date_range(merged, start=start, end=end, limit=limit)
        if result.empty:
            raise DataProviderError(
                f"No OHLCV data available for {instrument.id} ({timeframe}) in requested window"
            )
        return result
