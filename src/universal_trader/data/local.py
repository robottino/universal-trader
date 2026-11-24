"""Offline local CSV / Parquet DataProvider for reproducible research and testing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from universal_trader.core.models import Instrument
from universal_trader.core.registry import register_data_provider
from universal_trader.data.utils import (
    DataProviderError,
    normalize_ohlcv_dataframe,
    slice_date_range,
)


@register_data_provider("local")
@register_data_provider("csv_parquet")
class CsvParquetProvider:
    """Loads historical OHLCV data from a local .csv or .parquet file."""

    name = "local"

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def _resolve_path(self, instrument: Instrument) -> Path:
        explicit = instrument.provider_symbols.get("local")
        if explicit:
            p = Path(explicit)
            if p.is_absolute():
                return p
            if self.file_path.is_dir():
                return self.file_path / p
            return self.file_path.parent / p
        if self.file_path.is_dir():
            sanitized = instrument.id.replace("/", "_").replace(":", "_")
            for stem in (sanitized, instrument.base, instrument.id):
                for ext in (".parquet", ".csv"):
                    cand = self.file_path / f"{stem}{ext}"
                    if cand.exists():
                        return cand
            return self.file_path / f"{sanitized}.parquet"
        return self.file_path

    def fetch_ohlcv(
        self,
        instrument: Instrument,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        target_path = self._resolve_path(instrument)
        if not target_path.exists():
            raise DataProviderError(f"Local dataset file not found: {target_path}")
        suffix = target_path.suffix.lower()
        try:
            if suffix == ".parquet":
                raw = pd.read_parquet(target_path)
            elif suffix == ".csv":
                raw = pd.read_csv(target_path)
            else:
                raise DataProviderError(
                    f"Unsupported local dataset extension '{suffix}'. Expected .csv or .parquet"
                )
        except DataProviderError:
            raise
        except Exception as exc:
            raise DataProviderError(f"Failed reading '{target_path}': {exc}") from exc

        normalized = normalize_ohlcv_dataframe(raw)
        result = slice_date_range(normalized, start=start, end=end, limit=limit)
        if result.empty:
            raise DataProviderError(
                f"Local dataset '{target_path}' contains 0 bars in requested date range"
            )
        return result
