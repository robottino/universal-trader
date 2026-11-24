"""Persistent disk ForecastCache indexed by deterministic config/feature hash."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from universal_trader.core.config import AppConfig
from universal_trader.core.logging import get_logger
from universal_trader.core.models import ForecastResult, ensure_utc

logger = get_logger(__name__)


class ForecastCache:
    """Disk cache storing precomputed probabilistic ForecastResults by model/instrument/feature hash."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def build_key(config: AppConfig, model_id: str | None = None) -> str:
        """Compute a deterministic SHA-256 cache key for forecast identity."""
        payload: dict[str, Any] = {
            "model_id": model_id or config.forecaster.params.model_id,
            "forecaster_type": config.forecaster.type,
            "instrument_id": config.instrument.id,
            "timeframe": config.data.timeframe,
            "price_column": config.data.price_column,
            "context_len": config.forecaster.params.context_len,
            "horizon": config.forecaster.params.horizon,
            "symmetric_averaging": config.forecaster.params.symmetric_averaging,
            "clip_nonnegative": config.forecaster.params.clip_nonnegative,
            "past_only": sorted(config.data.features.past_only),
            "known_future": sorted(config.data.features.known_future),
            "categorical_past": sorted(config.data.features.categorical_past),
            "categorical_known": sorted(config.data.features.categorical_known),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()[:24]

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"fc_{key}.json.gz"

    def load(self, key: str) -> dict[datetime, ForecastResult]:
        path = self._path(key)
        if not path.exists():
            return {}
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                raw = json.load(fh)
            items = raw.get("forecasts", {})
            return {
                ensure_utc(ts_str): ForecastResult.from_serializable_dict(fc_dict)
                for ts_str, fc_dict in items.items()
            }
        except Exception as exc:
            logger.warning("ForecastCache '%s' unreadable (%s); ignoring cache.", path, exc)
            return {}

    def save(self, key: str, forecasts: dict[datetime, ForecastResult]) -> Path:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": key,
            "count": len(forecasts),
            "forecasts": {
                ensure_utc(ts).isoformat(): fc.to_serializable_dict()
                for ts, fc in sorted(forecasts.items(), key=lambda kv: kv[0])
            },
        }
        tmp_path = path.with_suffix(f".tmp.{os.getpid()}.gz")
        with gzip.open(tmp_path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp_path, path)
        return path
