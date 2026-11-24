"""Lightweight NoopForecaster for pure technical/rule-based strategies (zero ML overhead)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from universal_trader.core.models import ForecastRequest, ForecastResult
from universal_trader.core.protocols import ForecasterCapabilities
from universal_trader.core.registry import register_forecaster


@register_forecaster("noop")
class NoopForecaster:
    """Zero-overhead flat forecaster for strategies that operate directly on market candles."""

    def __init__(self, model_id: str = "noop/flat", **_: Any) -> None:
        self.model_id = model_id
        self.capabilities = ForecasterCapabilities(
            supports_nan=True,
            supports_past_covariates=False,
            supports_known_future_covariates=False,
            supports_categorical_covariates=False,
            max_context=32768,
            max_horizon=1024,
        )

    def predict(self, request: ForecastRequest) -> ForecastResult:
        return self.predict_batch([request])[0]

    def predict_batch(
        self,
        requests: Sequence[ForecastRequest],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[ForecastResult]:
        if not requests:
            return []
        total = len(requests)
        if progress_callback is not None:
            progress_callback(0, total)
        out: list[ForecastResult] = []
        for req in requests:
            arr = np.asarray(req.target, dtype=np.float64).reshape(-1)
            finite = arr[np.isfinite(arr)]
            spot = float(finite[-1]) if len(finite) > 0 else 1.0
            flat = np.full(req.horizon, spot, dtype=np.float64)
            out.append(
                ForecastResult(
                    instrument_id=req.instrument_id,
                    model_id=self.model_id,
                    timestamp=req.timestamp,
                    horizon=req.horizon,
                    quantiles={0.05: flat, 0.10: flat, 0.50: flat, 0.90: flat, 0.95: flat},
                    future_timestamps=list(req.future_timestamps),
                    metadata={"noop": True},
                )
            )
        if progress_callback is not None:
            progress_callback(total, total)
        return out
