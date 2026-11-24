"""Deterministic multi-quantile MockForecaster for unit/integration testing & CI."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from universal_trader.core.models import ForecastRequest, ForecastResult
from universal_trader.core.protocols import ForecasterCapabilities
from universal_trader.core.registry import register_forecaster

STANDARD_21_QUANTILES: tuple[float, ...] = (
    0.01,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.99,
)

# Standard normal z-scores for the 21 Timestral quantile levels
_NORMAL_Z: dict[float, float] = {
    0.01: -2.3263,
    0.05: -1.6449,
    0.10: -1.2816,
    0.15: -1.0364,
    0.20: -0.8416,
    0.25: -0.6745,
    0.30: -0.5244,
    0.35: -0.3853,
    0.40: -0.2533,
    0.45: -0.1257,
    0.50: 0.0,
    0.55: 0.1257,
    0.60: 0.2533,
    0.65: 0.3853,
    0.70: 0.5244,
    0.75: 0.6745,
    0.80: 0.8416,
    0.85: 1.0364,
    0.90: 1.2816,
    0.95: 1.6449,
    0.99: 2.3263,
}


@register_forecaster("mock")
class MockForecaster:
    """Fast, strictly causal, deterministic 21-quantile forecaster for offline testing."""

    def __init__(
        self,
        model_id: str = "mock/deterministic-21q",
        drift_override: float | None = None,
        vol_override: float | None = None,
        momentum_window: int = 12,
        **_: Any,
    ) -> None:
        self.model_id = model_id
        self.drift_override = drift_override
        self.vol_override = vol_override
        self.momentum_window = max(2, int(momentum_window))
        self.capabilities = ForecasterCapabilities(
            supports_nan=True,
            supports_past_covariates=True,
            supports_known_future_covariates=True,
            supports_categorical_covariates=True,
            max_context=32768,
            max_horizon=1024,
        )
        self.call_count: int = 0
        self.request_count: int = 0

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
        self.call_count += 1
        self.request_count += len(requests)
        total = len(requests)
        out: list[ForecastResult] = []
        if progress_callback is not None:
            progress_callback(0, total)
        for req in requests:
            arr = np.asarray(req.target, dtype=np.float64).reshape(-1)
            finite = arr[np.isfinite(arr)]
            last_price = float(finite[-1]) if len(finite) > 0 else 100.0
            win = finite[-self.momentum_window :] if len(finite) >= 2 else finite

            if self.drift_override is not None:
                step_drift = float(self.drift_override)
            elif len(win) >= 2 and win[0] > 0:
                log_rets = np.diff(np.log(np.clip(win, 1e-9, None)))
                step_drift = float(np.clip(np.mean(log_rets), -0.02, 0.02))
            else:
                step_drift = 0.0

            if self.vol_override is not None:
                step_vol = float(self.vol_override)
            elif len(win) >= 3:
                log_rets = np.diff(np.log(np.clip(win, 1e-9, None)))
                step_vol = float(max(0.002, np.std(log_rets)))
            else:
                step_vol = 0.01

            steps = np.arange(1, req.horizon + 1, dtype=np.float64)
            q_map: dict[float, np.ndarray] = {}
            for q in STANDARD_21_QUANTILES:
                z = _NORMAL_Z[q]
                log_path = step_drift * steps + z * step_vol * np.sqrt(steps)
                prices = np.maximum(1e-6, last_price * np.exp(log_path))
                q_map[q] = prices

            out.append(
                ForecastResult(
                    instrument_id=req.instrument_id,
                    model_id=self.model_id,
                    timestamp=req.timestamp,
                    horizon=req.horizon,
                    quantiles=q_map,
                    future_timestamps=list(req.future_timestamps),
                    metadata={"device": "cpu", "mock": True},
                )
            )
        if progress_callback is not None:
            progress_callback(total, total)
        return out
