"""Mistral Timestral Forecaster adapter supporting 5M / 28M / 123M / 209M checkpoints."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from universal_trader.core.logging import get_logger
from universal_trader.core.models import ForecastRequest, ForecastResult
from universal_trader.core.protocols import ForecasterCapabilities
from universal_trader.core.registry import register_forecaster
from universal_trader.forecasters.device import DeviceInfo, DeviceManager

logger = get_logger(__name__)

TIMESTRAL_MODEL_SIZES: dict[str, str] = {
    "5m": "mistralai/Timestral-5M",
    "28m": "mistralai/Timestral-28M",
    "123m": "mistralai/Timestral-123M",
    "209m": "mistralai/Timestral-209M",
}


def resolve_timestral_model_id(model_id: str) -> str:
    """Resolve short aliases like '28M' or '123M' to official Hugging Face repo IDs."""
    return TIMESTRAL_MODEL_SIZES.get(model_id.strip().lower(), model_id.strip())


@register_forecaster("timestral")
class TimestralForecaster:
    """Adapter around official `mistralai/timestral` (`TimestralForecaster`).

    Features:
    - Lazy model loading and single-load memory reuse across walk-forward / backtest.
    - Adaptive batch chunking to respect GPU VRAM / Apple unified memory.
    - Native support for past-only, known-future, and categorical covariates + NaN.
    - Full mapping of all 21 probabilistic quantiles (0.01..0.99).
    - Controlled fallback from Apple Silicon MPS to CPU if an op (e.g. float64 scaler)
      is unsupported on MPS, logged explicitly once.
    """

    def __init__(
        self,
        model_id: str = "mistralai/Timestral-28M",
        context_len: int = 512,
        horizon: int = 24,
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 32,
        max_context: int = 32768,
        max_horizon: int = 1024,
        symmetric_averaging: bool = True,
        clip_nonnegative: bool = True,
        local_files_only: bool = False,
        **_: Any,
    ) -> None:
        self.model_id = resolve_timestral_model_id(model_id)
        self.context_len = context_len
        self.default_horizon = horizon
        self.device_manager = DeviceManager(requested_device=device, requested_dtype=dtype)
        self.device_info: DeviceInfo = self.device_manager.resolve()
        self.effective_device: str = self.device_info.device
        self.batch_size: int = DeviceManager.recommended_batch_size(self.device_info, batch_size)
        self.symmetric_averaging = symmetric_averaging
        self.clip_nonnegative = clip_nonnegative
        self.local_files_only = local_files_only
        self.capabilities = ForecasterCapabilities(
            supports_nan=True,
            supports_past_covariates=True,
            supports_known_future_covariates=True,
            supports_categorical_covariates=True,
            max_context=min(max_context, 32768),
            max_horizon=max_horizon,
        )
        self._backend: Any | None = None

    def _ensure_loaded(self) -> Any:
        if self._backend is not None:
            return self._backend
        from timestral import TimestralForecaster as _OfficialTimestral

        cache_dir = self.device_info.hf_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Loading Timestral checkpoint '%s' on device='%s' (cache='%s')",
            self.model_id,
            self.effective_device,
            cache_dir,
        )
        self._backend = _OfficialTimestral.from_pretrained(
            self.model_id,
            device=self.effective_device,
            max_context=self.capabilities.max_context,
            cache_dir=str(cache_dir),
            local_files_only=self.local_files_only,
        )
        if self.effective_device == "cuda" and self.device_info.dtype_name == "bfloat16":
            self._backend.model.to(dtype=torch.bfloat16)
        return self._backend

    def _fallback_to_cpu(self, reason: Exception) -> Any:
        logger.warning(
            "MPS execution encountered an unsupported operation (%s). "
            "Falling back to CPU for '%s' for the remainder of this session.",
            reason,
            self.model_id,
        )
        self.effective_device = "cpu"
        if self._backend is not None:
            self._backend.model.to(device="cpu", dtype=torch.float32)
            if hasattr(self._backend.model, "transformer") and hasattr(
                self._backend.model.transformer, "rope"
            ):
                self._backend.model.transformer.rope.rebuild_inv_freq()
            return self._backend
        return self._ensure_loaded()

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

        backend = self._ensure_loaded()
        total = len(requests)
        completed = 0
        results: list[ForecastResult | None] = [None] * total
        if progress_callback is not None:
            progress_callback(0, total)

        # Group requests with identical structural signature for tight zero-pad batching
        groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for idx, req in enumerate(requests):
            sig = (
                int(req.target.shape[-1]),
                req.horizon,
                tuple(sorted(req.past_covariates.keys())),
                tuple(sorted(req.known_future_covariates.keys())),
                tuple(sorted(req.categorical_past_covariates.keys())),
                tuple(sorted(req.categorical_known_covariates.keys())),
            )
            groups[sig].append(idx)

        for sig, indices in groups.items():
            _, horizon, past_keys, known_keys, cat_past_keys, cat_known_keys = sig
            for start in range(0, len(indices), self.batch_size):
                chunk_indices = indices[start : start + self.batch_size]
                chunk_reqs = [requests[i] for i in chunk_indices]

                context_arr = np.stack(
                    [np.asarray(r.target, dtype=np.float32).reshape(1, -1) for r in chunk_reqs],
                    axis=0,
                )  # (B, 1, T)

                past_arr = (
                    np.stack(
                        [
                            np.stack(
                                [np.asarray(r.past_covariates[k], dtype=np.float32) for k in past_keys],
                                axis=0,
                            )
                            for r in chunk_reqs
                        ],
                        axis=0,
                    )
                    if past_keys
                    else None
                )
                known_arr = (
                    np.stack(
                        [
                            np.stack(
                                [
                                    np.asarray(r.known_future_covariates[k], dtype=np.float32)
                                    for k in known_keys
                                ],
                                axis=0,
                            )
                            for r in chunk_reqs
                        ],
                        axis=0,
                    )
                    if known_keys
                    else None
                )
                cat_past_arr = (
                    np.stack(
                        [
                            np.stack(
                                [
                                    np.asarray(r.categorical_past_covariates[k], dtype=np.float64)
                                    for k in cat_past_keys
                                ],
                                axis=0,
                            )
                            for r in chunk_reqs
                        ],
                        axis=0,
                    )
                    if cat_past_keys
                    else None
                )
                cat_known_arr = (
                    np.stack(
                        [
                            np.stack(
                                [
                                    np.asarray(r.categorical_known_covariates[k], dtype=np.float64)
                                    for k in cat_known_keys
                                ],
                                axis=0,
                            )
                            for r in chunk_reqs
                        ],
                        axis=0,
                    )
                    if cat_known_keys
                    else None
                )

                forecast_obj = self._run_forecast_with_fallback(
                    backend=backend,
                    context=context_arr,
                    prediction_length=horizon,
                    past_covariates=past_arr,
                    known_covariates=known_arr,
                    categorical_past_covariates=cat_past_arr,
                    categorical_known_covariates=cat_known_arr,
                )
                backend = self._backend

                quantiles_np = forecast_obj.quantiles.detach().cpu().numpy()  # (B, 1, H, Q)
                levels = tuple(float(q) for q in forecast_obj.quantile_levels)

                for local_i, orig_idx in enumerate(chunk_indices):
                    req = requests[orig_idx]
                    q_map = {
                        round(q_level, 6): quantiles_np[local_i, 0, :, q_pos].astype(np.float64)
                        for q_pos, q_level in enumerate(levels)
                    }
                    results[orig_idx] = ForecastResult(
                        instrument_id=req.instrument_id,
                        model_id=self.model_id,
                        timestamp=req.timestamp,
                        horizon=req.horizon,
                        quantiles=q_map,
                        future_timestamps=list(req.future_timestamps),
                        metadata={
                            "device": self.effective_device,
                            "num_quantiles": len(levels),
                            "past_covariates": list(past_keys),
                            "known_future_covariates": list(known_keys),
                        },
                    )

                completed += len(chunk_indices)
                if progress_callback is not None:
                    progress_callback(completed, total)

        return [r for r in results if r is not None]

    def _run_forecast_with_fallback(
        self,
        *,
        backend: Any,
        context: np.ndarray,
        prediction_length: int,
        past_covariates: np.ndarray | None,
        known_covariates: np.ndarray | None,
        categorical_past_covariates: np.ndarray | None,
        categorical_known_covariates: np.ndarray | None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "prediction_length": prediction_length,
            "past_covariates": past_covariates,
            "known_covariates": known_covariates,
            "categorical_past_covariates": categorical_past_covariates,
            "categorical_known_covariates": categorical_known_covariates,
            "symmetric_averaging": self.symmetric_averaging,
            "clip_nonnegative": self.clip_nonnegative,
        }
        try:
            return backend.forecast(context, **kwargs)
        except (TypeError, RuntimeError) as exc:
            if self.effective_device == "mps" and DeviceManager.is_mps_unsupported_error(exc):
                cpu_backend = self._fallback_to_cpu(exc)
                return cpu_backend.forecast(context, **kwargs)
            raise
