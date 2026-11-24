"""Hardware detection and execution policy for CUDA, Apple Silicon MPS, and CPU."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path

import torch

from universal_trader.core.config import RuntimePaths
from universal_trader.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved compute backend and hardware capabilities."""

    device: str  # "cuda" | "mps" | "cpu"
    dtype_name: str  # "float32" | "bfloat16"
    accelerator_name: str
    memory_gb: float | None
    cuda_available: bool
    mps_available: bool
    cpu_count: int
    hf_cache_dir: Path
    torch_version: str
    platform_info: str

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.dtype_name == "bfloat16":
            return torch.bfloat16
        return torch.float32


class DeviceManager:
    """Detects and resolves compute device (CUDA -> MPS -> CPU), safe dtype, and HF cache."""

    def __init__(
        self,
        requested_device: str = "auto",
        requested_dtype: str = "auto",
        base_dir: Path | None = None,
    ) -> None:
        self.requested_device = requested_device.strip().lower()
        self.requested_dtype = requested_dtype.strip().lower()
        self.paths = RuntimePaths.resolve(base_dir)
        self._configure_hf_cache()

    def _configure_hf_cache(self) -> None:
        """Ensure Hugging Face cache stays inside the portable project/volume cache dir."""
        if not os.environ.get("HF_HOME"):
            os.environ["HF_HOME"] = str(self.paths.hf_cache_dir)
        # Allow safe PyTorch MPS fallback for individual unsupported ops when possible
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    def resolve(self) -> DeviceInfo:
        cuda_ok = bool(torch.cuda.is_available())
        mps_ok = bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        )

        # Priority: explicit non-auto config > UTRADER_DEVICE env > auto (cuda -> mps -> cpu)
        env_override = os.environ.get("UTRADER_DEVICE", "").strip().lower()
        target = self.requested_device
        if target == "auto" and env_override in {"cuda", "mps", "cpu"}:
            target = env_override

        if target == "auto":
            if cuda_ok:
                selected = "cuda"
            elif mps_ok:
                selected = "mps"
            else:
                selected = "cpu"
        elif target == "cuda":
            if not cuda_ok:
                logger.warning("Requested device 'cuda' is unavailable; falling back to CPU.")
                selected = "cpu"
            else:
                selected = "cuda"
        elif target == "mps":
            if not mps_ok:
                logger.warning("Requested device 'mps' is unavailable; falling back to CPU.")
                selected = "cpu"
            else:
                selected = "mps"
        elif target == "cpu":
            selected = "cpu"
        else:
            raise ValueError(f"Unsupported device '{target}'. Expected auto|cuda|mps|cpu.")

        # Safe dtype policy: float32 on MPS & CPU; optional bfloat16 on CUDA
        dtype_name = "float32"
        if selected == "cuda" and self.requested_dtype == "bfloat16":
            if torch.cuda.is_bf16_supported():
                dtype_name = "bfloat16"
            else:
                logger.warning("CUDA device does not support bfloat16; using float32.")
        elif selected in {"mps", "cpu"} and self.requested_dtype not in {"auto", "float32"}:
            logger.warning(
                "Forcing float32 on %s for numerical stability (requested '%s').",
                selected,
                self.requested_dtype,
            )

        accel_name = platform.processor() or platform.machine()
        mem_gb: float | None = None
        if selected == "cuda":
            props = torch.cuda.get_device_properties(0)
            accel_name = props.name
            mem_gb = round(props.total_memory / (1024**3), 2)
        elif selected == "mps":
            accel_name = f"Apple Silicon MPS ({platform.machine()})"

        return DeviceInfo(
            device=selected,
            dtype_name=dtype_name,
            accelerator_name=accel_name,
            memory_gb=mem_gb,
            cuda_available=cuda_ok,
            mps_available=mps_ok,
            cpu_count=os.cpu_count() or 1,
            hf_cache_dir=Path(os.environ.get("HF_HOME", str(self.paths.hf_cache_dir))),
            torch_version=torch.__version__,
            platform_info=f"{platform.system()} {platform.machine()}",
        )

    @staticmethod
    def recommended_batch_size(device_info: DeviceInfo, requested: int) -> int:
        """Auto-cap batch size to protect discrete GPU VRAM and Apple unified memory."""
        req = max(1, int(requested))
        if device_info.device == "cuda" and device_info.memory_gb is not None:
            if device_info.memory_gb <= 8.5:
                return min(req, 16)
            if device_info.memory_gb <= 16.5:
                return min(req, 32)
            return min(req, 64)
        if device_info.device == "mps":
            return min(req, 16)
        return min(req, 32)

    @staticmethod
    def is_mps_unsupported_error(exc: Exception) -> bool:
        """Identify PyTorch MPS operator or float64 unsupported errors for controlled CPU fallback."""
        msg = str(exc).lower()
        return any(
            marker in msg
            for marker in (
                "mps",
                "float64",
                "not currently implemented for the mps",
                "cannot convert a mps tensor",
            )
        )
