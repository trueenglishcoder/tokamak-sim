from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast


ComputeBackend = Literal["cpu", "gpu"]


def normalize_compute_backend(value: object, *, field: str = "compute.backend") -> ComputeBackend:
    """Normalize a configured simulator compute backend."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    backend = value.strip().lower()
    if backend not in {"cpu", "gpu"}:
        raise ValueError(f"{field} must be 'cpu' or 'gpu', got {value!r}")
    return cast(ComputeBackend, backend)


@dataclass(frozen=True, slots=True)
class ComputeSettings:
    """Runtime compute backend settings."""

    backend: ComputeBackend = "cpu"
    gpu_device: str = "cuda:0"
    boundary_equivalence_mode: str = "strict"

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", normalize_compute_backend(self.backend))
        gpu_device = str(self.gpu_device).strip()
        if gpu_device == "":
            raise ValueError("compute.gpu_device must be a non-empty string")
        object.__setattr__(self, "gpu_device", gpu_device)
        mode = str(self.boundary_equivalence_mode).strip().lower()
        if mode not in {"strict"}:
            raise ValueError("compute.boundary_equivalence_mode must be 'strict'")
        object.__setattr__(self, "boundary_equivalence_mode", mode)


def compute_runtime_metadata(settings: ComputeSettings, *, validate: bool = False) -> dict[str, object]:
    """Return serializable runtime metadata for the selected compute backend."""
    if settings.backend == "cpu":
        return {
            "backend": "cpu",
            "plant_backend": "cpu",
            "boundary_backend": "cpu",
            "batched_env_backend": "cpu",
            "gpu_device": settings.gpu_device,
            "boundary_equivalence_mode": settings.boundary_equivalence_mode,
            "torch_version": None,
            "cuda_available": False,
            "device_name": None,
        }

    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        if validate:
            raise RuntimeError("GPU compute backend requires tokamak-sim[gpu] with torch installed") from exc
        return {
            "backend": "gpu",
            "plant_backend": "gpu",
            "boundary_backend": "gpu",
            "batched_env_backend": "gpu",
            "gpu_device": settings.gpu_device,
            "boundary_equivalence_mode": settings.boundary_equivalence_mode,
            "torch_version": None,
            "cuda_available": False,
            "device_name": None,
        }

    cuda_available = bool(torch.cuda.is_available())
    device_name = None
    if cuda_available:
        try:
            device = torch.device(settings.gpu_device)
            if device.type == "cuda":
                torch.empty((1,), device=device)
                index = 0 if device.index is None else int(device.index)
                device_name = str(torch.cuda.get_device_name(index))
            elif validate:
                raise RuntimeError(f"GPU compute backend requires a CUDA device, got {settings.gpu_device!r}")
        except Exception as exc:  # pragma: no cover - host CUDA dependent
            if validate:
                raise RuntimeError(f"GPU compute backend could not initialize device {settings.gpu_device!r}") from exc
    elif validate:
        raise RuntimeError("GPU compute backend requested, but torch.cuda.is_available() is False")

    return {
        "backend": "gpu",
        "plant_backend": "gpu",
        "boundary_backend": "gpu",
        "batched_env_backend": "gpu",
        "gpu_device": settings.gpu_device,
        "boundary_equivalence_mode": settings.boundary_equivalence_mode,
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "device_name": device_name,
    }
