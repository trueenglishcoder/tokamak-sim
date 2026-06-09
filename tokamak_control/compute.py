from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

ComputeBackend = Literal["cpu", "gpu"]


@dataclass(frozen=True, slots=True)
class ComputeSettings:
    """Runtime compute backend selection for tokamak-sim."""

    backend: ComputeBackend = "cpu"
    gpu_device: str = "cuda:0"

    def validate(self, *, require_available: bool = False) -> None:
        backend = normalize_compute_backend(self.backend)
        if backend == "gpu":
            if not str(self.gpu_device).strip():
                raise ValueError("compute.gpu_device must be non-empty for GPU backend")
            if require_available:
                require_gpu_available(self.gpu_device)


def normalize_compute_backend(value: ComputeBackend | str | None) -> ComputeBackend:
    text = "cpu" if value is None else str(value).strip().lower()
    if text not in {"cpu", "gpu"}:
        raise ValueError(f"compute backend must be 'cpu' or 'gpu', got {value!r}")
    return cast(ComputeBackend, text)


def require_gpu_available(device: str = "cuda:0") -> dict[str, object]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("GPU backend requested but PyTorch is not installed") from exc
    dev = torch.device(str(device))
    if dev.type != "cuda":
        raise RuntimeError(f"GPU backend requires a CUDA device, got {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU backend requested but torch.cuda.is_available() is false")
    try:
        index = 0 if dev.index is None else int(dev.index)
        name = torch.cuda.get_device_name(index)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"GPU backend requested but CUDA device {device!r} cannot be initialized") from exc
    return {
        "torch_version": str(torch.__version__),
        "cuda_available": True,
        "device": str(dev),
        "device_name": str(name),
    }


def compute_runtime_metadata(settings: ComputeSettings, *, validate: bool = False) -> dict[str, object]:
    backend = normalize_compute_backend(settings.backend)
    meta: dict[str, object] = {"backend": backend, "gpu_device": str(settings.gpu_device)}
    if backend == "gpu":
        if validate:
            meta.update(require_gpu_available(settings.gpu_device))
        else:
            try:
                import torch
                meta.update({
                    "torch_version": str(torch.__version__),
                    "cuda_available": bool(torch.cuda.is_available()),
                    "device_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
                })
            except Exception:
                meta.update({"torch_version": None, "cuda_available": False, "device_name": None})
    return meta
