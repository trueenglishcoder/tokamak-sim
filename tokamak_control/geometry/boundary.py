# tokamak_control/geometry/boundary.py
from __future__ import annotations

import numpy as np

from tokamak_control.compute import ComputeBackend, normalize_compute_backend
from tokamak_control.core.grid import Grid2D
from tokamak_control.geometry.boundary_common import (
    BoundaryMode,
    BoundaryNotFoundError,
    BoundaryStatus,
    boundary_status_is_real,
)
from tokamak_control.geometry.boundary_cpu import (
    boundary_profiling_snapshot as boundary_cpu_profiling_snapshot,
    configure_boundary_profiling as configure_boundary_cpu_profiling,
    find_plasma_boundary_cpu_with_status,
    log_boundary_profiling_summary as log_boundary_cpu_profiling_summary,
)
from tokamak_control.geometry.boundary_gpu import (
    boundary_gpu_profiling_snapshot,
    configure_boundary_gpu_profiling,
    find_plasma_boundary_gpu_with_status,
    log_boundary_gpu_profiling_summary,
)


def configure_boundary_profiling(*, enabled: bool, summary_every: int = 0, reset: bool = True) -> None:
    configure_boundary_cpu_profiling(enabled=enabled, summary_every=summary_every, reset=reset)
    configure_boundary_gpu_profiling(enabled=enabled, summary_every=summary_every, reset=reset)


def boundary_profiling_snapshot() -> dict[str, object]:
    return {
        "title": "boundary",
        "cpu": boundary_cpu_profiling_snapshot(),
        "gpu": boundary_gpu_profiling_snapshot(),
    }


def log_boundary_profiling_summary() -> None:
    log_boundary_cpu_profiling_summary()
    log_boundary_gpu_profiling_summary()


def find_plasma_boundary_with_status(
    psi: np.ndarray,
    grid: Grid2D,
    center: tuple[float, float],
    n_levels: int = 10,
    prev_level: float | None = None,
    prev_poly: np.ndarray | None = None,
    local_n_levels: int = 7,
    local_span_frac: float = 0.02,
    target_mean_radius: float | None = None,
    target_switch_ratio: float = 1.15,
    target_switch_abs_delta: float = 0.10,
    local_bbox_pad_r: float | None = None,
    local_bbox_pad_z: float | None = None,
    limiter_shape: np.ndarray | None = None,
    boundary_mode: BoundaryMode = "limited",
    compute_backend: ComputeBackend | str = "cpu",
    gpu_device: str = "cuda:0",
) -> tuple[np.ndarray, float, BoundaryStatus]:
    backend = normalize_compute_backend(compute_backend)
    if backend == "cpu":
        return find_plasma_boundary_cpu_with_status(
            psi,
            grid,
            center,
            n_levels=n_levels,
            prev_level=prev_level,
            prev_poly=prev_poly,
            local_n_levels=local_n_levels,
            local_span_frac=local_span_frac,
            target_mean_radius=target_mean_radius,
            target_switch_ratio=target_switch_ratio,
            target_switch_abs_delta=target_switch_abs_delta,
            local_bbox_pad_r=local_bbox_pad_r,
            local_bbox_pad_z=local_bbox_pad_z,
            limiter_shape=limiter_shape,
            boundary_mode=boundary_mode,
        )
    return find_plasma_boundary_gpu_with_status(
        psi,
        grid,
        center,
        n_levels=n_levels,
        prev_level=prev_level,
        prev_poly=prev_poly,
        local_n_levels=local_n_levels,
        local_span_frac=local_span_frac,
        target_mean_radius=target_mean_radius,
        target_switch_ratio=target_switch_ratio,
        target_switch_abs_delta=target_switch_abs_delta,
        local_bbox_pad_r=local_bbox_pad_r,
        local_bbox_pad_z=local_bbox_pad_z,
        limiter_shape=limiter_shape,
        boundary_mode=boundary_mode,
        gpu_device=gpu_device,
    )


__all__ = [
    "BoundaryMode",
    "BoundaryNotFoundError",
    "BoundaryStatus",
    "boundary_cpu_profiling_snapshot",
    "boundary_gpu_profiling_snapshot",
    "boundary_profiling_snapshot",
    "boundary_status_is_real",
    "configure_boundary_cpu_profiling",
    "configure_boundary_gpu_profiling",
    "configure_boundary_profiling",
    "find_plasma_boundary_cpu_with_status",
    "find_plasma_boundary_gpu_with_status",
    "find_plasma_boundary_with_status",
    "log_boundary_cpu_profiling_summary",
    "log_boundary_gpu_profiling_summary",
    "log_boundary_profiling_summary",
]
