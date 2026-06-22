# tokamak_control/geometry/boundary.py
from __future__ import annotations

import numpy as np

from tokamak_control.compute import ComputeBackend
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


def configure_boundary_profiling(*, enabled: bool, summary_every: int = 0, reset: bool = True) -> None:
    configure_boundary_cpu_profiling(enabled=enabled, summary_every=summary_every, reset=reset)


def boundary_profiling_snapshot() -> dict[str, object]:
    return boundary_cpu_profiling_snapshot()


def log_boundary_profiling_summary() -> None:
    log_boundary_cpu_profiling_summary()


def _as_cpu_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


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
    boundary_mode: BoundaryMode = "legacy_contour",
    boundary_base_mode: BoundaryMode = "legacy_contour_limited",
    legacy_precision_index2: float = 1.0e-3,
    track_level: bool = False,
    level_smoothing_alpha: float = 1.0,
    level_search_span_fraction: float = 0.02,
    continuity_weight_radii: float = 1.0,
    continuity_weight_mean_radius: float = 0.3,
    continuity_weight_center: float = 0.2,
    continuity_weight_area: float = 0.2,
    continuity_weight_level: float = 0.1,
    compute_backend: ComputeBackend | str = "cpu",
    gpu_device: str = "cuda:0",
) -> tuple[np.ndarray, float, BoundaryStatus]:
    del compute_backend, gpu_device
    return find_plasma_boundary_cpu_with_status(
        _as_cpu_numpy(psi),
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
        boundary_base_mode=boundary_base_mode,
        legacy_precision_index2=legacy_precision_index2,
        track_level=track_level,
        level_smoothing_alpha=level_smoothing_alpha,
        level_search_span_fraction=level_search_span_fraction,
        continuity_weight_radii=continuity_weight_radii,
        continuity_weight_mean_radius=continuity_weight_mean_radius,
        continuity_weight_center=continuity_weight_center,
        continuity_weight_area=continuity_weight_area,
        continuity_weight_level=continuity_weight_level,
    )


__all__ = [
    "BoundaryMode",
    "BoundaryNotFoundError",
    "BoundaryStatus",
    "boundary_cpu_profiling_snapshot",
    "boundary_profiling_snapshot",
    "boundary_status_is_real",
    "configure_boundary_cpu_profiling",
    "configure_boundary_profiling",
    "find_plasma_boundary_cpu_with_status",
    "find_plasma_boundary_with_status",
    "log_boundary_cpu_profiling_summary",
    "log_boundary_profiling_summary",
]
