from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from tokamak_control.compute import require_gpu_available
from tokamak_control.core.grid import Grid2D
from tokamak_control.core.torch_sampling import bilinear_sample_torch_points
from tokamak_control.geometry.boundary_common import BoundaryMode, BoundaryNotFoundError, BoundaryStatus
from tokamak_control.geometry.boundary_suchkov import (
    SUCHKOV_CONTROL_COUNT,
    SUCHKOV_SEARCH_ITERATIONS,
    SUCHKOV_VALIDATION_COUNT,
    SuchkovSplineTorchPlan,
    build_suchkov_spline_torch_plan,
    interpolate_closed_curve_torch,
    uniform_periodic_angles,
)


@dataclass(slots=True)
class FixedAngleBoundaryGpuResult:
    """Результат поиска границы плазмы на GPU."""

    found: object
    status_code: object
    level: object
    points: object
    radii: object
    axis_points: object


@dataclass(frozen=True, slots=True, repr=True)
class FixedAngleBoundaryGpuGeometry:
    """Предвычисленная геометрия фиксированных лучей для GPU-поиска."""

    angles: object
    limiter: object
    max_radii: object
    suchkov_plan: SuchkovSplineTorchPlan | None
    suchkov_control_max_radii: object | None
    suchkov_limiter_samples: object | None


def _torch(device: str):
    require_gpu_available(device)
    import torch
    return torch


def prepare_fixed_angle_boundary_gpu_geometry(
    *,
    grid: Grid2D,
    center: tuple[float, float],
    angles_rad: object,
    limiter_shape: object,
    boundary_mode: BoundaryMode,
    gpu_device: str,
    dtype: object,
) -> FixedAngleBoundaryGpuGeometry:
    """Предвычислить неизменную геометрию лучей и сплайна на GPU."""
    torch = _torch(gpu_device)
    device = torch.device(gpu_device)
    angles = torch.as_tensor(angles_rad, dtype=dtype, device=device).reshape(-1)
    limiter = torch.as_tensor(limiter_shape, dtype=dtype, device=device).reshape(-1, 2)
    if int(limiter.shape[0]) < 3:
        raise BoundaryNotFoundError("fixed-angle GPU geometry requires limiter geometry")
    max_radii = _ray_limit_radii(
        grid=grid,
        center=torch.as_tensor(center, dtype=dtype, device=device),
        angles=angles,
        limiter=limiter,
    )
    plan = None
    control_limits = None
    limiter_samples = None
    if str(boundary_mode) == "suchkov_spline_contour":
        dense_angles = torch.as_tensor(
            uniform_periodic_angles(SUCHKOV_VALIDATION_COUNT),
            dtype=dtype,
            device=device,
        )
        plan = build_suchkov_spline_torch_plan(
            dense_angles,
            control_count=SUCHKOV_CONTROL_COUNT,
        )
        control_limits = _ray_limit_radii(
            grid=grid,
            center=torch.as_tensor(center, dtype=dtype, device=device),
            angles=torch.as_tensor(plan.control_angles, dtype=dtype, device=device),
            limiter=limiter,
        )
        limiter_samples = torch.as_tensor(
            _sample_closed_polyline_numpy(
                np.asarray(limiter_shape, dtype=np.float64),
                _suchkov_limiter_sample_count(np.asarray(limiter_shape, dtype=np.float64), grid),
            ),
            dtype=dtype,
            device=device,
        )
    return FixedAngleBoundaryGpuGeometry(
        angles=angles,
        limiter=limiter,
        max_radii=max_radii,
        suchkov_plan=plan,
        suchkov_control_max_radii=control_limits,
        suchkov_limiter_samples=limiter_samples,
    )


def find_plasma_boundary_gpu_with_status(
    psi,
    grid: Grid2D,
    center: tuple[float, float],
    n_levels: int = 80,
    prev_level: float | None = None,
    prev_poly=None,
    local_n_levels: int = 7,
    local_span_frac: float = 0.02,
    target_mean_radius: float | None = None,
    target_switch_ratio: float = 1.15,
    target_switch_abs_delta: float = 0.10,
    local_bbox_pad_r: float | None = None,
    local_bbox_pad_z: float | None = None,
    limiter_shape=None,
    boundary_mode: BoundaryMode = "legacy_contour",
    gpu_device: str = "cuda:0",
) -> tuple[np.ndarray, float, BoundaryStatus]:
    del psi, grid, center, n_levels, prev_level, prev_poly, local_n_levels, local_span_frac
    del target_mean_radius, target_switch_ratio, target_switch_abs_delta
    del local_bbox_pad_r, local_bbox_pad_z, limiter_shape, boundary_mode, gpu_device
    raise BoundaryNotFoundError("legacy_contour boundary extraction is routed through the CPU dispatcher")


def fixed_angle_boundary_gpu(
    *,
    psi,
    grid: Grid2D,
    center: tuple[float, float],
    angles_rad,
    limiter_shape: np.ndarray,
    boundary_mode: BoundaryMode = "legacy_contour",
    gpu_device: str = "cuda:0",
    ray_samples: int = 256,
    prev_level=None,
    prev_points=None,
    prev_radii=None,
    legacy_precision_index2: float = 1.0e-3,
    smooth_selected_level: bool = False,
    soft_level_selection: bool = False,
    soft_level_candidates: int = 64,
    soft_level_temperature: float = 0.05,
    soft_level_radius_weight: float = 1.0,
    soft_level_missing_penalty: float = 4.0,
    soft_level_roughness_penalty: float = 0.2,
    level_smoothing_alpha: float = 1.0,
    level_search_span_fraction: float = 0.02,
    continuity_weight_radii: float = 1.0,
    continuity_weight_mean_radius: float = 0.3,
    continuity_weight_level: float = 0.1,
    suchkov_plan: SuchkovSplineTorchPlan | None = None,
    prepared_geometry: FixedAngleBoundaryGpuGeometry | None = None,
) -> FixedAngleBoundaryGpuResult:
    """CUDA fixed-angle boundary samples for batched RL training.

    This mirrors the active legacy boundary *signal* used by RL: center-origin
    radii at configured angles. It does not try to build full contour polygons.
    ``legacy_contour_limited`` means "legacy contour that fits inside the
    limiter", not "limiter-touching contour".
    """
    torch = _torch(gpu_device)
    field = torch.as_tensor(psi, device=gpu_device)
    if field.ndim != 3:
        raise ValueError(f"psi must have shape (B, Z, R), got {tuple(field.shape)}")
    B = int(field.shape[0])
    dtype = field.dtype
    device = field.device
    mode = str(boundary_mode)
    if mode not in {"legacy_contour", "legacy_contour_limited", "tracked_flux_contour", "suchkov_spline_contour"}:
        raise ValueError(f"unsupported boundary_mode for fixed_angle_boundary_gpu: {boundary_mode!r}")

    center_t = torch.tensor(center, dtype=dtype, device=device).reshape(1, 2).repeat(B, 1)
    use_limiter = mode in {"legacy_contour_limited", "tracked_flux_contour", "suchkov_spline_contour"}
    if prepared_geometry is not None:
        angles = torch.as_tensor(prepared_geometry.angles, dtype=dtype, device=device).reshape(-1)
        limiter_t = torch.as_tensor(prepared_geometry.limiter, dtype=dtype, device=device).reshape(-1, 2)
        max_radii = torch.as_tensor(prepared_geometry.max_radii, dtype=dtype, device=device).reshape(-1)
    else:
        angles = torch.as_tensor(angles_rad, dtype=dtype, device=device).reshape(-1)
        limiter_t = None
        if limiter_shape is not None:
            limiter_t = torch.as_tensor(limiter_shape, dtype=dtype, device=device).reshape(-1, 2)
        max_radii = _ray_limit_radii(
            grid=grid,
            center=center_t[0],
            angles=angles,
            limiter=limiter_t if use_limiter else None,
        )
    if use_limiter and (limiter_t is None or int(limiter_t.shape[0]) < 3):
        raise BoundaryNotFoundError(f"{mode} requires limiter geometry")
    center_level = _sample_points(field, grid, center_t[:, None, :]).reshape(B)

    tracked = None
    if mode == "tracked_flux_contour" and prev_level is not None and prev_points is not None and prev_radii is not None:
        tracked = _tracked_fixed_angle_boundary(
            psi=field,
            grid=grid,
            center_points=center_t,
            center_level=center_level,
            angles=angles,
            max_radii=max_radii,
            prev_level=torch.as_tensor(prev_level, dtype=dtype, device=device).reshape(B),
            prev_points=torch.as_tensor(prev_points, dtype=dtype, device=device).reshape(B, int(angles.numel()), 2),
            prev_radii=torch.as_tensor(prev_radii, dtype=dtype, device=device).reshape(B, int(angles.numel())),
            ray_samples=int(ray_samples),
            level_smoothing_alpha=float(level_smoothing_alpha),
            level_search_span_fraction=float(level_search_span_fraction),
            continuity_weight_radii=float(continuity_weight_radii),
            continuity_weight_mean_radius=float(continuity_weight_mean_radius),
            continuity_weight_level=float(continuity_weight_level),
        )

    if mode == "suchkov_spline_contour":
        active_plan = suchkov_plan
        if active_plan is None and prepared_geometry is not None:
            active_plan = prepared_geometry.suchkov_plan
        if active_plan is None:
            dense_angles = torch.as_tensor(
                uniform_periodic_angles(SUCHKOV_VALIDATION_COUNT),
                dtype=dtype,
                device=device,
            )
            active_plan = build_suchkov_spline_torch_plan(
                dense_angles,
                control_count=SUCHKOV_CONTROL_COUNT,
            )
        reset = _suchkov_fixed_angle_search(
            psi=field,
            grid=grid,
            center_points=center_t,
            center_level=center_level,
            measurement_angles=angles,
            limiter=limiter_t,
            ray_samples=int(ray_samples),
            plan=active_plan,
            precomputed_control_max_radii=(
                prepared_geometry.suchkov_control_max_radii
                if prepared_geometry is not None
                else None
            ),
            precomputed_limiter_samples=(
                prepared_geometry.suchkov_limiter_samples
                if prepared_geometry is not None
                else None
            ),
        )
    else:
        reset = _legacy_fixed_angle_search(
            psi=field,
            grid=grid,
            center=center,
            center_points=center_t,
            center_level=center_level,
            angles=angles,
            max_radii=max_radii,
            ray_samples=int(ray_samples),
            precision_index2=float(legacy_precision_index2),
        )

    if tracked is None:
        points, radii, found, level = reset
        status_code = torch.where(
            found,
            torch.full((B,), 7 if mode == "suchkov_spline_contour" else (4 if mode == "tracked_flux_contour" else 2), dtype=torch.int64, device=device),
            torch.zeros((B,), dtype=torch.int64, device=device),
        )
        if bool(soft_level_selection) and mode in {"legacy_contour", "legacy_contour_limited"}:
            soft_points, soft_radii, soft_found, soft_level = _soft_fixed_angle_boundary(
                psi=field,
                grid=grid,
                center_points=center_t,
                center_level=center_level,
                angles=angles,
                max_radii=max_radii,
                ray_samples=int(ray_samples),
                candidate_count=int(soft_level_candidates),
                temperature=float(soft_level_temperature),
                radius_weight=float(soft_level_radius_weight),
                missing_penalty=float(soft_level_missing_penalty),
                roughness_penalty=float(soft_level_roughness_penalty),
            )
            use_soft = soft_found & torch.isfinite(soft_level)
            points = torch.where(use_soft[:, None, None], soft_points, points)
            radii = torch.where(use_soft[:, None], soft_radii, radii)
            found = torch.where(use_soft, soft_found, found)
            level = torch.where(use_soft, soft_level, level)
            status_code = torch.where(
                found,
                torch.where(
                    use_soft,
                    torch.full((B,), 6, dtype=torch.int64, device=device),
                    torch.full((B,), 2, dtype=torch.int64, device=device),
                ),
                torch.zeros((B,), dtype=torch.int64, device=device),
            )
        elif bool(smooth_selected_level) and mode in {"legacy_contour", "legacy_contour_limited"} and prev_level is not None:
            reset_points, reset_radii, reset_found, reset_level = reset
            prev = torch.as_tensor(prev_level, dtype=dtype, device=device).reshape(B)
            prev_finite = torch.isfinite(prev)
            alpha = max(0.0, min(float(level_smoothing_alpha), 1.0))
            smooth_level = torch.where(
                prev_finite & torch.isfinite(reset_level) & reset_found,
                alpha * prev + (1.0 - alpha) * reset_level,
                reset_level,
            )
            smooth_points, smooth_radii, smooth_found = _center_ray_crossings(
                psi=field,
                grid=grid,
                center_points=center_t,
                center_level=center_level,
                level=smooth_level,
                angles=angles,
                max_radii=max_radii,
                ray_samples=int(ray_samples),
            )
            use_smooth = prev_finite & reset_found & smooth_found & torch.isfinite(smooth_level)
            points = torch.where(use_smooth[:, None, None], smooth_points, reset_points)
            radii = torch.where(use_smooth[:, None], smooth_radii, reset_radii)
            found = reset_found
            level = torch.where(use_smooth, smooth_level, reset_level)
            status_code = torch.where(
                found,
                torch.where(
                    use_smooth,
                    torch.full((B,), 5, dtype=torch.int64, device=device),
                    torch.full((B,), 2, dtype=torch.int64, device=device),
                ),
                torch.zeros((B,), dtype=torch.int64, device=device),
            )
    else:
        tracked_points, tracked_radii, tracked_found, tracked_level = tracked
        reset_points, reset_radii, reset_found, reset_level = reset
        use_tracked = tracked_found
        points = torch.where(use_tracked[:, None, None], tracked_points, reset_points)
        radii = torch.where(use_tracked[:, None], tracked_radii, reset_radii)
        found = tracked_found | reset_found
        level = torch.where(use_tracked, tracked_level, reset_level)
        status_code = torch.where(
            found,
            torch.where(use_tracked, torch.full((B,), 3, dtype=torch.int64, device=device), torch.full((B,), 4, dtype=torch.int64, device=device)),
            torch.zeros((B,), dtype=torch.int64, device=device),
        )

    return FixedAngleBoundaryGpuResult(
        found=found,
        status_code=status_code,
        level=level,
        points=points,
        radii=radii,
        axis_points=center_t,
    )


def _sample_points(psi, grid: Grid2D, points):
    return bilinear_sample_torch_points(psi, grid, points)


def _ray_limit_radii(*, grid: Grid2D, center, angles, limiter):
    torch = __import__("torch")
    dtype = angles.dtype
    device = angles.device
    dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    if limiter is not None:
        poly = limiter
        if not torch.allclose(poly[0], poly[-1]):
            poly = torch.cat([poly, poly[:1]], dim=0)
        a = poly[:-1]
        b = poly[1:]
        seg = b - a
        rel = a - center.reshape(1, 2)
        # Solve center + t * dir = a + u * seg for every ray/segment pair.
        d = dirs[:, None, :]
        s = seg[None, :, :]
        r = rel[None, :, :]
        denom = _cross2(d, s)
        numer_t = _cross2(r, s)
        numer_u = _cross2(r, d)
        eps = torch.finfo(dtype).eps * 128.0
        valid = torch.abs(denom) > eps
        t = torch.where(valid, numer_t / torch.where(valid, denom, torch.ones_like(denom)), torch.full_like(denom, float("inf")))
        u = torch.where(valid, numer_u / torch.where(valid, denom, torch.ones_like(denom)), torch.full_like(denom, float("inf")))
        valid = valid & (t > 0.0) & (u >= -1.0e-9) & (u <= 1.0 + 1.0e-9)
        t = torch.where(valid, t, torch.full_like(t, float("inf")))
        out = torch.min(t, dim=1).values
        return torch.where(torch.isfinite(out), out, torch.full_like(out, float("nan")))

    r_coords = torch.as_tensor(grid.r.coords(), dtype=dtype, device=device)
    z_coords = torch.as_tensor(grid.z.coords(), dtype=dtype, device=device)
    r_min, r_max = torch.min(r_coords), torch.max(r_coords)
    z_min, z_max = torch.min(z_coords), torch.max(z_coords)
    cx, cz = center[0], center[1]
    dx, dz = dirs[:, 0], dirs[:, 1]
    inf = torch.full_like(dx, float("inf"))
    candidates = []
    for bound, comp, other, other_min, other_max, origin_main, origin_cross in (
        (r_min, dx, dz, z_min, z_max, cx, cz),
        (r_max, dx, dz, z_min, z_max, cx, cz),
        (z_min, dz, dx, r_min, r_max, cz, cx),
        (z_max, dz, dx, r_min, r_max, cz, cx),
    ):
        origin_comp = origin_main
        origin_other = origin_cross
        t = (bound - origin_comp) / torch.where(torch.abs(comp) > 1.0e-12, comp, torch.ones_like(comp))
        other_value = origin_other + t * other
        valid = (torch.abs(comp) > 1.0e-12) & (t > 0.0) & (other_value >= other_min) & (other_value <= other_max)
        candidates.append(torch.where(valid, t, inf))
    return torch.min(torch.stack(candidates, dim=0), dim=0).values


def _suchkov_limiter_sample_count(limiter_shape: np.ndarray, grid: Grid2D) -> int:
    """Выбрать дискретизацию лимитера не грубее половины ячейки сетки."""
    points = np.asarray(limiter_shape, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 3:
        return 256
    if not np.allclose(points[0], points[-1], rtol=0.0, atol=1.0e-12):
        points = np.vstack([points, points[0]])
    perimeter = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    grid_step = 0.5 * float(min(abs(float(grid.r.step)), abs(float(grid.z.step))))
    if not np.isfinite(perimeter) or perimeter <= 0.0 or not np.isfinite(grid_step) or grid_step <= 0.0:
        return 256
    return int(np.clip(np.ceil(perimeter / grid_step), 256, 2048))


def _suchkov_containment_tolerance(grid: Grid2D) -> float:
    """Численный допуск для условия ``Gamma subset closure(Omega_limiter)``."""
    cell = float(max(abs(float(grid.r.step)), abs(float(grid.z.step))))
    if not np.isfinite(cell) or cell <= 0.0:
        return 1.0e-7
    return max(1.0e-7, 1.0e-5 * cell)


def _suchkov_contact_tolerance(grid: Grid2D) -> float:
    """Допуск первого контакта LCFS с лимитером для сеточного поля."""
    cell = float(max(abs(float(grid.r.step)), abs(float(grid.z.step))))
    if not np.isfinite(cell) or cell <= 0.0:
        return 1.0e-4
    return max(2.0 * cell, 1.0e-6)


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _center_ray_crossings(*, psi, grid: Grid2D, center_points, center_level, level, angles, max_radii, ray_samples: int):
    torch = __import__("torch")
    B = int(psi.shape[0])
    A = int(angles.numel())
    dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    max_r_input = torch.as_tensor(max_radii, dtype=psi.dtype, device=psi.device)
    if max_r_input.ndim == 1:
        max_r = max_r_input.reshape(1, A).expand(B, A)
    elif tuple(max_r_input.shape) == (B, A):
        max_r = max_r_input
    else:
        raise ValueError(
            f"max_radii must have shape ({A},) or ({B}, {A}), got {tuple(max_r_input.shape)}"
        )
    valid_ray = torch.isfinite(max_r) & (max_r > 0.0)
    t = torch.linspace(0.0, 1.0, max(int(ray_samples), 4), dtype=psi.dtype, device=psi.device)
    radii_grid = max_r[:, :, None] * t[None, None, :]
    pts = center_points[:, None, None, :] + radii_grid[..., None] * dirs[None, :, None, :]
    vals = _sample_points(psi, grid, pts.reshape(B, A * int(t.numel()), 2)).reshape(B, A, int(t.numel()))
    lv = level.reshape(B, 1, 1)
    inside_high = center_level.reshape(B, 1, 1) >= lv
    cond = torch.where(inside_high, vals <= lv, vals >= lv)
    cond = cond & torch.isfinite(vals) & valid_ray[:, :, None]
    cond[:, :, 0] = False
    first_idx = torch.argmax(cond.to(torch.int64), dim=2)
    has = torch.any(cond, dim=2)
    idx0 = torch.clamp(first_idx - 1, 0, int(t.numel()) - 1)
    idx1 = first_idx
    b = torch.arange(B, device=psi.device)[:, None]
    a = torch.arange(A, device=psi.device)[None, :]
    v0 = vals[b, a, idx0]
    v1 = vals[b, a, idx1]
    r0 = radii_grid[b, a, idx0]
    r1 = radii_grid[b, a, idx1]
    denom = v1 - v0
    safe = torch.abs(denom) > torch.finfo(psi.dtype).eps * 128.0
    frac = torch.clamp((level[:, None] - v0) / torch.where(safe, denom, torch.ones_like(denom)), 0.0, 1.0)
    radii = r0 + frac * (r1 - r0)
    radii = torch.where(has, radii, torch.full_like(radii, float("nan")))
    points = center_points[:, None, :] + radii[..., None] * dirs[None, :, :]
    found = torch.all(has, dim=1)
    return points, radii, found


def _nanmean_count(values, dim: int):
    torch = __import__("torch")
    finite = torch.isfinite(values)
    count = torch.sum(finite.to(torch.int64), dim=dim)
    total = torch.sum(torch.where(finite, values, torch.zeros_like(values)), dim=dim)
    mean = total / torch.clamp(count.to(values.dtype), min=1.0)
    return mean, count


def _soft_fixed_angle_boundary(
    *,
    psi,
    grid: Grid2D,
    center_points,
    center_level,
    angles,
    max_radii,
    ray_samples: int,
    candidate_count: int,
    temperature: float,
    radius_weight: float,
    missing_penalty: float,
    roughness_penalty: float,
):
    torch = __import__("torch")
    B = int(psi.shape[0])
    A = int(angles.numel())
    K = max(int(candidate_count), 3)
    dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    endpoint_radii = 0.995 * max_radii.reshape(1, A).repeat(B, 1)
    endpoint_points = center_points[:, None, :] + endpoint_radii[..., None] * dirs[None, :, :]
    endpoint_values = _sample_points(psi, grid, endpoint_points)
    finite_endpoint = torch.isfinite(endpoint_values)
    edge_level = torch.nanmedian(endpoint_values, dim=1).values

    finite_psi = torch.isfinite(psi)
    vmax = torch.max(torch.where(finite_psi, psi, torch.full_like(psi, -float("inf"))).reshape(B, -1), dim=1).values
    vmin = torch.min(torch.where(finite_psi, psi, torch.full_like(psi, float("inf"))).reshape(B, -1), dim=1).values
    fallback_span = 0.25 * (vmax - vmin)
    edge_level = torch.where(
        torch.isfinite(edge_level) & torch.any(finite_endpoint, dim=1),
        edge_level,
        center_level + torch.where(torch.isfinite(fallback_span), fallback_span, torch.ones_like(center_level)),
    )

    fractions = torch.linspace(0.02, 0.98, K, dtype=psi.dtype, device=psi.device)
    levels = center_level[:, None] + fractions[None, :] * (edge_level - center_level)[:, None]
    max_radius_scale, _count_scale = _nanmean_count(max_radii.reshape(1, A).repeat(B, 1), dim=1)
    max_radius_scale = torch.clamp(max_radius_scale, min=1.0e-6)

    score_parts = []
    for k in range(K):
        level = levels[:, k]
        _points, radii, _found = _center_ray_crossings(
            psi=psi,
            grid=grid,
            center_points=center_points,
            center_level=center_level,
            level=level,
            angles=angles,
            max_radii=max_radii,
            ray_samples=ray_samples,
        )
        mean_radius, count = _nanmean_count(radii, dim=1)
        missing_fraction = 1.0 - count.to(psi.dtype) / float(A)
        fill = mean_radius[:, None]
        filled = torch.where(torch.isfinite(radii), radii, fill)
        adjacent = torch.abs(filled - torch.roll(filled, shifts=-1, dims=1))
        roughness = torch.mean(adjacent, dim=1) / max_radius_scale
        radius_score = mean_radius / max_radius_scale
        score = (
            float(radius_weight) * radius_score
            - float(missing_penalty) * missing_fraction
            - float(roughness_penalty) * roughness
        )
        valid = (count > 0) & torch.isfinite(level) & torch.isfinite(score)
        score_parts.append(torch.where(valid, score, torch.full_like(score, -float("inf"))))

    scores = torch.stack(score_parts, dim=1)
    any_valid = torch.any(torch.isfinite(scores), dim=1)
    safe_scores = torch.where(torch.isfinite(scores), scores, torch.full_like(scores, -1.0e30))
    temp = max(float(temperature), 1.0e-6)
    weights = torch.softmax(safe_scores / temp, dim=1)
    weights = torch.where(any_valid[:, None], weights, torch.zeros_like(weights))
    soft_level = torch.sum(weights * levels, dim=1)
    points, radii, found = _center_ray_crossings(
        psi=psi,
        grid=grid,
        center_points=center_points,
        center_level=center_level,
        level=soft_level,
        angles=angles,
        max_radii=max_radii,
        ray_samples=ray_samples,
    )
    found = found & any_valid & torch.isfinite(soft_level)
    return points, radii, found, soft_level


def _legacy_sample_center_level_gpu(psi, grid: Grid2D, p_index):
    torch = __import__("torch")
    B = int(psi.shape[0])
    rows, cols = int(psi.shape[1]), int(psi.shape[2])
    p = p_index.reshape(B, 2)
    r1 = torch.floor(p).to(torch.long)
    r2 = torch.ceil(p).to(torch.long)
    use_r2 = (r1[:, 0] < 1) | (r1[:, 1] < 1)
    r1 = torch.where(use_r2[:, None], r2, r1)
    valid = (r1[:, 0] >= 1) & (r1[:, 1] >= 1) & (r2[:, 0] >= 1) & (r2[:, 1] >= 1)
    valid = valid & (r1[:, 0] <= cols) & (r2[:, 0] <= cols) & (r1[:, 1] <= rows) & (r2[:, 1] <= rows)
    i1 = torch.clamp(r1[:, 0] - 1, 0, cols - 1)
    j1 = torch.clamp(r1[:, 1] - 1, 0, rows - 1)
    i2 = torch.clamp(r2[:, 0] - 1, 0, cols - 1)
    j2 = torch.clamp(r2[:, 1] - 1, 0, rows - 1)
    b = torch.arange(B, device=psi.device)
    value1 = psi[b, j1, i1]
    value2 = psi[b, j2, i2]
    rr1 = torch.sum(r1.to(dtype=psi.dtype).pow(2), dim=1)
    rr2 = torch.sum(r2.to(dtype=psi.dtype).pow(2), dim=1)
    rr = torch.sum(p.to(dtype=psi.dtype).pow(2), dim=1)
    interp = ((rr2 - rr) * value1 + (rr - rr1) * value2) / torch.where(rr2 > rr1, rr2 - rr1, torch.ones_like(rr2))
    out = torch.where(rr2 > rr1, interp, value2)
    valid = valid & torch.isfinite(value1) & torch.isfinite(value2)
    return torch.where(valid, out, torch.full_like(out, float("nan")))


def _legacy_fixed_angle_search(
    *,
    psi,
    grid: Grid2D,
    center,
    center_points,
    center_level,
    angles,
    max_radii,
    ray_samples: int,
    precision_index2: float,
):
    torch = __import__("torch")
    B = int(psi.shape[0])
    A = int(angles.numel())
    r0 = float(grid.r.coords()[0])
    z0 = float(grid.z.coords()[0])
    o = torch.tensor(
        [
            1.0 + (float(center[0]) - r0) / float(grid.r.step),
            1.0 + (float(center[1]) - z0) / float(grid.z.step),
        ],
        dtype=psi.dtype,
        device=psi.device,
    )
    p = o.reshape(1, 2).repeat(B, 1)
    new_step = (-o / 2.0).reshape(1, 2).repeat(B, 1)
    precision_value = float(precision_index2)
    if not np.isfinite(precision_value) or precision_value <= 0.0:
        raise ValueError(f"legacy_precision_index2 must be finite and > 0, got {precision_index2!r}")
    precision = torch.as_tensor(precision_value, dtype=psi.dtype, device=psi.device)
    best_level = torch.full((B,), float("nan"), dtype=psi.dtype, device=psi.device)
    best_radii = torch.full((B, A), float("nan"), dtype=psi.dtype, device=psi.device)
    best_points = torch.full((B, A, 2), float("nan"), dtype=psi.dtype, device=psi.device)
    best_score = torch.full((B,), -float("inf"), dtype=psi.dtype, device=psi.device)
    best_found = torch.zeros((B,), dtype=torch.bool, device=psi.device)
    for _ in range(64):
        active = (torch.sum(new_step * new_step, dim=1) >= precision) & (p[:, 0] >= 1.0) & (p[:, 1] >= 1.0)
        p = torch.where(active[:, None], p + new_step, p)
        level = _legacy_sample_center_level_gpu(psi, grid, p)
        points, radii, found = _center_ray_crossings(
            psi=psi,
            grid=grid,
            center_points=center_points,
            center_level=center_level,
            level=level,
            angles=angles,
            max_radii=max_radii,
            ray_samples=ray_samples,
        )
        accepted = active & found & torch.isfinite(level)
        score = torch.nanmean(radii, dim=1)
        improve = accepted & (score > best_score)
        best_level = torch.where(improve, level, best_level)
        best_radii = torch.where(improve[:, None], radii, best_radii)
        best_points = torch.where(improve[:, None, None], points, best_points)
        best_score = torch.where(improve, score, best_score)
        best_found = best_found | improve
        new_step = torch.where(accepted[:, None], -torch.abs(new_step) / 2.0, torch.abs(new_step) / 2.0)
    return best_points, best_radii, best_found, best_level


def _tracked_fixed_angle_boundary(
    *,
    psi,
    grid: Grid2D,
    center_points,
    center_level,
    angles,
    max_radii,
    prev_level,
    prev_points,
    prev_radii,
    ray_samples: int,
    level_smoothing_alpha: float,
    level_search_span_fraction: float,
    continuity_weight_radii: float,
    continuity_weight_mean_radius: float,
    continuity_weight_level: float,
):
    torch = __import__("torch")
    B = int(psi.shape[0])
    A = int(angles.numel())
    sampled = _sample_points(psi, grid, prev_points)
    continued = torch.nanmedian(sampled, dim=1).values
    alpha = max(0.0, min(float(level_smoothing_alpha), 1.0))
    level0 = torch.where(
        torch.isfinite(prev_level) & torch.isfinite(continued),
        alpha * prev_level + (1.0 - alpha) * continued,
        continued,
    )
    finite = torch.isfinite(psi)
    vmax = torch.max(torch.where(finite, psi, torch.full_like(psi, -float("inf"))).reshape(B, -1), dim=1).values
    vmin = torch.min(torch.where(finite, psi, torch.full_like(psi, float("inf"))).reshape(B, -1), dim=1).values
    value_span = vmax - vmin
    level_span = max(float(level_search_span_fraction), 0.0) * torch.maximum(
        torch.maximum(value_span, torch.abs(level0)),
        torch.full_like(level0, 1.0e-12),
    )
    offsets_base = torch.tensor([0.0, -0.25, 0.25, -0.5, 0.5, -0.75, 0.75, -1.0, 1.0], dtype=psi.dtype, device=psi.device)
    best_level = torch.full((B,), float("nan"), dtype=psi.dtype, device=psi.device)
    best_radii = torch.full((B, A), float("nan"), dtype=psi.dtype, device=psi.device)
    best_points = torch.full((B, A, 2), float("nan"), dtype=psi.dtype, device=psi.device)
    best_score = torch.full((B,), float("inf"), dtype=psi.dtype, device=psi.device)
    best_found = torch.zeros((B,), dtype=torch.bool, device=psi.device)
    prev_mean = torch.nanmean(prev_radii, dim=1)
    for offset in offsets_base:
        level = level0 + offset * level_span
        points, radii, found = _center_ray_crossings(
            psi=psi,
            grid=grid,
            center_points=center_points,
            center_level=center_level,
            level=level,
            angles=angles,
            max_radii=max_radii,
            ray_samples=ray_samples,
        )
        mean = torch.nanmean(radii, dim=1)
        radii_score = torch.nanmean(torch.abs(radii - prev_radii), dim=1)
        mean_score = torch.abs(mean - prev_mean)
        level_score = torch.abs(level - level0) / torch.clamp(level_span, min=1.0e-12)
        score = (
            float(continuity_weight_radii) * radii_score
            + float(continuity_weight_mean_radius) * mean_score
            + float(continuity_weight_level) * level_score
        )
        accepted = found & torch.isfinite(level) & torch.isfinite(score)
        improve = accepted & (score < best_score)
        best_level = torch.where(improve, level, best_level)
        best_radii = torch.where(improve[:, None], radii, best_radii)
        best_points = torch.where(improve[:, None, None], points, best_points)
        best_score = torch.where(improve, score, best_score)
        best_found = best_found | improve
    return best_points, best_radii, best_found, best_level


def _axis_search(psi, grid: Grid2D, center: tuple[float, float], limiter):
    torch = __import__("torch")
    B, nz, nr = psi.shape
    r0 = float(grid.r.coords()[0])
    z0 = float(grid.z.coords()[0])
    r = torch.linspace(r0, r0 + float(grid.r.step) * (int(grid.r.size) - 1), int(grid.r.size), dtype=psi.dtype, device=psi.device)
    z = torch.linspace(z0, z0 + float(grid.z.step) * (int(grid.z.size) - 1), int(grid.z.size), dtype=psi.dtype, device=psi.device)
    Z, R = torch.meshgrid(z, r, indexing="ij")
    c = torch.tensor(center, dtype=psi.dtype, device=psi.device)
    dist = (R - c[0]) ** 2 + (Z - c[1]) ** 2
    finite = torch.isfinite(psi)
    # Select the nearer of the strongest max/min around the configured center.
    max_val = torch.where(finite, psi, torch.full_like(psi, -torch.inf))
    min_val = torch.where(finite, psi, torch.full_like(psi, torch.inf))
    max_flat = torch.argmax(max_val.reshape(B, -1), dim=1)
    min_flat = torch.argmin(min_val.reshape(B, -1), dim=1)
    max_j = max_flat // nr; max_i = max_flat % nr
    min_j = min_flat // nr; min_i = min_flat % nr
    b = torch.arange(B, device=psi.device)
    max_d = dist[max_j, max_i]
    min_d = dist[min_j, min_i]
    use_max = max_d <= min_d
    i = torch.where(use_max, max_i, min_i)
    j = torch.where(use_max, max_j, min_j)
    points = torch.stack([r[i], z[j]], dim=1)
    levels = psi[b, j, i]
    kind = torch.where(use_max, torch.ones((B,), dtype=torch.int64, device=psi.device), -torch.ones((B,), dtype=torch.int64, device=psi.device))
    return points, levels, kind


def _sample_limiter_psi(psi, grid, limiter):
    from tokamak_control.core.torch_sampling import bilinear_sample_torch
    return bilinear_sample_torch(psi, grid, limiter)


def _xpoint_level(psi, grid: Grid2D, axis_points):
    torch = __import__("torch")
    B, nz, nr = psi.shape
    dz = float(grid.z.step); dr = float(grid.r.step)
    grad_z, grad_r = torch.gradient(psi, spacing=(dz, dr), dim=(1, 2))
    score = grad_z[:, 1:-1, 1:-1] ** 2 + grad_r[:, 1:-1, 1:-1] ** 2
    flat = torch.argmin(score.reshape(B, -1), dim=1)
    jj = flat // (nr - 2) + 1
    ii = flat % (nr - 2) + 1
    b = torch.arange(B, device=psi.device)
    level = psi[b, jj, ii]
    has = torch.isfinite(level)
    return level, has


def _ray_crossings(psi, grid: Grid2D, axis_points, level, axis_kind, angles, limiter, *, ray_samples: int):
    torch = __import__("torch")
    B = int(psi.shape[0]); A = int(angles.numel())
    dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    max_radius = torch.max(torch.linalg.norm(limiter[None, :, :] - axis_points[:, None, :], dim=2), dim=1).values
    t = torch.linspace(0.0, 1.0, int(ray_samples), dtype=psi.dtype, device=psi.device)
    radii_grid = max_radius[:, None, None] * t[None, None, :]
    pts = axis_points[:, None, None, :] + radii_grid[..., None] * dirs[None, :, None, :]
    pts_flat = pts.reshape(B, A * int(ray_samples), 2)
    vals = bilinear_sample_torch_points(psi, grid, pts_flat).reshape(B, A, int(ray_samples))
    lv = level[:, None, None]
    if torch.any(axis_kind > 0):
        cond_max = vals <= lv
        cond_min = vals >= lv
        cond = torch.where((axis_kind[:, None, None] > 0), cond_max, cond_min)
    else:
        cond = vals >= lv
    cond = cond & torch.isfinite(vals)
    first_idx = torch.argmax(cond.to(torch.int64), dim=2)
    has = torch.any(cond, dim=2)
    idx0 = torch.clamp(first_idx - 1, 0, int(ray_samples) - 1)
    idx1 = first_idx
    b = torch.arange(B, device=psi.device)[:, None]
    a = torch.arange(A, device=psi.device)[None, :]
    v0 = vals[b, a, idx0]
    v1 = vals[b, a, idx1]
    r0 = radii_grid[b, 0, idx0]
    r1 = radii_grid[b, 0, idx1]
    denom = torch.where(torch.abs(v1 - v0) > 1e-30, v1 - v0, torch.ones_like(v1))
    frac = torch.clamp((level[:, None] - v0) / denom, 0.0, 1.0)
    radii = r0 + frac * (r1 - r0)
    radii = torch.where(has, radii, torch.full_like(radii, float("nan")))
    points = axis_points[:, None, :] + radii[..., None] * dirs[None, :, :]
    found = torch.all(has, dim=1)
    return points, radii, found



def _suchkov_fixed_angle_search(
    *,
    psi,
    grid: Grid2D,
    center_points,
    center_level,
    measurement_angles,
    limiter,
    ray_samples: int,
    plan: SuchkovSplineTorchPlan,
    precomputed_control_max_radii=None,
    precomputed_limiter_samples=None,
):
    """Найти максимальную по ширине допустимую сплайновую линию уровня ``psi``.

    Поиск выполняется в нормированной координате между значением потока в
    центре плазмы и первым уровнем контакта с лимитером. Двоичный поиск
    находит самую внешнюю допустимую поверхность. Для вложенных магнитных
    поверхностей она одновременно имеет максимальную горизонтальную ширину.
    """
    torch = __import__("torch")
    if limiter is None or int(limiter.shape[0]) < 3:
        raise BoundaryNotFoundError("suchkov_spline_contour requires limiter geometry")
    batch_size = int(psi.shape[0])
    control_angles = torch.as_tensor(
        plan.control_angles,
        dtype=psi.dtype,
        device=psi.device,
    ).reshape(-1)
    if precomputed_control_max_radii is None:
        control_limits = _ray_limit_radii(
            grid=grid,
            center=center_points[0],
            angles=control_angles,
            limiter=limiter,
        )
    else:
        control_limits = torch.as_tensor(
            precomputed_control_max_radii,
            dtype=psi.dtype,
            device=psi.device,
        ).reshape(-1)
    if precomputed_limiter_samples is None:
        limiter_samples = torch.as_tensor(
            _sample_closed_polyline_numpy(
                limiter.detach().cpu().numpy(),
                _suchkov_limiter_sample_count(limiter.detach().cpu().numpy(), grid),
            ),
            dtype=psi.dtype,
            device=psi.device,
        )
    else:
        limiter_samples = torch.as_tensor(
            precomputed_limiter_samples,
            dtype=psi.dtype,
            device=psi.device,
        ).reshape(-1, 2)

    axis_kind = _center_extremum_kind(
        psi=psi,
        grid=grid,
        center_points=center_points,
        center_level=center_level,
    )
    limiter_values = _sample_points(
        psi,
        grid,
        limiter_samples.reshape(1, -1, 2).expand(batch_size, -1, -1),
    )
    contact_level, contact_found = _suchkov_contact_levels(
        limiter_values=limiter_values,
        center_level=center_level,
        axis_kind=axis_kind,
    )

    lower_fraction = torch.zeros((batch_size,), dtype=psi.dtype, device=psi.device)
    upper_fraction = torch.ones((batch_size,), dtype=psi.dtype, device=psi.device)
    best_level = torch.full(
        (batch_size,),
        float("nan"),
        dtype=psi.dtype,
        device=psi.device,
    )
    best_polyline = torch.full(
        (batch_size, int(plan.output_angles.numel()), 2),
        float("nan"),
        dtype=psi.dtype,
        device=psi.device,
    )
    best_width = torch.full(
        (batch_size,),
        -float("inf"),
        dtype=psi.dtype,
        device=psi.device,
    )
    best_found = torch.zeros((batch_size,), dtype=torch.bool, device=psi.device)

    containment_tolerance = _suchkov_containment_tolerance(grid)

    for _ in range(SUCHKOV_SEARCH_ITERATIONS):
        fraction = 0.5 * (lower_fraction + upper_fraction)
        level = center_level + fraction * (contact_level - center_level)
        control_points, _control_radii, control_found = _center_ray_crossings(
            psi=psi,
            grid=grid,
            center_points=center_points,
            center_level=center_level,
            level=level,
            angles=control_angles,
            max_radii=control_limits,
            ray_samples=ray_samples,
        )
        spline_polyline = interpolate_closed_curve_torch(
            control_points,
            plan.interpolation_matrix,
        )
        inside_limiter = torch.all(
            _points_in_or_on_polygon_torch(
                spline_polyline,
                limiter,
                tolerance=containment_tolerance,
            ),
            dim=1,
        )
        axis_inside = _points_in_batched_polygons_torch(
            center_points,
            spline_polyline,
        )
        width = torch.max(spline_polyline[:, :, 0], dim=1).values - torch.min(
            spline_polyline[:, :, 0],
            dim=1,
        ).values
        accepted = (
            contact_found
            & control_found
            & inside_limiter
            & axis_inside
            & torch.isfinite(level)
            & torch.isfinite(width)
        )
        improve = accepted & (width > best_width)
        best_level = torch.where(improve, level, best_level)
        best_polyline = torch.where(
            improve[:, None, None],
            spline_polyline,
            best_polyline,
        )
        best_width = torch.where(improve, width, best_width)
        best_found = best_found | improve
        lower_fraction = torch.where(accepted, fraction, lower_fraction)
        upper_fraction = torch.where(accepted, upper_fraction, fraction)

    points, radii, measurement_found = _radii_from_closed_polylines_torch(
        best_polyline,
        center_points,
        measurement_angles,
    )
    contact_gap = _minimum_points_to_polygon_distance_torch(
        best_polyline,
        limiter,
    )
    contact_tolerance = _suchkov_contact_tolerance(grid)
    touches_limiter = torch.isfinite(contact_gap) & (contact_gap <= contact_tolerance)
    found = best_found & measurement_found & touches_limiter
    points = torch.where(
        found[:, None, None],
        points,
        torch.full_like(points, float("nan")),
    )
    radii = torch.where(
        found[:, None],
        radii,
        torch.full_like(radii, float("nan")),
    )
    level = torch.where(
        found,
        best_level,
        torch.full_like(best_level, float("nan")),
    )
    return points, radii, found, level


def _center_extremum_kind(*, psi, grid: Grid2D, center_points, center_level):
    """Определить знак экстремума потока около заданного центра плазмы."""
    torch = __import__("torch")
    offsets = torch.tensor(
        [
            [float(grid.r.step), 0.0],
            [-float(grid.r.step), 0.0],
            [0.0, float(grid.z.step)],
            [0.0, -float(grid.z.step)],
        ],
        dtype=psi.dtype,
        device=psi.device,
    )
    neighbours = center_points[:, None, :] + offsets[None, :, :]
    values = _sample_points(psi, grid, neighbours)
    neighbour_mean = torch.nanmean(values, dim=1)
    return torch.where(
        center_level >= neighbour_mean,
        torch.ones_like(center_level, dtype=torch.int64),
        -torch.ones_like(center_level, dtype=torch.int64),
    )


def _suchkov_contact_levels(*, limiter_values, center_level, axis_kind):
    """Найти первый достижимый уровень контакта поверхности с лимитером."""
    torch = __import__("torch")
    finite = torch.isfinite(limiter_values)
    center = center_level[:, None]
    below = finite & (limiter_values < center)
    above = finite & (limiter_values > center)
    max_below = torch.max(
        torch.where(below, limiter_values, torch.full_like(limiter_values, -float("inf"))),
        dim=1,
    ).values
    min_above = torch.min(
        torch.where(above, limiter_values, torch.full_like(limiter_values, float("inf"))),
        dim=1,
    ).values
    use_maximum = axis_kind > 0
    contact = torch.where(use_maximum, max_below, min_above)
    found = torch.where(
        use_maximum,
        torch.isfinite(max_below),
        torch.isfinite(min_above),
    )
    return contact, found


def _sample_closed_polyline_numpy(polyline: np.ndarray, count: int) -> np.ndarray:
    """Равномерно дискретизировать замкнутую ломаную по длине дуги."""
    points = np.asarray(polyline, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 3:
        raise ValueError("polyline must contain at least three points")
    if not np.allclose(points[0], points[-1], rtol=0.0, atol=1.0e-12):
        points = np.vstack([points, points[0]])
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([np.asarray([0.0]), np.cumsum(segment_lengths)])
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("polyline must have positive finite length")
    targets = np.linspace(0.0, total, max(int(count), 4), endpoint=False)
    segment_indices = np.searchsorted(cumulative, targets, side="right") - 1
    segment_indices = np.clip(segment_indices, 0, points.shape[0] - 2)
    starts = cumulative[segment_indices]
    lengths = segment_lengths[segment_indices]
    local = (targets - starts) / np.where(lengths > 0.0, lengths, 1.0)
    return points[segment_indices] + local[:, None] * (
        points[segment_indices + 1] - points[segment_indices]
    )

def _points_in_polygon_torch(points, polygon):
    """Проверить принадлежность batch точек замкнутому многоугольнику."""
    torch = __import__("torch")
    poly = polygon
    if not torch.allclose(poly[0], poly[-1]):
        poly = torch.cat([poly, poly[:1]], dim=0)
    x = points[..., 0, None]
    y = points[..., 1, None]
    x0 = poly[:-1, 0].reshape(1, 1, -1)
    y0 = poly[:-1, 1].reshape(1, 1, -1)
    x1 = poly[1:, 0].reshape(1, 1, -1)
    y1 = poly[1:, 1].reshape(1, 1, -1)
    crosses = (y0 > y) != (y1 > y)
    denominator = y1 - y0
    safe = torch.where(
        torch.abs(denominator) > torch.finfo(points.dtype).eps,
        denominator,
        torch.ones_like(denominator),
    )
    x_cross = x0 + (y - y0) * (x1 - x0) / safe
    crossing_count = torch.sum((crosses & (x_cross >= x)).to(torch.int64), dim=2)
    return (crossing_count % 2) == 1


def _point_to_polygon_distances_torch(points, polygon):
    """Расстояния batch точек до ближайшего ребра статического полигона."""
    torch = __import__("torch")
    poly = polygon
    if not torch.allclose(poly[0], poly[-1]):
        poly = torch.cat([poly, poly[:1]], dim=0)
    starts = poly[:-1]
    vectors = poly[1:] - poly[:-1]
    denom = torch.sum(vectors * vectors, dim=1)
    relative = points[:, :, None, :] - starts[None, None, :, :]
    safe_denom = torch.where(denom > 0.0, denom, torch.ones_like(denom))
    fraction = torch.clamp(
        torch.sum(relative * vectors[None, None, :, :], dim=3)
        / safe_denom[None, None, :],
        0.0,
        1.0,
    )
    nearest = starts[None, None, :, :] + fraction[..., None] * vectors[None, None, :, :]
    squared = torch.sum((points[:, :, None, :] - nearest) ** 2, dim=3)
    return torch.sqrt(torch.clamp(torch.min(squared, dim=2).values, min=0.0))


def _minimum_points_to_polygon_distance_torch(points, polygon):
    """Минимальный зазор каждой batch кривой до границы лимитера."""
    torch = __import__("torch")
    distances = _point_to_polygon_distances_torch(points, polygon)
    return torch.min(distances, dim=1).values


def _points_in_or_on_polygon_torch(points, polygon, *, tolerance: float):
    """Проверить ``point in closure(polygon)`` с метрическим допуском."""
    inside = _points_in_polygon_torch(points, polygon)
    if float(tolerance) <= 0.0:
        return inside
    distance = _point_to_polygon_distances_torch(points, polygon)
    return inside | (distance <= float(tolerance))


def _points_in_batched_polygons_torch(points, polygons):
    """Проверить принадлежность одной точки каждому полигону batch."""
    torch = __import__("torch")
    closed = torch.cat([polygons, polygons[:, :1, :]], dim=1)
    x = points[:, 0, None]
    y = points[:, 1, None]
    x0 = closed[:, :-1, 0]
    y0 = closed[:, :-1, 1]
    x1 = closed[:, 1:, 0]
    y1 = closed[:, 1:, 1]
    crosses = (y0 > y) != (y1 > y)
    denominator = y1 - y0
    safe = torch.where(
        torch.abs(denominator) > torch.finfo(points.dtype).eps,
        denominator,
        torch.ones_like(denominator),
    )
    x_cross = x0 + (y - y0) * (x1 - x0) / safe
    crossing_count = torch.sum((crosses & (x_cross > x)).to(torch.int64), dim=1)
    return (crossing_count % 2) == 1


def _radii_from_closed_polylines_torch(polylines, centers, angles):
    """Пересечь фиксированные лучи с batch замкнутых сплайновых ломаных."""
    torch = __import__("torch")
    batch_size = int(polylines.shape[0])
    angle_count = int(angles.numel())
    closed = torch.cat([polylines, polylines[:, :1, :]], dim=1)
    segment_starts = closed[:, :-1, :]
    segment_vectors = closed[:, 1:, :] - closed[:, :-1, :]
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    relative = segment_starts[:, None, :, :] - centers[:, None, None, :]
    ray = directions[None, :, None, :]
    segment = segment_vectors[:, None, :, :]
    denominator = _cross2(ray, segment)
    numerator_t = _cross2(relative, segment)
    numerator_u = _cross2(relative, ray)
    valid_denominator = torch.abs(denominator) > torch.finfo(polylines.dtype).eps * 128.0
    safe = torch.where(valid_denominator, denominator, torch.ones_like(denominator))
    t_ray = numerator_t / safe
    u_segment = numerator_u / safe
    valid = (
        valid_denominator
        & (t_ray >= 0.0)
        & (u_segment >= -1.0e-9)
        & (u_segment <= 1.0 + 1.0e-9)
    )
    hits = torch.where(valid, t_ray, torch.full_like(t_ray, -float("inf")))
    radii = torch.max(hits, dim=2).values
    found = torch.all(torch.isfinite(radii) & (radii >= 0.0), dim=1)
    radii = torch.where(
        torch.isfinite(radii),
        radii,
        torch.full_like(radii, float("nan")),
    )
    points = centers[:, None, :] + radii[..., None] * directions[None, :, :]
    if tuple(points.shape) != (batch_size, angle_count, 2):
        raise RuntimeError("unexpected fixed-angle spline intersection shape")
    return points, radii, found
