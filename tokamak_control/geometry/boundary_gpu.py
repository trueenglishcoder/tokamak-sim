from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from tokamak_control.compute import require_gpu_available
from tokamak_control.core.grid import Grid2D
from tokamak_control.core.torch_sampling import bilinear_sample_torch_points
from tokamak_control.geometry.boundary_common import BoundaryMode, BoundaryNotFoundError, BoundaryStatus


@dataclass(slots=True)
class FixedAngleBoundaryGpuResult:
    """Fixed-angle projection and topology of the physical GPU LCFS."""

    found: object
    status_code: object
    topology_code: object
    level: object
    points: object
    radii: object
    axis_points: object
    x_points: object


@dataclass(frozen=True, slots=True, repr=True)
class FixedAngleBoundaryGpuGeometry:
    """Precomputed limiter geometry used by batched GPU LCFS extraction."""

    angles: object
    limiter: object
    max_radii: object
    limiter_samples: object


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
    """Precompute fixed limiter samples and configured-angle ray limits."""
    torch = _torch(gpu_device)
    device = torch.device(gpu_device)
    angles = torch.as_tensor(angles_rad, dtype=dtype, device=device).reshape(-1)
    limiter_np = np.asarray(limiter_shape, dtype=np.float64).reshape(-1, 2)
    limiter = torch.as_tensor(limiter_np, dtype=dtype, device=device).reshape(-1, 2)
    if int(limiter.shape[0]) < 3:
        raise BoundaryNotFoundError("fixed-angle GPU geometry requires limiter geometry")
    max_radii = _ray_limit_radii(
        grid=grid,
        center=torch.as_tensor(center, dtype=dtype, device=device),
        angles=angles,
        limiter=limiter,
    )
    sample_count = max(_limiter_sample_count(limiter_np, grid), 256)
    limiter_samples = torch.as_tensor(
        _sample_closed_polyline_numpy(limiter_np, sample_count),
        dtype=dtype,
        device=device,
    )
    return FixedAngleBoundaryGpuGeometry(
        angles=angles,
        limiter=limiter,
        max_radii=max_radii,
        limiter_samples=limiter_samples,
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
    prepared_geometry: FixedAngleBoundaryGpuGeometry | None = None,
) -> FixedAngleBoundaryGpuResult:
    """Compute the batched fixed-angle projection used by RL.

    ``equilibrium_lcfs`` selects the wall- or saddle-limited flux level and
    returns topology and subgrid critical-point data. The canonical dense LCFS
    is produced by the CPU extractor for artifacts and validation.
    """
    torch = _torch(gpu_device)
    field = torch.as_tensor(psi, device=gpu_device)
    if field.ndim != 3:
        raise ValueError(f"psi must have shape (B, Z, R), got {tuple(field.shape)}")
    B = int(field.shape[0])
    dtype = field.dtype
    device = field.device
    mode = str(boundary_mode)
    if mode not in {"legacy_contour", "legacy_contour_limited", "tracked_flux_contour", "equilibrium_lcfs"}:
        raise ValueError(f"unsupported boundary_mode for fixed_angle_boundary_gpu: {boundary_mode!r}")

    center_t = torch.tensor(center, dtype=dtype, device=device).reshape(1, 2).repeat(B, 1)
    use_limiter = mode in {"legacy_contour_limited", "tracked_flux_contour", "equilibrium_lcfs"}
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
    axis_points = center_t
    axis_level = center_level
    axis_kind = torch.zeros((B,), dtype=torch.int64, device=device)
    axis_valid = torch.ones((B,), dtype=torch.bool, device=device)
    if mode == "equilibrium_lcfs":
        axis_points, axis_level, axis_kind, axis_valid = _axis_search(field, grid, center, limiter_t)

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

    topology_code = torch.zeros((B,), dtype=torch.int64, device=device)
    selected_x_points = torch.full((B, 8, 2), float("nan"), dtype=dtype, device=device)
    if mode == "equilibrium_lcfs":
        limiter_samples = (
            torch.as_tensor(prepared_geometry.limiter_samples, dtype=dtype, device=device)
            if prepared_geometry is not None
            else torch.as_tensor(
                _sample_closed_polyline_numpy(np.asarray(limiter_shape, dtype=np.float64), 256),
                dtype=dtype,
                device=device,
            )
        )
        reset, topology_code, selected_x_points = _equilibrium_lcfs_fixed_angle_search(
            psi=field,
            grid=grid,
            axis_points=axis_points,
            projection_center=center_t,
            axis_level=axis_level,
            axis_kind=axis_kind,
            axis_valid=axis_valid,
            measurement_angles=angles,
            limiter=limiter_t,
            limiter_samples=limiter_samples,
            ray_samples=int(ray_samples),
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
        success_code = (
            topology_code + 7
            if mode == "equilibrium_lcfs"
            else torch.full((B,), 4 if mode == "tracked_flux_contour" else 2, dtype=torch.int64, device=device)
        )
        status_code = torch.where(
            found,
            success_code,
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
        topology_code=topology_code,
        level=level,
        points=points,
        radii=radii,
        axis_points=axis_points,
        x_points=selected_x_points,
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


def _limiter_sample_count(limiter_shape: np.ndarray, grid: Grid2D) -> int:
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


def _equilibrium_lcfs_fixed_angle_search(
    *,
    psi,
    grid: Grid2D,
    axis_points,
    projection_center,
    axis_level,
    axis_kind,
    axis_valid,
    measurement_angles,
    limiter,
    limiter_samples,
    ray_samples: int,
):
    """Vectorized LCFS topology and fixed-angle projection for RL batches.

    The canonical dense boundary is produced by the CPU extractor. This path
    evaluates the same wall-versus-saddle flux rule directly on batched tensors
    and returns only the derived fixed-angle signal required by training.
    """
    torch = __import__("torch")
    B = int(psi.shape[0])
    dtype = psi.dtype
    device = psi.device
    orientation = -axis_kind.to(dtype=dtype)
    flux_scale = torch.amax(psi, dim=(1, 2)) - torch.amin(psi, dim=(1, 2))
    flux_floor = torch.clamp(flux_scale * 1.0e-9, min=torch.finfo(dtype).eps)

    limiter_batch = limiter_samples.reshape(1, -1, 2).expand(B, -1, -1)
    limiter_values = _sample_points(psi, grid, limiter_batch)
    wall_chi_raw = orientation[:, None] * (limiter_values - axis_level[:, None])
    positive_wall = torch.isfinite(wall_chi_raw) & (wall_chi_raw > flux_floor[:, None])
    local_wall_minimum = (
        positive_wall
        & (wall_chi_raw <= torch.roll(wall_chi_raw, shifts=1, dims=1))
        & (wall_chi_raw <= torch.roll(wall_chi_raw, shifts=-1, dims=1))
    )
    wall_score = torch.where(local_wall_minimum, wall_chi_raw, torch.full_like(wall_chi_raw, float("inf")))
    wall_candidate_count = min(16, int(wall_score.shape[1]))
    wall_chi_candidates, wall_indices = torch.topk(
        wall_score, k=wall_candidate_count, dim=1, largest=False
    )
    wall_candidate_valid = torch.zeros((B, wall_candidate_count), dtype=torch.bool, device=device)
    wall_candidate_levels = axis_level[:, None] + orientation[:, None] * wall_chi_candidates
    wall_points = limiter_samples[wall_indices]
    validation_t = torch.linspace(0.0, 1.0, max(int(ray_samples), 128), dtype=dtype, device=device)
    for candidate_index in range(wall_candidate_count):
        candidate_level = wall_candidate_levels[:, candidate_index]
        points, _radii, crossing_found, counts = _ray_crossings_with_counts(
            psi=psi,
            grid=grid,
            axis_points=projection_center,
            level=candidate_level,
            axis_kind=axis_kind,
            angles=measurement_angles,
            limiter=limiter,
            ray_samples=ray_samples,
        )
        inside = torch.all(
            _points_in_or_on_polygon_torch(
                points,
                limiter,
                tolerance=1.5 * max(float(grid.r.step), float(grid.z.step)),
            ),
            dim=1,
        )
        single_valued = torch.all(counts == 1, dim=1)

        contact_point = wall_points[:, candidate_index, :]
        contact_line = projection_center[:, None, :] + validation_t[None, :, None] * (
            contact_point - projection_center
        )[:, None, :]
        contact_values = _sample_points(psi, grid, contact_line)
        contact_condition = torch.where(
            axis_kind[:, None] > 0,
            contact_values <= candidate_level[:, None],
            contact_values >= candidate_level[:, None],
        ) & torch.isfinite(contact_values)
        transitions = (~contact_condition[:, :-1]) & contact_condition[:, 1:]
        transition_count = torch.sum(transitions.to(torch.int64), dim=1)
        first_index = torch.argmax(transitions.to(torch.int64), dim=1)
        first_fraction = validation_t[torch.clamp(first_index + 1, max=int(validation_t.numel()) - 1)]
        contact_tolerance = 3.0 / float(max(int(validation_t.numel()) - 1, 1))
        reaches_contact = (transition_count == 1) & (first_fraction >= 1.0 - contact_tolerance)
        wall_candidate_valid[:, candidate_index] = (
            torch.isfinite(wall_chi_candidates[:, candidate_index])
            & crossing_found
            & inside
            & single_valued
            & reaches_contact
        )

    valid_wall_chi = torch.where(
        wall_candidate_valid, wall_chi_candidates, torch.full_like(wall_chi_candidates, float("inf"))
    )
    wall_chi, best_wall_index = torch.min(valid_wall_chi, dim=1)
    wall_level = wall_candidate_levels[torch.arange(B, device=device), best_wall_index]

    x_points, x_levels, x_valid = _xpoint_candidates_gpu(
        psi,
        grid,
        axis_points=axis_points,
        axis_level=axis_level,
        orientation=orientation,
        max_points=8,
    )
    K = int(x_levels.shape[1])
    x_chi = orientation[:, None] * (x_levels - axis_level[:, None])
    x_chi = torch.where(
        x_valid & torch.isfinite(x_chi) & (x_chi > flux_floor[:, None]),
        x_chi,
        torch.full_like(x_chi, float("inf")),
    )

    candidate_valid = torch.zeros((B, K), dtype=torch.bool, device=device)
    for candidate_index in range(K):
        candidate_level = x_levels[:, candidate_index]
        points, _radii, found, counts = _ray_crossings_with_counts(
            psi=psi,
            grid=grid,
            axis_points=projection_center,
            level=candidate_level,
            axis_kind=axis_kind,
            angles=measurement_angles,
            limiter=limiter,
            ray_samples=ray_samples,
        )
        inside = torch.all(
            _points_in_or_on_polygon_torch(points, limiter, tolerance=1.5 * max(float(grid.r.step), float(grid.z.step))),
            dim=1,
        )
        candidate_valid[:, candidate_index] = x_valid[:, candidate_index] & found & inside

    valid_x_chi = torch.where(candidate_valid, x_chi, torch.full_like(x_chi, float("inf")))
    best_x_chi, best_x_index = torch.min(valid_x_chi, dim=1)
    has_x = torch.isfinite(best_x_chi)
    has_wall = torch.isfinite(wall_chi)
    use_x = has_x & ((best_x_chi < wall_chi) | ~has_wall)
    selected_chi = torch.where(use_x, best_x_chi, wall_chi)
    selected_level = axis_level + orientation * selected_chi

    points, radii, crossing_found, counts = _ray_crossings_with_counts(
        psi=psi,
        grid=grid,
        axis_points=projection_center,
        level=selected_level,
        axis_kind=axis_kind,
        angles=measurement_angles,
        limiter=limiter,
        ray_samples=ray_samples,
    )
    inside = torch.all(
        _points_in_or_on_polygon_torch(points, limiter, tolerance=1.5 * max(float(grid.r.step), float(grid.z.step))),
        dim=1,
    )
    found = axis_valid & torch.isfinite(selected_chi) & crossing_found & inside

    topology_code = torch.where(
        found & ~use_x,
        torch.ones((B,), dtype=torch.int64, device=device),
        torch.zeros((B,), dtype=torch.int64, device=device),
    )
    selected_x = torch.full((B, K, 2), float("nan"), dtype=dtype, device=device)
    if K > 0:
        relative_tolerance = torch.maximum(2.0e-4 * selected_chi, 1.0e-10 * flux_scale)
        same_level = candidate_valid & (torch.abs(x_chi - selected_chi[:, None]) <= relative_tolerance[:, None])
        same_level_count = torch.sum(same_level.to(torch.int64), dim=1)
        topology_code = torch.where(
            found & use_x,
            torch.where(
                same_level_count >= 3,
                torch.full((B,), 4, dtype=torch.int64, device=device),
                torch.where(
                    same_level_count == 2,
                    torch.full((B,), 3, dtype=torch.int64, device=device),
                    torch.full((B,), 2, dtype=torch.int64, device=device),
                ),
            ),
            topology_code,
        )
        for batch_index in range(B):
            if not bool(use_x[batch_index]):
                continue
            indices = torch.nonzero(same_level[batch_index], as_tuple=False).reshape(-1)
            count = min(int(indices.numel()), K)
            if count:
                selected_x[batch_index, :count, :] = x_points[batch_index, indices[:count], :]

    points = torch.where(found[:, None, None], points, torch.full_like(points, float("nan")))
    radii = torch.where(found[:, None], radii, torch.full_like(radii, float("nan")))
    selected_level = torch.where(found, selected_level, torch.full_like(selected_level, float("nan")))
    return (points, radii, found, selected_level), topology_code, selected_x


def _xpoint_candidates_gpu(
    psi,
    grid: Grid2D,
    *,
    axis_points,
    axis_level,
    orientation,
    max_points: int,
):
    torch = __import__("torch")
    import torch.nn.functional as F

    B, nz, nr = psi.shape
    dz = float(grid.z.step)
    dr = float(grid.r.step)
    grad_z, grad_r = torch.gradient(psi, spacing=(dz, dr), dim=(1, 2))
    dzz, dzr = torch.gradient(grad_z, spacing=(dz, dr), dim=(1, 2))
    drz, drr = torch.gradient(grad_r, spacing=(dz, dr), dim=(1, 2))
    drz = 0.5 * (drz + dzr)
    grad2 = grad_r * grad_r + grad_z * grad_z
    det = drr * dzz - drz * drz
    pooled = -F.max_pool2d((-grad2).unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    local_min = grad2 <= pooled + torch.finfo(psi.dtype).eps * 16.0
    interior = torch.zeros_like(local_min)
    interior[:, 1:-1, 1:-1] = True
    saddle = local_min & interior & (det < 0.0) & torch.isfinite(grad2)
    score = torch.where(saddle, grad2, torch.full_like(grad2, float("inf")))
    K = min(max(int(max_points), 1), int(nz * nr))
    values, flat = torch.topk(score.reshape(B, -1), k=K, dim=1, largest=False)
    jj = flat // nr
    ii = flat % nr
    batch = torch.arange(B, device=psi.device)[:, None]
    r0 = float(grid.r.coords()[0])
    z0 = float(grid.z.coords()[0])
    points = torch.stack((r0 + ii.to(psi.dtype) * dr, z0 + jj.to(psi.dtype) * dz), dim=2)

    g_r = grad_r[batch, jj, ii]
    g_z = grad_z[batch, jj, ii]
    h_rr = drr[batch, jj, ii]
    h_rz = drz[batch, jj, ii]
    h_zz = dzz[batch, jj, ii]
    determinant = h_rr * h_zz - h_rz * h_rz
    safe_det = torch.where(torch.abs(determinant) > torch.finfo(psi.dtype).eps, determinant, torch.ones_like(determinant))
    delta_r = (h_zz * g_r - h_rz * g_z) / safe_det
    delta_z = (-h_rz * g_r + h_rr * g_z) / safe_det
    delta_r = torch.clamp(delta_r, -dr, dr)
    delta_z = torch.clamp(delta_z, -dz, dz)
    points[:, :, 0] -= delta_r
    points[:, :, 1] -= delta_z
    levels = _sample_points(psi, grid, points)
    finite = torch.isfinite(values) & torch.isfinite(levels) & (determinant < 0.0)

    samples = torch.linspace(0.0, 1.0, 64, dtype=psi.dtype, device=psi.device)
    line = axis_points[:, None, None, :] + samples[None, None, :, None] * (points[:, :, None, :] - axis_points[:, None, None, :])
    line_values = _sample_points(psi, grid, line.reshape(B, K * 64, 2)).reshape(B, K, 64)
    delta = levels - axis_level[:, None]
    safe_delta = torch.where(torch.abs(delta) > torch.finfo(psi.dtype).eps, delta, torch.ones_like(delta))
    normalized = (line_values - axis_level[:, None, None]) / safe_delta[:, :, None]
    monotonic = (
        torch.all(torch.isfinite(normalized), dim=2)
        & (torch.amin(normalized, dim=2) >= -0.02)
        & (torch.amax(normalized, dim=2) <= 1.02)
        & (torch.sum((torch.diff(normalized, dim=2) < -0.02).to(torch.int64), dim=2) <= 2)
    )
    positive = orientation[:, None] * delta > torch.finfo(psi.dtype).eps
    valid = finite & monotonic & positive
    dedup_distance = 0.75 * max(float(grid.r.step), float(grid.z.step))
    for index in range(K):
        if index == 0:
            continue
        distances = torch.linalg.norm(points[:, index : index + 1, :] - points[:, :index, :], dim=2)
        duplicate = torch.any(valid[:, :index] & (distances <= dedup_distance), dim=1)
        valid[:, index] = valid[:, index] & ~duplicate
    return points, levels, valid


def _ray_crossings_with_counts(
    *,
    psi,
    grid: Grid2D,
    axis_points,
    level,
    axis_kind,
    angles,
    limiter,
    ray_samples: int,
):
    torch = __import__("torch")
    B = int(psi.shape[0])
    A = int(angles.numel())
    sample_count = max(int(ray_samples), 64)
    directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    max_radius = torch.max(torch.linalg.norm(limiter[None, :, :] - axis_points[:, None, :], dim=2), dim=1).values
    t = torch.linspace(0.0, 1.0, sample_count, dtype=psi.dtype, device=psi.device)
    radius_grid = max_radius[:, None, None] * t[None, None, :]
    points = axis_points[:, None, None, :] + radius_grid[..., None] * directions[None, :, None, :]
    values = _sample_points(psi, grid, points.reshape(B, A * sample_count, 2)).reshape(B, A, sample_count)
    condition = torch.where(
        axis_kind[:, None, None] > 0,
        values <= level[:, None, None],
        values >= level[:, None, None],
    ) & torch.isfinite(values)
    transitions = (~condition[:, :, :-1]) & condition[:, :, 1:]
    counts = torch.sum(transitions.to(torch.int64), dim=2)
    first_transition = torch.argmax(transitions.to(torch.int64), dim=2)
    has = counts > 0
    idx0 = first_transition
    idx1 = torch.clamp(first_transition + 1, max=sample_count - 1)
    batch = torch.arange(B, device=psi.device)[:, None]
    angle_index = torch.arange(A, device=psi.device)[None, :]
    v0 = values[batch, angle_index, idx0]
    v1 = values[batch, angle_index, idx1]
    r0 = radius_grid[batch, 0, idx0]
    r1 = radius_grid[batch, 0, idx1]
    denominator = torch.where(torch.abs(v1 - v0) > 1.0e-30, v1 - v0, torch.ones_like(v1))
    fraction = torch.clamp((level[:, None] - v0) / denominator, 0.0, 1.0)
    radii = r0 + fraction * (r1 - r0)
    radii = torch.where(has, radii, torch.full_like(radii, float("nan")))
    boundary_points = axis_points[:, None, :] + radii[..., None] * directions[None, :, :]
    found = torch.all(has, dim=1)
    return boundary_points, radii, found, counts

def _axis_search(psi, grid: Grid2D, center: tuple[float, float], limiter):
    """Locate the nearest subgrid magnetic O-point in every batch lane."""
    torch = __import__("torch")
    import torch.nn.functional as F

    B, nz, nr = psi.shape
    dr = float(grid.r.step)
    dz = float(grid.z.step)
    r0 = float(grid.r.coords()[0])
    z0 = float(grid.z.coords()[0])
    r = torch.arange(nr, dtype=psi.dtype, device=psi.device) * dr + r0
    z = torch.arange(nz, dtype=psi.dtype, device=psi.device) * dz + z0
    Z, R = torch.meshgrid(z, r, indexing="ij")
    grid_points = torch.stack((R.reshape(-1), Z.reshape(-1)), dim=1)
    if limiter is not None:
        inside = _points_in_polygon_torch(
            grid_points.reshape(1, -1, 2).expand(B, -1, -1), limiter
        ).reshape(B, nz, nr)
    else:
        inside = torch.ones((B, nz, nr), dtype=torch.bool, device=psi.device)

    finite = torch.isfinite(psi) & inside
    local_max = psi >= F.max_pool2d(psi.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    local_min = psi <= -F.max_pool2d((-psi).unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    interior = torch.zeros_like(finite)
    interior[:, 1:-1, 1:-1] = True

    grad_z, grad_r = torch.gradient(psi, spacing=(dz, dr), dim=(1, 2))
    dzz, dzr = torch.gradient(grad_z, spacing=(dz, dr), dim=(1, 2))
    drz, drr = torch.gradient(grad_r, spacing=(dz, dr), dim=(1, 2))
    drz = 0.5 * (drz + dzr)
    determinant = drr * dzz - drz * drz
    maximum = local_max & (determinant > 0.0) & (drr < 0.0) & (dzz < 0.0)
    minimum = local_min & (determinant > 0.0) & (drr > 0.0) & (dzz > 0.0)
    candidates = finite & interior & (maximum | minimum)

    center_t = torch.tensor(center, dtype=psi.dtype, device=psi.device)
    distance2 = (R - center_t[0]) ** 2 + (Z - center_t[1]) ** 2
    score = torch.where(candidates, distance2.unsqueeze(0), torch.full_like(psi, float("inf")))
    flat = torch.argmin(score.reshape(B, -1), dim=1)
    valid = torch.isfinite(torch.min(score.reshape(B, -1), dim=1).values)
    jj = flat // nr
    ii = flat % nr
    batch = torch.arange(B, device=psi.device)
    kind = torch.where(
        maximum[batch, jj, ii],
        torch.ones((B,), dtype=torch.int64, device=psi.device),
        -torch.ones((B,), dtype=torch.int64, device=psi.device),
    )

    g_r = grad_r[batch, jj, ii]
    g_z = grad_z[batch, jj, ii]
    h_rr = drr[batch, jj, ii]
    h_rz = drz[batch, jj, ii]
    h_zz = dzz[batch, jj, ii]
    det = h_rr * h_zz - h_rz * h_rz
    safe = torch.where(torch.abs(det) > torch.finfo(psi.dtype).eps, det, torch.ones_like(det))
    delta_r = torch.clamp((h_zz * g_r - h_rz * g_z) / safe, -dr, dr)
    delta_z = torch.clamp((-h_rz * g_r + h_rr * g_z) / safe, -dz, dz)
    points = torch.stack((r[ii] - delta_r, z[jj] - delta_z), dim=1)
    levels = _sample_points(psi, grid, points[:, None, :]).reshape(B)
    valid = valid & torch.isfinite(levels) & (det > 0.0)
    points = torch.where(valid[:, None], points, torch.full_like(points, float("nan")))
    levels = torch.where(valid, levels, torch.full_like(levels, float("nan")))
    return points, levels, kind, valid

def _sample_limiter_psi(psi, grid, limiter):
    from tokamak_control.core.torch_sampling import bilinear_sample_torch
    return bilinear_sample_torch(psi, grid, limiter)


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


def _points_in_or_on_polygon_torch(points, polygon, *, tolerance: float):
    """Проверить ``point in closure(polygon)`` с метрическим допуском."""
    inside = _points_in_polygon_torch(points, polygon)
    if float(tolerance) <= 0.0:
        return inside
    distance = _point_to_polygon_distances_torch(points, polygon)
    return inside | (distance <= float(tolerance))
