from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from tokamak_control.compute import require_gpu_available
from tokamak_control.core.grid import Grid2D
from tokamak_control.core.torch_sampling import bilinear_sample_torch_points
from tokamak_control.geometry.boundary_common import BoundaryMode, BoundaryNotFoundError, BoundaryStatus


@dataclass(slots=True, repr=True)
class FixedAngleBoundaryGpuResult:
    """Полный результат GPU-поиска LCFS и его фиксированная проекция."""

    found: object
    status_code: object
    topology_code: object
    level: object
    points: object
    radii: object
    intersection_counts: object
    axis_points: object
    x_points: object
    core_boundary: object
    core_boundary_count: object
    limiter_contacts: object
    limiter_contact_count: object
    quality: object


@dataclass(frozen=True, slots=True, repr=True)
class FixedAngleBoundaryGpuGeometry:
    """Предвычисленная геометрия для полного batched GPU-поиска LCFS."""

    angles: object
    validation_angles: object
    dense_angles: object
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
    validation_count = max(int(angles.numel()), 32)
    validation_angles = torch.linspace(
        -float(np.pi),
        float(np.pi),
        validation_count + 1,
        dtype=dtype,
        device=device,
    )[:-1]
    dense_count = max(validation_count * 8, 256)
    dense_angles = torch.linspace(
        -float(np.pi),
        float(np.pi),
        dense_count + 1,
        dtype=dtype,
        device=device,
    )[:-1]
    return FixedAngleBoundaryGpuGeometry(
        angles=angles,
        validation_angles=validation_angles,
        dense_angles=dense_angles,
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
    return_dense_boundary: bool = False,
) -> FixedAngleBoundaryGpuResult:
    """Вычислить LCFS и фиксированную проекцию полностью на GPU.

    В режиме ``equilibrium_lcfs`` один GPU-путь выбирает физический уровень,
    определяет топологию, строит радиусы и при запросе материализует плотный
    замкнутый контур. CPU-экстрактор в этом пути не используется.
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
        validation_angles = torch.as_tensor(
            prepared_geometry.validation_angles,
            dtype=dtype,
            device=device,
        ).reshape(-1)
        dense_angles = torch.as_tensor(
            prepared_geometry.dense_angles,
            dtype=dtype,
            device=device,
        ).reshape(-1)
        limiter_t = torch.as_tensor(prepared_geometry.limiter, dtype=dtype, device=device).reshape(-1, 2)
        max_radii = torch.as_tensor(prepared_geometry.max_radii, dtype=dtype, device=device).reshape(-1)
    else:
        angles = torch.as_tensor(angles_rad, dtype=dtype, device=device).reshape(-1)
        validation_count = max(int(angles.numel()), 32)
        validation_angles = torch.linspace(
            -float(np.pi),
            float(np.pi),
            validation_count + 1,
            dtype=dtype,
            device=device,
        )[:-1]
        dense_count = max(validation_count * 8, 256)
        dense_angles = torch.linspace(
            -float(np.pi),
            float(np.pi),
            dense_count + 1,
            dtype=dtype,
            device=device,
        )[:-1]
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
    intersection_counts = torch.zeros((B, int(angles.numel())), dtype=torch.int64, device=device)
    core_boundary = torch.empty((B, 0, 2), dtype=dtype, device=device)
    core_boundary_count = torch.zeros((B,), dtype=torch.int64, device=device)
    limiter_contacts = torch.empty((B, 0, 2), dtype=dtype, device=device)
    limiter_contact_count = torch.zeros((B,), dtype=torch.int64, device=device)
    quality = torch.full((B, 6), float("nan"), dtype=dtype, device=device)
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
        (
            reset,
            topology_code,
            selected_x_points,
            intersection_counts,
            core_boundary,
            core_boundary_count,
            limiter_contacts,
            limiter_contact_count,
            quality,
        ) = _equilibrium_lcfs_fixed_angle_search(
            psi=field,
            grid=grid,
            axis_points=axis_points,
            projection_center=center_t,
            axis_level=axis_level,
            axis_kind=axis_kind,
            axis_valid=axis_valid,
            measurement_angles=angles,
            validation_angles=validation_angles,
            dense_angles=dense_angles,
            limiter=limiter_t,
            limiter_samples=limiter_samples,
            ray_samples=int(ray_samples),
            return_dense_boundary=bool(return_dense_boundary),
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

    if mode != "equilibrium_lcfs":
        intersection_counts = torch.where(
            torch.isfinite(radii),
            torch.ones_like(radii, dtype=torch.int64),
            torch.zeros_like(radii, dtype=torch.int64),
        )

    return FixedAngleBoundaryGpuResult(
        found=found,
        status_code=status_code,
        topology_code=topology_code,
        level=level,
        points=points,
        radii=radii,
        intersection_counts=intersection_counts,
        axis_points=axis_points,
        x_points=selected_x_points,
        core_boundary=core_boundary,
        core_boundary_count=core_boundary_count,
        limiter_contacts=limiter_contacts,
        limiter_contact_count=limiter_contact_count,
        quality=quality,
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
    validation_angles,
    dense_angles,
    limiter,
    limiter_samples,
    ray_samples: int,
    return_dense_boundary: bool,
):
    """Extract the physical LCFS with a batched GPU level-set graph.

    Wall and X-point candidates are ordered by oriented flux exactly as in the
    CPU reference. Each candidate is converted to marching-squares segments,
    selected X-points are promoted to graph nodes, and the primary-core cycle
    containing the magnetic axis is traced on the GPU. Fixed-angle radii are
    projected from that cycle and never define the LCFS themselves.
    """
    del ray_samples
    torch = __import__("torch")
    B = int(psi.shape[0])
    dtype = psi.dtype
    device = psi.device
    orientation = -axis_kind.to(dtype=dtype)
    flux_scale = torch.amax(psi, dim=(1, 2)) - torch.amin(psi, dim=(1, 2))
    flux_floor = torch.clamp(flux_scale * 1.0e-9, min=torch.finfo(dtype).eps)
    grid_scale = max(float(grid.r.step), float(grid.z.step))

    limiter_batch = limiter_samples.reshape(1, -1, 2).expand(B, -1, -1)
    limiter_values = _sample_points(psi, grid, limiter_batch)
    wall_chi_raw = orientation[:, None] * (limiter_values - axis_level[:, None])
    positive_wall = torch.isfinite(wall_chi_raw) & (wall_chi_raw > flux_floor[:, None])
    local_wall_minimum = (
        positive_wall
        & (wall_chi_raw <= torch.roll(wall_chi_raw, shifts=1, dims=1))
        & (wall_chi_raw <= torch.roll(wall_chi_raw, shifts=-1, dims=1))
    )
    wall_score = torch.where(
        local_wall_minimum,
        wall_chi_raw,
        torch.full_like(wall_chi_raw, float("inf")),
    )
    wall_candidate_count = min(16, int(wall_score.shape[1]))
    wall_chi_candidates, wall_indices = torch.topk(
        wall_score,
        k=wall_candidate_count,
        dim=1,
        largest=False,
        sorted=True,
    )
    wall_candidate_levels = axis_level[:, None] + orientation[:, None] * wall_chi_candidates
    wall_points = limiter_samples[wall_indices]
    wall_used = ~torch.isfinite(wall_chi_candidates)

    x_points_raw, x_levels_raw, x_valid_raw = _xpoint_candidates_gpu(
        psi,
        grid,
        axis_points=axis_points,
        axis_level=axis_level,
        orientation=orientation,
        max_points=8,
    )
    x_chi_raw = orientation[:, None] * (x_levels_raw - axis_level[:, None])
    x_chi_raw = torch.where(
        x_valid_raw & torch.isfinite(x_chi_raw) & (x_chi_raw > flux_floor[:, None]),
        x_chi_raw,
        torch.full_like(x_chi_raw, float("inf")),
    )
    x_chi_candidates, x_order = torch.sort(x_chi_raw, dim=1)
    x_levels = torch.gather(x_levels_raw, 1, x_order)
    x_points = torch.gather(
        x_points_raw,
        1,
        x_order[:, :, None].expand(-1, -1, 2),
    )
    x_used = ~torch.isfinite(x_chi_candidates)
    x_count = int(x_chi_candidates.shape[1])

    selected = torch.zeros((B,), dtype=torch.bool, device=device)
    selected_level = torch.full((B,), float("nan"), dtype=dtype, device=device)
    selected_use_x = torch.zeros((B,), dtype=torch.bool, device=device)
    selected_x_mask = torch.zeros((B, x_count), dtype=torch.bool, device=device)
    selected_wall_point = torch.full((B, 2), float("nan"), dtype=dtype, device=device)
    selected_raw_boundary = None
    selected_raw_count = None
    selected_measurement_points = torch.full(
        (B, int(measurement_angles.numel()), 2),
        float("nan"),
        dtype=dtype,
        device=device,
    )
    selected_measurement_radii = torch.full(
        (B, int(measurement_angles.numel())),
        float("nan"),
        dtype=dtype,
        device=device,
    )
    selected_measurement_counts = torch.zeros(
        (B, int(measurement_angles.numel())),
        dtype=torch.int64,
        device=device,
    )

    anchor_angles = torch.as_tensor(
        [0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi],
        dtype=dtype,
        device=device,
    )
    max_attempts = wall_candidate_count + x_count
    batch_index = torch.arange(B, device=device)
    for _attempt in range(max_attempts):
        active = axis_valid & ~selected
        if not bool(torch.any(active).item()):
            break

        wall_available = ~wall_used
        wall_choice_chi, wall_choice_index = torch.min(
            torch.where(
                wall_available,
                wall_chi_candidates,
                torch.full_like(wall_chi_candidates, float("inf")),
            ),
            dim=1,
        )
        wall_choice_level = wall_candidate_levels[batch_index, wall_choice_index]
        wall_choice_point = wall_points[batch_index, wall_choice_index]

        x_available = ~x_used
        x_choice_chi, _x_choice_index = torch.min(
            torch.where(
                x_available,
                x_chi_candidates,
                torch.full_like(x_chi_candidates, float("inf")),
            ),
            dim=1,
        )
        x_group_tolerance = torch.maximum(
            1.0e-10 * flux_scale,
            2.0e-4 * torch.clamp(x_choice_chi, min=1.0e-12),
        )
        x_group = x_available & (
            torch.abs(x_chi_candidates - x_choice_chi[:, None])
            <= x_group_tolerance[:, None]
        )
        x_group_count = torch.sum(x_group.to(torch.int64), dim=1)
        x_group_level = torch.sum(
            torch.where(x_group, x_levels, torch.zeros_like(x_levels)),
            dim=1,
        ) / torch.clamp(x_group_count.to(dtype), min=1.0)
        has_wall = torch.isfinite(wall_choice_chi)
        has_x = (x_group_count > 0) & torch.isfinite(x_choice_chi)
        choose_x = has_x & ((x_choice_chi < wall_choice_chi) | ~has_wall)
        has_candidate = has_wall | has_x
        candidate_level = torch.where(choose_x, x_group_level, wall_choice_level)
        safe_level = torch.where(active & has_candidate, candidate_level, axis_level)
        candidate_x_mask = x_group & choose_x[:, None]
        candidate_x_points = torch.where(
            candidate_x_mask[:, :, None],
            x_points,
            torch.full_like(x_points, float("nan")),
        )

        segment_points, segment_nodes, segment_valid = _marching_squares_segments_gpu(
            psi=psi,
            grid=grid,
            level=safe_level,
        )
        _anchor_points, _anchor_radii, anchor_found, _anchor_counts, anchor_hits = (
            _ray_intersections_from_segments_gpu(
                segment_points=segment_points,
                segment_valid=segment_valid,
                origins=axis_points,
                angles=anchor_angles,
                deduplicate_counts=False,
            )
        )
        anchor_valid = anchor_hits >= 0
        first_anchor_slot = torch.argmax(anchor_valid.to(torch.int64), dim=1)
        start_segment = torch.gather(
            anchor_hits,
            1,
            first_anchor_slot[:, None],
        ).reshape(B)
        start_segment = torch.where(
            anchor_found,
            start_segment,
            torch.full_like(start_segment, -1),
        )

        raw_boundary, raw_count, cycle_found = _trace_core_cycle_gpu(
            segment_points=segment_points,
            segment_nodes=segment_nodes,
            segment_valid=segment_valid,
            start_segment=start_segment,
            axis_points=axis_points,
            x_points=candidate_x_points,
            grid=grid,
        )
        cycle_segments, cycle_segment_valid = _padded_boundary_segments_gpu(
            boundaries=raw_boundary,
            boundary_count=raw_count,
        )
        axis_boundary_points, _axis_radii, axis_found, axis_counts, _axis_hits = (
            _ray_intersections_from_segments_gpu(
                segment_points=cycle_segments,
                segment_valid=cycle_segment_valid,
                origins=axis_points,
                angles=validation_angles,
            )
        )
        measurement_points, measurement_radii, measurement_found, measurement_counts, _measurement_hits = (
            _ray_intersections_from_segments_gpu(
                segment_points=cycle_segments,
                segment_valid=cycle_segment_valid,
                origins=projection_center,
                angles=measurement_angles,
            )
        )
        axis_inside = torch.all(
            _points_in_or_on_polygon_torch(
                axis_boundary_points,
                limiter,
                tolerance=1.5 * grid_scale,
            ),
            dim=1,
        )
        measurement_inside = torch.all(
            _points_in_or_on_polygon_torch(
                measurement_points,
                limiter,
                tolerance=1.5 * grid_scale,
            ),
            dim=1,
        )

        wall_distance = _point_to_segments_min_distance_gpu(
            points=wall_choice_point,
            segment_points=cycle_segments,
            segment_valid=cycle_segment_valid,
        )
        wall_contact_ok = wall_distance <= 2.5 * grid_scale

        x_distances = []
        for x_index in range(x_count):
            x_distances.append(
                _point_to_segments_min_distance_gpu(
                    points=x_points[:, x_index, :],
                    segment_points=cycle_segments,
                    segment_valid=cycle_segment_valid,
                )
            )
        if x_distances:
            x_distance_matrix = torch.stack(x_distances, dim=1)
            x_contact_ok = torch.all(
                (~x_group) | (x_distance_matrix <= 2.5 * grid_scale),
                dim=1,
            ) & (x_group_count > 0)
        else:
            x_contact_ok = torch.zeros((B,), dtype=torch.bool, device=device)

        candidate_valid = (
            active
            & has_candidate
            & cycle_found
            & axis_found
            & measurement_found
            & torch.all(axis_counts == 1, dim=1)
            & torch.all(measurement_counts == 1, dim=1)
            & axis_inside
            & measurement_inside
            & torch.where(choose_x, x_contact_ok, wall_contact_ok)
        )

        if selected_raw_boundary is None:
            selected_raw_boundary = torch.full_like(raw_boundary, float("nan"))
            selected_raw_count = torch.zeros_like(raw_count)
        selected_level = torch.where(candidate_valid, candidate_level, selected_level)
        selected_use_x = torch.where(candidate_valid, choose_x, selected_use_x)
        selected_x_mask = torch.where(
            candidate_valid[:, None],
            candidate_x_mask,
            selected_x_mask,
        )
        selected_wall_point = torch.where(
            (candidate_valid & ~choose_x)[:, None],
            wall_choice_point,
            selected_wall_point,
        )
        selected_raw_boundary = torch.where(
            candidate_valid[:, None, None],
            raw_boundary,
            selected_raw_boundary,
        )
        selected_raw_count = torch.where(candidate_valid, raw_count, selected_raw_count)
        selected_measurement_points = torch.where(
            candidate_valid[:, None, None],
            measurement_points,
            selected_measurement_points,
        )
        selected_measurement_radii = torch.where(
            candidate_valid[:, None],
            measurement_radii,
            selected_measurement_radii,
        )
        selected_measurement_counts = torch.where(
            candidate_valid[:, None],
            measurement_counts,
            selected_measurement_counts,
        )
        selected = selected | candidate_valid

        rejected = active & has_candidate & ~candidate_valid
        rejected_wall = rejected & ~choose_x
        wall_used[batch_index, wall_choice_index] = (
            wall_used[batch_index, wall_choice_index] | rejected_wall
        )
        x_used = x_used | (rejected[:, None] & choose_x[:, None] & x_group)

    if selected_raw_boundary is None or selected_raw_count is None:
        raise RuntimeError("equilibrium LCFS candidate search produced no graph tensors")

    found = selected & axis_valid
    selected_x = torch.where(
        selected_x_mask[:, :, None],
        x_points,
        torch.full_like(x_points, float("nan")),
    )
    selected_x_count = torch.sum(selected_x_mask.to(torch.int64), dim=1)
    topology_code = torch.where(
        found & ~selected_use_x,
        torch.ones((B,), dtype=torch.int64, device=device),
        torch.zeros((B,), dtype=torch.int64, device=device),
    )
    topology_code = torch.where(
        found & selected_use_x,
        torch.where(
            selected_x_count >= 3,
            torch.full((B,), 4, dtype=torch.int64, device=device),
            torch.where(
                selected_x_count == 2,
                torch.full((B,), 3, dtype=torch.int64, device=device),
                torch.full((B,), 2, dtype=torch.int64, device=device),
            ),
        ),
        topology_code,
    )

    limiter_contacts = torch.where(
        (found & ~selected_use_x)[:, None, None],
        selected_wall_point[:, None, :],
        torch.full((B, 1, 2), float("nan"), dtype=dtype, device=device),
    )
    limiter_contact_count = (found & ~selected_use_x).to(torch.int64)

    core_boundary = torch.empty((B, 0, 2), dtype=dtype, device=device)
    core_boundary_count = torch.zeros((B,), dtype=torch.int64, device=device)
    quality = torch.full((B, 6), float("nan"), dtype=dtype, device=device)
    if bool(return_dense_boundary):
        dense_count = max(int(dense_angles.numel()), 32)
        core_boundary, core_boundary_count, dense_found = _resample_closed_cycles_gpu(
            boundaries=selected_raw_boundary,
            boundary_count=selected_raw_count,
            output_count=dense_count + 1,
        )
        core_boundary = _snap_dense_boundary_nodes(core_boundary, nodes=selected_x)
        core_boundary = _snap_dense_boundary_nodes(core_boundary, nodes=limiter_contacts)
        core_boundary[:, -1, :] = core_boundary[:, 0, :].clone()
        found = found & dense_found
        core_boundary = torch.where(
            found[:, None, None],
            core_boundary,
            torch.full_like(core_boundary, float("nan")),
        )
        core_boundary_count = torch.where(
            found,
            core_boundary_count,
            torch.zeros_like(core_boundary_count),
        )
        quality = _gpu_boundary_quality(
            psi=psi,
            grid=grid,
            boundary=core_boundary,
            boundary_count=core_boundary_count,
            level=selected_level,
            axis_level=axis_level,
            axis_kind=axis_kind,
            limiter=limiter,
            flux_scale=flux_scale,
        )

    points = torch.where(
        found[:, None, None],
        selected_measurement_points,
        torch.full_like(selected_measurement_points, float("nan")),
    )
    radii = torch.where(
        found[:, None],
        selected_measurement_radii,
        torch.full_like(selected_measurement_radii, float("nan")),
    )
    measurement_counts = torch.where(
        found[:, None],
        selected_measurement_counts,
        torch.zeros_like(selected_measurement_counts),
    )
    selected_level = torch.where(
        found,
        selected_level,
        torch.full_like(selected_level, float("nan")),
    )
    topology_code = torch.where(found, topology_code, torch.zeros_like(topology_code))
    selected_x = torch.where(
        found[:, None, None],
        selected_x,
        torch.full_like(selected_x, float("nan")),
    )
    limiter_contacts = torch.where(
        found[:, None, None],
        limiter_contacts,
        torch.full_like(limiter_contacts, float("nan")),
    )
    limiter_contact_count = torch.where(
        found,
        limiter_contact_count,
        torch.zeros_like(limiter_contact_count),
    )
    return (
        (points, radii, found, selected_level),
        topology_code,
        selected_x,
        measurement_counts,
        core_boundary,
        core_boundary_count,
        limiter_contacts,
        limiter_contact_count,
        quality,
    )

def _marching_squares_segments_gpu(*, psi, grid: Grid2D, level):
    """Build batched marching-squares segments and shared edge-node IDs."""
    torch = __import__("torch")
    B, nz, nr = psi.shape
    cell_nz = int(nz) - 1
    cell_nr = int(nr) - 1
    cell_count = cell_nz * cell_nr
    dtype = psi.dtype
    device = psi.device

    v0 = psi[:, :-1, :-1].reshape(B, cell_count)
    v1 = psi[:, :-1, 1:].reshape(B, cell_count)
    v2 = psi[:, 1:, 1:].reshape(B, cell_count)
    v3 = psi[:, 1:, :-1].reshape(B, cell_count)
    below0 = v0 <= level[:, None]
    below1 = v1 <= level[:, None]
    below2 = v2 <= level[:, None]
    below3 = v3 <= level[:, None]
    case_index = (
        below0.to(torch.int64)
        + 2 * below1.to(torch.int64)
        + 4 * below2.to(torch.int64)
        + 8 * below3.to(torch.int64)
    )

    table = torch.as_tensor(
        [
            [[-1, -1], [-1, -1]],
            [[3, 0], [-1, -1]],
            [[0, 1], [-1, -1]],
            [[3, 1], [-1, -1]],
            [[1, 2], [-1, -1]],
            [[3, 0], [1, 2]],
            [[0, 2], [-1, -1]],
            [[3, 2], [-1, -1]],
            [[2, 3], [-1, -1]],
            [[0, 2], [-1, -1]],
            [[0, 1], [2, 3]],
            [[1, 2], [-1, -1]],
            [[1, 3], [-1, -1]],
            [[0, 1], [-1, -1]],
            [[3, 0], [-1, -1]],
            [[-1, -1], [-1, -1]],
        ],
        dtype=torch.int64,
        device=device,
    )
    edge_pairs = table[case_index]
    center_below = 0.25 * (v0 + v1 + v2 + v3) <= level[:, None]
    case5 = case_index == 5
    case10 = case_index == 10
    pair_inside_5 = torch.as_tensor([[0, 1], [2, 3]], dtype=torch.int64, device=device)
    pair_outside_5 = torch.as_tensor([[3, 0], [1, 2]], dtype=torch.int64, device=device)
    pair_inside_10 = torch.as_tensor([[3, 0], [1, 2]], dtype=torch.int64, device=device)
    pair_outside_10 = torch.as_tensor([[0, 1], [2, 3]], dtype=torch.int64, device=device)
    edge_pairs = torch.where(
        case5[:, :, None, None],
        torch.where(
            center_below[:, :, None, None],
            pair_inside_5.reshape(1, 1, 2, 2),
            pair_outside_5.reshape(1, 1, 2, 2),
        ),
        edge_pairs,
    )
    edge_pairs = torch.where(
        case10[:, :, None, None],
        torch.where(
            center_below[:, :, None, None],
            pair_inside_10.reshape(1, 1, 2, 2),
            pair_outside_10.reshape(1, 1, 2, 2),
        ),
        edge_pairs,
    )

    jj = torch.arange(cell_nz, device=device).reshape(-1, 1).expand(cell_nz, cell_nr).reshape(-1)
    ii = torch.arange(cell_nr, device=device).reshape(1, -1).expand(cell_nz, cell_nr).reshape(-1)
    r0 = float(grid.r.coords()[0])
    z0 = float(grid.z.coords()[0])
    dr = float(grid.r.step)
    dz = float(grid.z.step)
    cell_r = r0 + ii.to(dtype) * dr
    cell_z = z0 + jj.to(dtype) * dz

    def interpolate(first_value, second_value):
        """Interpolate an edge crossing at the requested level."""
        denominator = second_value - first_value
        safe = torch.where(
            torch.abs(denominator) > torch.finfo(dtype).eps,
            denominator,
            torch.ones_like(denominator),
        )
        return torch.clamp((level[:, None] - first_value) / safe, 0.0, 1.0)

    t0 = interpolate(v0, v1)
    t1 = interpolate(v1, v2)
    t2 = interpolate(v3, v2)
    t3 = interpolate(v0, v3)
    edge_points = torch.stack(
        (
            torch.stack((cell_r[None, :] + t0 * dr, cell_z[None, :].expand(B, -1)), dim=2),
            torch.stack(((cell_r + dr)[None, :].expand(B, -1), cell_z[None, :] + t1 * dz), dim=2),
            torch.stack((cell_r[None, :] + t2 * dr, (cell_z + dz)[None, :].expand(B, -1)), dim=2),
            torch.stack((cell_r[None, :].expand(B, -1), cell_z[None, :] + t3 * dz), dim=2),
        ),
        dim=2,
    )

    horizontal_count = int(nz) * (int(nr) - 1)
    edge_nodes = torch.stack(
        (
            jj * (int(nr) - 1) + ii,
            horizontal_count + jj * int(nr) + (ii + 1),
            (jj + 1) * (int(nr) - 1) + ii,
            horizontal_count + jj * int(nr) + ii,
        ),
        dim=1,
    )
    safe_edges = torch.clamp(edge_pairs, min=0)
    point_index = safe_edges[..., None].expand(-1, -1, -1, -1, 2)
    all_edge_points = edge_points[:, :, None, :, :].expand(-1, -1, 2, -1, -1)
    segment_points = torch.gather(all_edge_points, 3, point_index).reshape(B, 2 * cell_count, 2, 2)
    all_edge_nodes = edge_nodes[None, :, None, :].expand(B, -1, 2, -1)
    segment_nodes = torch.gather(all_edge_nodes, 3, safe_edges).reshape(B, 2 * cell_count, 2)
    segment_valid = (edge_pairs[..., 0] >= 0).reshape(B, 2 * cell_count)
    segment_length = torch.linalg.norm(segment_points[:, :, 1, :] - segment_points[:, :, 0, :], dim=2)
    segment_valid = segment_valid & torch.isfinite(segment_length) & (segment_length > 1.0e-12)
    segment_points = torch.where(
        segment_valid[:, :, None, None],
        segment_points,
        torch.full_like(segment_points, float("nan")),
    )
    segment_nodes = torch.where(
        segment_valid[:, :, None],
        segment_nodes,
        torch.full_like(segment_nodes, -1),
    )
    return segment_points, segment_nodes, segment_valid


def _ray_intersections_from_segments_gpu(
    *,
    segment_points,
    segment_valid,
    origins,
    angles,
    deduplicate_counts: bool = True,
):
    """Project a level-set segment graph onto center-origin rays on the GPU."""
    torch = __import__("torch")
    B, _segment_count = segment_valid.shape
    angle_count = int(angles.numel())
    dtype = segment_points.dtype
    device = segment_points.device
    starts = segment_points[:, :, 0, :]
    vectors = segment_points[:, :, 1, :] - starts
    points = torch.full((B, angle_count, 2), float("nan"), dtype=dtype, device=device)
    radii = torch.full((B, angle_count), float("nan"), dtype=dtype, device=device)
    counts = torch.zeros((B, angle_count), dtype=torch.int64, device=device)
    hit_segments = torch.full((B, angle_count), -1, dtype=torch.int64, device=device)
    directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)

    for angle_index in range(angle_count):
        direction = directions[angle_index]
        rhs = starts - origins[:, None, :]
        denominator = direction[0] * vectors[:, :, 1] - direction[1] * vectors[:, :, 0]
        valid_denominator = torch.abs(denominator) > 1.0e-14
        safe_denominator = torch.where(valid_denominator, denominator, torch.ones_like(denominator))
        ray_t = (
            rhs[:, :, 0] * vectors[:, :, 1]
            - rhs[:, :, 1] * vectors[:, :, 0]
        ) / safe_denominator
        segment_u = (
            rhs[:, :, 0] * direction[1]
            - rhs[:, :, 1] * direction[0]
        ) / safe_denominator
        hit = (
            segment_valid
            & valid_denominator
            & torch.isfinite(ray_t)
            & (ray_t >= 0.0)
            & (segment_u >= -1.0e-10)
            & (segment_u <= 1.0 + 1.0e-10)
        )
        candidate_t = torch.where(hit, ray_t, torch.full_like(ray_t, float("inf")))
        nearest_t, nearest_segment = torch.min(candidate_t, dim=1)
        has = torch.isfinite(nearest_t)
        radii[:, angle_index] = torch.where(
            has,
            nearest_t,
            torch.full_like(nearest_t, float("nan")),
        )
        points[:, angle_index, :] = torch.where(
            has[:, None],
            origins + nearest_t[:, None] * direction[None, :],
            torch.full((B, 2), float("nan"), dtype=dtype, device=device),
        )
        hit_segments[:, angle_index] = torch.where(
            has,
            nearest_segment,
            torch.full_like(nearest_segment, -1),
        )
        if deduplicate_counts:
            sorted_t = torch.sort(candidate_t, dim=1).values
            finite = torch.isfinite(sorted_t)
            first = finite[:, 0].to(torch.int64)
            separated = (
                finite[:, 1:]
                & (
                    ~finite[:, :-1]
                    | (torch.abs(sorted_t[:, 1:] - sorted_t[:, :-1]) > 1.0e-9)
                )
            )
            counts[:, angle_index] = first + torch.sum(separated.to(torch.int64), dim=1)
        else:
            counts[:, angle_index] = torch.sum(hit.to(torch.int64), dim=1)

    found = torch.all(torch.isfinite(radii), dim=1)
    return points, radii, found, counts, hit_segments


def _padded_boundary_segments_gpu(*, boundaries, boundary_count):
    """Convert padded closed cycles to segment tensors and validity masks."""
    torch = __import__("torch")
    starts = boundaries[:, :-1, :]
    ends = boundaries[:, 1:, :]
    segment_points = torch.stack((starts, ends), dim=2)
    positions = torch.arange(segment_points.shape[1], device=boundaries.device)
    segment_valid = positions[None, :] < (boundary_count - 1)[:, None]
    finite = torch.all(torch.isfinite(segment_points), dim=(2, 3))
    length = torch.linalg.norm(ends - starts, dim=2)
    segment_valid = segment_valid & finite & (length > 1.0e-12)
    segment_points = torch.where(
        segment_valid[:, :, None, None],
        segment_points,
        torch.full_like(segment_points, float("nan")),
    )
    return segment_points, segment_valid

def _point_to_segments_min_distance_gpu(*, points, segment_points, segment_valid):
    """Return the minimum point-to-segment distance for each batch lane."""
    torch = __import__("torch")
    starts = segment_points[:, :, 0, :]
    vectors = segment_points[:, :, 1, :] - starts
    denominator = torch.sum(vectors * vectors, dim=2)
    safe_denominator = torch.where(denominator > 0.0, denominator, torch.ones_like(denominator))
    fraction = torch.clamp(
        torch.sum((points[:, None, :] - starts) * vectors, dim=2) / safe_denominator,
        0.0,
        1.0,
    )
    closest = starts + fraction[:, :, None] * vectors
    distance = torch.linalg.norm(points[:, None, :] - closest, dim=2)
    distance = torch.where(
        segment_valid & torch.isfinite(distance),
        distance,
        torch.full_like(distance, float("inf")),
    )
    return torch.amin(distance, dim=1)


def _trace_core_cycle_gpu(
    *,
    segment_points,
    segment_nodes,
    segment_valid,
    start_segment,
    axis_points,
    x_points,
    grid: Grid2D,
):
    """Trace the primary-core cycle while treating X-points as graph nodes."""
    torch = __import__("torch")
    B, segment_count = segment_valid.shape
    device = segment_points.device
    dtype = segment_points.dtype
    base_node_count = int(grid.z.size) * (int(grid.r.size) - 1) + (
        int(grid.z.size) - 1
    ) * int(grid.r.size)
    x_count = int(x_points.shape[1])
    node_count = base_node_count + x_count
    sentinel_segment = int(segment_count)
    sentinel_state = 2 * int(segment_count)
    tolerance = 1.75 * max(float(grid.r.step), float(grid.z.step))

    points_work = segment_points.clone()
    nodes_work = segment_nodes.clone()
    if x_count > 0:
        endpoint_distance = torch.linalg.norm(
            points_work[:, :, :, None, :] - x_points[:, None, None, :, :],
            dim=4,
        )
        finite_x = torch.all(torch.isfinite(x_points), dim=2)
        endpoint_distance = torch.where(
            finite_x[:, None, None, :],
            endpoint_distance,
            torch.full_like(endpoint_distance, float("inf")),
        )
        nearest_distance, nearest_x = torch.min(endpoint_distance, dim=3)
        snap = segment_valid[:, :, None] & (nearest_distance <= tolerance)
        special_nodes = base_node_count + nearest_x
        nodes_work = torch.where(snap, special_nodes, nodes_work)
        batch = torch.arange(B, device=device)[:, None, None]
        snapped_points = x_points[batch, nearest_x, :]
        points_work = torch.where(snap[:, :, :, None], snapped_points, points_work)

    adjacency_min = torch.full(
        (B, node_count),
        sentinel_segment,
        dtype=torch.int64,
        device=device,
    )
    adjacency_max = torch.full(
        (B, node_count),
        -1,
        dtype=torch.int64,
        device=device,
    )
    segment_ids = torch.arange(segment_count, device=device, dtype=torch.int64)[None, :].expand(B, -1)
    for endpoint in range(2):
        node = torch.clamp(nodes_work[:, :, endpoint], min=0)
        source_min = torch.where(segment_valid, segment_ids, torch.full_like(segment_ids, sentinel_segment))
        source_max = torch.where(segment_valid, segment_ids, torch.full_like(segment_ids, -1))
        adjacency_min.scatter_reduce_(1, node, source_min, reduce="amin", include_self=True)
        adjacency_max.scatter_reduce_(1, node, source_max, reduce="amax", include_self=True)

    max_x_degree = 8
    if x_count > 0:
        special_ids = base_node_count + torch.arange(x_count, device=device, dtype=torch.int64)
        incident = (
            (nodes_work[:, None, :, 0] == special_ids[None, :, None])
            | (nodes_work[:, None, :, 1] == special_ids[None, :, None])
        ) & segment_valid[:, None, :]
        incident_ids = torch.where(
            incident,
            segment_ids[:, None, :],
            torch.full((B, x_count, segment_count), sentinel_segment, dtype=torch.int64, device=device),
        )
        x_adjacency = torch.topk(
            incident_ids,
            k=min(max_x_degree, segment_count),
            dim=2,
            largest=False,
            sorted=True,
        ).values
        if int(x_adjacency.shape[2]) < max_x_degree:
            padding = torch.full(
                (B, x_count, max_x_degree - int(x_adjacency.shape[2])),
                sentinel_segment,
                dtype=torch.int64,
                device=device,
            )
            x_adjacency = torch.cat((x_adjacency, padding), dim=2)
    else:
        x_adjacency = torch.empty((B, 0, max_x_degree), dtype=torch.int64, device=device)

    state_ids = torch.arange(2 * segment_count, device=device, dtype=torch.int64)
    state_segment = state_ids // 2
    state_forward = (state_ids % 2) == 0
    state_segment_batch = state_segment[None, :].expand(B, -1)
    state_end_node = torch.where(
        state_forward[None, :],
        nodes_work[:, state_segment, 1],
        nodes_work[:, state_segment, 0],
    )
    safe_end_node = torch.clamp(state_end_node, min=0)
    first_adjacent = torch.gather(adjacency_min, 1, safe_end_node)
    last_adjacent = torch.gather(adjacency_max, 1, safe_end_node)
    regular_next_segment = torch.where(
        first_adjacent != state_segment_batch,
        first_adjacent,
        last_adjacent,
    )

    is_x_node = (state_end_node >= base_node_count) & (state_end_node < node_count)
    if x_count > 0:
        x_index = torch.clamp(state_end_node - base_node_count, min=0, max=x_count - 1)
        batch_state = torch.arange(B, device=device)[:, None].expand(B, 2 * segment_count)
        candidates = x_adjacency[batch_state, x_index, :]
        candidate_valid = (
            is_x_node[:, :, None]
            & (candidates >= 0)
            & (candidates < segment_count)
            & (candidates != state_segment_batch[:, :, None])
        )
        safe_candidates = torch.clamp(candidates, min=0, max=max(segment_count - 1, 0))
        batch_candidate = torch.arange(B, device=device)[:, None, None]
        candidate_nodes0 = nodes_work[batch_candidate, safe_candidates, 0]
        candidate_nodes1 = nodes_work[batch_candidate, safe_candidates, 1]
        candidate_points0 = points_work[batch_candidate, safe_candidates, 0, :]
        candidate_points1 = points_work[batch_candidate, safe_candidates, 1, :]
        candidate_other = torch.where(
            (candidate_nodes0 == state_end_node[:, :, None])[:, :, :, None],
            candidate_points1,
            candidate_points0,
        )
        current_x = x_points[batch_state, x_index, :]
        axis_direction = axis_points[:, None, :] - current_x
        candidate_direction = candidate_other - current_x[:, :, None, :]
        axis_norm = torch.linalg.norm(axis_direction, dim=2)
        candidate_norm = torch.linalg.norm(candidate_direction, dim=3)
        denominator = torch.clamp(
            axis_norm[:, :, None] * candidate_norm,
            min=torch.finfo(dtype).eps,
        )
        alignment = torch.sum(
            candidate_direction * axis_direction[:, :, None, :],
            dim=3,
        ) / denominator
        alignment = torch.where(
            candidate_valid,
            alignment,
            torch.full_like(alignment, -float("inf")),
        )
        best_x_slot = torch.argmax(alignment, dim=2)
        x_next_segment = torch.gather(candidates, 2, best_x_slot[:, :, None]).squeeze(2)
        x_has_next = torch.any(candidate_valid, dim=2)
    else:
        x_next_segment = torch.full_like(regular_next_segment, sentinel_segment)
        x_has_next = torch.zeros_like(is_x_node)

    next_segment = torch.where(is_x_node, x_next_segment, regular_next_segment)
    next_valid = (
        segment_valid[:, state_segment]
        & (state_end_node >= 0)
        & torch.where(
            is_x_node,
            x_has_next,
            (next_segment >= 0)
            & (next_segment < segment_count)
            & (next_segment != state_segment_batch),
        )
    )
    safe_next_segment = torch.clamp(next_segment, min=0, max=max(segment_count - 1, 0))
    next_starts_at_node = torch.gather(
        nodes_work[:, :, 0],
        1,
        safe_next_segment,
    ) == state_end_node
    successor = 2 * safe_next_segment + torch.where(
        next_starts_at_node,
        torch.zeros_like(safe_next_segment),
        torch.ones_like(safe_next_segment),
    )
    successor = torch.where(
        next_valid,
        successor,
        torch.full_like(successor, sentinel_state),
    )
    successor = torch.cat(
        (
            successor,
            torch.full((B, 1), sentinel_state, dtype=torch.int64, device=device),
        ),
        dim=1,
    )

    max_cycle_points = min(4096, 8 * (int(grid.r.size) + int(grid.z.size)))
    jump_tables = [successor]
    while (1 << len(jump_tables)) < max_cycle_points:
        previous = jump_tables[-1]
        jump_tables.append(torch.gather(previous, 1, previous))

    safe_start_segment = torch.clamp(start_segment, min=0, max=max(segment_count - 1, 0))
    start_state = 2 * safe_start_segment
    positions = torch.arange(max_cycle_points, device=device, dtype=torch.int64)
    states = start_state[:, None].expand(B, max_cycle_points).clone()
    for bit, jump in enumerate(jump_tables):
        moved = torch.gather(jump, 1, states)
        states = torch.where(
            ((positions >> bit) & 1)[None, :].to(torch.bool),
            moved,
            states,
        )

    returns = states[:, 1:] == start_state[:, None]
    has_return = torch.any(returns, dim=1)
    first_return = torch.argmax(returns.to(torch.int64), dim=1) + 1
    cycle_length = torch.where(has_return, first_return, torch.zeros_like(first_return))
    trace_valid = (
        (start_segment >= 0)
        & segment_valid[torch.arange(B, device=device), safe_start_segment]
        & has_return
        & (cycle_length >= 3)
    )

    safe_states = torch.clamp(states, max=2 * segment_count - 1)
    traced_segments = safe_states // 2
    traced_forward = (safe_states % 2) == 0
    batch = torch.arange(B, device=device)[:, None]
    start_points = torch.where(
        traced_forward[:, :, None],
        points_work[batch, traced_segments, 0, :],
        points_work[batch, traced_segments, 1, :],
    )
    position_valid = positions[None, :] < cycle_length[:, None]
    raw = torch.full(
        (B, max_cycle_points + 1, 2),
        float("nan"),
        dtype=dtype,
        device=device,
    )
    raw[:, :max_cycle_points, :] = torch.where(
        position_valid[:, :, None],
        start_points,
        torch.full_like(start_points, float("nan")),
    )
    closing_index = torch.clamp(cycle_length, max=max_cycle_points)
    raw[torch.arange(B, device=device), closing_index, :] = raw[:, 0, :].clone()
    count = torch.where(trace_valid, cycle_length + 1, torch.zeros_like(cycle_length))
    raw = torch.where(
        trace_valid[:, None, None],
        raw,
        torch.full_like(raw, float("nan")),
    )
    return raw, count, trace_valid

def _resample_closed_cycles_gpu(*, boundaries, boundary_count, output_count: int):
    """Resample padded closed GPU cycles to a fixed number of arc-length points."""
    torch = __import__("torch")
    B, max_points, _ = boundaries.shape
    dtype = boundaries.dtype
    device = boundaries.device
    edge_count = max_points - 1
    positions = torch.arange(edge_count, device=device)
    valid_edges = positions[None, :] < (boundary_count - 1)[:, None]
    starts = boundaries[:, :-1, :]
    ends = boundaries[:, 1:, :]
    lengths = torch.linalg.norm(ends - starts, dim=2)
    lengths = torch.where(valid_edges & torch.isfinite(lengths), lengths, torch.zeros_like(lengths))
    cumulative = torch.cat(
        (torch.zeros((B, 1), dtype=dtype, device=device), torch.cumsum(lengths, dim=1)),
        dim=1,
    )
    total = cumulative[:, -1]
    valid = (boundary_count >= 4) & torch.isfinite(total) & (total > 0.0)
    fractions = torch.linspace(0.0, 1.0, int(output_count), dtype=dtype, device=device)
    targets = total[:, None] * fractions[None, :]
    indices = torch.searchsorted(cumulative, targets, right=True) - 1
    indices = torch.clamp(indices, min=0, max=edge_count - 1)
    batch = torch.arange(B, device=device)[:, None]
    left_s = cumulative[batch, indices]
    edge_length = lengths[batch, indices]
    alpha = torch.where(
        edge_length > 0.0,
        torch.clamp((targets - left_s) / edge_length, 0.0, 1.0),
        torch.zeros_like(targets),
    )
    sampled = starts[batch, indices, :] + alpha[:, :, None] * (
        ends[batch, indices, :] - starts[batch, indices, :]
    )
    sampled[:, -1, :] = sampled[:, 0, :]
    sampled = torch.where(
        valid[:, None, None],
        sampled,
        torch.full_like(sampled, float("nan")),
    )
    count = torch.where(
        valid,
        torch.full((B,), int(output_count), dtype=torch.int64, device=device),
        torch.zeros((B,), dtype=torch.int64, device=device),
    )
    return sampled, count, valid

def _snap_dense_boundary_nodes(points, *, nodes):
    """Вставить точные X-точки и контакты в ближайшие узлы плотного контура."""
    torch = __import__("torch")
    if int(points.shape[1]) == 0 or int(nodes.shape[1]) == 0:
        return points
    out = points.clone()
    B = int(points.shape[0])
    batch = torch.arange(B, device=points.device)
    for node_index in range(int(nodes.shape[1])):
        node = nodes[:, node_index, :]
        valid = torch.all(torch.isfinite(node), dim=1)
        distances = torch.linalg.norm(out - node[:, None, :], dim=2)
        nearest = torch.argmin(distances, dim=1)
        previous = out[batch, nearest, :]
        out[batch, nearest, :] = torch.where(valid[:, None], node, previous)
    return out


def _gpu_boundary_quality(
    *,
    psi,
    grid: Grid2D,
    boundary,
    boundary_count,
    level,
    axis_level,
    axis_kind,
    limiter,
    flux_scale,
):
    """Вычислить диагностические метрики плотного GPU-контура."""
    torch = __import__("torch")
    B = int(psi.shape[0])
    if int(boundary.shape[1]) == 0:
        return torch.full((B, 6), float("nan"), dtype=psi.dtype, device=psi.device)
    points = boundary[:, :-1, :]
    valid_points = torch.all(torch.isfinite(points), dim=2)
    values = _sample_points(psi, grid, points)
    residual = torch.abs(values - level[:, None])
    residual = torch.where(valid_points, residual, torch.zeros_like(residual))
    max_residual = torch.amax(residual, dim=1)
    normalized = max_residual / torch.clamp(flux_scale, min=torch.finfo(psi.dtype).eps)
    closure = torch.linalg.norm(boundary[:, 0, :] - boundary[:, -1, :], dim=1)

    dz = float(grid.z.step)
    dr = float(grid.r.step)
    grad_z, grad_r = torch.gradient(psi, spacing=(dz, dr), dim=(1, 2))
    grad_r_values = _sample_points(grad_r, grid, points)
    grad_z_values = _sample_points(grad_z, grid, points)
    gradient_norm = torch.sqrt(grad_r_values * grad_r_values + grad_z_values * grad_z_values)
    gradient_norm = torch.where(
        valid_points & torch.isfinite(gradient_norm),
        gradient_norm,
        torch.full_like(gradient_norm, float("inf")),
    )
    minimum_gradient = torch.amin(gradient_norm, dim=1)
    inside = _points_in_or_on_polygon_torch(
        points,
        limiter,
        tolerance=1.5 * max(dr, dz),
    )
    limiter_violations = torch.sum((valid_points & ~inside).to(psi.dtype), dim=1)

    r_coords = torch.as_tensor(grid.r.coords(), dtype=psi.dtype, device=psi.device)
    z_coords = torch.as_tensor(grid.z.coords(), dtype=psi.dtype, device=psi.device)
    Z, R = torch.meshgrid(z_coords, r_coords, indexing="ij")
    grid_points = torch.stack((R.reshape(-1), Z.reshape(-1)), dim=1)
    limiter_mask = _points_in_or_on_polygon_torch(
        grid_points.reshape(1, -1, 2).expand(B, -1, -1),
        limiter,
        tolerance=0.0,
    ).reshape_as(psi)
    core_mask = torch.where(
        axis_kind[:, None, None] > 0,
        psi > level[:, None, None],
        psi < level[:, None, None],
    ) & limiter_mask
    component_size = torch.sum(core_mask.to(psi.dtype), dim=(1, 2))

    valid_boundary = boundary_count > 0
    result = torch.stack(
        (
            max_residual,
            normalized,
            closure,
            minimum_gradient,
            limiter_violations,
            component_size,
        ),
        dim=1,
    )
    return torch.where(
        valid_boundary[:, None],
        result,
        torch.full_like(result, float("nan")),
    )

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
