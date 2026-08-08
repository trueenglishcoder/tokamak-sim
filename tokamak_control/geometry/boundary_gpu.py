from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tokamak_control.compute import require_gpu_available
from tokamak_control.core.grid import Grid2D
from tokamak_control.core.torch_sampling import bilinear_sample_torch_points
from tokamak_control.geometry.boundary_common import BoundaryMode, BoundaryNotFoundError, BoundaryStatus
from tokamak_control.geometry.equilibrium_lcfs_gpu import (
    ExactSplineGpuField,
    ExactSplineGpuGeometry,
    build_level_set_segments_exact_gpu,
    combine_exact_spline_gpu_field,
    evaluate_exact_spline_value_gradient,
    find_critical_points_exact_gpu,
    limiter_flux_candidates_exact_gpu,
    prepare_exact_spline_gpu_geometry,
    repeat_exact_spline_gpu_field,
)


_EQUILIBRIUM_GRAPH_LANE_CHUNK = 32


@dataclass(slots=True, repr=True)
class FixedAngleBoundaryGpuResult:
    """Полный результат GPU-поиска LCFS и его фиксированная проекция."""

    found: object
    status_code: object
    topology_code: object
    level: object
    psi_axis: object
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
    projection_valid: object
    projection_error_code: object


@dataclass(frozen=True, slots=True, repr=True)
class FixedAngleBoundaryGpuGeometry:
    """Предвычисленная геометрия для полного batched GPU-поиска LCFS."""

    angles: object
    limiter: object
    max_radii: object
    exact_spline: ExactSplineGpuGeometry | None


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
    basis_fields: np.ndarray | None = None,
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
    exact_spline = None
    if str(boundary_mode) == "equilibrium_lcfs" and basis_fields is not None:
        exact_spline = prepare_exact_spline_gpu_geometry(
            grid=grid,
            basis_fields=np.asarray(basis_fields, dtype=np.float64),
            limiter_samples=_sample_closed_polyline_numpy(limiter_np, sample_count),
            limiter_poly=limiter_np,
            device=device,
            dtype=dtype,
        )
    return FixedAngleBoundaryGpuGeometry(
        angles=angles,
        limiter=limiter,
        max_radii=max_radii,
        exact_spline=exact_spline,
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
    amplitudes: object | None = None,
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
    exact_spline = prepared_geometry.exact_spline if prepared_geometry is not None else None
    amplitudes_t = None
    if amplitudes is not None:
        amplitudes_t = torch.as_tensor(amplitudes, dtype=dtype, device=device).reshape(B, -1)
    exact_field = None
    exact_x_points = None
    exact_x_levels = None
    exact_x_valid = None
    if mode == "equilibrium_lcfs":
        if exact_spline is None or amplitudes_t is None:
            raise BoundaryNotFoundError(
                "equilibrium_lcfs GPU path requires exact spline basis amplitudes"
            )
        exact_field = combine_exact_spline_gpu_field(
            amplitudes=amplitudes_t,
            geometry=exact_spline,
        )
        (
            axis_points,
            axis_level,
            axis_kind,
            axis_valid,
            exact_x_points,
            exact_x_levels,
            exact_x_valid,
        ) = find_critical_points_exact_gpu(
            psi=field,
            field=exact_field,
            center_hint=center,
            max_candidates=256,
            max_o_points=16,
            max_x_points=32,
        )

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
    selected_x_points = torch.full((B, 32, 2), float("nan"), dtype=dtype, device=device)
    intersection_counts = torch.zeros((B, int(angles.numel())), dtype=torch.int64, device=device)
    core_boundary = torch.empty((B, 0, 2), dtype=dtype, device=device)
    core_boundary_count = torch.zeros((B,), dtype=torch.int64, device=device)
    limiter_contacts = torch.empty((B, 0, 2), dtype=dtype, device=device)
    limiter_contact_count = torch.zeros((B,), dtype=torch.int64, device=device)
    quality = torch.full((B, 6), float("nan"), dtype=dtype, device=device)
    if mode == "equilibrium_lcfs":
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
            limiter=limiter_t,
            exact_field=exact_field,
            critical_x_points=exact_x_points,
            critical_x_levels=exact_x_levels,
            critical_x_valid=exact_x_valid,
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

    projection_valid = (
        found
        & torch.all(intersection_counts == 1, dim=1)
        & torch.all(torch.isfinite(radii), dim=1)
    )
    projection_error_code = torch.where(
        projection_valid,
        torch.zeros((B,), dtype=torch.int64, device=device),
        torch.where(
            found,
            torch.full((B,), 2, dtype=torch.int64, device=device),
            torch.ones((B,), dtype=torch.int64, device=device),
        ),
    )
    return FixedAngleBoundaryGpuResult(
        found=found,
        status_code=status_code,
        topology_code=topology_code,
        level=level,
        psi_axis=torch.where(axis_valid, axis_level, torch.full_like(axis_level, float("nan"))),
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
        projection_valid=projection_valid,
        projection_error_code=projection_error_code,
    )


def _sample_points(psi, grid: Grid2D, points):
    return bilinear_sample_torch_points(psi, grid, points)


def _ray_limit_radii(*, grid: Grid2D, center, angles, limiter):
    torch = __import__("torch")
    dtype = angles.dtype
    device = angles.device
    dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    if limiter is not None:
        # Appending the first vertex unconditionally avoids a device-to-host
        # boolean synchronization.  If the polygon is already closed, the
        # extra zero-length edge is ignored by the intersection mask.
        poly = torch.cat((limiter, limiter[:1]), dim=0)
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


def _pad_point_batches_for_concat(batches):
    """Pad variable point capacity on dimension one before batch concatenation."""
    torch = __import__("torch")

    if not batches:
        raise ValueError("at least one point batch is required")
    target = max(int(batch.shape[1]) for batch in batches)
    padded = []
    for batch in batches:
        missing = target - int(batch.shape[1])
        if missing > 0:
            batch = torch.nn.functional.pad(
                batch, (0, 0, 0, missing), value=float("nan")
            )
        padded.append(batch)
    return torch.cat(padded, dim=0)


def _slice_exact_spline_field(
    field: ExactSplineGpuField,
    start: int,
    stop: int,
) -> ExactSplineGpuField:
    """Take a batch slice without rebuilding static spline geometry."""
    return ExactSplineGpuField(
        geometry=field.geometry,
        coefficients=field.coefficients[start:stop],
        limiter_segment_values=field.limiter_segment_values[start:stop],
    )


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
    exact_field: ExactSplineGpuField | None = None,
    critical_x_points: object | None = None,
    critical_x_levels: object | None = None,
    critical_x_valid: object | None = None,
    return_dense_boundary: bool = False,
):
    """Run the exact LCFS search in bounded-memory lane chunks.

    Topology-grid extraction has a large temporary cell dimension.  Splitting
    only the independent batch lanes bounds peak memory without changing any
    candidate ordering or per-lane mathematics.
    """
    torch = __import__("torch")

    if exact_field is None:
        raise RuntimeError("equilibrium_lcfs requires a combined exact spline field")
    if critical_x_points is None or critical_x_levels is None or critical_x_valid is None:
        raise RuntimeError("equilibrium_lcfs requires the critical-point set")
    batch_size = int(psi.shape[0])
    if batch_size <= _EQUILIBRIUM_GRAPH_LANE_CHUNK:
        return _equilibrium_lcfs_fixed_angle_search_impl(
            psi=psi,
            grid=grid,
            axis_points=axis_points,
            projection_center=projection_center,
            axis_level=axis_level,
            axis_kind=axis_kind,
            axis_valid=axis_valid,
            measurement_angles=measurement_angles,
            limiter=limiter,
            exact_field=exact_field,
            critical_x_points=critical_x_points,
            critical_x_levels=critical_x_levels,
            critical_x_valid=critical_x_valid,
            return_dense_boundary=return_dense_boundary,
        )

    chunks = []
    for start in range(0, batch_size, _EQUILIBRIUM_GRAPH_LANE_CHUNK):
        stop = min(start + _EQUILIBRIUM_GRAPH_LANE_CHUNK, batch_size)
        chunks.append(
            _equilibrium_lcfs_fixed_angle_search_impl(
                psi=psi[start:stop],
                grid=grid,
                axis_points=axis_points[start:stop],
                projection_center=projection_center[start:stop],
                axis_level=axis_level[start:stop],
                axis_kind=axis_kind[start:stop],
                axis_valid=axis_valid[start:stop],
                measurement_angles=measurement_angles,
                limiter=limiter,
                exact_field=_slice_exact_spline_field(exact_field, start, stop),
                critical_x_points=critical_x_points[start:stop],
                critical_x_levels=critical_x_levels[start:stop],
                critical_x_valid=critical_x_valid[start:stop],
                return_dense_boundary=return_dense_boundary,
            )
        )

    resets = [chunk[0] for chunk in chunks]
    points = torch.cat([reset[0] for reset in resets], dim=0)
    radii = torch.cat([reset[1] for reset in resets], dim=0)
    found = torch.cat([reset[2] for reset in resets], dim=0)
    level = torch.cat([reset[3] for reset in resets], dim=0)
    return (
        (points, radii, found, level),
        torch.cat([chunk[1] for chunk in chunks], dim=0),
        torch.cat([chunk[2] for chunk in chunks], dim=0),
        torch.cat([chunk[3] for chunk in chunks], dim=0),
        _pad_point_batches_for_concat([chunk[4] for chunk in chunks]),
        torch.cat([chunk[5] for chunk in chunks], dim=0),
        _pad_point_batches_for_concat([chunk[6] for chunk in chunks]),
        torch.cat([chunk[7] for chunk in chunks], dim=0),
        torch.cat([chunk[8] for chunk in chunks], dim=0),
    )


def _equilibrium_lcfs_fixed_angle_search_impl(
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
    exact_field: ExactSplineGpuField | None = None,
    critical_x_points: object | None = None,
    critical_x_levels: object | None = None,
    critical_x_valid: object | None = None,
    return_dense_boundary: bool = False,
):
    """Torch execution of the CPU ``find_equilibrium_lcfs`` algorithm.

    The CPU candidate ordering, tailored topology grid, exact edge roots and
    explicit X-point graph define the level set.  Half-edge face traversal then
    selects the bounded face containing the magnetic axis.  Fixed-angle rays
    are evaluated only after that physical LCFS has been selected.
    """
    torch = __import__("torch")
    if exact_field is None:
        raise RuntimeError("equilibrium_lcfs requires a combined exact spline field")
    if critical_x_points is None or critical_x_levels is None or critical_x_valid is None:
        raise RuntimeError("equilibrium_lcfs requires the topology critical-point set")

    B = int(psi.shape[0])
    dtype = psi.dtype
    device = psi.device
    orientation = -axis_kind.to(dtype=dtype)
    flux_span = torch.amax(psi, dim=(1, 2)) - torch.amin(psi, dim=(1, 2))
    flux_abs = torch.amax(torch.abs(psi), dim=(1, 2))
    flux_scale = torch.maximum(torch.maximum(flux_span, flux_abs), torch.full_like(flux_span, 1.0e-12))
    flux_floor = 1.0e-9 * flux_scale
    grid_scale = max(float(grid.r.step), float(grid.z.step))

    (
        wall_group_chi,
        wall_group_level,
        wall_group_valid,
        wall_raw_points,
        _wall_raw_segments,
        wall_raw_group,
        wall_raw_valid,
    ) = limiter_flux_candidates_exact_gpu(
        field=exact_field,
        axis_level=axis_level,
        orientation=orientation,
        flux_scale=flux_scale,
        flux_floor=flux_floor,
    )
    (
        x_group_chi,
        x_group_level,
        x_group_valid,
        x_points,
        x_levels,
        x_valid,
        x_group_id,
    ) = _group_xpoint_candidates_exact_gpu(
        x_points=critical_x_points,
        x_levels=critical_x_levels,
        x_valid=critical_x_valid,
        axis_level=axis_level,
        orientation=orientation,
        flux_scale=flux_scale,
        flux_floor=flux_floor,
        relative_tolerance=2.0e-4,
    )

    wall_capacity = int(wall_group_valid.shape[1])
    x_capacity = int(x_group_valid.shape[1])
    wall_index = torch.arange(wall_capacity, dtype=torch.int64, device=device)
    x_index = torch.arange(x_capacity, dtype=torch.int64, device=device)
    candidate_chi = torch.cat((wall_group_chi, x_group_chi), dim=1)
    candidate_level = torch.cat((wall_group_level, x_group_level), dim=1)
    candidate_valid = torch.cat((wall_group_valid, x_group_valid), dim=1)
    candidate_is_x = torch.cat(
        (
            torch.zeros((B, wall_capacity), dtype=torch.bool, device=device),
            torch.ones((B, x_capacity), dtype=torch.bool, device=device),
        ),
        dim=1,
    )
    candidate_group = torch.cat(
        (
            wall_index[None, :].expand(B, -1),
            x_index[None, :].expand(B, -1),
        ),
        dim=1,
    )
    candidate_order = torch.argsort(
        torch.where(candidate_valid, candidate_chi, torch.full_like(candidate_chi, float("inf"))),
        dim=1,
        stable=True,
    )
    candidate_level = torch.gather(candidate_level, 1, candidate_order)
    candidate_valid = torch.gather(candidate_valid, 1, candidate_order)
    candidate_is_x = torch.gather(candidate_is_x, 1, candidate_order)
    candidate_group = torch.gather(candidate_group, 1, candidate_order)
    # Wall groups are already compact and X groups have a fixed small capacity.
    # Keep this static candidate width so no device synchronization is needed
    # merely to determine a Python loop bound.  The loop exits after the first
    # block that resolves every active lane.
    maximum_candidate_count = int(candidate_valid.shape[1])

    x_count = int(x_points.shape[1])
    raw_wall_capacity = int(wall_raw_points.shape[1])
    selected = torch.zeros((B,), dtype=torch.bool, device=device)
    selected_level = torch.full((B,), float("nan"), dtype=dtype, device=device)
    selected_use_x = torch.zeros((B,), dtype=torch.bool, device=device)
    selected_x_mask = torch.zeros((B, x_count), dtype=torch.bool, device=device)
    selected_wall_contacts = torch.full(
        (B, raw_wall_capacity, 2),
        float("nan"),
        dtype=dtype,
        device=device,
    )
    selected_wall_contact_count = torch.zeros((B,), dtype=torch.int64, device=device)
    selected_raw_boundary = None
    selected_raw_count = None

    # Candidate levels are evaluated in one flattened tensor block whenever
    # memory permits.  Limiting the product ``B * block_size`` to sixteen lanes
    # keeps graph tensors bounded while avoiding repeated topology/root kernels
    # for single-lane replay, where the first physical candidate is normally
    # among the first several ordered levels.
    if device.type == "cuda":
        candidate_block_size = max(1, min(8, 16 // max(B, 1)))
    else:
        # CPU tensor execution is a correctness oracle only.  A smaller block
        # avoids allocating large temporary graph tensors during the test suite
        # without changing candidate order or selection semantics.
        candidate_block_size = max(1, 4 // max(B, 1))
    for block_start in range(0, maximum_candidate_count, candidate_block_size):
        block_stop = min(block_start + candidate_block_size, maximum_candidate_count)
        block_size = block_stop - block_start
        block_level = candidate_level[:, block_start:block_stop]
        block_valid = candidate_valid[:, block_start:block_stop]
        block_is_x = candidate_is_x[:, block_start:block_stop]
        block_group = candidate_group[:, block_start:block_stop]
        block_active = axis_valid[:, None] & ~selected[:, None] & block_valid

        block_x_mask = (
            x_valid[:, None, :]
            & (x_group_id[:, None, :] == block_group[:, :, None])
            & block_is_x[:, :, None]
        )
        block_x_count = torch.sum(block_x_mask.to(torch.int64), dim=2)
        compact_x_capacity = max(
            1, int(torch.max(block_x_count).item())
        )
        compact_order = torch.argsort(
            (~block_x_mask).to(torch.int64),
            dim=2,
            stable=True,
        )[:, :, :compact_x_capacity]
        expanded_x_points = x_points[:, None, :, :].expand(-1, block_size, -1, -1)
        expanded_x_levels = x_levels[:, None, :].expand(-1, block_size, -1)
        compact_x_points = torch.gather(
            expanded_x_points,
            2,
            compact_order[:, :, :, None].expand(-1, -1, -1, 2),
        )
        compact_x_levels = torch.gather(expanded_x_levels, 2, compact_order)
        compact_x_valid = torch.gather(block_x_mask, 2, compact_order)
        compact_x_points = torch.where(
            compact_x_valid[:, :, :, None],
            compact_x_points,
            torch.full_like(compact_x_points, float("nan")),
        )
        compact_x_levels = torch.where(
            compact_x_valid,
            compact_x_levels,
            torch.full_like(compact_x_levels, float("nan")),
        )

        flat_batch = B * block_size
        block_field = repeat_exact_spline_gpu_field(exact_field, block_size)
        flat_axis = axis_points[:, None, :].expand(-1, block_size, -1).reshape(flat_batch, 2)
        flat_level = torch.where(
            block_active,
            block_level,
            axis_level[:, None],
        ).reshape(flat_batch)
        flat_x_points = compact_x_points.reshape(flat_batch, compact_x_capacity, 2)
        flat_x_levels = compact_x_levels.reshape(flat_batch, compact_x_capacity)
        flat_x_valid = compact_x_valid.reshape(flat_batch, compact_x_capacity)
        segment_points, segment_nodes, segment_valid, graph_base_node_count = (
            build_level_set_segments_exact_gpu(
                field=block_field,
                grid=grid,
                level=flat_level,
                axis_points=flat_axis,
                limiter=limiter,
                x_points=flat_x_points,
                x_levels=flat_x_levels,
                x_valid=flat_x_valid,
                refinement=2,
            )
        )
        (
            segment_points,
            segment_nodes,
            segment_valid,
            graph_node_valid,
            graph_node_is_x,
            graph_node_x_slot,
        ) = _compact_level_set_segments_gpu(
            segment_points=segment_points,
            segment_nodes=segment_nodes,
            segment_valid=segment_valid,
            base_node_count=int(graph_base_node_count),
            x_count=compact_x_capacity,
        )
        raw_boundary, raw_count, cycle_found, compact_used_x = (
            _trace_level_set_faces_gpu(
                segment_points=segment_points,
                segment_nodes=segment_nodes,
                segment_valid=segment_valid,
                node_valid=graph_node_valid,
                node_is_x=graph_node_is_x,
                node_x_slot=graph_node_x_slot,
                axis_points=flat_axis,
                require_x=block_is_x.reshape(flat_batch),
                limiter=limiter,
                grid=grid,
                x_count=compact_x_capacity,
            )
        )
        cycle_segments, cycle_segment_valid = _padded_boundary_segments_gpu(
            boundaries=raw_boundary,
            boundary_count=raw_count,
        )

        starts = cycle_segments[:, :, 0, :]
        vectors = cycle_segments[:, :, 1, :] - starts
        denominator = torch.sum(vectors * vectors, dim=2)
        safe_denominator = torch.where(
            denominator > 0.0,
            denominator,
            torch.ones_like(denominator),
        )
        flat_compact_order = compact_order.reshape(
            flat_batch, compact_x_capacity
        )
        flat_candidate_x_mask = block_x_mask.reshape(flat_batch, x_count)
        flat_used_x_mask = torch.zeros_like(flat_candidate_x_mask)
        flat_used_x_mask.scatter_(1, flat_compact_order, compact_used_x)
        flat_used_x_mask &= flat_candidate_x_mask
        used_x_any = torch.any(flat_used_x_mask, dim=1)

        flat_block_group = block_group.reshape(flat_batch)
        flat_block_is_x = block_is_x.reshape(flat_batch)
        repeated_wall_points = wall_raw_points[:, None, :, :].expand(
            -1,
            block_size,
            -1,
            -1,
        ).reshape(flat_batch, raw_wall_capacity, 2)
        repeated_wall_group = wall_raw_group[:, None, :].expand(
            -1,
            block_size,
            -1,
        ).reshape(flat_batch, raw_wall_capacity)
        repeated_wall_valid = wall_raw_valid[:, None, :].expand(
            -1,
            block_size,
            -1,
        ).reshape(flat_batch, raw_wall_capacity)
        wall_member = (
            repeated_wall_valid
            & (repeated_wall_group == flat_block_group[:, None])
            & ~flat_block_is_x[:, None]
        )
        wall_relative = repeated_wall_points[:, :, None, :] - starts[:, None, :, :]
        wall_fraction = torch.clamp(
            torch.sum(wall_relative * vectors[:, None, :, :], dim=3)
            / safe_denominator[:, None, :],
            0.0,
            1.0,
        )
        wall_closest = starts[:, None, :, :] + wall_fraction[:, :, :, None] * vectors[:, None, :, :]
        wall_distance = torch.linalg.norm(
            repeated_wall_points[:, :, None, :] - wall_closest,
            dim=3,
        )
        wall_distance = torch.where(
            cycle_segment_valid[:, None, :] & torch.isfinite(wall_distance),
            wall_distance,
            torch.full_like(wall_distance, float("inf")),
        )
        wall_contact_mask = wall_member & (
            torch.amin(wall_distance, dim=2) <= 0.75 * grid_scale
        )
        wall_contact_ok = torch.any(wall_contact_mask, dim=1)

        flat_active = block_active.reshape(flat_batch)
        flat_candidate_valid = (
            flat_active
            & cycle_found
            & torch.where(flat_block_is_x, used_x_any, wall_contact_ok)
        )
        block_candidate_valid = flat_candidate_valid.reshape(B, block_size)
        has_valid = torch.any(block_candidate_valid, dim=1)
        first_valid = torch.argmax(block_candidate_valid.to(torch.int64), dim=1)
        batch_index = torch.arange(B, device=device)

        raw_boundary_block = raw_boundary.reshape(B, block_size, raw_boundary.shape[1], 2)
        raw_count_block = raw_count.reshape(B, block_size)
        used_x_mask_block = flat_used_x_mask.reshape(B, block_size, x_count)
        current_boundary_capacity = int(raw_boundary_block.shape[2])
        previous_boundary_capacity = (
            0 if selected_raw_boundary is None else int(selected_raw_boundary.shape[1])
        )
        target_boundary_capacity = max(
            current_boundary_capacity,
            previous_boundary_capacity,
        )
        if current_boundary_capacity < target_boundary_capacity:
            raw_boundary_block = torch.nn.functional.pad(
                raw_boundary_block,
                (0, 0, 0, target_boundary_capacity - current_boundary_capacity),
                value=float("nan"),
            )
        if selected_raw_boundary is None:
            selected_raw_boundary = torch.full(
                (B, target_boundary_capacity, 2),
                float("nan"),
                dtype=dtype,
                device=device,
            )
            selected_raw_count = torch.zeros((B,), dtype=torch.int64, device=device)
        elif previous_boundary_capacity < target_boundary_capacity:
            selected_raw_boundary = torch.nn.functional.pad(
                selected_raw_boundary,
                (0, 0, 0, target_boundary_capacity - previous_boundary_capacity),
                value=float("nan"),
            )
        candidate_boundary = raw_boundary_block[batch_index, first_valid]
        candidate_count_value = raw_count_block[batch_index, first_valid]
        candidate_level_value = block_level[batch_index, first_valid]
        candidate_is_x_value = block_is_x[batch_index, first_valid]
        candidate_used_x = used_x_mask_block[batch_index, first_valid]

        contact_rank = torch.cumsum(wall_contact_mask.to(torch.int64), dim=1) - 1
        chosen_contact = wall_contact_mask & (contact_rank < raw_wall_capacity)
        flat_contacts = torch.full_like(repeated_wall_points, float("nan"))
        contact_batch = torch.arange(flat_batch, device=device)[:, None].expand_as(contact_rank)
        flat_contacts[
            contact_batch[chosen_contact],
            contact_rank[chosen_contact],
            :,
        ] = repeated_wall_points[chosen_contact]
        contact_count = torch.sum(wall_contact_mask.to(torch.int64), dim=1)
        contacts_block = flat_contacts.reshape(B, block_size, raw_wall_capacity, 2)
        contact_count_block = contact_count.reshape(B, block_size)
        candidate_contacts = contacts_block[batch_index, first_valid]
        candidate_contact_count = contact_count_block[batch_index, first_valid]

        selected_level = torch.where(has_valid, candidate_level_value, selected_level)
        selected_use_x = torch.where(has_valid, candidate_is_x_value, selected_use_x)
        selected_x_mask = torch.where(has_valid[:, None], candidate_used_x, selected_x_mask)
        selected_raw_boundary = torch.where(
            has_valid[:, None, None],
            candidate_boundary,
            selected_raw_boundary,
        )
        selected_raw_count = torch.where(has_valid, candidate_count_value, selected_raw_count)
        selected_wall_contacts = torch.where(
            (has_valid & ~candidate_is_x_value)[:, None, None],
            candidate_contacts,
            selected_wall_contacts,
        )
        selected_wall_contact_count = torch.where(
            has_valid & ~candidate_is_x_value,
            candidate_contact_count,
            selected_wall_contact_count,
        )
        selected |= has_valid
        if bool(torch.all(selected | ~axis_valid).item()):
            break

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

    cycle_segments, cycle_segment_valid = _padded_boundary_segments_gpu(
        boundaries=selected_raw_boundary,
        boundary_count=selected_raw_count,
    )
    points, radii, _projection_found, measurement_counts, _measurement_hits = (
        _ray_intersections_from_segments_gpu(
            segment_points=cycle_segments,
            segment_valid=cycle_segment_valid,
            origins=projection_center,
            angles=measurement_angles,
        )
    )
    points = torch.where(found[:, None, None], points, torch.full_like(points, float("nan")))
    radii = torch.where(found[:, None], radii, torch.full_like(radii, float("nan")))
    measurement_counts = torch.where(
        found[:, None],
        measurement_counts,
        torch.zeros_like(measurement_counts),
    )

    # CPU branch contacts are derived from non-core graph edges.  Limited
    # contacts are carried exactly here; diverted branch endpoints remain a
    # diagnostic output and do not participate in LCFS selection or radii.
    limiter_contacts = torch.where(
        (found & ~selected_use_x)[:, None, None],
        selected_wall_contacts,
        torch.full_like(selected_wall_contacts, float("nan")),
    )
    limiter_contact_count = torch.where(
        found & ~selected_use_x,
        selected_wall_contact_count,
        torch.zeros_like(selected_wall_contact_count),
    )

    quality = _gpu_boundary_quality_exact(
        psi=psi,
        grid=grid,
        boundary=selected_raw_boundary,
        boundary_count=selected_raw_count,
        level=selected_level,
        axis_level=axis_level,
        axis_kind=axis_kind,
        limiter=limiter,
        x_points=selected_x,
        exact_field=exact_field,
    )

    core_boundary = torch.empty((B, 0, 2), dtype=dtype, device=device)
    core_boundary_count = torch.zeros((B,), dtype=torch.int64, device=device)
    if bool(return_dense_boundary):
        # The selected topology cycle is already the dense physical LCFS.  Keep
        # its ordered graph vertices instead of resampling it to a fixed angular
        # or arc-length representation.  This preserves X-point cusps, limiter
        # contacts and CPU graph geometry within the accepted edge-root tolerance.
        # Variable-length batching is represented by the
        # existing padded tensor plus ``core_boundary_count``.
        core_boundary = torch.where(
            found[:, None, None],
            selected_raw_boundary,
            torch.full_like(selected_raw_boundary, float("nan")),
        )
        core_boundary_count = torch.where(
            found,
            selected_raw_count,
            torch.zeros_like(selected_raw_count),
        )

    selected_level = torch.where(found, selected_level, torch.full_like(selected_level, float("nan")))
    topology_code = torch.where(found, topology_code, torch.zeros_like(topology_code))
    selected_x = torch.where(found[:, None, None], selected_x, torch.full_like(selected_x, float("nan")))
    limiter_contacts = torch.where(found[:, None, None], limiter_contacts, torch.full_like(limiter_contacts, float("nan")))
    limiter_contact_count = torch.where(found, limiter_contact_count, torch.zeros_like(limiter_contact_count))
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




def _ray_intersections_from_segments_gpu(
    *,
    segment_points,
    segment_valid,
    origins,
    angles,
    deduplicate_counts: bool = True,
):
    """Project all fixed-angle rays against all selected segments at once."""
    torch = __import__("torch")
    starts = segment_points[:, :, 0, :]
    vectors = segment_points[:, :, 1, :] - starts
    directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    rhs = starts - origins[:, None, :]

    denominator = (
        directions[None, :, None, 0] * vectors[:, None, :, 1]
        - directions[None, :, None, 1] * vectors[:, None, :, 0]
    )
    valid_denominator = torch.abs(denominator) > 1.0e-14
    safe_denominator = torch.where(
        valid_denominator,
        denominator,
        torch.ones_like(denominator),
    )
    ray_t = (
        rhs[:, None, :, 0] * vectors[:, None, :, 1]
        - rhs[:, None, :, 1] * vectors[:, None, :, 0]
    ) / safe_denominator
    segment_u = (
        rhs[:, None, :, 0] * directions[None, :, None, 1]
        - rhs[:, None, :, 1] * directions[None, :, None, 0]
    ) / safe_denominator
    hit = (
        segment_valid[:, None, :]
        & valid_denominator
        & torch.isfinite(ray_t)
        & (ray_t >= 0.0)
        & (segment_u >= -1.0e-10)
        & (segment_u <= 1.0 + 1.0e-10)
    )
    candidate_t = torch.where(hit, ray_t, torch.full_like(ray_t, float("inf")))
    radii, hit_segments = torch.min(candidate_t, dim=2)
    found_per_ray = torch.isfinite(radii)
    radii = torch.where(found_per_ray, radii, torch.full_like(radii, float("nan")))
    points = origins[:, None, :] + radii[:, :, None] * directions[None, :, :]
    points = torch.where(
        found_per_ray[:, :, None],
        points,
        torch.full_like(points, float("nan")),
    )
    hit_segments = torch.where(
        found_per_ray,
        hit_segments,
        torch.full_like(hit_segments, -1),
    )

    if deduplicate_counts:
        sorted_t = torch.sort(candidate_t, dim=2).values
        finite = torch.isfinite(sorted_t)
        first = finite[:, :, 0].to(torch.int64)
        separated = finite[:, :, 1:] & (
            ~finite[:, :, :-1]
            | (torch.abs(sorted_t[:, :, 1:] - sorted_t[:, :, :-1]) > 1.0e-9)
        )
        counts = first + torch.sum(separated.to(torch.int64), dim=2)
    else:
        counts = torch.sum(hit.to(torch.int64), dim=2)
    found = torch.all(found_per_ray, dim=1)
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









def _gpu_boundary_quality_exact(
    *,
    psi,
    grid: Grid2D,
    boundary,
    boundary_count,
    level,
    axis_level,
    axis_kind,
    limiter,
    x_points,
    exact_field: ExactSplineGpuField,
):
    """Compute CPU-equivalent quality metrics on the exact graph boundary."""
    torch = __import__("torch")
    B = int(psi.shape[0])
    if int(boundary.shape[1]) == 0:
        return torch.full((B, 6), float("nan"), dtype=psi.dtype, device=psi.device)

    positions = torch.arange(boundary.shape[1] - 1, device=psi.device)
    valid_points = positions[None, :] < (boundary_count - 1)[:, None]
    points = boundary[:, :-1, :]
    values, gradients, contains = evaluate_exact_spline_value_gradient(
        field=exact_field,
        points=points,
    )
    valid_points &= contains & torch.all(torch.isfinite(points), dim=2)
    residual = torch.abs(values - level[:, None])
    max_residual = torch.amax(
        torch.where(valid_points, residual, torch.zeros_like(residual)),
        dim=1,
    )
    flux_span = torch.abs(level - axis_level)
    r_span = abs(float(grid.r.coords()[-1] - grid.r.coords()[0]))
    z_span = abs(float(grid.z.coords()[-1] - grid.z.coords()[0]))
    grid_scale = max(float(grid.r.step), float(grid.z.step))
    domain_diagonal = max(float(np.hypot(r_span, z_span)), grid_scale)
    interpolation_floor = (grid_scale / domain_diagonal) ** 4
    normalized = torch.maximum(
        max_residual / torch.clamp(flux_span, min=1.0e-30),
        torch.full_like(max_residual, interpolation_floor),
    )

    last_index = torch.clamp(boundary_count.to(torch.int64) - 1, min=0, max=boundary.shape[1] - 1)
    batch_index = torch.arange(B, device=psi.device)
    closure = torch.linalg.norm(boundary[:, 0, :] - boundary[batch_index, last_index, :], dim=1)

    gradient_norm = torch.linalg.norm(gradients, dim=2)
    finite_x = torch.all(torch.isfinite(x_points), dim=2)
    if int(x_points.shape[1]) > 0:
        distance = torch.linalg.norm(
            points[:, :, None, :] - x_points[:, None, :, :],
            dim=3,
        )
        distance = torch.where(
            finite_x[:, None, :],
            distance,
            torch.full_like(distance, float("inf")),
        )
        regular = valid_points & (
            torch.amin(distance, dim=2) > 1.5 * max(float(grid.r.step), float(grid.z.step))
        )
    else:
        regular = valid_points
    minimum_gradient = torch.amin(
        torch.where(
            regular & torch.isfinite(gradient_norm),
            gradient_norm,
            torch.full_like(gradient_norm, float("inf")),
        ),
        dim=1,
    )
    minimum_gradient = torch.where(
        torch.isfinite(minimum_gradient),
        minimum_gradient,
        torch.zeros_like(minimum_gradient),
    )

    inside = _points_in_or_on_polygon_torch(
        points,
        limiter,
        tolerance=0.35 * max(float(grid.r.step), float(grid.z.step)),
    )
    limiter_violations = torch.sum((valid_points & ~inside).to(psi.dtype), dim=1)

    orientation = -axis_kind.to(dtype=psi.dtype)
    chi = orientation[:, None, None] * (psi - axis_level[:, None, None])
    chi_boundary = orientation * (level - axis_level)
    component_size = torch.sum((chi < chi_boundary[:, None, None]).to(psi.dtype), dim=(1, 2))

    valid_boundary = boundary_count > 1
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
    # Always append the first vertex.  A pre-existing closing vertex only
    # creates a harmless zero-length edge and avoids synchronizing CUDA merely
    # to inspect polygon closure.
    poly = torch.cat((polygon, polygon[:1]), dim=0)
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
    # Always append the first vertex.  A pre-existing closing vertex only
    # creates a harmless zero-length edge and avoids synchronizing CUDA merely
    # to inspect polygon closure.
    poly = torch.cat((polygon, polygon[:1]), dim=0)
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



def _compact_level_set_segments_gpu(
    *,
    segment_points,
    segment_nodes,
    segment_valid,
    base_node_count: int,
    x_count: int,
):
    """Compact sparse primitive segments and remap their node ids per lane.

    ``build_level_set_segments_exact_gpu`` uses deterministic cell-local slots,
    most of which are empty for one physical contour.  Carrying those padded
    slots into half-edge traversal multiplies memory by the full topology-grid
    area.  This operation keeps every valid segment, packs it without
    truncation, and maps the sparse original node ids to a dense lane-local
    range.  One scalar synchronization obtains the exact required capacity.
    """
    torch = __import__("torch")

    points = torch.as_tensor(segment_points)
    nodes = torch.as_tensor(segment_nodes, dtype=torch.int64, device=points.device)
    valid = torch.as_tensor(segment_valid, dtype=torch.bool, device=points.device)
    if points.ndim != 4 or points.shape[2:] != (2, 2):
        raise ValueError("segment_points must have shape (B,S,2,2)")
    if nodes.shape != points.shape[:2] + (2,) or valid.shape != points.shape[:2]:
        raise ValueError("segment graph tensor shapes are inconsistent")

    batch_size = int(valid.shape[0])
    segment_count = torch.sum(valid.to(torch.int64), dim=1)
    compact_capacity = max(1, int(torch.max(segment_count).item()))
    rank = torch.cumsum(valid.to(torch.int64), dim=1) - 1
    keep = valid & (rank >= 0)
    batch = torch.arange(batch_size, device=points.device)[:, None].expand_as(rank)

    compact_points = torch.full(
        (batch_size, compact_capacity, 2, 2),
        float("nan"),
        dtype=points.dtype,
        device=points.device,
    )
    compact_original_nodes = torch.full(
        (batch_size, compact_capacity, 2),
        -1,
        dtype=torch.int64,
        device=points.device,
    )
    compact_valid = (
        torch.arange(compact_capacity, device=points.device)[None, :]
        < segment_count[:, None]
    )
    compact_points[batch[keep], rank[keep], :, :] = points[keep]
    compact_original_nodes[batch[keep], rank[keep], :] = nodes[keep]

    endpoint_capacity = 2 * compact_capacity
    flat_nodes = compact_original_nodes.reshape(batch_size, endpoint_capacity)
    flat_valid = compact_valid[:, :, None].expand(-1, -1, 2).reshape(
        batch_size, endpoint_capacity
    ) & (flat_nodes >= 0)
    sentinel = torch.iinfo(torch.int64).max
    sortable = torch.where(flat_valid, flat_nodes, torch.full_like(flat_nodes, sentinel))
    sorted_nodes, order = torch.sort(sortable, dim=1, stable=True)
    sorted_valid = sorted_nodes != sentinel
    unique = sorted_valid.clone()
    unique[:, 1:] &= sorted_nodes[:, 1:] != sorted_nodes[:, :-1]
    dense_rank = torch.cumsum(unique.to(torch.int64), dim=1) - 1
    dense_for_sorted = torch.where(
        sorted_valid,
        dense_rank,
        torch.full_like(dense_rank, -1),
    )

    inverse = torch.full_like(flat_nodes, -1)
    inverse.scatter_(1, order, dense_for_sorted)
    compact_nodes = inverse.reshape(batch_size, compact_capacity, 2)

    dense_original_nodes = torch.full(
        (batch_size, endpoint_capacity),
        -1,
        dtype=torch.int64,
        device=points.device,
    )
    sorted_batch = torch.arange(batch_size, device=points.device)[:, None].expand_as(dense_rank)
    dense_original_nodes[
        sorted_batch[unique],
        dense_rank[unique],
    ] = sorted_nodes[unique]
    node_count = torch.sum(unique.to(torch.int64), dim=1)
    node_valid = (
        torch.arange(endpoint_capacity, device=points.device)[None, :]
        < node_count[:, None]
    )
    node_is_x = (
        node_valid
        & (dense_original_nodes >= int(base_node_count))
        & (dense_original_nodes < int(base_node_count) + int(x_count))
    )
    node_x_slot = torch.where(
        node_is_x,
        dense_original_nodes - int(base_node_count),
        torch.full_like(dense_original_nodes, -1),
    )
    return (
        compact_points,
        compact_nodes,
        compact_valid,
        node_valid,
        node_is_x,
        node_x_slot,
    )

def _trace_level_set_faces_gpu(
    *,
    segment_points,
    segment_nodes,
    segment_valid,
    node_valid,
    node_is_x,
    node_x_slot,
    axis_points,
    require_x,
    limiter,
    grid: Grid2D,
    x_count: int,
):
    """Select the bounded axis-containing face of a compact half-edge graph.

    Every primitive segment contributes two directed half-edges.  Degree-two
    regular nodes continue through their other incident segment.  At an
    explicit X-point, outgoing half-edges are ordered by their one-sided
    tangent and the clockwise predecessor of the incoming twin is selected.
    The resulting successor map traces the face on the left of each half-edge.
    Pointer doubling labels every closed orbit without DFS or subset search.
    """
    torch = __import__("torch")

    points = torch.as_tensor(segment_points)
    nodes = torch.as_tensor(segment_nodes, dtype=torch.int64, device=points.device)
    valid = torch.as_tensor(segment_valid, dtype=torch.bool, device=points.device)
    node_valid_t = torch.as_tensor(node_valid, dtype=torch.bool, device=points.device)
    node_is_x_t = torch.as_tensor(node_is_x, dtype=torch.bool, device=points.device)
    node_x_slot_t = torch.as_tensor(node_x_slot, dtype=torch.int64, device=points.device)
    require_x_t = torch.as_tensor(require_x, dtype=torch.bool, device=points.device).reshape(-1)
    B, segment_count = valid.shape
    node_capacity = int(node_valid_t.shape[1])
    device = points.device
    dtype = points.dtype
    state_count = 2 * int(segment_count)
    sentinel_state = state_count
    sentinel_segment = int(segment_count)

    state_ids = torch.arange(state_count, dtype=torch.int64, device=device)
    state_segment = state_ids // 2
    state_forward = (state_ids % 2) == 0
    state_valid = valid[:, state_segment]
    state_start_node = torch.where(
        state_forward[None, :],
        nodes[:, state_segment, 0],
        nodes[:, state_segment, 1],
    )
    state_end_node = torch.where(
        state_forward[None, :],
        nodes[:, state_segment, 1],
        nodes[:, state_segment, 0],
    )
    state_start = torch.where(
        state_forward[None, :, None],
        points[:, state_segment, 0, :],
        points[:, state_segment, 1, :],
    )
    state_end = torch.where(
        state_forward[None, :, None],
        points[:, state_segment, 1, :],
        points[:, state_segment, 0, :],
    )

    segment_ids = torch.arange(
        segment_count, dtype=torch.int64, device=device
    )[None, :].expand(B, -1)
    degree = torch.zeros((B, node_capacity), dtype=torch.int32, device=device)
    adjacency_min = torch.full(
        (B, node_capacity), sentinel_segment, dtype=torch.int64, device=device
    )
    adjacency_max = torch.full(
        (B, node_capacity), -1, dtype=torch.int64, device=device
    )
    for endpoint in range(2):
        endpoint_node = nodes[:, :, endpoint]
        safe_node = torch.clamp(endpoint_node, min=0, max=max(node_capacity - 1, 0))
        endpoint_valid = valid & (endpoint_node >= 0)
        degree.scatter_add_(1, safe_node, endpoint_valid.to(torch.int32))
        adjacency_min.scatter_reduce_(
            1,
            safe_node,
            torch.where(
                endpoint_valid,
                segment_ids,
                torch.full_like(segment_ids, sentinel_segment),
            ),
            reduce="amin",
            include_self=True,
        )
        adjacency_max.scatter_reduce_(
            1,
            safe_node,
            torch.where(
                endpoint_valid,
                segment_ids,
                torch.full_like(segment_ids, -1),
            ),
            reduce="amax",
            include_self=True,
        )

    safe_start_node = torch.clamp(
        state_start_node, min=0, max=max(node_capacity - 1, 0)
    )
    safe_end_node = torch.clamp(
        state_end_node, min=0, max=max(node_capacity - 1, 0)
    )
    end_node_valid = (state_end_node >= 0) & torch.gather(
        node_valid_t, 1, safe_end_node
    )
    end_is_x = end_node_valid & torch.gather(node_is_x_t, 1, safe_end_node)
    state_segment_batch = state_segment[None, :].expand(B, -1)
    first_adjacent = torch.gather(adjacency_min, 1, safe_end_node)
    last_adjacent = torch.gather(adjacency_max, 1, safe_end_node)
    next_segment = torch.where(
        first_adjacent != state_segment_batch, first_adjacent, last_adjacent
    )
    regular_end = (
        state_valid
        & end_node_valid
        & ~end_is_x
        & (torch.gather(degree, 1, safe_end_node) == 2)
        & (next_segment >= 0)
        & (next_segment < segment_count)
        & (next_segment != state_segment_batch)
    )
    safe_next_segment = torch.clamp(
        next_segment, min=0, max=max(segment_count - 1, 0)
    )
    next_starts_here = (
        torch.gather(nodes[:, :, 0], 1, safe_next_segment) == state_end_node
    )
    regular_successor = 2 * safe_next_segment + torch.where(
        next_starts_here,
        torch.zeros_like(safe_next_segment),
        torch.ones_like(safe_next_segment),
    )

    x_successor = torch.full(
        (B, state_count), sentinel_state, dtype=torch.int64, device=device
    )
    if int(x_count) > 0:
        source_node_valid = (state_start_node >= 0) & torch.gather(
            node_valid_t, 1, safe_start_node
        )
        source_is_x = source_node_valid & torch.gather(
            node_is_x_t, 1, safe_start_node
        )
        available = state_valid & source_is_x
        outgoing = torch.full(
            (B, node_capacity, 4),
            sentinel_state,
            dtype=torch.int64,
            device=device,
        )
        state_grid = state_ids[None, :].expand(B, -1)
        for slot in range(4):
            minimum = torch.full(
                (B, node_capacity),
                sentinel_state,
                dtype=torch.int64,
                device=device,
            )
            minimum.scatter_reduce_(
                1,
                safe_start_node,
                torch.where(
                    available,
                    state_grid,
                    torch.full_like(state_grid, sentinel_state),
                ),
                reduce="amin",
                include_self=True,
            )
            outgoing[:, :, slot] = minimum
            chosen_for_state = torch.gather(minimum, 1, safe_start_node)
            available &= state_grid != chosen_for_state

        outgoing_valid = outgoing < state_count
        safe_outgoing = torch.clamp(
            outgoing, min=0, max=max(state_count - 1, 0)
        )
        batch_node = torch.arange(B, device=device)[:, None, None]
        outgoing_start = state_start[batch_node, safe_outgoing]
        outgoing_end = state_end[batch_node, safe_outgoing]
        tangent = outgoing_end - outgoing_start
        angle = torch.atan2(tangent[:, :, :, 1], tangent[:, :, :, 0])
        angle = torch.where(
            outgoing_valid, angle, torch.full_like(angle, float("inf"))
        )
        order = torch.argsort(angle, dim=2, stable=True)
        outgoing_sorted = torch.gather(outgoing, 2, order)

        sorted_at_end = outgoing_sorted[
            torch.arange(B, device=device)[:, None],
            safe_end_node,
            :,
        ]
        twin = (state_ids ^ 1)[None, :, None]
        twin_match = sorted_at_end == twin
        twin_found = torch.any(twin_match, dim=2)
        twin_rank = torch.argmax(twin_match.to(torch.int64), dim=2)
        predecessor_rank = torch.remainder(twin_rank - 1, 4)
        selected_outgoing = torch.gather(
            sorted_at_end, 2, predecessor_rank[:, :, None]
        )[:, :, 0]
        x_transition = (
            state_valid
            & end_is_x
            & twin_found
            & (torch.gather(degree, 1, safe_end_node) == 4)
            & (selected_outgoing < state_count)
        )
        x_successor = torch.where(
            x_transition, selected_outgoing, x_successor
        )

    successor_core = torch.where(regular_end, regular_successor, x_successor)
    successor_core = torch.where(
        state_valid,
        successor_core,
        torch.full_like(successor_core, sentinel_state),
    )
    successor = torch.cat(
        (
            successor_core,
            torch.full(
                (B, 1), sentinel_state, dtype=torch.int64, device=device
            ),
        ),
        dim=1,
    )

    labels = torch.arange(
        state_count + 1, dtype=torch.int64, device=device
    )[None, :].expand(B, -1).clone()
    jump = successor
    jump_tables = [successor]
    span = 1
    while span <= max(state_count, 1):
        labels = torch.minimum(labels, torch.gather(labels, 1, jump))
        jump = torch.gather(jump, 1, jump)
        jump_tables.append(jump)
        span <<= 1
    representative = labels[:, :state_count]
    closed_state = (
        state_valid
        & end_node_valid
        & (jump[:, :state_count] != sentinel_state)
    )

    cross = 0.5 * (
        state_start[:, :, 0] * state_end[:, :, 1]
        - state_end[:, :, 0] * state_start[:, :, 1]
    )
    axis_x = axis_points[:, 0][:, None]
    axis_y = axis_points[:, 1][:, None]
    y0 = state_start[:, :, 1]
    y1 = state_end[:, :, 1]
    denominator = y1 - y0
    safe_denominator = torch.where(
        torch.abs(denominator) > torch.finfo(dtype).eps,
        denominator,
        torch.ones_like(denominator),
    )
    x_cross = state_start[:, :, 0] + (axis_y - y0) * (
        state_end[:, :, 0] - state_start[:, :, 0]
    ) / safe_denominator
    crossing = ((y0 > axis_y) != (y1 > axis_y)) & (x_cross > axis_x)
    limiter_inside = _points_in_or_on_polygon_torch(
        state_start,
        limiter,
        tolerance=0.35 * max(float(grid.r.step), float(grid.z.step)),
    )
    state_start_x_slot = torch.gather(
        node_x_slot_t, 1, safe_start_node
    )
    state_uses_x = state_valid & (state_start_x_slot >= 0)

    safe_rep = torch.clamp(
        representative, min=0, max=max(state_count - 1, 0)
    )
    area_by_rep = torch.zeros((B, state_count), dtype=dtype, device=device)
    parity_by_rep = torch.zeros(
        (B, state_count), dtype=torch.int32, device=device
    )
    invalid_by_rep = torch.zeros(
        (B, state_count), dtype=torch.int32, device=device
    )
    count_by_rep = torch.zeros(
        (B, state_count), dtype=torch.int32, device=device
    )
    x_visits_by_rep = torch.zeros(
        (B, state_count), dtype=torch.int32, device=device
    )
    area_by_rep.scatter_add_(
        1, safe_rep, torch.where(closed_state, cross, torch.zeros_like(cross))
    )
    parity_by_rep.scatter_add_(
        1, safe_rep, (closed_state & crossing).to(torch.int32)
    )
    invalid_by_rep.scatter_add_(
        1, safe_rep, (closed_state & ~limiter_inside).to(torch.int32)
    )
    count_by_rep.scatter_add_(
        1, safe_rep, closed_state.to(torch.int32)
    )
    x_visits_by_rep.scatter_add_(
        1, safe_rep, (closed_state & state_uses_x).to(torch.int32)
    )

    candidate = (
        (count_by_rep >= 3)
        & ((parity_by_rep % 2) == 1)
        & (invalid_by_rep == 0)
        & torch.isfinite(area_by_rep)
        & (area_by_rep > 0.0)
        & (~require_x_t[:, None] | (x_visits_by_rep > 0))
    )
    score = torch.where(
        candidate, area_by_rep, torch.full_like(area_by_rep, -float("inf"))
    )
    best_score, start_state = torch.max(score, dim=1)
    found = torch.isfinite(best_score) & (best_score > 0.0)
    start_state = torch.where(
        found, start_state, torch.full_like(start_state, sentinel_state)
    )
    safe_start_state = torch.clamp(
        start_state, min=0, max=max(state_count - 1, 0)
    )
    selected_cycle_segments = count_by_rep[
        torch.arange(B, device=device), safe_start_state
    ].to(torch.int64)
    selected_cycle_segments = torch.where(
        found, selected_cycle_segments, torch.zeros_like(selected_cycle_segments)
    )
    # Materialize only the selected cycle.  This scalar synchronization avoids
    # allocating and pointer-jumping across the entire compact graph when the
    # physical face is much smaller.
    maximum_cycle_segments = max(
        3, int(torch.max(selected_cycle_segments).item())
    )
    positions = torch.arange(
        maximum_cycle_segments + 1, dtype=torch.int64, device=device
    )
    states = start_state[:, None].expand(
        B, maximum_cycle_segments + 1
    ).clone()
    for bit, table in enumerate(jump_tables):
        moved = torch.gather(table, 1, states)
        states = torch.where(
            (((positions >> bit) & 1) != 0)[None, :], moved, states
        )
    batch_index = torch.arange(B, device=device)
    safe_cycle_length = torch.clamp(
        selected_cycle_segments, min=0, max=maximum_cycle_segments
    )
    closes_exactly = (
        states[batch_index, safe_cycle_length] == start_state
    )
    found &= closes_exactly & (selected_cycle_segments >= 3)
    point_count = torch.where(
        found,
        selected_cycle_segments + 1,
        torch.zeros_like(selected_cycle_segments),
    )

    safe_states = torch.clamp(
        states, min=0, max=max(state_count - 1, 0)
    )
    cycle_segment = safe_states // 2
    cycle_forward = (safe_states % 2) == 0
    batch = torch.arange(B, device=device)[:, None]
    cycle_points = torch.where(
        cycle_forward[:, :, None],
        points[batch, cycle_segment, 0, :],
        points[batch, cycle_segment, 1, :],
    )
    point_valid = positions[None, :] < point_count[:, None]
    cycle_points = torch.where(
        found[:, None, None] & point_valid[:, :, None],
        cycle_points,
        torch.full_like(cycle_points, float("nan")),
    )

    cycle_start_node = torch.gather(
        state_start_node, 1, safe_states
    )
    safe_cycle_node = torch.clamp(
        cycle_start_node, min=0, max=max(node_capacity - 1, 0)
    )
    cycle_x_slot = torch.gather(
        node_x_slot_t, 1, safe_cycle_node
    )
    cycle_edge_position = (
        positions[None, :] < selected_cycle_segments[:, None]
    )
    x_visit_valid = (
        found[:, None]
        & cycle_edge_position
        & (cycle_x_slot >= 0)
        & (cycle_x_slot < int(x_count))
    )
    used_x_counts = torch.zeros(
        (B, int(x_count)), dtype=torch.int32, device=device
    )
    if int(x_count) > 0:
        used_x_counts.scatter_add_(
            1,
            torch.clamp(cycle_x_slot, min=0, max=int(x_count) - 1),
            x_visit_valid.to(torch.int32),
        )
    used_x_mask = used_x_counts > 0
    return cycle_points, point_count, found, used_x_mask


def _group_xpoint_candidates_exact_gpu(
    *,
    x_points,
    x_levels,
    x_valid,
    axis_level,
    orientation,
    flux_scale,
    flux_floor,
    relative_tolerance: float = 2.0e-4,
):
    """Tensor port of ``lcfs._group_xpoint_candidates``."""
    torch = __import__("torch")
    points = torch.as_tensor(x_points)
    levels = torch.as_tensor(x_levels, dtype=points.dtype, device=points.device)
    valid = torch.as_tensor(x_valid, dtype=torch.bool, device=points.device)
    B, K = valid.shape
    chi = torch.as_tensor(orientation, dtype=points.dtype, device=points.device)[:, None] * (
        levels - torch.as_tensor(axis_level, dtype=points.dtype, device=points.device)[:, None]
    )
    valid &= torch.isfinite(chi) & (
        chi > torch.as_tensor(flux_floor, dtype=points.dtype, device=points.device)[:, None]
    )
    chi = torch.where(valid, chi, torch.full_like(chi, float("inf")))
    order = torch.argsort(chi, dim=1, stable=True)
    sorted_chi = torch.gather(chi, 1, order)
    sorted_levels = torch.gather(levels, 1, order)
    sorted_points = torch.gather(points, 1, order[:, :, None].expand(-1, -1, 2))
    sorted_valid = torch.gather(valid, 1, order)

    # Critical-point outputs use a fixed padded capacity.  Compact that padding
    # once before grouping so empty X slots do not become boundary candidates or
    # participate in subsequent graph tensors.
    compact_x_capacity = max(
        1,
        int(torch.max(torch.sum(sorted_valid.to(torch.int64), dim=1)).item()),
    )
    sorted_chi = sorted_chi[:, :compact_x_capacity]
    sorted_levels = sorted_levels[:, :compact_x_capacity]
    sorted_points = sorted_points[:, :compact_x_capacity]
    sorted_valid = sorted_valid[:, :compact_x_capacity]
    K = compact_x_capacity

    group_id = torch.full((B, K), -1, dtype=torch.int64, device=points.device)
    current_group = torch.full((B,), -1, dtype=torch.int64, device=points.device)
    current_sum = torch.zeros((B,), dtype=points.dtype, device=points.device)
    current_count = torch.zeros((B,), dtype=torch.int64, device=points.device)
    for index in range(K):
        value = sorted_chi[:, index]
        valid_now = sorted_valid[:, index]
        current_mean = current_sum / torch.clamp(current_count.to(points.dtype), min=1.0)
        tolerance = torch.maximum(
            1.0e-10 * torch.as_tensor(flux_scale, dtype=points.dtype, device=points.device),
            float(relative_tolerance) * torch.clamp(value, min=1.0e-12),
        )
        new_group = valid_now & ((current_group < 0) | (torch.abs(value - current_mean) > tolerance))
        current_group = torch.where(new_group, current_group + 1, current_group)
        current_sum = torch.where(new_group, value, torch.where(valid_now, current_sum + value, current_sum))
        current_count = torch.where(
            new_group,
            torch.ones_like(current_count),
            torch.where(valid_now, current_count + 1, current_count),
        )
        group_id[:, index] = torch.where(valid_now, current_group, torch.full_like(current_group, -1))

    safe_group = torch.clamp(group_id, min=0)
    group_count = torch.zeros((B, K), dtype=torch.int64, device=points.device)
    group_chi_sum = torch.zeros((B, K), dtype=points.dtype, device=points.device)
    group_level_sum = torch.zeros((B, K), dtype=points.dtype, device=points.device)
    group_count.scatter_add_(1, safe_group, sorted_valid.to(torch.int64))
    group_chi_sum.scatter_add_(1, safe_group, torch.where(sorted_valid, sorted_chi, torch.zeros_like(sorted_chi)))
    group_level_sum.scatter_add_(1, safe_group, torch.where(sorted_valid, sorted_levels, torch.zeros_like(sorted_levels)))
    group_valid = group_count > 0
    group_chi = torch.where(
        group_valid,
        group_chi_sum / torch.clamp(group_count.to(points.dtype), min=1.0),
        torch.full_like(group_chi_sum, float("inf")),
    )
    group_level = torch.where(
        group_valid,
        group_level_sum / torch.clamp(group_count.to(points.dtype), min=1.0),
        torch.full_like(group_level_sum, float("nan")),
    )
    return (
        group_chi,
        group_level,
        group_valid,
        sorted_points,
        sorted_levels,
        sorted_valid,
        group_id,
    )
