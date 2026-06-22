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
    found: object
    status_code: object
    level: object
    points: object
    radii: object
    axis_points: object


def _torch(device: str):
    require_gpu_available(device)
    import torch
    return torch


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
    level_smoothing_alpha: float = 1.0,
    level_search_span_fraction: float = 0.02,
    continuity_weight_radii: float = 1.0,
    continuity_weight_mean_radius: float = 0.3,
    continuity_weight_level: float = 0.1,
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
    if mode not in {"legacy_contour", "legacy_contour_limited", "tracked_flux_contour"}:
        raise ValueError(f"unsupported boundary_mode for fixed_angle_boundary_gpu: {boundary_mode!r}")

    angles = torch.as_tensor(angles_rad, dtype=dtype, device=device).reshape(-1)
    center_t = torch.tensor(center, dtype=dtype, device=device).reshape(1, 2).repeat(B, 1)
    limiter_t = None
    if limiter_shape is not None:
        limiter_t = torch.as_tensor(np.asarray(limiter_shape, dtype=float), dtype=dtype, device=device).reshape(-1, 2)
    use_limiter = mode in {"legacy_contour_limited", "tracked_flux_contour"}
    if use_limiter and (limiter_t is None or int(limiter_t.shape[0]) < 3):
        raise BoundaryNotFoundError(f"{mode} requires limiter geometry")

    max_radii = _ray_limit_radii(grid=grid, center=center_t[0], angles=angles, limiter=limiter_t if use_limiter else None)
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

    reset = _legacy_fixed_angle_search(
        psi=field,
        grid=grid,
        center=center,
        center_points=center_t,
        center_level=center_level,
        angles=angles,
        max_radii=max_radii,
        ray_samples=int(ray_samples),
    )

    if tracked is None:
        points, radii, found, level = reset
        status_code = torch.where(found, torch.full((B,), 4 if mode == "tracked_flux_contour" else 2, dtype=torch.int64, device=device), torch.zeros((B,), dtype=torch.int64, device=device))
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


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _center_ray_crossings(*, psi, grid: Grid2D, center_points, center_level, level, angles, max_radii, ray_samples: int):
    torch = __import__("torch")
    B = int(psi.shape[0])
    A = int(angles.numel())
    dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    max_r = max_radii.reshape(1, A).repeat(B, 1)
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


def _legacy_fixed_angle_search(*, psi, grid: Grid2D, center, center_points, center_level, angles, max_radii, ray_samples: int):
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
    precision = torch.as_tensor(1.0e-3, dtype=psi.dtype, device=psi.device)
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
