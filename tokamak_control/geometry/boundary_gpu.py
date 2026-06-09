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
    boundary_mode: BoundaryMode = "limited",
    gpu_device: str = "cuda:0",
) -> tuple[np.ndarray, float, BoundaryStatus]:
    del n_levels, prev_level, prev_poly, local_n_levels, local_span_frac, target_mean_radius, target_switch_ratio, target_switch_abs_delta, local_bbox_pad_r, local_bbox_pad_z
    if limiter_shape is None:
        raise BoundaryNotFoundError("GPU boundary requires limiter geometry")
    torch = _torch(gpu_device)
    psi_t = torch.as_tensor(psi, dtype=torch.float64, device=torch.device(gpu_device))
    if psi_t.ndim == 2:
        psi_t = psi_t.unsqueeze(0)
    angles = torch.linspace(-torch.pi, torch.pi, 128, device=psi_t.device, dtype=torch.float64)[:-1]
    result = fixed_angle_boundary_gpu(
        psi=psi_t,
        grid=grid,
        center=center,
        angles_rad=angles,
        limiter_shape=np.asarray(limiter_shape, dtype=float),
        boundary_mode=boundary_mode,
        gpu_device=gpu_device,
    )
    if not bool(result.found[0].detach().cpu().item()):
        raise BoundaryNotFoundError("No GPU plasma boundary found")
    poly = result.points[0].detach().cpu().numpy().astype(float)
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    level = float(result.level[0].detach().cpu().item())
    status: BoundaryStatus = "limited_success" if int(result.status_code[0].detach().cpu().item()) == 1 else "separatrix_success"
    return poly, level, status


def fixed_angle_boundary_gpu(
    *,
    psi,
    grid: Grid2D,
    center: tuple[float, float],
    angles_rad,
    limiter_shape: np.ndarray,
    boundary_mode: BoundaryMode = "limited",
    gpu_device: str = "cuda:0",
    ray_samples: int = 256,
) -> FixedAngleBoundaryGpuResult:
    """Return fixed-angle physical boundary samples on CUDA.

    Limited mode uses the flux level sampled on the limiter in outward physical
    order. The boundary point for each configured angle is the first ray
    crossing of that limiter-contact level from the magnetic axis outward.
    """
    torch = _torch(gpu_device)
    dev = torch.device(gpu_device)
    psi_t = torch.as_tensor(psi, dtype=torch.float64, device=dev)
    if psi_t.ndim == 2:
        psi_t = psi_t.unsqueeze(0)
    B = int(psi_t.shape[0])
    angles = torch.as_tensor(angles_rad, dtype=torch.float64, device=dev).reshape(-1)
    limiter = torch.as_tensor(np.asarray(limiter_shape, dtype=float).reshape(-1, 2), dtype=torch.float64, device=dev)
    axis_points, axis_levels, axis_kind = _axis_search(psi_t, grid, center, limiter)
    if str(boundary_mode) == "limited":
        limiter_psi = _sample_limiter_psi(psi_t, grid, limiter)
        high = torch.nanmax(limiter_psi, dim=1).values
        low = torch.nanmin(limiter_psi, dim=1).values
        level = torch.where(axis_kind > 0, high, low)
        status_code = torch.ones((B,), dtype=torch.int64, device=dev)
    elif str(boundary_mode) == "diverted":
        level, has_x = _xpoint_level(psi_t, grid, axis_points)
        status_code = torch.full((B,), 2, dtype=torch.int64, device=dev)
        level = torch.where(has_x, level, torch.full_like(level, float("nan")))
    else:
        raise ValueError(f"boundary_mode must be 'limited' or 'diverted', got {boundary_mode!r}")
    points, radii, found_rays = _ray_crossings(psi_t, grid, axis_points, level, axis_kind, angles, limiter, ray_samples=ray_samples)
    found = found_rays & torch.isfinite(level) & torch.isfinite(axis_levels)
    return FixedAngleBoundaryGpuResult(found=found, status_code=status_code, level=level, points=points, radii=radii, axis_points=axis_points)


def _axis_search(psi, grid: Grid2D, center: tuple[float, float], limiter):
    torch = __import__("torch")
    B, nz, nr = psi.shape
    r = torch.linspace(float(grid.r.start), float(grid.r.start) + float(grid.r.step) * (int(grid.r.size) - 1), int(grid.r.size), dtype=psi.dtype, device=psi.device)
    z = torch.linspace(float(grid.z.start), float(grid.z.start) + float(grid.z.step) * (int(grid.z.size) - 1), int(grid.z.size), dtype=psi.dtype, device=psi.device)
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
