from __future__ import annotations

import torch

from tokamak_control.core.grid import Grid2D


def bilinear_sample_torch(field: torch.Tensor, grid: Grid2D, points: torch.Tensor) -> torch.Tensor:
    """Bilinear sample ``field[B, Z, R]`` at fixed ``points[M, 2]``."""
    if field.ndim != 3:
        raise ValueError(f"field must have shape (B, Z, R), got {tuple(field.shape)}")
    pts = points.to(device=field.device, dtype=field.dtype).reshape(-1, 2)
    u = (pts[:, 0] - float(grid.r.start)) / float(grid.r.step)
    v = (pts[:, 1] - float(grid.z.start)) / float(grid.z.step)
    i0 = torch.floor(u).long()
    j0 = torch.floor(v).long()
    i1 = i0 + 1
    j1 = j0 + 1
    valid = (i0 >= 0) & (j0 >= 0) & (i1 < int(grid.r.size)) & (j1 < int(grid.z.size))
    i0c = torch.clamp(i0, 0, int(grid.r.size) - 1)
    i1c = torch.clamp(i1, 0, int(grid.r.size) - 1)
    j0c = torch.clamp(j0, 0, int(grid.z.size) - 1)
    j1c = torch.clamp(j1, 0, int(grid.z.size) - 1)
    du = (u - i0.to(field.dtype)).to(field.dtype)
    dv = (v - j0.to(field.dtype)).to(field.dtype)
    q00 = field[:, j0c, i0c]
    q10 = field[:, j0c, i1c]
    q01 = field[:, j1c, i0c]
    q11 = field[:, j1c, i1c]
    out = (1.0 - dv)[None, :] * ((1.0 - du)[None, :] * q00 + du[None, :] * q10) + dv[None, :] * ((1.0 - du)[None, :] * q01 + du[None, :] * q11)
    return torch.where(valid[None, :], out, torch.full_like(out, float("nan")))


def bilinear_sample_torch_points(field: torch.Tensor, grid: Grid2D, points: torch.Tensor) -> torch.Tensor:
    """Bilinear sample ``field[B, Z, R]`` at per-batch ``points[B, M, 2]``."""
    if field.ndim != 3:
        raise ValueError(f"field must have shape (B, Z, R), got {tuple(field.shape)}")
    pts = points.to(device=field.device, dtype=field.dtype)
    B, M, _ = pts.shape
    if B != field.shape[0]:
        raise ValueError("points batch size must match field batch size")
    u = (pts[..., 0] - float(grid.r.start)) / float(grid.r.step)
    v = (pts[..., 1] - float(grid.z.start)) / float(grid.z.step)
    i0 = torch.floor(u).long()
    j0 = torch.floor(v).long()
    i1 = i0 + 1
    j1 = j0 + 1
    valid = (i0 >= 0) & (j0 >= 0) & (i1 < int(grid.r.size)) & (j1 < int(grid.z.size))
    i0c = torch.clamp(i0, 0, int(grid.r.size) - 1)
    i1c = torch.clamp(i1, 0, int(grid.r.size) - 1)
    j0c = torch.clamp(j0, 0, int(grid.z.size) - 1)
    j1c = torch.clamp(j1, 0, int(grid.z.size) - 1)
    b = torch.arange(B, device=field.device)[:, None]
    du = (u - i0.to(field.dtype)).to(field.dtype)
    dv = (v - j0.to(field.dtype)).to(field.dtype)
    q00 = field[b, j0c, i0c]
    q10 = field[b, j0c, i1c]
    q01 = field[b, j1c, i0c]
    q11 = field[b, j1c, i1c]
    out = (1.0 - dv) * ((1.0 - du) * q00 + du * q10) + dv * ((1.0 - du) * q01 + du * q11)
    return torch.where(valid, out, torch.full_like(out, float("nan")))
