from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokamak_control.core.grid import Grid2D
from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections


@dataclass(frozen=True, slots=True)
class MagneticDiagnosticLayout:
    """Fixed virtual magnetic diagnostic geometry."""

    flux_points: np.ndarray
    field_points: np.ndarray
    field_angles: np.ndarray

    @property
    def flux_count(self) -> int:
        return int(np.asarray(self.flux_points).reshape(-1, 2).shape[0])

    @property
    def field_count(self) -> int:
        return int(np.asarray(self.field_points).reshape(-1, 2).shape[0])


def default_t15_diagnostic_layout(
    *,
    grid: Grid2D,
    limiter_shape: np.ndarray | None,
    center: tuple[float, float],
    flux_count: int = 38,
    field_count: int = 38,
) -> MagneticDiagnosticLayout:
    """Create a deterministic virtual T15-like magnetic diagnostic layout.

    The actor observes these diagnostics instead of reconstructed boundary/radii.
    Points are fixed machine diagnostics, not target-dependent samples.
    """
    if limiter_shape is not None and np.asarray(limiter_shape).size:
        limiter = np.asarray(limiter_shape, dtype=float).reshape(-1, 2)
        angles = np.linspace(-np.pi, np.pi, max(flux_count, field_count), endpoint=False)
        radii = radii_from_polyline_ray_intersections(limiter, center, angles)
        c = np.asarray(center, dtype=float).reshape(2)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        # Put virtual probes inside the limiter so bilinear sampling is stable.
        flux_points = c[None, :] + 0.92 * radii[:flux_count, None] * dirs[:flux_count]
        field_points = c[None, :] + 0.86 * radii[:field_count, None] * dirs[:field_count]
        field_angles = angles[:field_count]
        return MagneticDiagnosticLayout(flux_points=flux_points, field_points=field_points, field_angles=field_angles)

    r = grid.r.coords()
    z = grid.z.coords()
    rr = np.linspace(float(r[1]), float(r[-2]), flux_count)
    zz = np.linspace(float(z[1]), float(z[-2]), field_count)
    flux_points = np.column_stack([rr, np.full_like(rr, float(center[1]))])
    field_points = np.column_stack([np.full_like(zz, float(center[0])), zz])
    field_angles = np.full((field_count,), np.pi / 2.0, dtype=float)
    return MagneticDiagnosticLayout(flux_points=flux_points, field_points=field_points, field_angles=field_angles)


def bilinear_sample_numpy(field: np.ndarray, grid: Grid2D, points: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=float)
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    u = (pts[:, 0] - float(grid.r.start)) / float(grid.r.step)
    v = (pts[:, 1] - float(grid.z.start)) / float(grid.z.step)
    i0 = np.floor(u).astype(int)
    j0 = np.floor(v).astype(int)
    i1 = i0 + 1
    j1 = j0 + 1
    valid = (i0 >= 0) & (j0 >= 0) & (i1 < int(grid.r.size)) & (j1 < int(grid.z.size))
    out = np.full((pts.shape[0],), np.nan, dtype=float)
    if not bool(np.any(valid)):
        return out
    du = u[valid] - i0[valid]
    dv = v[valid] - j0[valid]
    q00 = arr[j0[valid], i0[valid]]
    q10 = arr[j0[valid], i1[valid]]
    q01 = arr[j1[valid], i0[valid]]
    q11 = arr[j1[valid], i1[valid]]
    out[valid] = (1.0 - dv) * ((1.0 - du) * q00 + du * q10) + dv * ((1.0 - du) * q01 + du * q11)
    return out


def magnetic_diagnostics_numpy(
    *,
    psi: np.ndarray,
    grid: Grid2D,
    layout: MagneticDiagnosticLayout,
    previous_flux: np.ndarray | None = None,
    dt: float | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate virtual magnetic diagnostics from a psi field."""
    flux = bilinear_sample_numpy(psi, grid, layout.flux_points)
    dpsi_dz, dpsi_dr = np.gradient(np.asarray(psi, dtype=float), float(grid.z.step), float(grid.r.step))
    br = bilinear_sample_numpy(-dpsi_dz, grid, layout.field_points)
    bz = bilinear_sample_numpy(dpsi_dr, grid, layout.field_points)
    angle = np.asarray(layout.field_angles, dtype=float).reshape(-1)
    field = br * np.cos(angle) + bz * np.sin(angle)
    if previous_flux is not None and dt is not None and float(dt) > 0.0:
        bdot = (flux - np.asarray(previous_flux, dtype=float).reshape(-1)) / float(dt)
    else:
        bdot = np.zeros_like(flux)
    return {"flux": flux, "field": field, "bdot": bdot}


def magnetic_diagnostics_torch(
    *,
    psi,
    grid: Grid2D,
    layout: MagneticDiagnosticLayout,
    device: str,
    previous_flux=None,
    dt: float | None = None,
) -> dict[str, object]:
    import torch
    from tokamak_control.core.torch_sampling import bilinear_sample_torch

    dev = torch.device(device)
    psi_t = torch.as_tensor(psi, dtype=torch.float64, device=dev)
    if psi_t.ndim == 2:
        psi_t = psi_t.unsqueeze(0)
    flux_points = torch.as_tensor(layout.flux_points, dtype=torch.float64, device=dev)
    field_points = torch.as_tensor(layout.field_points, dtype=torch.float64, device=dev)
    flux = bilinear_sample_torch(psi_t, grid, flux_points)
    grad_z = torch.gradient(psi_t, spacing=(float(grid.z.step), float(grid.r.step)), dim=(1, 2))[0]
    grad_r = torch.gradient(psi_t, spacing=(float(grid.z.step), float(grid.r.step)), dim=(1, 2))[1]
    br = bilinear_sample_torch(-grad_z, grid, field_points)
    bz = bilinear_sample_torch(grad_r, grid, field_points)
    angle = torch.as_tensor(layout.field_angles, dtype=torch.float64, device=dev)
    field = br * torch.cos(angle)[None, :] + bz * torch.sin(angle)[None, :]
    if previous_flux is not None and dt is not None and float(dt) > 0.0:
        prev = torch.as_tensor(previous_flux, dtype=torch.float64, device=dev).reshape_as(flux)
        bdot = (flux - prev) / float(dt)
    else:
        bdot = torch.zeros_like(flux)
    return {"flux": flux, "field": field, "bdot": bdot}
