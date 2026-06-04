from __future__ import annotations

import numpy as np

from tokamak_control.core.grid import Grid2D
from tokamak_control.core.plasma_model import PlasmaModel
from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections


def gradient_psi(psi: np.ndarray, grid: Grid2D) -> tuple[np.ndarray, np.ndarray]:
    """Compute finite-difference gradients of ψ on the simulation grid."""
    dpsi_dR = np.zeros_like(psi, dtype=float)
    dpsi_dZ = np.zeros_like(psi, dtype=float)

    dR = float(grid.r.step)
    dpsi_dR[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2.0 * dR)
    dpsi_dR[:, 0] = (psi[:, 1] - psi[:, 0]) / dR
    dpsi_dR[:, -1] = (psi[:, -1] - psi[:, -2]) / dR

    dZ = float(grid.z.step)
    dpsi_dZ[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2.0 * dZ)
    dpsi_dZ[0, :] = (psi[1, :] - psi[0, :]) / dZ
    dpsi_dZ[-1, :] = (psi[-1, :] - psi[-2, :]) / dZ

    return dpsi_dR, dpsi_dZ


def _bilinear_sample(field: np.ndarray, grid: Grid2D, points: np.ndarray) -> np.ndarray:
    """Sample a scalar field at (R, Z) points using bilinear interpolation."""
    vals = np.empty(points.shape[0], dtype=float)

    r_coords = grid.r.coords()
    z_coords = grid.z.coords()
    r0 = float(r_coords[0])
    z0 = float(z_coords[0])
    dR = float(grid.r.step)
    dZ = float(grid.z.step)

    for k, (R0, Z0) in enumerate(points):
        u = (float(R0) - r0) / dR
        v = (float(Z0) - z0) / dZ

        i0 = int(np.floor(u))
        j0 = int(np.floor(v))
        i1 = i0 + 1
        j1 = j0 + 1

        if i0 < 0 or j0 < 0 or i1 >= grid.r.size or j1 >= grid.z.size:
            vals[k] = np.nan
            continue

        du = u - i0
        dv = v - j0

        q00 = field[j0, i0]
        q10 = field[j0, i1]
        q01 = field[j1, i0]
        q11 = field[j1, i1]

        q0 = (1.0 - du) * q00 + du * q10
        q1 = (1.0 - du) * q01 + du * q11
        vals[k] = (1.0 - dv) * q0 + dv * q1

    return vals


def _boundary_points_from_radii(
    center: tuple[float, float],
    angles: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    angles = np.asarray(angles, dtype=float).reshape(-1)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    if angles.shape != radii.shape:
        raise ValueError(f"angles shape {angles.shape} != radii shape {radii.shape}")

    R = float(center[0]) + radii * np.cos(angles)
    Z = float(center[1]) + radii * np.sin(angles)
    return np.column_stack([R, Z])


def boundary_sensitivities(
    model: PlasmaModel,
    psi: np.ndarray,
    boundary_poly: np.ndarray,
    center: tuple[float, float],
    measure_angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute local boundary sensitivities from the already known boundary.

    This removes the redundant full-boundary re-extraction loop that used to run
    once per actuator perturbation. The sensitivities are computed from the
    local implicit-contour relation on the current boundary:

        δr ≈ - δψ / (∂ψ/∂r)

    evaluated at the measurement-angle intersection points of the current
    boundary, where

        ∂ψ/∂r = ∇ψ · e_r

    and ``e_r`` is the outward radial direction from ``center``.
    """
    psi = np.asarray(psi, dtype=float)
    measure_angles = np.asarray(measure_angles, dtype=float).reshape(-1)

    base_radii = radii_from_polyline_ray_intersections(boundary_poly, center, measure_angles)
    points = _boundary_points_from_radii(center, measure_angles, base_radii)

    dpsi_dR, dpsi_dZ = gradient_psi(psi, model.grid)
    grad_R = _bilinear_sample(dpsi_dR, model.grid, points)
    grad_Z = _bilinear_sample(dpsi_dZ, model.grid, points)

    e_r_R = np.cos(measure_angles)
    e_r_Z = np.sin(measure_angles)
    dpsi_dr = grad_R * e_r_R + grad_Z * e_r_Z

    safe_scale = np.maximum(np.abs(dpsi_dr), 1e-12)
    signed_scale = np.where(dpsi_dr >= 0.0, safe_scale, -safe_scale)

    G_pfc = np.asarray(model.sample_green_pfc(points), dtype=float)
    G_sol = np.asarray(model.sample_green_sol(points), dtype=float).reshape(points.shape[0], -1)
    G_ip = np.asarray(model.sample_green_plasma(points), dtype=float).reshape(-1)

    if G_pfc.ndim != 2:
        raise ValueError(f"sample_green_pfc returned shape {G_pfc.shape}, expected 2D")
    if G_sol.ndim != 2:
        raise ValueError(f"sample_green_sol returned shape {G_sol.shape}, expected 2D")
    if G_ip.ndim != 1 or G_ip.shape[0] != points.shape[0]:
        raise ValueError(f"sample_green_plasma returned shape {G_ip.shape}, expected ({points.shape[0]},)")

    C_pfc = -G_pfc / signed_scale[:, None] if G_pfc.size else np.zeros((measure_angles.size, 0), dtype=float)
    C_sol = -G_sol / signed_scale[:, None] if G_sol.size else np.zeros((measure_angles.size, 0), dtype=float)
    C_Ip = -G_ip / signed_scale

    C_pfc = np.nan_to_num(C_pfc, nan=0.0, posinf=0.0, neginf=0.0)
    C_sol = np.nan_to_num(C_sol, nan=0.0, posinf=0.0, neginf=0.0)
    C_Ip = np.nan_to_num(C_Ip, nan=0.0, posinf=0.0, neginf=0.0)
    return C_pfc, C_sol, C_Ip


def discrete_B_from_derivative_sensitivities(
    C_pfc: np.ndarray,
    C_sol: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert current sensitivities to derivative sensitivities over one time step."""
    return float(dt) * np.asarray(C_pfc, dtype=float), float(dt) * np.asarray(C_sol, dtype=float)
