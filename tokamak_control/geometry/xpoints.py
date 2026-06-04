"""Utilities for locating magnetic X-points in sampled psi fields."""

from __future__ import annotations

import numpy as np

from tokamak_control.core.grid import Grid2D


def find_x_points(
    psi: np.ndarray,
    grid: Grid2D,
    *,
    max_points: int = 8,
    min_separation: float | None = None,
) -> np.ndarray:
    """Return candidate X-points as ``(R, Z)`` coordinates.

    The detector looks for interior grid points where ``|grad psi|`` has a
    local minimum and the Hessian has negative determinant, which is the
    sampled-field signature of a saddle point.
    """
    psi_arr = np.asarray(psi, dtype=float)
    if psi_arr.shape != grid.shape:
        raise ValueError(f"psi shape {psi_arr.shape} != grid shape {grid.shape}")

    if psi_arr.shape[0] < 3 or psi_arr.shape[1] < 3:
        return np.zeros((0, 2), dtype=float)

    d_z = float(grid.z.step)
    d_r = float(grid.r.step)
    dpsi_dz, dpsi_dr = np.gradient(psi_arr, d_z, d_r, edge_order=2)
    d2psi_dz2, d2psi_dzdr = np.gradient(dpsi_dz, d_z, d_r, edge_order=2)
    d2psi_drdz, d2psi_dr2 = np.gradient(dpsi_dr, d_z, d_r, edge_order=2)

    grad_norm = np.hypot(dpsi_dr, dpsi_dz)
    det_hessian = d2psi_dr2 * d2psi_dz2 - d2psi_drdz * d2psi_dzdr

    r_coords = np.asarray(grid.r.coords(), dtype=float)
    z_coords = np.asarray(grid.z.coords(), dtype=float)
    candidates: list[tuple[float, float, float]] = []

    for j in range(1, psi_arr.shape[0] - 1):
        for i in range(1, psi_arr.shape[1] - 1):
            if not np.isfinite(grad_norm[j, i]) or not np.isfinite(det_hessian[j, i]):
                continue
            if det_hessian[j, i] >= 0.0:
                continue
            neighborhood = grad_norm[j - 1 : j + 2, i - 1 : i + 2]
            if float(grad_norm[j, i]) > float(np.nanmin(neighborhood)):
                continue
            candidates.append((float(grad_norm[j, i]), float(r_coords[i]), float(z_coords[j])))

    candidates.sort(key=lambda item: item[0])
    return _select_separated_candidates(
        candidates,
        max_points=max_points,
        min_separation=min_separation,
    )


def _select_separated_candidates(
    candidates: list[tuple[float, float, float]],
    *,
    max_points: int,
    min_separation: float | None,
) -> np.ndarray:
    """Select the strongest saddle candidates with optional spatial spacing."""
    limit = max(int(max_points), 0)
    if limit == 0 or not candidates:
        return np.zeros((0, 2), dtype=float)

    sep = 0.0 if min_separation is None else max(float(min_separation), 0.0)
    selected: list[tuple[float, float]] = []

    for _score, r_value, z_value in candidates:
        point = (r_value, z_value)
        if sep > 0.0 and any(np.hypot(point[0] - old[0], point[1] - old[1]) < sep for old in selected):
            continue
        selected.append(point)
        if len(selected) >= limit:
            break

    return np.asarray(selected, dtype=float).reshape(-1, 2)
