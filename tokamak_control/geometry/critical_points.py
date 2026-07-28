"""Sub-grid magnetic O-point and X-point detection for a known equilibrium."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import minimum_filter

from tokamak_control.geometry.boundary_common import points_in_or_on_polygon
from tokamak_control.geometry.equilibrium_field import EquilibriumField


CriticalPointKind = Literal["o_point", "x_point"]


@dataclass(frozen=True, slots=True, repr=True)
class CriticalPoint:
    point: tuple[float, float]
    level: float
    kind: CriticalPointKind
    gradient_norm: float
    eigenvalues: tuple[float, float]
    hessian: np.ndarray

    @property
    def determinant(self) -> float:
        return float(self.eigenvalues[0] * self.eigenvalues[1])


@dataclass(frozen=True, slots=True, repr=True)
class CriticalPointSet:
    o_points: tuple[CriticalPoint, ...]
    x_points: tuple[CriticalPoint, ...]
    primary_axis: CriticalPoint


def find_critical_points(
    field: EquilibriumField,
    *,
    center_hint: tuple[float, float],
    limiter_poly: np.ndarray | None,
    max_candidates: int = 256,
    max_o_points: int = 16,
    max_x_points: int = 32,
) -> CriticalPointSet:
    """Find and refine all relevant stationary points of ``psi``.

    Grid-local minima of ``|grad psi|`` and cells with simultaneous derivative
    sign changes are used only as Newton seeds. Returned coordinates and flux
    values are evaluated from the continuous bicubic field.
    """
    seeds = _critical_point_seeds(field, max_candidates=max_candidates)
    center_arr = np.asarray(center_hint, dtype=float).reshape(1, 2)
    if bool(field.contains(center_arr)[0]):
        seeds = np.vstack((center_arr, seeds)) if seeds.size else center_arr.copy()

    refined: list[CriticalPoint] = []
    for seed in seeds:
        point = _refine_stationary_point(field, seed)
        if point is None:
            continue
        if limiter_poly is not None:
            inside = points_in_or_on_polygon(point.reshape(1, 2), limiter_poly, tol=field.grid_scale)
            if not bool(inside[0]):
                continue
        cp = _classify_stationary_point(field, point)
        if cp is None:
            continue
        if _is_duplicate(cp, refined, tolerance=0.35 * field.grid_scale):
            continue
        refined.append(cp)

    o_points = [point for point in refined if point.kind == "o_point"]
    x_points = [point for point in refined if point.kind == "x_point"]
    if not o_points:
        raise RuntimeError("No magnetic O-point was found inside the limiter")

    hint = np.asarray(center_hint, dtype=float)
    o_points.sort(key=lambda p: float(np.linalg.norm(np.asarray(p.point) - hint)))
    primary = o_points[0]
    x_points = _filter_x_points_by_axis_connection(field, primary, x_points)
    x_points.sort(key=lambda p: abs(float(p.level) - float(primary.level)))
    return CriticalPointSet(
        o_points=tuple(o_points[: max(int(max_o_points), 1)]),
        x_points=tuple(x_points[: max(int(max_x_points), 0)]),
        primary_axis=primary,
    )


def _critical_point_seeds(field: EquilibriumField, *, max_candidates: int) -> np.ndarray:
    R, Z = field.grid.mesh()
    points = np.column_stack((R.reshape(-1), Z.reshape(-1)))
    gradients = field.gradient(points).reshape(*field.grid.shape, 2)
    grad2 = np.sum(gradients * gradients, axis=2)
    local_min = grad2 <= minimum_filter(grad2, size=3, mode="nearest")
    local_min[[0, -1], :] = False
    local_min[:, [0, -1]] = False

    seeds: list[tuple[float, float, float]] = []
    z_coords = np.asarray(field.grid.z.coords(), dtype=float)
    r_coords = np.asarray(field.grid.r.coords(), dtype=float)
    for j, i in np.argwhere(local_min):
        score = float(grad2[j, i])
        if np.isfinite(score):
            seeds.append((score, float(r_coords[i]), float(z_coords[j])))

    d_r = gradients[:, :, 0]
    d_z = gradients[:, :, 1]
    for j in range(field.grid.z.size - 1):
        for i in range(field.grid.r.size - 1):
            cell_r = d_r[j : j + 2, i : i + 2]
            cell_z = d_z[j : j + 2, i : i + 2]
            if not np.all(np.isfinite(cell_r)) or not np.all(np.isfinite(cell_z)):
                continue
            if float(np.min(cell_r)) <= 0.0 <= float(np.max(cell_r)) and float(np.min(cell_z)) <= 0.0 <= float(np.max(cell_z)):
                r_mid = 0.5 * (r_coords[i] + r_coords[i + 1])
                z_mid = 0.5 * (z_coords[j] + z_coords[j + 1])
                score = float(np.mean(cell_r * cell_r + cell_z * cell_z))
                seeds.append((score, float(r_mid), float(z_mid)))

    seeds.sort(key=lambda item: item[0])
    out: list[np.ndarray] = []
    separation = 0.4 * field.grid_scale
    for _score, r_value, z_value in seeds:
        p = np.array([r_value, z_value], dtype=float)
        if any(float(np.linalg.norm(p - old)) < separation for old in out):
            continue
        out.append(p)
        if len(out) >= max(int(max_candidates), 1):
            break
    return np.asarray(out, dtype=float).reshape(-1, 2)


def _refine_stationary_point(field: EquilibriumField, seed: np.ndarray) -> np.ndarray | None:
    x0 = np.asarray(seed, dtype=float).reshape(2)
    x = x0.copy()
    search_radius = 3.5 * np.hypot(float(field.grid.r.step), float(field.grid.z.step))
    grad_tol = max(1.0e-11, 1.0e-8 * field.flux_scale / max(field.grid_scale, 1.0e-12))
    for _ in range(40):
        grad = field.gradient(x)[0]
        hess = field.hessian(x)[0]
        grad_norm = float(np.linalg.norm(grad))
        if not np.isfinite(grad_norm) or not np.all(np.isfinite(hess)):
            return None
        if grad_norm <= grad_tol:
            return x
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(hess, rcond=1.0e-12) @ grad
        if not np.all(np.isfinite(delta)):
            return None
        accepted = False
        alpha = 1.0
        for _ in range(10):
            trial = x - alpha * delta
            if not bool(field.contains(trial.reshape(1, 2), margin=0.25 * field.grid_scale)[0]):
                alpha *= 0.5
                continue
            if float(np.linalg.norm(trial - x0)) > search_radius:
                alpha *= 0.5
                continue
            trial_norm = float(np.linalg.norm(field.gradient(trial)[0]))
            if trial_norm < grad_norm:
                x = trial
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            return None
    final_norm = float(np.linalg.norm(field.gradient(x)[0]))
    return x if final_norm <= 10.0 * grad_tol else None


def _classify_stationary_point(field: EquilibriumField, point: np.ndarray) -> CriticalPoint | None:
    hess = field.hessian(point)[0]
    eig = np.linalg.eigvalsh(hess)
    scale = max(float(np.max(np.abs(eig))), 1.0e-30)
    if float(np.min(np.abs(eig))) <= 1.0e-8 * scale:
        return None
    if eig[0] < 0.0 < eig[1]:
        kind: CriticalPointKind = "x_point"
    elif eig[0] > 0.0 or eig[1] < 0.0:
        kind = "o_point"
    else:
        return None
    grad_norm = float(np.linalg.norm(field.gradient(point)[0]))
    return CriticalPoint(
        point=(float(point[0]), float(point[1])),
        level=float(field.value(point)[0]),
        kind=kind,
        gradient_norm=grad_norm,
        eigenvalues=(float(eig[0]), float(eig[1])),
        hessian=np.asarray(hess, dtype=float).copy(),
    )


def _is_duplicate(candidate: CriticalPoint, existing: list[CriticalPoint], *, tolerance: float) -> bool:
    p = np.asarray(candidate.point, dtype=float)
    for old in existing:
        if float(np.linalg.norm(p - np.asarray(old.point, dtype=float))) <= float(tolerance):
            return True
    return False


def _filter_x_points_by_axis_connection(
    field: EquilibriumField,
    axis: CriticalPoint,
    x_points: list[CriticalPoint],
) -> list[CriticalPoint]:
    axis_point = np.asarray(axis.point, dtype=float)
    kept: list[CriticalPoint] = []
    for x_point in x_points:
        delta = float(x_point.level - axis.level)
        if not np.isfinite(delta) or abs(delta) <= 1.0e-12 * field.flux_scale:
            continue
        samples = np.linspace(0.0, 1.0, 96, dtype=float)
        target = np.asarray(x_point.point, dtype=float)
        line = axis_point[None, :] + samples[:, None] * (target - axis_point)[None, :]
        normalized = (field.value(line) - float(axis.level)) / delta
        if not np.all(np.isfinite(normalized)):
            continue
        # The direct path from the primary axis to a primary separatrix saddle
        # must stay within the same flux interval. Small interpolation ripple is
        # tolerated, but a separate extremum along the path rejects the saddle.
        if float(np.min(normalized)) < -0.02 or float(np.max(normalized)) > 1.02:
            continue
        adverse = np.diff(normalized) < -0.02
        if int(np.count_nonzero(adverse)) > 2:
            continue
        kept.append(x_point)
    return kept
