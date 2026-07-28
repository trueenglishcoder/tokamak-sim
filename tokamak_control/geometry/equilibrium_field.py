"""Continuous interpolation and differential evaluation of an equilibrium flux field."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RectBivariateSpline

from tokamak_control.core.grid import Grid2D


@dataclass(slots=True)
class EquilibriumField:
    """Bicubic representation of ``psi(R, Z)`` on a rectangular grid.

    The interpolant is used only to evaluate the already known equilibrium,
    refine critical points, and project level-set samples back onto the same
    flux surface. It does not fit or smooth a separate boundary curve.
    """

    grid: Grid2D
    psi: np.ndarray
    _spline: RectBivariateSpline = field(init=False, repr=False)
    _r: np.ndarray = field(init=False, repr=False)
    _z: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        psi = np.asarray(self.psi, dtype=float)
        if psi.shape != self.grid.shape:
            raise ValueError(f"psi shape {psi.shape} != grid shape {self.grid.shape}")
        if not np.all(np.isfinite(psi)):
            raise ValueError("EquilibriumField requires finite psi values on the complete grid")
        self.psi = psi.copy()
        self._r = np.asarray(self.grid.r.coords(), dtype=float)
        self._z = np.asarray(self.grid.z.coords(), dtype=float)
        kx = min(3, int(self._r.size) - 1)
        ky = min(3, int(self._z.size) - 1)
        self._spline = RectBivariateSpline(self._r, self._z, self.psi.T, kx=kx, ky=ky, s=0.0)

    @property
    def r_bounds(self) -> tuple[float, float]:
        return float(self._r[0]), float(self._r[-1])

    @property
    def z_bounds(self) -> tuple[float, float]:
        return float(self._z[0]), float(self._z[-1])

    @property
    def grid_scale(self) -> float:
        return float(max(abs(float(self.grid.r.step)), abs(float(self.grid.z.step))))

    @property
    def flux_scale(self) -> float:
        span = float(np.ptp(self.psi))
        return max(span, float(np.max(np.abs(self.psi))), 1.0e-12)

    def contains(self, points: np.ndarray, *, margin: float = 0.0) -> np.ndarray:
        pts = _as_points(points)
        r0, r1 = self.r_bounds
        z0, z1 = self.z_bounds
        m = max(float(margin), 0.0)
        return (
            (pts[:, 0] >= r0 + m)
            & (pts[:, 0] <= r1 - m)
            & (pts[:, 1] >= z0 + m)
            & (pts[:, 1] <= z1 - m)
        )

    def value(self, points: np.ndarray | tuple[float, float]) -> np.ndarray:
        pts = _as_points(points)
        return np.asarray(self._spline.ev(pts[:, 0], pts[:, 1]), dtype=float)

    def gradient(self, points: np.ndarray | tuple[float, float]) -> np.ndarray:
        pts = _as_points(points)
        d_r = self._spline.ev(pts[:, 0], pts[:, 1], dx=1, dy=0)
        d_z = self._spline.ev(pts[:, 0], pts[:, 1], dx=0, dy=1)
        return np.column_stack((d_r, d_z)).astype(float, copy=False)

    def hessian(self, points: np.ndarray | tuple[float, float]) -> np.ndarray:
        pts = _as_points(points)
        d_rr = self._spline.ev(pts[:, 0], pts[:, 1], dx=2, dy=0)
        d_rz = self._spline.ev(pts[:, 0], pts[:, 1], dx=1, dy=1)
        d_zz = self._spline.ev(pts[:, 0], pts[:, 1], dx=0, dy=2)
        out = np.empty((pts.shape[0], 2, 2), dtype=float)
        out[:, 0, 0] = d_rr
        out[:, 0, 1] = d_rz
        out[:, 1, 0] = d_rz
        out[:, 1, 1] = d_zz
        return out

    def project_to_level(
        self,
        points: np.ndarray,
        level: float,
        *,
        max_iterations: int = 12,
        absolute_tolerance: float | None = None,
        max_step: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project regular points to ``psi=level`` by normal Newton correction.

        Points where the gradient is too small are left unchanged and reported
        as not converged. X-points are inserted explicitly by the level-set
        graph code and are not projected through this routine.
        """
        pts = _as_points(points).copy()
        level_f = float(level)
        tol = (
            max(1.0e-12, 1.0e-10 * self.flux_scale)
            if absolute_tolerance is None
            else max(float(absolute_tolerance), 0.0)
        )
        step_limit = 0.75 * self.grid_scale if max_step is None else max(float(max_step), 0.0)
        converged = np.zeros((pts.shape[0],), dtype=bool)
        valid = self.contains(pts)
        for _ in range(max(int(max_iterations), 1)):
            active = valid & ~converged
            if not bool(np.any(active)):
                break
            values = self.value(pts[active]) - level_f
            grads = self.gradient(pts[active])
            grad2 = np.sum(grads * grads, axis=1)
            local_ok = np.isfinite(values) & np.isfinite(grad2) & (grad2 > 1.0e-28)
            active_indices = np.flatnonzero(active)
            if bool(np.any(local_ok)):
                corrections = values[local_ok, None] * grads[local_ok] / grad2[local_ok, None]
                if step_limit > 0.0:
                    norms = np.linalg.norm(corrections, axis=1)
                    scale = np.minimum(1.0, step_limit / np.maximum(norms, 1.0e-30))
                    corrections *= scale[:, None]
                pts[active_indices[local_ok]] -= corrections
            if bool(np.any(~local_ok)):
                valid[active_indices[~local_ok]] = False
            valid &= self.contains(pts)
            residual = np.full((pts.shape[0],), np.inf, dtype=float)
            if bool(np.any(valid)):
                residual[valid] = np.abs(self.value(pts[valid]) - level_f)
            converged = valid & (residual <= tol)
        return pts, converged


def _as_points(points: np.ndarray | tuple[float, float]) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.shape == (2,):
        return arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {arr.shape}")
    return arr
