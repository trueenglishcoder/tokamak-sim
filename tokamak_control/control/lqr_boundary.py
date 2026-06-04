# tokamak_control/control/lqr_boundary.py
from __future__ import annotations

import numpy as np

from tokamak_control.control.base import ControlAction, Controller
from tokamak_control.control.linearization import (
    boundary_sensitivities,
    discrete_B_from_derivative_sensitivities,
)
from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections


class LQRBoundaryController(Controller):
    """
    Boundary controller using the one-step quadratic solution.

    The controller assumes a local one-step model

        e_{k+1} ≈ e_k - B u

    and chooses derivatives u that reduce the predicted boundary error.
    """

    def __init__(
        self,
        *,
        q_error: float = 1e6,
        r_pfc: float = 1e-10,
        r_sol: float = 1e-10,
        u_clip: float = 5e6,
        ridge: float = 1e-14,
    ) -> None:
        for name, value in (
            ("q_error", q_error),
            ("r_pfc", r_pfc),
            ("r_sol", r_sol),
            ("u_clip", u_clip),
            ("ridge", ridge),
        ):
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")

        self.q_error = float(q_error)
        self.r_pfc = float(r_pfc)
        self.r_sol = float(r_sol)
        self.u_clip = float(u_clip)
        self.ridge = float(ridge)
        self._last_K: np.ndarray | None = None

    def reset(self) -> None:
        self._last_K = None

    def compute_control(
        self,
        *,
        model,
        psi: np.ndarray,
        boundary_poly: np.ndarray | None,
        center: tuple[float, float],
        measure_angles: np.ndarray,
        ref_radii: np.ndarray,
    ) -> ControlAction:
        n_pfc = int(model.pfc.n_coils)
        n_sol = int(model.sol.n_coils)

        if boundary_poly is None:
            return ControlAction(
                pfc_derivs=np.zeros((n_pfc,), dtype=float),
                sol_derivs=np.zeros((n_sol,), dtype=float),
            )

        radii = radii_from_polyline_ray_intersections(boundary_poly, center, measure_angles)
        e = np.asarray(ref_radii - radii, dtype=float)

        C_pfc, C_sol, _ = boundary_sensitivities(
            model=model,
            psi=psi,
            boundary_poly=boundary_poly,
            center=center,
            measure_angles=measure_angles,
        )

        Bp, Bs = discrete_B_from_derivative_sensitivities(C_pfc, C_sol, model.t_step)
        B = np.concatenate([Bp, Bs], axis=1)
        n_u = B.shape[1]
        n_pfc = C_pfc.shape[1]
        n_sol = C_sol.shape[1]
        n_err = e.shape[0]

        if n_u == 0:
            return ControlAction(
                pfc_derivs=np.zeros((n_pfc,), dtype=float),
                sol_derivs=np.zeros((n_sol,), dtype=float),
            )

        Q = self.q_error * np.eye(n_err)

        R = np.zeros((n_u, n_u), dtype=float)
        if n_pfc:
            R[:n_pfc, :n_pfc] = self.r_pfc * np.eye(n_pfc)
        if n_sol:
            R[n_pfc:, n_pfc:] = self.r_sol * np.eye(n_sol)

        H = B.T @ Q @ B + R + self.ridge * np.eye(n_u)
        G = B.T @ Q
        K = np.linalg.solve(H, G)
        self._last_K = K

        u = K @ e
        u = np.clip(u, -self.u_clip, self.u_clip)

        u_pfc = u[:n_pfc] if n_pfc else np.zeros((0,), dtype=float)
        u_sol = u[n_pfc:] if n_sol else np.zeros((0,), dtype=float)
        return ControlAction(pfc_derivs=u_pfc, sol_derivs=u_sol)
