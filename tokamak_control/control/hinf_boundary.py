# tokamak_control/control/hinf_boundary.py
from __future__ import annotations

import numpy as np

from tokamak_control.control.base import ControlAction, Controller
from tokamak_control.control.linearization import (
    boundary_sensitivities,
    discrete_B_from_derivative_sensitivities,
)
from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections


class HinftyBoundaryController(Controller):
    """
    H∞ boundary controller using a one-step robust minimax solution.

    Disturbance enters measurements via E, default identity.
    As gamma grows large, the controller approaches the one-step LQR solution.
    """

    def __init__(
        self,
        *,
        q_error: float = 1.0,
        r_pfc: float = 1e-2,
        r_sol: float = 1e-2,
        gamma: float = 10.0,
        u_clip: float = 1e3,
        ridge: float = 1e-12,
    ) -> None:
        for name, value in (
            ("q_error", q_error),
            ("r_pfc", r_pfc),
            ("r_sol", r_sol),
            ("gamma", gamma),
            ("u_clip", u_clip),
            ("ridge", ridge),
        ):
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")

        if float(q_error) < 0.0:
            raise ValueError("q_error must be >= 0")
        if float(r_pfc) < 0.0:
            raise ValueError("r_pfc must be >= 0")
        if float(r_sol) < 0.0:
            raise ValueError("r_sol must be >= 0")
        if float(gamma) <= 0.0:
            raise ValueError("gamma must be > 0")
        if float(u_clip) < 0.0:
            raise ValueError("u_clip must be >= 0")
        if float(ridge) < 0.0:
            raise ValueError("ridge must be >= 0")

        self.q_error = float(q_error)
        self.r_pfc = float(r_pfc)
        self.r_sol = float(r_sol)
        self.gamma = float(gamma)
        self.u_clip = float(u_clip)
        self.ridge = float(ridge)
        self._last_gain: np.ndarray | None = None

    def reset(self) -> None:
        self._last_gain = None

    def compute_control(
        self,
        *,
        model,
        psi: np.ndarray,
        boundary_poly: np.ndarray | None,
        center: tuple[float, float],
        measure_angles: np.ndarray,
        ref_radii: np.ndarray,
        E: np.ndarray | None = None,
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
        M = e.shape[0]

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

        if n_u == 0:
            return ControlAction(
                pfc_derivs=np.zeros((n_pfc,), dtype=float),
                sol_derivs=np.zeros((n_sol,), dtype=float),
            )

        Q = self.q_error * np.eye(M)

        R = np.zeros((n_u, n_u), dtype=float)
        if n_pfc:
            R[:n_pfc, :n_pfc] = self.r_pfc * np.eye(n_pfc)
        if n_sol:
            R[n_pfc:, n_pfc:] = self.r_sol * np.eye(n_sol)

        if E is None:
            E = np.eye(M, dtype=float)
        else:
            E = np.asarray(E, dtype=float)
            if E.ndim != 2 or E.shape[0] != M:
                raise ValueError(
                    f"E must have shape (M, M_d) with M={M}, got {E.shape!r}"
                )

        EtQE = E.T @ Q @ E
        gamma_mat = self.gamma**2 * np.eye(EtQE.shape[0], dtype=float) - EtQE
        evals = np.linalg.eigvalsh(gamma_mat)
        if float(np.min(evals)) <= 0.0:
            raise ValueError(
                "HinftyBoundaryController infeasible: gamma^2 I - E^T Q E is not positive definite. "
                "Increase gamma or reduce q_error."
            )

        Q_hat = Q + Q @ E @ np.linalg.solve(gamma_mat, E.T @ Q)

        H = B.T @ Q_hat @ B + R + self.ridge * np.eye(n_u)
        G = B.T @ Q_hat
        K = np.linalg.solve(H, G)
        self._last_gain = K

        u = K @ e
        u = np.clip(u, -self.u_clip, self.u_clip)

        u_pfc = u[:n_pfc] if n_pfc else np.zeros((0,), dtype=float)
        u_sol = u[n_pfc:] if n_sol else np.zeros((0,), dtype=float)
        return ControlAction(pfc_derivs=u_pfc, sol_derivs=u_sol)
