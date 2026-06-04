from __future__ import annotations

import numpy as np

from tokamak_control.control.base import ControlAction, Controller
from tokamak_control.control.linearization import (
    boundary_sensitivities,
    discrete_B_from_derivative_sensitivities,
)
from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections


class HinftyJointController(Controller):
    """
    One-step H∞ controller for joint boundary and Ip regulation.

    This is the robust analogue of LQRJointController. It stacks boundary and
    current tracking into one regulated-error vector and solves a one-step
    minimax problem in that combined measurement space.

    Disturbance enters the regulated output through E, default identity.
    As gamma grows large, the controller approaches the one-step LQR joint solution.
    """

    def __init__(
        self,
        *,
        q_error: float = 1e6,
        q_ip: float = 1.0,
        r_pfc: float = 1e-10,
        r_sol: float = 1e-10,
        gamma: float = 10.0,
        ridge: float = 1e-14,
        u_clip: float | None = None,
        ip_ref: float | None = None,
        j_curr: float = 0,
    ) -> None:
        for name, value in (
            ("q_error", q_error),
            ("q_ip", q_ip),
            ("r_pfc", r_pfc),
            ("r_sol", r_sol),
            ("gamma", gamma),
            ("ridge", ridge),
            ("j_curr", j_curr),
        ):
            v = float(value)
            if not np.isfinite(v):
                raise ValueError(f"{name} must be finite, got {value!r}")
            if name == "gamma":
                if v <= 0.0:
                    raise ValueError("gamma must be > 0")
            else:
                if v < 0.0:
                    raise ValueError(f"{name} must be >= 0")

        if u_clip is not None:
            u_clip = float(u_clip)
            if not np.isfinite(u_clip):
                raise ValueError(f"u_clip must be finite if set, got {u_clip!r}")
            if u_clip < 0.0:
                raise ValueError("u_clip must be >= 0 if set")

        if ip_ref is not None:
            ip_ref = float(ip_ref)
            if not np.isfinite(ip_ref):
                raise ValueError(f"ip_ref must be finite if set, got {ip_ref!r}")

        self.q_error = float(q_error)
        self.q_ip = float(q_ip)
        self.r_pfc = float(r_pfc)
        self.r_sol = float(r_sol)
        self.gamma = float(gamma)
        self.ridge = float(ridge)
        self.u_clip = u_clip
        self.ip_ref = ip_ref
        self.j_curr = float(j_curr)

        self._pfc_curr_ref: np.ndarray | None = None
        self._sol_curr_ref: np.ndarray | None = None
        self._last_gain: np.ndarray | None = None

    def reset(self) -> None:
        self._pfc_curr_ref = None
        self._sol_curr_ref = None
        self._last_gain = None

    def _ensure_current_refs(self, model) -> None:
        if self._pfc_curr_ref is None:
            self._pfc_curr_ref = np.asarray(model.state.pfc_currents, dtype=float).copy()
        if self._sol_curr_ref is None:
            self._sol_curr_ref = np.asarray(model.state.sol_currents, dtype=float).copy()

    def compute_control(
        self,
        *,
        model,
        psi: np.ndarray,
        boundary_poly: np.ndarray | None,
        center: tuple[float, float],
        measure_angles: np.ndarray,
        ref_radii: np.ndarray,
        Ip_ref: float | None = None,
        E: np.ndarray | None = None,
    ) -> ControlAction:
        n_pfc = int(model.pfc.n_coils)
        n_sol = int(model.sol.n_coils)
        n_u = n_pfc + n_sol

        if n_u == 0:
            return ControlAction(
                pfc_derivs=np.zeros((0,), dtype=float),
                sol_derivs=np.zeros((0,), dtype=float),
            )

        dt = float(model.t_step)

        if boundary_poly is None:
            e_r = np.zeros((0,), dtype=float)
            B_r = np.zeros((0, n_u), dtype=float)
        else:
            radii = radii_from_polyline_ray_intersections(boundary_poly, center, measure_angles)
            e_r = np.asarray(ref_radii - radii, dtype=float)

            C_pfc, C_sol, _ = boundary_sensitivities(
                model=model,
                psi=psi,
                boundary_poly=boundary_poly,
                center=center,
                measure_angles=measure_angles,
            )
            Bp, Bs = discrete_B_from_derivative_sensitivities(C_pfc, C_sol, dt)
            if Bp.size or Bs.size:
                B_r = np.concatenate([Bp, Bs], axis=1)
            else:
                B_r = np.zeros((e_r.shape[0], n_u), dtype=float)

        B_ip = np.asarray(model.get_ip_B_row(), dtype=float).reshape(-1)
        if B_ip.size != n_u:
            raise ValueError(
                f"model.get_ip_B_row returned size {B_ip.size}, expected {n_u} "
                f"for {n_pfc} PFC and {n_sol} SOL inputs"
            )

        ip_target = (
            float(Ip_ref)
            if Ip_ref is not None
            else (float(self.ip_ref) if self.ip_ref is not None else float(model.Ip0))
        )
        e_ip = float(model.state.Ip - ip_target)

        # Fit current error into e_{k+1} ≈ e_k - B u
        B_ip_row = (-dt * B_ip).reshape(1, -1)

        if e_r.size:
            e = np.concatenate([e_r, np.asarray([e_ip], dtype=float)], axis=0)
            B = np.vstack([B_r, B_ip_row])
        else:
            e = np.asarray([e_ip], dtype=float)
            B = B_ip_row

        M = e.shape[0]

        Q = np.zeros((M, M), dtype=float)
        if e_r.size:
            Q[: e_r.size, : e_r.size] = self.q_error * np.eye(e_r.size, dtype=float)
        Q[-1, -1] = self.q_ip

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
                raise ValueError(f"E must have shape (M, M_d) with M={M}, got {E.shape!r}")

        EtQE = E.T @ Q @ E
        gamma_mat = self.gamma**2 * np.eye(EtQE.shape[0], dtype=float) - EtQE
        evals = np.linalg.eigvalsh(gamma_mat)
        if float(np.min(evals)) <= 0.0:
            raise ValueError(
                "HinftyJointController infeasible: gamma^2 I - E^T Q E is not positive definite. "
                "Increase gamma or reduce q_error/q_ip."
            )

        Q_hat = Q + Q @ E @ np.linalg.solve(gamma_mat, E.T @ Q)

        H = B.T @ Q_hat @ B + R + self.ridge * np.eye(n_u)
        rhs = B.T @ Q_hat @ e

        if self.j_curr > 0.0:
            self._ensure_current_refs(model)

            J_pfc = (
                np.asarray(model.state.pfc_currents, dtype=float).reshape(-1)
                if n_pfc
                else np.zeros((0,), dtype=float)
            )
            J_sol = (
                np.asarray(model.state.sol_currents, dtype=float).reshape(-1)
                if n_sol
                else np.zeros((0,), dtype=float)
            )
            Jref_pfc = self._pfc_curr_ref.reshape(-1) if self._pfc_curr_ref is not None else J_pfc
            Jref_sol = self._sol_curr_ref.reshape(-1) if self._sol_curr_ref is not None else J_sol

            if n_pfc:
                w_p = self.j_curr / float(n_pfc)
                H[:n_pfc, :n_pfc] += (w_p * dt * dt) * np.eye(n_pfc)
                rhs[:n_pfc] += -(w_p * dt) * (J_pfc - Jref_pfc)

            if n_sol:
                w_s = self.j_curr / float(n_sol)
                H[n_pfc:, n_pfc:] += (w_s * dt * dt) * np.eye(n_sol)
                rhs[n_pfc:] += -(w_s * dt) * (J_sol - Jref_sol)

        K = np.linalg.solve(H, B.T @ Q_hat)
        self._last_gain = K

        u = np.linalg.solve(H, rhs)

        if self.u_clip is not None:
            u = np.clip(u, -self.u_clip, self.u_clip)

        u_pfc = u[:n_pfc] if n_pfc else np.zeros((0,), dtype=float)
        u_sol = u[n_pfc:] if n_sol else np.zeros((0,), dtype=float)
        return ControlAction(pfc_derivs=u_pfc, sol_derivs=u_sol)
