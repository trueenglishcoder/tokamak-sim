from __future__ import annotations

import numpy as np

from tokamak_control.control.base import ControlAction, Controller
from tokamak_control.control.linearization import (
    boundary_sensitivities,
    discrete_B_from_derivative_sensitivities,
)
from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections


class LQRJointController(Controller):
    """One-step quadratic controller for joint boundary and recursive Ip regulation."""

    def __init__(
        self,
        *,
        q_error: float = 1e6,
        q_ip: float = 1.0,
        r_pfc: float = 1e-10,
        r_sol: float = 1e-10,
        ridge: float = 1e-14,
        u_clip: float | None = None,
        ip_ref: float | None = None,
    ) -> None:
        for name, value in (("q_error", q_error), ("q_ip", q_ip), ("r_pfc", r_pfc), ("r_sol", r_sol), ("ridge", ridge)):
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
            if value < 0.0:
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
        self.ridge = float(ridge)
        self.u_clip = u_clip
        self.ip_ref = ip_ref

    def reset(self) -> None:
        return None

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
    ) -> ControlAction:
        n_pfc = int(model.pfc.n_coils)
        n_sol = int(model.sol.n_coils)
        n_u = n_pfc + n_sol
        if n_u == 0:
            return ControlAction(pfc_derivs=np.zeros((0,), dtype=float), sol_derivs=np.zeros((0,), dtype=float))

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
            Bp, Bs = discrete_B_from_derivative_sensitivities(C_pfc, C_sol, model.t_step)
            B_r = np.concatenate([Bp, Bs], axis=1) if (Bp.size or Bs.size) else np.zeros((e_r.shape[0], n_u), dtype=float)

        Q = self.q_error * np.eye(e_r.shape[0], dtype=float)

        R = np.zeros((n_u, n_u), dtype=float)
        if n_pfc:
            R[:n_pfc, :n_pfc] = self.r_pfc * np.eye(n_pfc)
        if n_sol:
            R[n_pfc:, n_pfc:] = self.r_sol * np.eye(n_sol)

        B_ip = np.asarray(model.get_ip_B_row(), dtype=float).reshape(-1)
        if B_ip.size != n_u:
            raise ValueError(
                f"model.get_ip_B_row returned size {B_ip.size}, expected {n_u} for {n_pfc} PFC and {n_sol} SOL inputs"
            )

        ip_target = float(Ip_ref) if Ip_ref is not None else (float(self.ip_ref) if self.ip_ref is not None else float(model.Ip0))
        ip_next0 = float(model.predict_Ip_decay_baseline_next())
        e_ip_next0 = float(ip_next0 - ip_target)

        H = B_r.T @ Q @ B_r + self.q_ip * np.outer(B_ip, B_ip) + R + self.ridge * np.eye(n_u)
        rhs = B_r.T @ Q @ e_r + (-self.q_ip) * (B_ip * e_ip_next0)

        u = np.linalg.solve(H, rhs)
        if self.u_clip is not None:
            u = np.clip(u, -self.u_clip, self.u_clip)

        u_pfc = u[:n_pfc] if n_pfc else np.zeros((0,), dtype=float)
        u_sol = u[n_pfc:] if n_sol else np.zeros((0,), dtype=float)
        return ControlAction(pfc_derivs=u_pfc, sol_derivs=u_sol)
