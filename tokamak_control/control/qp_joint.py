from __future__ import annotations

import numpy as np

from tokamak_control.control.base import ControlAction, Controller
from tokamak_control.control.linearization import (
    boundary_sensitivities,
    discrete_B_from_derivative_sensitivities,
)
from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections


class QPJointController(Controller):
    """
    Joint boundary and Ip controller via constrained quadratic minimization.

    Controls coil current derivatives ``u = [u_pfc; u_sol]`` in A/s subject to
    componentwise box bounds derived from ``model.{pfc,sol}_deriv_limit`` when set.
    """

    def __init__(
        self,
        *,
        w_radii: float = 1.0,
        w_Ip: float = 0.2,
        w_u: float = 1e-9,
        k_i_radii: float = 0.5,
        k_i_Ip: float = 0.5,
        radii_int_clip: float = 0.5,
        Ip_int_clip: float = 5.0e3,
        pgd_iters: int = 400,
        ridge: float = 1e-8,
    ) -> None:
        for name, value in (
            ("w_radii", w_radii),
            ("w_Ip", w_Ip),
            ("w_u", w_u),
            ("k_i_radii", k_i_radii),
            ("k_i_Ip", k_i_Ip),
            ("radii_int_clip", radii_int_clip),
            ("Ip_int_clip", Ip_int_clip),
            ("ridge", ridge),
        ):
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if int(pgd_iters) != pgd_iters:
            raise ValueError(f"pgd_iters must be an int, got {pgd_iters!r}")
        if int(pgd_iters) <= 0:
            raise ValueError("pgd_iters must be > 0")
        for name, value in (
            ("w_radii", w_radii),
            ("w_Ip", w_Ip),
            ("w_u", w_u),
            ("radii_int_clip", radii_int_clip),
            ("Ip_int_clip", Ip_int_clip),
            ("ridge", ridge),
        ):
            if float(value) < 0.0:
                raise ValueError(f"{name} must be >= 0")

        self.w_radii = float(w_radii)
        self.w_Ip = float(w_Ip)
        self.w_u = float(w_u)
        self.k_i_radii = float(k_i_radii)
        self.k_i_Ip = float(k_i_Ip)
        self.radii_int_clip = float(radii_int_clip)
        self.Ip_int_clip = float(Ip_int_clip)
        self.pgd_iters = int(pgd_iters)
        self.ridge = float(ridge)
        self.reset()

    def reset(self) -> None:
        self._radii_int: np.ndarray | None = None
        self._Ip_int: float = 0.0
        self._last_u: np.ndarray | None = None

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
        psi = np.asarray(psi, dtype=float)
        measure_angles = np.asarray(measure_angles, dtype=float).reshape(-1)
        ref_radii = np.asarray(ref_radii, dtype=float).reshape(-1)

        dt = float(getattr(model, "t_step"))
        Ip = float(getattr(model.state, "Ip"))
        if Ip_ref is None:
            Ip_ref = float(getattr(model, "Ip0"))

        n_pfc = int(getattr(model.pfc, "n_coils", 0))
        n_sol = int(getattr(model.sol, "n_coils", 0))
        N = n_pfc + n_sol
        if N == 0:
            return ControlAction(
                pfc_derivs=np.zeros((0,), dtype=float),
                sol_derivs=np.zeros((0,), dtype=float),
            )

        B_ip = np.asarray(model.get_ip_B_row(), dtype=float).reshape(-1)
        if B_ip.size != N:
            raise ValueError(
                f"model.get_ip_B_row returned size {B_ip.size}, expected {N} "
                f"for {n_pfc} PFC and {n_sol} SOL inputs"
            )

        ip_next0 = float(model.predict_Ip_decay_baseline_next())

        if boundary_poly is None:
            e_r = np.zeros((0,), dtype=float)
            A_r = np.zeros((0, N), dtype=float)
            b_r = np.zeros((0,), dtype=float)
        else:
            boundary_poly = np.asarray(boundary_poly, dtype=float)

            radii = radii_from_polyline_ray_intersections(boundary_poly, center, measure_angles)
            C_pfc, C_sol, C_Ip = boundary_sensitivities(
                model=model,
                psi=psi,
                boundary_poly=boundary_poly,
                center=center,
                measure_angles=measure_angles,
            )
            Bpfc, Bsol = discrete_B_from_derivative_sensitivities(C_pfc, C_sol, dt)

            n_pfc = Bpfc.shape[1]
            n_sol = Bsol.shape[1]
            N = n_pfc + n_sol
            M = radii.shape[0]

            if B_ip.size != N:
                raise ValueError(
                    f"model.get_ip_B_row returned size {B_ip.size}, expected {N} "
                    f"for {n_pfc} PFC and {n_sol} SOL inputs"
                )

            if self._radii_int is None or self._radii_int.shape != (M,):
                self._radii_int = np.zeros((M,), dtype=float)

            e_r = radii - ref_radii
            self._radii_int = self._radii_int + dt * e_r
            if self.radii_int_clip > 0.0:
                self._radii_int = np.clip(
                    self._radii_int,
                    -self.radii_int_clip,
                    self.radii_int_clip,
                )

            A_r_base = (
                np.concatenate([Bpfc, Bsol], axis=1)
                if N
                else np.zeros((M, 0), dtype=float)
            )

            A_r = A_r_base + (C_Ip.reshape(M, 1) * B_ip.reshape(1, N))
            bias_r = C_Ip * (ip_next0 - Ip)
            b_r = e_r + self.k_i_radii * self._radii_int + bias_r

        e_Ip = Ip - float(Ip_ref)
        self._Ip_int = float(self._Ip_int + dt * e_Ip)
        if self.Ip_int_clip > 0.0:
            self._Ip_int = float(np.clip(self._Ip_int, -self.Ip_int_clip, self.Ip_int_clip))

        sqrt_wr = float(np.sqrt(max(self.w_radii, 0.0)))
        sqrt_wI = float(np.sqrt(max(self.w_Ip, 0.0)))
        sqrt_wu = float(np.sqrt(max(self.w_u, 0.0)))

        b_I = (ip_next0 - float(Ip_ref)) + self.k_i_Ip * self._Ip_int

        A_ls = np.vstack([
            sqrt_wr * A_r,
            sqrt_wI * B_ip.reshape(1, N),
            sqrt_wu * np.eye(N, dtype=float),
        ])
        b_ls = np.concatenate([
            sqrt_wr * b_r,
            np.array([sqrt_wI * b_I], dtype=float),
            np.zeros((N,), dtype=float),
        ], axis=0)

        H = 2.0 * (A_ls.T @ A_ls)
        if self.ridge > 0.0:
            H = H + self.ridge * np.eye(N, dtype=float)
        f = 2.0 * (A_ls.T @ b_ls)

        pfc_lim = getattr(model, "pfc_deriv_limit", None)
        sol_lim = getattr(model, "sol_deriv_limit", None)

        if pfc_lim is None:
            lo_pfc = -np.inf * np.ones((n_pfc,), dtype=float)
            hi_pfc = np.inf * np.ones((n_pfc,), dtype=float)
        else:
            lim = float(pfc_lim)
            lo_pfc = -lim * np.ones((n_pfc,), dtype=float)
            hi_pfc = lim * np.ones((n_pfc,), dtype=float)

        if sol_lim is None:
            lo_sol = -np.inf * np.ones((n_sol,), dtype=float)
            hi_sol = np.inf * np.ones((n_sol,), dtype=float)
        else:
            lim = float(sol_lim)
            lo_sol = -lim * np.ones((n_sol,), dtype=float)
            hi_sol = lim * np.ones((n_sol,), dtype=float)

        lo = np.concatenate([lo_pfc, lo_sol], axis=0)
        hi = np.concatenate([hi_pfc, hi_sol], axis=0)

        if self._last_u is None or self._last_u.shape != (N,):
            u = np.zeros((N,), dtype=float)
        else:
            u = self._last_u.copy()

        eigs = np.linalg.eigvalsh(H)
        L = float(np.max(eigs)) if eigs.size else 1.0
        step = 1.0 / max(L, 1e-12)

        for _ in range(self.pgd_iters):
            grad = H @ u + f
            u = u - step * grad
            u = np.clip(u, lo, hi)

        self._last_u = u.copy()

        return ControlAction(
            pfc_derivs=u[:n_pfc].copy(),
            sol_derivs=u[n_pfc:].copy(),
        )
