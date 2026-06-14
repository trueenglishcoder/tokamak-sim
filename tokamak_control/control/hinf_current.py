from __future__ import annotations

import numpy as np

from tokamak_control.control.base import ControlAction, Controller


class HinftyCurrentController(Controller):
    """
    Ip-only one-step H∞ controller.

    This is the robust analogue of LQRCurrentController. It treats the scalar
    Ip tracking error as the regulated output and solves a one-step minimax
    problem using the same local discrete derivative-input row used by the
    LQR current controller.

    Disturbance enters the regulated output through E, default identity.
    As gamma grows large, the controller approaches the one-step LQR solution.
    """

    def __init__(
        self,
        *,
        q_ip: float = 1.0,
        r_pfc: float = 1e-10,
        r_sol: float = 1e-10,
        gamma: float = 10.0,
        ridge: float = 1e-14,
        u_clip: float | None = None,
        ip_ref: float | None = None,
    ) -> None:
        for name, value in (
            ("q_ip", q_ip),
            ("r_pfc", r_pfc),
            ("r_sol", r_sol),
            ("gamma", gamma),
            ("ridge", ridge),
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

        self.q_ip = float(q_ip)
        self.r_pfc = float(r_pfc)
        self.r_sol = float(r_sol)
        self.gamma = float(gamma)
        self.ridge = float(ridge)
        self.u_clip = u_clip
        self.ip_ref = ip_ref
        self._last_gain: np.ndarray | None = None

    def reset(self) -> None:
        self._last_gain = None

    def compute_control(
        self,
        *,
        model,
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

        ip_target = (
            float(Ip_ref)
            if Ip_ref is not None
            else (float(self.ip_ref) if self.ip_ref is not None else float(model.Ip0))
        )
        ip_next0 = float(model.predict_Ip_decay_baseline_next())
        e = float(ip_target - ip_next0)

        B_ip = np.asarray(model.get_ip_B_row(), dtype=float).reshape(-1)
        if B_ip.size != n_u:
            raise ValueError(
                f"model.get_ip_B_row returned size {B_ip.size}, expected {n_u} "
                f"for {n_pfc} PFC and {n_sol} SOL inputs"
            )

        # Same one-step derivative-input convention as LQRCurrentController:
        # Ip_{k+1} ≈ ip_next0 + B_ip u, so target error is e - B_ip u.
        B = B_ip.reshape(1, -1)

        Q = self.q_ip * np.eye(1, dtype=float)

        R = np.zeros((n_u, n_u), dtype=float)
        if n_pfc:
            R[:n_pfc, :n_pfc] = self.r_pfc * np.eye(n_pfc)
        if n_sol:
            R[n_pfc:, n_pfc:] = self.r_sol * np.eye(n_sol)

        if E is None:
            E = np.eye(1, dtype=float)
        else:
            E = np.asarray(E, dtype=float)
            if E.ndim != 2 or E.shape[0] != 1:
                raise ValueError(f"E must have shape (1, M_d), got {E.shape!r}")

        EtQE = E.T @ Q @ E
        gamma_mat = self.gamma**2 * np.eye(EtQE.shape[0], dtype=float) - EtQE
        evals = np.linalg.eigvalsh(gamma_mat)
        if float(np.min(evals)) <= 0.0:
            raise ValueError(
                "HinftyCurrentController infeasible: gamma^2 I - E^T Q E is not positive definite. "
                "Increase gamma or reduce q_ip."
            )

        Q_hat = Q + Q @ E @ np.linalg.solve(gamma_mat, E.T @ Q)

        H = B.T @ Q_hat @ B + R + self.ridge * np.eye(n_u)
        G = B.T @ Q_hat
        K = np.linalg.solve(H, G)
        self._last_gain = K

        u = (K @ np.asarray([e], dtype=float)).reshape(-1)

        if self.u_clip is not None:
            u = np.clip(u, -self.u_clip, self.u_clip)

        u_pfc = u[:n_pfc] if n_pfc else np.zeros((0,), dtype=float)
        u_sol = u[n_pfc:] if n_sol else np.zeros((0,), dtype=float)
        return ControlAction(pfc_derivs=u_pfc, sol_derivs=u_sol)
