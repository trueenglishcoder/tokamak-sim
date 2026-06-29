from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tokamak_control.core.grid import Grid2D
from tokamak_control.core.coils import CoilGroup
from tokamak_control.core.green import (
    build_green_for_coils,
    build_green_for_plasma_center,
    build_green_for_eind,
)
from tokamak_control.core.plasma_state import PlasmaState
from tokamak_control.config.settings import PhysicsSettings
from tokamak_control.io.logger import get_logger
from tokamak_control.io.profiling import Profiler


_PROFILER = Profiler(
    enabled=False,
    summary_every=0,
    logger=get_logger("core.plasma_model.profiling"),
)
_time_block = _PROFILER.time_block


def configure_plasma_model_profiling(
    *,
    enabled: bool,
    summary_every: int = 0,
    reset: bool = True,
) -> None:
    _PROFILER.configure(
        enabled=enabled,
        summary_every=summary_every,
        logger=get_logger("core.plasma_model.profiling"),
        reset=reset,
    )


def plasma_model_profiling_snapshot() -> dict[str, object]:
    return _PROFILER.summary_dict(
        total_key="step_total",
        keys=(
            "step_derive_current_change",
            "step_ip_update",
            "step_compose_psi",
            "step_commit_state",
            "compute_psi",
            "compose_psi",
            "init_green_build",
        ),
        title="plasma_model",
    )


def log_plasma_model_profiling_summary() -> None:
    _PROFILER.log_summary(
        total_key="step_total",
        keys=(
            "step_derive_current_change",
            "step_ip_update",
            "step_compose_psi",
            "step_commit_state",
            "compute_psi",
            "compose_psi",
            "init_green_build",
        ),
        title="plasma_model",
    )


def _clip_currents(currents: np.ndarray, limit: float | None) -> np.ndarray:
    if limit is None:
        return currents
    lim = float(limit)
    if lim <= 0.0:
        return currents
    return np.clip(currents, -lim, lim)


def _clip_derivs(derivs: np.ndarray, limit: float | None) -> np.ndarray:
    if limit is None:
        return derivs
    lim = float(limit)
    if not np.isfinite(lim) or lim < 0.0:
        return derivs
    return np.clip(derivs, -lim, lim)


@dataclass(slots=True)
class PlasmaModel:
    """
    Deterministic, single-process plasma model.

    Each call to ``step_currents`` treats the input as absolute next coil
    currents, derives ``Jdot`` over the time step, and advances ``Ip`` as a
    causal state. The LittleSCOPE coil-drive expression is interpreted as a
    dIp/dt term and integrated over ``t_step`` instead of overwriting Ip.
    Current/derivative limits and actuator lag are metadata for controllers and
    diagnostics, not plant-side clamps.
    """

    grid: Grid2D
    pfc: CoilGroup
    sol: CoilGroup
    R0: float
    Z0: float
    Ip0: float
    mu0: float
    sigma: float
    inductance_L: float
    t_step: float
    ip_coupling_sign: float = -1.0
    plasma_psi_sign: float = 1.0
    actuator_tau: float = 0.0

    pfc_current_limit: float | None = None
    sol_current_limit: float | None = None

    pfc_deriv_limit: float | None = None
    sol_deriv_limit: float | None = None

    ip_coupling_pfc: tuple[float, ...] | None = None
    ip_coupling_sol: tuple[float, ...] | None = None

    _G_pfc: np.ndarray | None = None
    _G_sol: np.ndarray | None = None
    _G_plasma: np.ndarray | None = None
    state: PlasmaState | None = None
    _initial_state: PlasmaState | None = None

    g: np.ndarray | None = None
    g2: np.ndarray | None = None

    @classmethod
    def from_settings(
        cls,
        grid: Grid2D,
        pfc: CoilGroup,
        sol: CoilGroup,
        settings: PhysicsSettings,
        *,
        ip0: float,
    ) -> "PlasmaModel":
        settings.validate()
        return cls(
            grid=grid,
            pfc=pfc,
            sol=sol,
            R0=settings.R0,
            Z0=settings.Z0,
            Ip0=float(ip0),
            mu0=settings.mu0,
            sigma=settings.sigma,
            inductance_L=settings.inductance_L,
            ip_coupling_sign=float(getattr(settings, "ip_coupling_sign", -1.0)),
            plasma_psi_sign=float(getattr(settings, "plasma_psi_sign", 1.0)),
            t_step=settings.t_step,
            actuator_tau=float(settings.actuator_tau),
            pfc_current_limit=settings.pfc_current_limit,
            sol_current_limit=settings.sol_current_limit,
            pfc_deriv_limit=settings.pfc_deriv_limit,
            sol_deriv_limit=settings.sol_deriv_limit,
            ip_coupling_pfc=settings.ip_coupling_pfc,
            ip_coupling_sol=settings.ip_coupling_sol,
        )

    def __post_init__(self) -> None:
        with _time_block("init_green_build"):
            R, Z = self.grid.mesh()
            self._G_pfc = build_green_for_coils(R, Z, self.pfc.element_positions, self.pfc.element_weights) if self.pfc.n_coils else np.zeros((0, *self.grid.shape))
            self._G_sol = build_green_for_coils(R, Z, self.sol.element_positions, self.sol.element_weights) if self.sol.n_coils else np.zeros((0, *self.grid.shape))
            self._G_plasma = build_green_for_plasma_center(R, Z, self.R0, self.Z0)

            self.g = build_green_for_eind(self.R0, self.Z0, self.pfc.element_positions, self.pfc.element_weights) if self.pfc.n_coils else np.zeros((0,), dtype=float)
            self.g2 = build_green_for_eind(self.R0, self.Z0, self.sol.element_positions, self.sol.element_weights) if self.sol.n_coils else np.zeros((0,), dtype=float)

            if self.ip_coupling_pfc is not None:
                g_pfc = np.asarray(self.ip_coupling_pfc, dtype=float).reshape(-1)
                if g_pfc.shape != (self.pfc.n_coils,):
                    raise ValueError(
                        f"ip_coupling_pfc length {g_pfc.size} != number of PFC actuators {self.pfc.n_coils}"
                    )
                self.g = g_pfc.copy()

            if self.ip_coupling_sol is not None:
                g_sol = np.asarray(self.ip_coupling_sol, dtype=float).reshape(-1)
                if g_sol.shape != (self.sol.n_coils,):
                    raise ValueError(
                        f"ip_coupling_sol length {g_sol.size} != number of SOL actuators {self.sol.n_coils}"
                    )
                self.g2 = g_sol.copy()

        pfc0 = np.asarray(self.pfc.initial_currents, dtype=float).copy()
        sol0 = np.asarray(self.sol.initial_currents, dtype=float).copy()
        psi0 = self._compose_psi(self.Ip0, pfc0, sol0)

        self.state = PlasmaState(
            t=0.0,
            step=0,
            Ip=self.Ip0,
            Ip0=self.Ip0,
            psi=psi0,
            pfc_currents=pfc0.copy(),
            pfc_current_derivs=np.zeros_like(pfc0),
            sol_currents=sol0.copy(),
            sol_current_derivs=np.zeros_like(sol0),
        )
        self._initial_state = self.state.copied()

    def time_constant(self) -> float:
        """Return the passive Ip loss time constant."""
        return float(self.sigma * self.inductance_L)

    def decay_factor(self, dt: float | None = None) -> float:
        """Return the one-step explicit-Euler passive Ip multiplier."""
        step = float(self.t_step if dt is None else dt)
        return float(1.0 - step / max(self.time_constant(), 1.0e-30))

    def ip_decay_baseline_at(self, t: float, *, Ip0: float | None = None) -> float:
        """Return the continuous passive-loss baseline for diagnostics."""
        ip0 = float(self.Ip0 if Ip0 is None else Ip0)
        return float(ip0 * np.exp(-float(t) / max(self.time_constant(), 1.0e-30)))

    def predict_Ip_decay_baseline_next(self) -> float:
        """Return next-step passive-only Ip from the current state."""
        s = self._require_state()
        return float(s.Ip) * self.decay_factor()

    def get_ip_B_row(self) -> np.ndarray:
        """Return one-step Ip sensitivity to derivative commands [PFC..., SOL...]."""
        gp = self.g if self.g is not None else np.zeros((0,), dtype=float)
        gs = self.g2 if self.g2 is not None else np.zeros((0,), dtype=float)
        scale = float(self.ip_coupling_sign) * (float(self.mu0) * float(self.sigma) / float(self.R0))
        return float(self.t_step) * scale * np.concatenate([gp, gs])

    def _advance_ip(self, ip_now: float, pfc_jdot: np.ndarray, sol_jdot: np.ndarray) -> float:
        """Advance plasma current by integrating passive decay plus coil drive."""
        gp = self.g if self.g is not None else np.zeros((0,), dtype=float)
        gs = self.g2 if self.g2 is not None else np.zeros((0,), dtype=float)
        drive = 0.0
        if gp.size:
            drive += float(np.dot(gp, np.asarray(pfc_jdot, dtype=float)))
        if gs.size:
            drive += float(np.dot(gs, np.asarray(sol_jdot, dtype=float)))
        scale = float(self.ip_coupling_sign) * (float(self.mu0) * float(self.sigma) / float(self.R0))
        d_ip_dt = -float(ip_now) / max(self.time_constant(), 1.0e-30) + scale * drive
        return float(float(ip_now) + float(self.t_step) * d_ip_dt)

    def snapshot_state(self) -> PlasmaState:
        return self._require_state().copied()

    def restore_state(self, state: PlasmaState) -> PlasmaState:
        self._validate_state_compatibility(state)
        restored = state.copied()
        restored.psi = self._compose_psi(restored.Ip, restored.pfc_currents, restored.sol_currents)
        self.Ip0 = float(restored.Ip0)
        self.state = restored
        return self.state

    def reset_state(self) -> PlasmaState:
        if self._initial_state is None:
            raise RuntimeError("PlasmaModel initial state is not initialized")
        return self.restore_state(self._initial_state)

    def compute_psi(self) -> np.ndarray:
        with _time_block("compute_psi"):
            s = self._require_state()
            return self._compose_psi(s.Ip, s.pfc_currents, s.sol_currents)

    def _actuator_alpha(self) -> float:
        tau = float(self.actuator_tau)
        if tau <= 0.0:
            return 0.0
        return float(np.exp(-float(self.t_step) / tau))

    def step_currents(
        self,
        pfc_currents_next: np.ndarray | None = None,
        sol_currents_next: np.ndarray | None = None,
    ) -> PlasmaState:
        """Advance one old-parity step from absolute next coil currents."""
        with _time_block("step_total"):
            s = self._require_state()
            dt = float(self.t_step)
            if dt <= 0.0:
                raise ValueError(f"t_step must be > 0, got {dt!r}")

            with _time_block("step_derive_current_change"):
                prev_pfc = np.asarray(s.pfc_currents, dtype=float)
                prev_sol = np.asarray(s.sol_currents, dtype=float)
                next_pfc = prev_pfc.copy() if pfc_currents_next is None else np.asarray(pfc_currents_next, dtype=float).reshape(self.pfc.n_coils)
                next_sol = prev_sol.copy() if sol_currents_next is None else np.asarray(sol_currents_next, dtype=float).reshape(self.sol.n_coils)
                applied_pfc = (next_pfc - prev_pfc) / dt
                applied_sol = (next_sol - prev_sol) / dt

            with _time_block("step_ip_update"):
                t_next = float(s.t) + dt
                ip_next = self._advance_ip(float(s.Ip), applied_pfc, applied_sol)

            with _time_block("step_compose_psi"):
                psi_next = self._compose_psi(ip_next, next_pfc, next_sol)

            with _time_block("step_commit_state"):
                self.state = PlasmaState(
                    t=t_next,
                    step=int(s.step) + 1,
                    Ip=ip_next,
                    Ip0=float(s.Ip0),
                    psi=psi_next,
                    pfc_currents=next_pfc,
                    pfc_current_derivs=applied_pfc,
                    sol_currents=next_sol,
                    sol_current_derivs=applied_sol,
                )
            _PROFILER.step()
            return self.state

    def step(self, *args: object, **kwargs: object) -> PlasmaState:
        """Reject the removed derivative-command API."""
        raise RuntimeError("PlasmaModel.step() was removed; use step_currents(J_next) with absolute next currents")

    def _compose_psi(self, Ip: float, pfc_currents: np.ndarray, sol_currents: np.ndarray) -> np.ndarray:
        with _time_block("compose_psi"):
            if self._G_plasma is None or self._G_pfc is None or self._G_sol is None:
                raise RuntimeError("Green arrays are not initialized")

            psi = float(self.plasma_psi_sign) * float(Ip) * self._G_plasma.copy()
            if self._G_pfc.shape[0]:
                psi += np.tensordot(np.asarray(pfc_currents, dtype=float), self._G_pfc, axes=(0, 0))
            if self._G_sol.shape[0]:
                psi += np.tensordot(np.asarray(sol_currents, dtype=float), self._G_sol, axes=(0, 0))
            return self.mu0 * np.asarray(psi, dtype=float)

    def _require_state(self) -> PlasmaState:
        if self.state is None:
            raise RuntimeError("PlasmaModel state is not initialized")
        return self.state

    def _validate_state_compatibility(self, state: PlasmaState) -> None:
        if np.asarray(state.psi).shape != self.grid.shape:
            raise ValueError(f"state.psi shape {np.asarray(state.psi).shape} != {self.grid.shape}")
        if np.asarray(state.pfc_currents).shape != (self.pfc.n_coils,):
            raise ValueError(f"state.pfc_currents shape {np.asarray(state.pfc_currents).shape} != ({self.pfc.n_coils},)")
        if np.asarray(state.pfc_current_derivs).shape != (self.pfc.n_coils,):
            raise ValueError(f"state.pfc_current_derivs shape {np.asarray(state.pfc_current_derivs).shape} != ({self.pfc.n_coils},)")
        if np.asarray(state.sol_currents).shape != (self.sol.n_coils,):
            raise ValueError(f"state.sol_currents shape {np.asarray(state.sol_currents).shape} != ({self.sol.n_coils},)")
        if np.asarray(state.sol_current_derivs).shape != (self.sol.n_coils,):
            raise ValueError(f"state.sol_current_derivs shape {np.asarray(state.sol_current_derivs).shape} != ({self.sol.n_coils},)")

    def sample_green_pfc(self, points: np.ndarray) -> np.ndarray:
        if self._G_pfc is None:
            raise RuntimeError("PFC Green array not initialized")
        M = points.shape[0]
        n = self._G_pfc.shape[0]
        if n == 0:
            return np.zeros((M, 0), dtype=float)
        vals = np.empty((M, n), dtype=float)
        for i in range(n):
            vals[:, i] = self._bilinear_sample_slice(self._G_pfc[i], points)
        return vals

    def sample_green_sol(self, points: np.ndarray) -> np.ndarray:
        if self._G_sol is None:
            raise RuntimeError("SOL Green array not initialized")
        M = points.shape[0]
        n = self._G_sol.shape[0]
        if n == 0:
            return np.zeros((M, 0), dtype=float)
        vals = np.empty((M, n), dtype=float)
        for i in range(n):
            vals[:, i] = self._bilinear_sample_slice(self._G_sol[i], points)
        return vals

    def sample_green_plasma(self, points: np.ndarray) -> np.ndarray:
        if self._G_plasma is None:
            raise RuntimeError("Plasma Green array not initialized")
        return self._bilinear_sample_slice(self._G_plasma, points)

    def _bilinear_sample_slice(self, field2d: np.ndarray, points: np.ndarray) -> np.ndarray:
        R0 = float(self.grid.r.coords()[0])
        Z0 = float(self.grid.z.coords()[0])
        dR = self.grid.r.step
        dZ = self.grid.z.step
        NR = self.grid.r.size
        NZ = self.grid.z.size

        vals = np.empty(points.shape[0], dtype=float)
        for k, (R, Z) in enumerate(points):
            u = (R - R0) / dR
            v = (Z - Z0) / dZ
            i0 = int(np.floor(u))
            j0 = int(np.floor(v))
            i1 = i0 + 1
            j1 = j0 + 1
            if i0 < 0 or j0 < 0 or i1 >= NR or j1 >= NZ:
                vals[k] = np.nan
                continue
            du = u - i0
            dv = v - j0
            q00 = field2d[j0, i0]
            q10 = field2d[j0, i1]
            q01 = field2d[j1, i0]
            q11 = field2d[j1, i1]
            q0 = (1.0 - du) * q00 + du * q10
            q1 = (1.0 - du) * q01 + du * q11
            vals[k] = (1.0 - dv) * q0 + dv * q1
        return vals
