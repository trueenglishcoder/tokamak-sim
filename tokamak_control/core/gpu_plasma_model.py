from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tokamak_control.config.settings import PhysicsSettings
from tokamak_control.core.coils import CoilGroup
from tokamak_control.core.green import (
    build_green_for_coils,
    build_green_for_eind,
    build_green_for_plasma_center,
)
from tokamak_control.core.grid import Grid2D
from tokamak_control.core.plasma_state import PlasmaState
from tokamak_control.io.logger import get_logger
from tokamak_control.io.profiling import Profiler


_PROFILER = Profiler(
    enabled=False,
    summary_every=0,
    logger=get_logger("core.gpu_plasma_model.profiling"),
)
_time_block = _PROFILER.time_block


def configure_gpu_plasma_model_profiling(
    *,
    enabled: bool,
    summary_every: int = 0,
    reset: bool = True,
) -> None:
    _PROFILER.configure(
        enabled=enabled,
        summary_every=summary_every,
        logger=get_logger("core.gpu_plasma_model.profiling"),
        reset=reset,
    )


def gpu_plasma_model_profiling_snapshot() -> dict[str, object]:
    return _PROFILER.summary_dict(
        total_key="step_total",
        keys=(
            "step_to_device",
            "step_clip_cmd",
            "step_actuator_lag",
            "step_integrate_currents",
            "step_ip_update",
            "step_compose_psi",
            "step_commit_state",
            "compute_psi",
            "compose_psi",
            "to_cpu_state",
            "init_green_build",
            "init_tensor_upload",
        ),
        title="gpu_plasma_model",
    )


def log_gpu_plasma_model_profiling_summary() -> None:
    _PROFILER.log_summary(
        total_key="step_total",
        keys=(
            "step_to_device",
            "step_clip_cmd",
            "step_actuator_lag",
            "step_integrate_currents",
            "step_ip_update",
            "step_compose_psi",
            "step_commit_state",
            "compute_psi",
            "compose_psi",
            "to_cpu_state",
            "init_green_build",
            "init_tensor_upload",
        ),
        title="gpu_plasma_model",
    )


@dataclass(slots=True)
class GpuPlasmaModel:
    """Torch/CUDA plant model with the same public contract as ``PlasmaModel``."""

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
    gpu_device: str = "cuda:0"
    ip_coupling_sign: float = -1.0
    plasma_psi_sign: float = 1.0
    actuator_tau: float = 0.0
    pfc_current_limit: float | None = None
    sol_current_limit: float | None = None
    pfc_deriv_limit: float | None = None
    sol_deriv_limit: float | None = None
    ip_coupling_pfc: tuple[float, ...] | None = None
    ip_coupling_sol: tuple[float, ...] | None = None

    torch: Any | None = None
    device: Any | None = None
    _G_pfc: Any | None = None
    _G_sol: Any | None = None
    _G_plasma: Any | None = None
    g: Any | None = None
    g2: Any | None = None
    state: PlasmaState | None = None
    _initial_state: PlasmaState | None = None

    _t: float = 0.0
    _step: int = 0
    _Ip: Any | None = None
    _Ip0_tensor: Any | None = None
    _psi: Any | None = None
    _pfc_currents: Any | None = None
    _pfc_current_derivs: Any | None = None
    _sol_currents: Any | None = None
    _sol_current_derivs: Any | None = None

    @classmethod
    def from_settings(
        cls,
        grid: Grid2D,
        pfc: CoilGroup,
        sol: CoilGroup,
        settings: PhysicsSettings,
        *,
        gpu_device: str = "cuda:0",
    ) -> "GpuPlasmaModel":
        settings.validate()
        return cls(
            grid=grid,
            pfc=pfc,
            sol=sol,
            R0=settings.R0,
            Z0=settings.Z0,
            Ip0=settings.Ip0,
            mu0=settings.mu0,
            sigma=settings.sigma,
            inductance_L=settings.inductance_L,
            t_step=settings.t_step,
            gpu_device=str(gpu_device),
            ip_coupling_sign=float(getattr(settings, "ip_coupling_sign", -1.0)),
            plasma_psi_sign=float(getattr(settings, "plasma_psi_sign", 1.0)),
            actuator_tau=float(settings.actuator_tau),
            pfc_current_limit=settings.pfc_current_limit,
            sol_current_limit=settings.sol_current_limit,
            pfc_deriv_limit=settings.pfc_deriv_limit,
            sol_deriv_limit=settings.sol_deriv_limit,
            ip_coupling_pfc=settings.ip_coupling_pfc,
            ip_coupling_sol=settings.ip_coupling_sol,
        )

    def __post_init__(self) -> None:
        self.torch, self.device = _require_torch(self.gpu_device)
        with _time_block("init_green_build"):
            R, Z = self.grid.mesh()
            G_pfc = build_green_for_coils(R, Z, self.pfc.element_positions, self.pfc.element_weights) if self.pfc.n_coils else np.zeros((0, *self.grid.shape))
            G_sol = build_green_for_coils(R, Z, self.sol.element_positions, self.sol.element_weights) if self.sol.n_coils else np.zeros((0, *self.grid.shape))
            G_plasma = build_green_for_plasma_center(R, Z, self.R0, self.Z0)
            g = build_green_for_eind(self.R0, self.Z0, self.pfc.element_positions, self.pfc.element_weights) if self.pfc.n_coils else np.zeros((0,), dtype=float)
            g2 = build_green_for_eind(self.R0, self.Z0, self.sol.element_positions, self.sol.element_weights) if self.sol.n_coils else np.zeros((0,), dtype=float)

            if self.ip_coupling_pfc is not None:
                g_pfc = np.asarray(self.ip_coupling_pfc, dtype=float).reshape(-1)
                if g_pfc.shape != (self.pfc.n_coils,):
                    raise ValueError(f"ip_coupling_pfc length {g_pfc.size} != number of PFC actuators {self.pfc.n_coils}")
                g = g_pfc.copy()
            if self.ip_coupling_sol is not None:
                g_sol = np.asarray(self.ip_coupling_sol, dtype=float).reshape(-1)
                if g_sol.shape != (self.sol.n_coils,):
                    raise ValueError(f"ip_coupling_sol length {g_sol.size} != number of SOL actuators {self.sol.n_coils}")
                g2 = g_sol.copy()

        with _time_block("init_tensor_upload"):
            self._G_pfc = self._tensor(G_pfc)
            self._G_sol = self._tensor(G_sol)
            self._G_plasma = self._tensor(G_plasma)
            self.g = self._tensor(g)
            self.g2 = self._tensor(g2)

            pfc0 = self._clip_currents_t(self._tensor(self.pfc.initial_currents), self.pfc_current_limit)
            sol0 = self._clip_currents_t(self._tensor(self.sol.initial_currents), self.sol_current_limit)
            self._t = 0.0
            self._step = 0
            self._Ip = self._scalar(float(self.Ip0))
            self._Ip0_tensor = self._scalar(float(self.Ip0))
            self._pfc_currents = pfc0
            self._sol_currents = sol0
            self._pfc_current_derivs = self._zeros((self.pfc.n_coils,))
            self._sol_current_derivs = self._zeros((self.sol.n_coils,))
            self._psi = self._compose_psi_t(self._Ip, self._pfc_currents, self._sol_currents)

        self.state = self._cpu_state()
        self._initial_state = self.state.copied()

    @property
    def psi_tensor(self):
        if self._psi is None:
            raise RuntimeError("GpuPlasmaModel psi tensor is not initialized")
        return self._psi

    def time_constant(self) -> float:
        return float(self.sigma * self.inductance_L)

    def decay_factor(self, dt: float | None = None) -> float:
        tau = max(self.time_constant(), 1e-30)
        step = float(self.t_step if dt is None else dt)
        return float(np.exp(-step / tau))

    def ip_decay_baseline_at(self, t: float, *, Ip0: float | None = None) -> float:
        tau = max(self.time_constant(), 1e-30)
        ip0 = float(self.Ip0 if Ip0 is None else Ip0)
        return ip0 * float(np.exp(-float(t) / tau))

    def predict_Ip_decay_baseline_next(self) -> float:
        return float(self._require_ip().detach().cpu()) * self.decay_factor(float(self.t_step))

    def get_ip_B_row(self) -> np.ndarray:
        gp = self._to_numpy(self.g) if self.g is not None else np.zeros((0,), dtype=float)
        gs = self._to_numpy(self.g2) if self.g2 is not None else np.zeros((0,), dtype=float)
        return (
            float(self.t_step)
            * float(self.ip_coupling_sign)
            * (self.mu0 * self.sigma / self.R0)
            * np.concatenate([gp, gs])
        )

    def snapshot_state(self) -> PlasmaState:
        if self.state is None or int(self.state.step) != int(self._step):
            self.state = self._cpu_state()
        return self.state.copied()

    def restore_state(self, state: PlasmaState) -> PlasmaState:
        self._validate_state_compatibility(state)
        self._t = float(state.t)
        self._step = int(state.step)
        self.Ip0 = float(state.Ip0)
        self._Ip = self._scalar(float(state.Ip))
        self._Ip0_tensor = self._scalar(float(state.Ip0))
        self._pfc_currents = self._tensor(state.pfc_currents)
        self._pfc_current_derivs = self._tensor(state.pfc_current_derivs)
        self._sol_currents = self._tensor(state.sol_currents)
        self._sol_current_derivs = self._tensor(state.sol_current_derivs)
        self._psi = self._compose_psi_t(self._Ip, self._pfc_currents, self._sol_currents)
        self.state = self._cpu_state()
        return self.state

    def reset_state(self) -> PlasmaState:
        if self._initial_state is None:
            raise RuntimeError("GpuPlasmaModel initial state is not initialized")
        return self.restore_state(self._initial_state)

    def compute_psi(self) -> np.ndarray:
        with _time_block("compute_psi"):
            self._psi = self._compose_psi_t(self._require_ip(), self._require_pfc_currents(), self._require_sol_currents())
            return self._to_numpy(self._psi)

    def compute_psi_tensor(self):
        with _time_block("compute_psi"):
            self._psi = self._compose_psi_t(self._require_ip(), self._require_pfc_currents(), self._require_sol_currents())
            return self._psi

    def step(
        self,
        pfc_current_derivs: np.ndarray | None = None,
        sol_current_derivs: np.ndarray | None = None,
    ) -> PlasmaState:
        torch = self._torch()
        with _time_block("step_total"):
            dt = float(self.t_step)
            if dt <= 0.0:
                raise ValueError(f"t_step must be > 0, got {dt!r}")

            with _time_block("step_to_device"):
                cmd_pfc = self._zeros((self.pfc.n_coils,)) if pfc_current_derivs is None else self._tensor(np.asarray(pfc_current_derivs, dtype=float).reshape(self.pfc.n_coils))
                cmd_sol = self._zeros((self.sol.n_coils,)) if sol_current_derivs is None else self._tensor(np.asarray(sol_current_derivs, dtype=float).reshape(self.sol.n_coils))

            with _time_block("step_clip_cmd"):
                cmd_pfc = self._clip_derivs_t(cmd_pfc, self.pfc_deriv_limit)
                cmd_sol = self._clip_derivs_t(cmd_sol, self.sol_deriv_limit)

            with _time_block("step_actuator_lag"):
                alpha = self._actuator_alpha()
                prev_pfc_derivs = self._require_pfc_derivs()
                prev_sol_derivs = self._require_sol_derivs()
                applied_pfc = alpha * prev_pfc_derivs + (1.0 - alpha) * cmd_pfc
                applied_sol = alpha * prev_sol_derivs + (1.0 - alpha) * cmd_sol
                applied_pfc = self._clip_derivs_t(applied_pfc, self.pfc_deriv_limit)
                applied_sol = self._clip_derivs_t(applied_sol, self.sol_deriv_limit)

            with _time_block("step_integrate_currents"):
                prev_pfc = self._require_pfc_currents()
                prev_sol = self._require_sol_currents()
                next_pfc = self._clip_currents_t(prev_pfc + dt * applied_pfc, self.pfc_current_limit)
                next_sol = self._clip_currents_t(prev_sol + dt * applied_sol, self.sol_current_limit)
                delta_pfc = next_pfc - prev_pfc
                delta_sol = next_sol - prev_sol

            with _time_block("step_ip_update"):
                ip_base = self._require_ip() * self.decay_factor(dt)
                control_term = self._scalar(0.0)
                if int(self.g.numel()) > 0:
                    control_term = control_term + torch.dot(self.g, delta_pfc)
                if int(self.g2.numel()) > 0:
                    control_term = control_term + torch.dot(self.g2, delta_sol)
                ip_next = ip_base + (float(self.ip_coupling_sign) * (self.mu0 * self.sigma / self.R0)) * control_term

            with _time_block("step_compose_psi"):
                psi_next = self._compose_psi_t(ip_next, next_pfc, next_sol)

            with _time_block("step_commit_state"):
                self._t = float(self._t) + dt
                self._step = int(self._step) + 1
                self._Ip = ip_next
                self._pfc_currents = next_pfc
                self._pfc_current_derivs = applied_pfc
                self._sol_currents = next_sol
                self._sol_current_derivs = applied_sol
                self._psi = psi_next
                self.state = self._cpu_state()
            _PROFILER.step()
            return self.state

    def _actuator_alpha(self) -> float:
        tau = float(self.actuator_tau)
        if tau <= 0.0:
            return 0.0
        return float(np.exp(-float(self.t_step) / tau))

    def _compose_psi_t(self, Ip, pfc_currents, sol_currents):
        torch = self._torch()
        with _time_block("compose_psi"):
            if self._G_plasma is None or self._G_pfc is None or self._G_sol is None:
                raise RuntimeError("Green arrays are not initialized")
            psi = float(self.plasma_psi_sign) * Ip * self._G_plasma
            if int(self._G_pfc.shape[0]):
                psi = psi + torch.tensordot(pfc_currents, self._G_pfc, dims=([0], [0]))
            if int(self._G_sol.shape[0]):
                psi = psi + torch.tensordot(sol_currents, self._G_sol, dims=([0], [0]))
            return float(self.mu0) * psi

    def _cpu_state(self) -> PlasmaState:
        with _time_block("to_cpu_state"):
            return PlasmaState(
                t=float(self._t),
                step=int(self._step),
                Ip=float(self._require_ip().detach().cpu()),
                Ip0=float(self.Ip0),
                psi=self._to_numpy(self.psi_tensor),
                pfc_currents=self._to_numpy(self._require_pfc_currents()),
                pfc_current_derivs=self._to_numpy(self._require_pfc_derivs()),
                sol_currents=self._to_numpy(self._require_sol_currents()),
                sol_current_derivs=self._to_numpy(self._require_sol_derivs()),
            )

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

    def _clip_currents_t(self, currents, limit: float | None):
        if limit is None:
            return currents
        lim = float(limit)
        if lim <= 0.0:
            return currents
        return self._torch().clamp(currents, -lim, lim)

    def _clip_derivs_t(self, derivs, limit: float | None):
        if limit is None:
            return derivs
        lim = float(limit)
        if not np.isfinite(lim) or lim < 0.0:
            return derivs
        return self._torch().clamp(derivs, -lim, lim)

    def _tensor(self, value):
        return self._torch().as_tensor(value, dtype=self._torch().float64, device=self.device)

    def _scalar(self, value: float):
        return self._torch().tensor(float(value), dtype=self._torch().float64, device=self.device)

    def _zeros(self, shape: tuple[int, ...]):
        return self._torch().zeros(shape, dtype=self._torch().float64, device=self.device)

    def _to_numpy(self, tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(float, copy=True)

    def _torch(self):
        if self.torch is None:
            raise RuntimeError("Torch runtime is not initialized")
        return self.torch

    def _require_ip(self):
        if self._Ip is None:
            raise RuntimeError("Ip tensor is not initialized")
        return self._Ip

    def _require_pfc_currents(self):
        if self._pfc_currents is None:
            raise RuntimeError("PFC currents tensor is not initialized")
        return self._pfc_currents

    def _require_pfc_derivs(self):
        if self._pfc_current_derivs is None:
            raise RuntimeError("PFC derivative tensor is not initialized")
        return self._pfc_current_derivs

    def _require_sol_currents(self):
        if self._sol_currents is None:
            raise RuntimeError("SOL currents tensor is not initialized")
        return self._sol_currents

    def _require_sol_derivs(self):
        if self._sol_current_derivs is None:
            raise RuntimeError("SOL derivative tensor is not initialized")
        return self._sol_current_derivs


def _require_torch(gpu_device: str):
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("GPU plant backend requires tokamak-sim[gpu] with torch installed") from exc

    device = torch.device(str(gpu_device))
    if device.type != "cuda":
        raise RuntimeError(f"GPU plant backend requires a CUDA device, got {gpu_device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU plant backend requested, but torch.cuda.is_available() is False")
    try:
        torch.empty((1,), device=device)
    except Exception as exc:  # pragma: no cover - depends on host CUDA setup
        raise RuntimeError(f"GPU plant backend could not initialize device {gpu_device!r}") from exc
    return torch, device

