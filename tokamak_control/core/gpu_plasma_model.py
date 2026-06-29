from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokamak_control.compute import require_gpu_available
from tokamak_control.config.settings import PhysicsSettings
from tokamak_control.core.coils import CoilGroup
from tokamak_control.core.green import build_green_for_coils, build_green_for_eind, build_green_for_plasma_center
from tokamak_control.core.grid import Grid2D
from tokamak_control.core.plasma_state import PlasmaState


def _torch(device: str):
    require_gpu_available(device)
    import torch
    return torch


@dataclass(slots=True)
class GpuPlasmaModel:
    """Torch/CUDA mirror of PlasmaModel for explicit GPU backend runs."""

    grid: Grid2D
    pfc: CoilGroup
    sol: CoilGroup
    settings: PhysicsSettings
    ip0: float
    gpu_device: str = "cuda:0"

    def __post_init__(self) -> None:
        torch = _torch(self.gpu_device)
        self.torch = torch
        self.device = torch.device(self.gpu_device)
        self.dtype = torch.float64
        self.R0 = float(self.settings.R0)
        self.Z0 = float(self.settings.Z0)
        self.Ip0 = float(self.ip0)
        self.mu0 = float(self.settings.mu0)
        self.sigma = float(self.settings.sigma)
        self.inductance_L = float(self.settings.inductance_L)
        self.t_step = float(self.settings.t_step)
        self.ip_coupling_sign = float(getattr(self.settings, "ip_coupling_sign", -1.0))
        self.plasma_psi_sign = float(getattr(self.settings, "plasma_psi_sign", 1.0))
        self.actuator_tau = float(self.settings.actuator_tau)
        R, Z = self.grid.mesh()
        self._G_pfc = torch.as_tensor(build_green_for_coils(R, Z, self.pfc.element_positions, self.pfc.element_weights) if self.pfc.n_coils else np.zeros((0, *self.grid.shape)), dtype=self.dtype, device=self.device)
        self._G_sol = torch.as_tensor(build_green_for_coils(R, Z, self.sol.element_positions, self.sol.element_weights) if self.sol.n_coils else np.zeros((0, *self.grid.shape)), dtype=self.dtype, device=self.device)
        self._G_plasma = torch.as_tensor(build_green_for_plasma_center(R, Z, self.R0, self.Z0), dtype=self.dtype, device=self.device)
        gp = build_green_for_eind(self.R0, self.Z0, self.pfc.element_positions, self.pfc.element_weights) if self.pfc.n_coils else np.zeros((0,), dtype=float)
        gs = build_green_for_eind(self.R0, self.Z0, self.sol.element_positions, self.sol.element_weights) if self.sol.n_coils else np.zeros((0,), dtype=float)
        if self.settings.ip_coupling_pfc is not None:
            gp = np.asarray(self.settings.ip_coupling_pfc, dtype=float).reshape(-1)
        if self.settings.ip_coupling_sol is not None:
            gs = np.asarray(self.settings.ip_coupling_sol, dtype=float).reshape(-1)
        self.g = torch.as_tensor(gp, dtype=self.dtype, device=self.device)
        self.g2 = torch.as_tensor(gs, dtype=self.dtype, device=self.device)
        pfc0 = torch.as_tensor(self.pfc.initial_currents, dtype=self.dtype, device=self.device)
        sol0 = torch.as_tensor(self.sol.initial_currents, dtype=self.dtype, device=self.device)
        ip0 = torch.tensor(float(self.Ip0), dtype=self.dtype, device=self.device)
        self._ip = ip0
        self._pfc = pfc0.clone()
        self._sol = sol0.clone()
        self._pfc_deriv = torch.zeros_like(pfc0)
        self._sol_deriv = torch.zeros_like(sol0)
        self._step = 0
        self._time = 0.0
        self._ip_origin = float(self.Ip0)
        self.state = self._snapshot_state()

    @classmethod
    def from_settings(cls, grid: Grid2D, pfc: CoilGroup, sol: CoilGroup, settings: PhysicsSettings, *, ip0: float, gpu_device: str = "cuda:0") -> "GpuPlasmaModel":
        settings.validate()
        return cls(grid=grid, pfc=pfc, sol=sol, settings=settings, ip0=float(ip0), gpu_device=gpu_device)

    def _clip_currents(self, values, limit: float | None):
        if limit is None or float(limit) <= 0.0:
            return values
        return self.torch.clamp(values, -float(limit), float(limit))

    def _clip_derivs(self, values, limit: float | None):
        if limit is None or not np.isfinite(float(limit)) or float(limit) < 0.0:
            return values
        return self.torch.clamp(values, -float(limit), float(limit))

    def _actuator_alpha(self) -> float:
        tau = float(self.actuator_tau)
        if tau <= 0.0:
            return 0.0
        return float(np.exp(-float(self.t_step) / tau))

    def decay_factor(self, dt: float | None = None) -> float:
        """Return the one-step explicit-Euler passive Ip multiplier."""
        tau = max(float(self.sigma * self.inductance_L), 1.0e-30)
        return float(1.0 - float(self.t_step if dt is None else dt) / tau)

    def predict_Ip_decay_baseline_next(self) -> float:
        """Return next-step passive-only Ip from the current state."""
        return float(self._ip.detach().cpu().item()) * self.decay_factor()

    def get_ip_B_row(self) -> np.ndarray:
        """Return one-step Ip sensitivity to derivative commands [PFC..., SOL...]."""
        scale = float(self.ip_coupling_sign) * (float(self.mu0) * float(self.sigma) / float(self.R0))
        row = float(self.t_step) * scale * self.torch.cat([self.g, self.g2])
        return row.detach().cpu().numpy().astype(float)

    def _compose_psi_tensor(self, ip, pfc, sol):
        psi = float(self.plasma_psi_sign) * ip * self._G_plasma
        if self._G_pfc.shape[0]:
            psi = psi + self.torch.einsum("i,izr->zr", pfc, self._G_pfc)
        if self._G_sol.shape[0]:
            psi = psi + self.torch.einsum("i,izr->zr", sol, self._G_sol)
        return float(self.mu0) * psi

    def compute_psi_tensor(self):
        return self._compose_psi_tensor(self._ip, self._pfc, self._sol)

    def compute_psi(self) -> np.ndarray:
        return self.compute_psi_tensor().detach().cpu().numpy().astype(float)

    def step_currents(self, pfc_currents_next=None, sol_currents_next=None) -> PlasmaState:
        """Advance one old-parity step from absolute next coil currents."""
        torch = self.torch
        next_pfc = self._pfc.clone() if pfc_currents_next is None else torch.as_tensor(pfc_currents_next, dtype=self.dtype, device=self.device).reshape(self.pfc.n_coils)
        next_sol = self._sol.clone() if sol_currents_next is None else torch.as_tensor(sol_currents_next, dtype=self.dtype, device=self.device).reshape(self.sol.n_coils)
        applied_pfc = (next_pfc - self._pfc) / float(self.t_step)
        applied_sol = (next_sol - self._sol) / float(self.t_step)
        drive = torch.as_tensor(0.0, dtype=self.dtype, device=self.device)
        if self.g.numel():
            drive = drive + self.torch.dot(self.g, applied_pfc)
        if self.g2.numel():
            drive = drive + self.torch.dot(self.g2, applied_sol)
        scale = float(self.ip_coupling_sign) * (float(self.mu0) * float(self.sigma) / float(self.R0))
        d_ip_dt = -self._ip / max(float(self.sigma * self.inductance_L), 1.0e-30) + scale * drive
        ip_next = self._ip + float(self.t_step) * d_ip_dt
        self._ip = ip_next
        self._pfc = next_pfc
        self._sol = next_sol
        self._pfc_deriv = applied_pfc
        self._sol_deriv = applied_sol
        self._step += 1
        self._time += float(self.t_step)
        self.state = self._snapshot_state()
        return self.state

    def step(self, *args, **kwargs) -> PlasmaState:
        """Reject the removed derivative-command API."""
        raise RuntimeError("GpuPlasmaModel.step() was removed; use step_currents(J_next) with absolute next currents")

    def snapshot_state(self) -> PlasmaState:
        return self._snapshot_state().copied()

    def _snapshot_state(self) -> PlasmaState:
        return PlasmaState(
            t=float(self._time),
            step=int(self._step),
            Ip=float(self._ip.detach().cpu().item()),
            Ip0=float(self._ip_origin),
            psi=self.compute_psi(),
            pfc_currents=self._pfc.detach().cpu().numpy().astype(float),
            pfc_current_derivs=self._pfc_deriv.detach().cpu().numpy().astype(float),
            sol_currents=self._sol.detach().cpu().numpy().astype(float),
            sol_current_derivs=self._sol_deriv.detach().cpu().numpy().astype(float),
        )
