from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokamak_control.compute import require_gpu_available
from tokamak_control.config.settings import PhysicsSettings
from tokamak_control.core.coils import CoilGroup
from tokamak_control.core.green import build_green_for_coils, build_green_for_eind, build_green_for_plasma_center
from tokamak_control.core.grid import Grid2D
from tokamak_control.geometry.boundary_gpu import FixedAngleBoundaryGpuResult, fixed_angle_boundary_gpu


@dataclass(slots=True)
class BatchedGpuPlantState:
    t: object
    step: object
    Ip: object
    pfc_currents: object
    sol_currents: object
    pfc_current_derivs: object
    sol_current_derivs: object
    psi: object


@dataclass(slots=True)
class BatchedGpuSimulatorResult:
    state: BatchedGpuPlantState
    boundary: FixedAngleBoundaryGpuResult


class BatchedGpuTokamakSimulator:
    """Batched CUDA mirror of tokamak-sim's PlasmaModel dynamics."""

    def __init__(
        self,
        *,
        grid: Grid2D,
        pfc: CoilGroup,
        sol: CoilGroup,
        settings: PhysicsSettings,
        batch_size: int,
        angles_rad: np.ndarray,
        limiter_shape: np.ndarray,
        gpu_device: str = "cuda:0",
    ) -> None:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be > 0")
        require_gpu_available(gpu_device)
        import torch
        self.torch = torch
        self.device = torch.device(gpu_device)
        self.dtype = torch.float64
        self.grid = grid
        self.pfc = pfc
        self.sol = sol
        self.settings = settings
        self.batch_size = int(batch_size)
        self.angles_rad = np.asarray(angles_rad, dtype=float).reshape(-1)
        self.limiter_shape = np.asarray(limiter_shape, dtype=float).reshape(-1, 2)
        self.center = (float(settings.R0), float(settings.Z0))
        R, Z = grid.mesh()
        self.G_pfc = torch.as_tensor(build_green_for_coils(R, Z, pfc.element_positions, pfc.element_weights) if pfc.n_coils else np.zeros((0, *grid.shape)), dtype=self.dtype, device=self.device)
        self.G_sol = torch.as_tensor(build_green_for_coils(R, Z, sol.element_positions, sol.element_weights) if sol.n_coils else np.zeros((0, *grid.shape)), dtype=self.dtype, device=self.device)
        self.G_plasma = torch.as_tensor(build_green_for_plasma_center(R, Z, settings.R0, settings.Z0), dtype=self.dtype, device=self.device)
        gp = build_green_for_eind(settings.R0, settings.Z0, pfc.element_positions, pfc.element_weights) if pfc.n_coils else np.zeros((0,), dtype=float)
        gs = build_green_for_eind(settings.R0, settings.Z0, sol.element_positions, sol.element_weights) if sol.n_coils else np.zeros((0,), dtype=float)
        if settings.ip_coupling_pfc is not None:
            gp = np.asarray(settings.ip_coupling_pfc, dtype=float).reshape(-1)
        if settings.ip_coupling_sol is not None:
            gs = np.asarray(settings.ip_coupling_sol, dtype=float).reshape(-1)
        self.g_pfc = torch.as_tensor(gp, dtype=self.dtype, device=self.device)
        self.g_sol = torch.as_tensor(gs, dtype=self.dtype, device=self.device)
        self.angles_t = torch.as_tensor(self.angles_rad, dtype=self.dtype, device=self.device)
        self.reset()

    @classmethod
    def from_settings(cls, **kwargs) -> "BatchedGpuTokamakSimulator":
        return cls(**kwargs)

    def reset(self, *, ip=None, pfc_currents=None, sol_currents=None) -> BatchedGpuSimulatorResult:
        torch = self.torch
        B = self.batch_size
        ip_t = torch.full((B,), float(self.settings.Ip0), dtype=self.dtype, device=self.device) if ip is None else torch.as_tensor(ip, dtype=self.dtype, device=self.device).reshape(B)
        pfc_t = torch.as_tensor(np.asarray(self.pfc.initial_currents, dtype=float), dtype=self.dtype, device=self.device).reshape(1, self.pfc.n_coils).repeat(B, 1) if pfc_currents is None else torch.as_tensor(pfc_currents, dtype=self.dtype, device=self.device).reshape(B, self.pfc.n_coils)
        sol_t = torch.as_tensor(np.asarray(self.sol.initial_currents, dtype=float), dtype=self.dtype, device=self.device).reshape(1, self.sol.n_coils).repeat(B, 1) if sol_currents is None else torch.as_tensor(sol_currents, dtype=self.dtype, device=self.device).reshape(B, self.sol.n_coils)
        self.Ip = ip_t.clone()
        self.pfc_currents = self._clip_currents(pfc_t, self.settings.pfc_current_limit)
        self.sol_currents = self._clip_currents(sol_t, self.settings.sol_current_limit)
        self.pfc_derivs = torch.zeros_like(self.pfc_currents)
        self.sol_derivs = torch.zeros_like(self.sol_currents)
        self.step_index = torch.zeros((B,), dtype=torch.int64, device=self.device)
        self.time_s = torch.zeros((B,), dtype=self.dtype, device=self.device)
        self.psi = self.compose_psi(self.Ip, self.pfc_currents, self.sol_currents)
        return self._result()

    def reset_indices(self, indices: list[int], *, ip, pfc_currents, sol_currents) -> BatchedGpuSimulatorResult:
        if not indices:
            return self._result()
        torch = self.torch
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        self.Ip[idx] = torch.as_tensor(ip, dtype=self.dtype, device=self.device).reshape(-1)
        self.pfc_currents[idx] = self._clip_currents(torch.as_tensor(pfc_currents, dtype=self.dtype, device=self.device).reshape(len(indices), self.pfc.n_coils), self.settings.pfc_current_limit)
        self.sol_currents[idx] = self._clip_currents(torch.as_tensor(sol_currents, dtype=self.dtype, device=self.device).reshape(len(indices), self.sol.n_coils), self.settings.sol_current_limit)
        self.pfc_derivs[idx] = 0.0
        self.sol_derivs[idx] = 0.0
        self.step_index[idx] = 0
        self.time_s[idx] = 0.0
        self.psi[idx] = self.compose_psi(self.Ip[idx], self.pfc_currents[idx], self.sol_currents[idx])
        return self._result()

    def step(self, active_current_derivs) -> BatchedGpuSimulatorResult:
        torch = self.torch
        B = self.batch_size
        actions = torch.as_tensor(active_current_derivs, dtype=self.dtype, device=self.device).reshape(B, self.pfc.n_coils + self.sol.n_coils)
        cmd_pfc = self._clip_derivs(actions[:, : self.pfc.n_coils], self.settings.pfc_deriv_limit)
        cmd_sol = self._clip_derivs(actions[:, self.pfc.n_coils :], self.settings.sol_deriv_limit)
        alpha = self._actuator_alpha()
        applied_pfc = self._clip_derivs(alpha * self.pfc_derivs + (1.0 - alpha) * cmd_pfc, self.settings.pfc_deriv_limit)
        applied_sol = self._clip_derivs(alpha * self.sol_derivs + (1.0 - alpha) * cmd_sol, self.settings.sol_deriv_limit)
        next_pfc = self._clip_currents(self.pfc_currents + float(self.settings.t_step) * applied_pfc, self.settings.pfc_current_limit)
        next_sol = self._clip_currents(self.sol_currents + float(self.settings.t_step) * applied_sol, self.settings.sol_current_limit)
        delta_pfc = next_pfc - self.pfc_currents
        delta_sol = next_sol - self.sol_currents
        control = torch.zeros((B,), dtype=self.dtype, device=self.device)
        if self.g_pfc.numel():
            control = control + torch.einsum("bi,i->b", delta_pfc, self.g_pfc)
        if self.g_sol.numel():
            control = control + torch.einsum("bi,i->b", delta_sol, self.g_sol)
        decay = float(np.exp(-float(self.settings.t_step) / max(float(self.settings.sigma * self.settings.inductance_L), 1e-30)))
        ip_next = self.Ip * decay + float(getattr(self.settings, "ip_coupling_sign", -1.0)) * (float(self.settings.mu0) * float(self.settings.sigma) / float(self.settings.R0)) * control
        self.Ip = ip_next
        self.pfc_currents = next_pfc
        self.sol_currents = next_sol
        self.pfc_derivs = applied_pfc
        self.sol_derivs = applied_sol
        self.step_index = self.step_index + 1
        self.time_s = self.time_s + float(self.settings.t_step)
        self.psi = self.compose_psi(self.Ip, self.pfc_currents, self.sol_currents)
        return self._result()

    def compose_psi(self, ip, pfc, sol):
        psi = float(getattr(self.settings, "plasma_psi_sign", 1.0)) * ip[:, None, None] * self.G_plasma[None, :, :]
        if self.G_pfc.shape[0]:
            psi = psi + self.torch.einsum("bi,izr->bzr", pfc, self.G_pfc)
        if self.G_sol.shape[0]:
            psi = psi + self.torch.einsum("bi,izr->bzr", sol, self.G_sol)
        return float(self.settings.mu0) * psi

    def _result(self) -> BatchedGpuSimulatorResult:
        boundary = fixed_angle_boundary_gpu(
            psi=self.psi,
            grid=self.grid,
            center=self.center,
            angles_rad=self.angles_t,
            limiter_shape=self.limiter_shape,
            boundary_mode="limited",
            gpu_device=str(self.device),
        )
        state = BatchedGpuPlantState(
            t=self.time_s,
            step=self.step_index,
            Ip=self.Ip,
            pfc_currents=self.pfc_currents,
            sol_currents=self.sol_currents,
            pfc_current_derivs=self.pfc_derivs,
            sol_current_derivs=self.sol_derivs,
            psi=self.psi,
        )
        return BatchedGpuSimulatorResult(state=state, boundary=boundary)

    def _clip_currents(self, values, limit: float | None):
        if limit is None or float(limit) <= 0.0:
            return values
        return self.torch.clamp(values, -float(limit), float(limit))

    def _clip_derivs(self, values, limit: float | None):
        if limit is None or not np.isfinite(float(limit)) or float(limit) < 0.0:
            return values
        return self.torch.clamp(values, -float(limit), float(limit))

    def _actuator_alpha(self) -> float:
        tau = float(self.settings.actuator_tau)
        if tau <= 0.0:
            return 0.0
        return float(np.exp(-float(self.settings.t_step) / tau))
