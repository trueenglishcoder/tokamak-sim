from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokamak_control.compute import require_gpu_available
from tokamak_control.config.settings import PhysicsSettings
from tokamak_control.core.coils import CoilGroup
from tokamak_control.core.green import build_green_for_coils, build_green_for_eind, build_green_for_plasma_center
from tokamak_control.core.grid import Grid2D
from tokamak_control.geometry.boundary_gpu import FixedAngleBoundaryGpuResult


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
        boundary_mode: str = "legacy_contour",
        boundary_base_mode: str = "legacy_contour_limited",
        boundary_level_smoothing_alpha: float = 0.6,
        boundary_level_search_span_fraction: float = 0.02,
        boundary_continuity_weight_radii: float = 1.0,
        boundary_continuity_weight_mean_radius: float = 0.3,
        boundary_continuity_weight_center: float = 0.2,
        boundary_continuity_weight_area: float = 0.2,
        boundary_continuity_weight_level: float = 0.1,
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
        self.boundary_mode = str(boundary_mode)
        if self.boundary_mode not in {"legacy_contour", "legacy_contour_limited", "tracked_flux_contour"}:
            raise ValueError("BatchedGpuTokamakSimulator only supports legacy contour boundary modes")
        self.boundary_base_mode = str(boundary_base_mode)
        self.boundary_level_smoothing_alpha = float(boundary_level_smoothing_alpha)
        self.boundary_level_search_span_fraction = float(boundary_level_search_span_fraction)
        self.boundary_continuity_weight_radii = float(boundary_continuity_weight_radii)
        self.boundary_continuity_weight_mean_radius = float(boundary_continuity_weight_mean_radius)
        self.boundary_continuity_weight_center = float(boundary_continuity_weight_center)
        self.boundary_continuity_weight_area = float(boundary_continuity_weight_area)
        self.boundary_continuity_weight_level = float(boundary_continuity_weight_level)
        self._boundary_prev_polys: list[np.ndarray | None] = [None for _ in range(self.batch_size)]
        self._boundary_prev_levels = np.full((self.batch_size,), np.nan, dtype=float)
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
        self.Ip0 = ip_t.clone()
        self.pfc_currents = pfc_t.clone()
        self.sol_currents = sol_t.clone()
        self.pfc_derivs = torch.zeros_like(self.pfc_currents)
        self.sol_derivs = torch.zeros_like(self.sol_currents)
        self.step_index = torch.zeros((B,), dtype=torch.int64, device=self.device)
        self.time_s = torch.zeros((B,), dtype=self.dtype, device=self.device)
        self.psi = self.compose_psi(self.Ip, self.pfc_currents, self.sol_currents)
        self._boundary_prev_polys = [None for _ in range(B)]
        self._boundary_prev_levels = np.full((B,), np.nan, dtype=float)
        return self._result()

    def reset_indices(self, indices: list[int], *, ip, pfc_currents, sol_currents) -> BatchedGpuSimulatorResult:
        if not indices:
            return self._result()
        torch = self.torch
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        self.Ip[idx] = torch.as_tensor(ip, dtype=self.dtype, device=self.device).reshape(-1)
        self.Ip0[idx] = self.Ip[idx]
        self.pfc_currents[idx] = torch.as_tensor(pfc_currents, dtype=self.dtype, device=self.device).reshape(len(indices), self.pfc.n_coils)
        self.sol_currents[idx] = torch.as_tensor(sol_currents, dtype=self.dtype, device=self.device).reshape(len(indices), self.sol.n_coils)
        self.pfc_derivs[idx] = 0.0
        self.sol_derivs[idx] = 0.0
        self.step_index[idx] = 0
        self.time_s[idx] = 0.0
        self.psi[idx] = self.compose_psi(self.Ip[idx], self.pfc_currents[idx], self.sol_currents[idx])
        for lane in indices:
            self._boundary_prev_polys[int(lane)] = None
            self._boundary_prev_levels[int(lane)] = np.nan
        return self._result()

    def step_currents(self, active_currents_next) -> BatchedGpuSimulatorResult:
        """Advance one old-parity batched step from absolute next active currents."""
        torch = self.torch
        B = self.batch_size
        actions = torch.as_tensor(active_currents_next, dtype=self.dtype, device=self.device).reshape(B, self.pfc.n_coils + self.sol.n_coils)
        next_pfc = actions[:, : self.pfc.n_coils]
        next_sol = actions[:, self.pfc.n_coils :]
        applied_pfc = (next_pfc - self.pfc_currents) / float(self.settings.t_step)
        applied_sol = (next_sol - self.sol_currents) / float(self.settings.t_step)
        drive = torch.zeros((B,), dtype=self.dtype, device=self.device)
        if self.g_pfc.numel():
            drive = drive + torch.einsum("bi,i->b", applied_pfc, self.g_pfc)
        if self.g_sol.numel():
            drive = drive + torch.einsum("bi,i->b", applied_sol, self.g_sol)
        tau = max(float(self.settings.sigma * self.settings.inductance_L), 1.0e-30)
        scale = float(getattr(self.settings, "ip_coupling_sign", -1.0)) * (
            float(self.settings.mu0) * float(self.settings.sigma) / float(self.settings.R0)
        )
        d_ip_dt = -self.Ip / tau + scale * drive
        ip_next = self.Ip + float(self.settings.t_step) * d_ip_dt
        self.Ip = ip_next
        self.pfc_currents = next_pfc
        self.sol_currents = next_sol
        self.pfc_derivs = applied_pfc
        self.sol_derivs = applied_sol
        self.step_index = self.step_index + 1
        self.time_s = self.time_s + float(self.settings.t_step)
        self.psi = self.compose_psi(self.Ip, self.pfc_currents, self.sol_currents)
        return self._result()

    def step(self, *args, **kwargs) -> BatchedGpuSimulatorResult:
        """Reject the removed derivative-command API."""
        raise RuntimeError("BatchedGpuTokamakSimulator.step() was removed; use step_currents(J_next)")

    def compose_psi(self, ip, pfc, sol):
        psi = float(getattr(self.settings, "plasma_psi_sign", 1.0)) * ip[:, None, None] * self.G_plasma[None, :, :]
        if self.G_pfc.shape[0]:
            psi = psi + self.torch.einsum("bi,izr->bzr", pfc, self.G_pfc)
        if self.G_sol.shape[0]:
            psi = psi + self.torch.einsum("bi,izr->bzr", sol, self.G_sol)
        return float(self.settings.mu0) * psi

    def _result(self) -> BatchedGpuSimulatorResult:
        from tokamak_control.geometry.boundary import BoundaryNotFoundError, find_plasma_boundary_with_status
        from tokamak_control.geometry.legacy_metrics import legacy_radii_at_angles

        torch = self.torch
        psi_cpu = self.psi.detach().cpu().numpy().astype(float)
        angles = np.asarray(self.angles_rad, dtype=float).reshape(-1)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        B = int(psi_cpu.shape[0])
        A = int(angles.shape[0])
        found_np = np.zeros((B,), dtype=bool)
        status_np = np.zeros((B,), dtype=np.int64)
        level_np = np.full((B,), np.nan, dtype=float)
        radii_np = np.full((B, A), np.nan, dtype=float)
        points_np = np.full((B, A, 2), np.nan, dtype=float)
        center_np = np.asarray(self.center, dtype=float).reshape(2)
        axis_np = np.repeat(center_np.reshape(1, 2), B, axis=0)
        for b in range(B):
            try:
                use_limiter = self.boundary_mode in {"legacy_contour_limited", "tracked_flux_contour"}
                poly, level, _status = find_plasma_boundary_with_status(
                    psi_cpu[b],
                    self.grid,
                    self.center,
                    prev_level=None if not np.isfinite(self._boundary_prev_levels[b]) else float(self._boundary_prev_levels[b]),
                    prev_poly=self._boundary_prev_polys[b],
                    limiter_shape=(self.limiter_shape if use_limiter else None),
                    boundary_mode=self.boundary_mode,
                    boundary_base_mode=self.boundary_base_mode,
                    track_level=False,
                    level_smoothing_alpha=self.boundary_level_smoothing_alpha,
                    level_search_span_fraction=self.boundary_level_search_span_fraction,
                    continuity_weight_radii=self.boundary_continuity_weight_radii,
                    continuity_weight_mean_radius=self.boundary_continuity_weight_mean_radius,
                    continuity_weight_center=self.boundary_continuity_weight_center,
                    continuity_weight_area=self.boundary_continuity_weight_area,
                    continuity_weight_level=self.boundary_continuity_weight_level,
                    compute_backend="cpu",
                )
                radii = legacy_radii_at_angles(poly, self.center, angles)
                found_np[b] = True
                status_np[b] = 3
                level_np[b] = float(level)
                radii_np[b] = radii
                points_np[b] = center_np.reshape(1, 2) + radii.reshape(-1, 1) * dirs
                self._boundary_prev_polys[b] = np.asarray(poly, dtype=float)
                self._boundary_prev_levels[b] = float(level)
            except (BoundaryNotFoundError, ValueError, FloatingPointError):
                self._boundary_prev_polys[b] = None
                self._boundary_prev_levels[b] = np.nan
                continue
        boundary = FixedAngleBoundaryGpuResult(
            found=torch.as_tensor(found_np, dtype=torch.bool, device=self.device),
            status_code=torch.as_tensor(status_np, dtype=torch.int64, device=self.device),
            level=torch.as_tensor(level_np, dtype=self.dtype, device=self.device),
            points=torch.as_tensor(points_np, dtype=self.dtype, device=self.device),
            radii=torch.as_tensor(radii_np, dtype=self.dtype, device=self.device),
            axis_points=torch.as_tensor(axis_np, dtype=self.dtype, device=self.device),
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
