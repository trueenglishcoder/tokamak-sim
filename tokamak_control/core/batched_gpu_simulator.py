from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tokamak_control.config.settings import PhysicsSettings
from tokamak_control.core.coils import CoilGroup
from tokamak_control.core.green import build_green_for_coils, build_green_for_eind, build_green_for_plasma_center
from tokamak_control.core.grid import Grid2D
from tokamak_control.geometry.boundary_common import points_in_or_on_polygon, sample_limiter_points
from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections
from tokamak_control.io.logger import get_logger
from tokamak_control.io.profiling import Profiler


_PROFILER = Profiler(enabled=False, summary_every=0, logger=get_logger("core.batched_gpu_simulator.profiling"))
_time_block = _PROFILER.time_block


def configure_batched_gpu_simulator_profiling(*, enabled: bool, summary_every: int = 0, reset: bool = True) -> None:
    _PROFILER.configure(enabled=enabled, summary_every=summary_every, logger=get_logger("core.batched_gpu_simulator.profiling"), reset=reset)


def batched_gpu_simulator_profiling_snapshot() -> dict[str, object]:
    return _PROFILER.summary_dict(
        total_key="step_total",
        keys=(
            "plant_step",
            "boundary_axis",
            "boundary_limiter_levels",
            "boundary_fixed_angle",
            "observation",
            "reward",
            "cpu_transfer",
        ),
        title="batched_gpu_simulator",
    )


@dataclass(frozen=True, slots=True)
class BatchedPlantState:
    step: Any
    time_s: Any
    ip: Any
    pfc_currents: Any
    sol_currents: Any
    pfc_derivatives: Any
    sol_derivatives: Any
    psi: Any


@dataclass(frozen=True, slots=True)
class BatchedBoundaryResult:
    found: Any
    status: Any
    level: Any
    radii: Any
    points: Any
    axis_points: Any
    axis_kind: Any


@dataclass(frozen=True, slots=True)
class BatchedStepResult:
    state: BatchedPlantState
    boundary: BatchedBoundaryResult
    current_margin: Any | None
    derivative_margin: Any | None


@dataclass(slots=True)
class BatchedGpuTokamakSimulator:
    """GPU-resident batched plant and fixed-angle boundary backend for RL training."""

    grid: Grid2D
    pfc: CoilGroup
    sol: CoilGroup
    settings: PhysicsSettings
    batch_size: int
    angles_rad: np.ndarray
    limiter_shape: np.ndarray
    gpu_device: str = "cuda:0"

    torch: Any | None = None
    device: Any | None = None
    _G_pfc: Any | None = None
    _G_sol: Any | None = None
    _G_plasma: Any | None = None
    _g_pfc: Any | None = None
    _g_sol: Any | None = None
    _r: Any | None = None
    _z: Any | None = None
    _angles: Any | None = None
    _dirs: Any | None = None
    _ray_caps: Any | None = None
    _ray_sample_fracs: Any | None = None
    _limiter_points: Any | None = None
    _limiter_mask: Any | None = None
    _pfc_current_limit: float | None = None
    _sol_current_limit: float | None = None
    _pfc_deriv_limit: float | None = None
    _sol_deriv_limit: float | None = None
    _ip_coupling_sign: float = -1.0
    _plasma_psi_sign: float = 1.0

    _step: Any | None = None
    _time_s: Any | None = None
    _ip: Any | None = None
    _pfc_currents: Any | None = None
    _sol_currents: Any | None = None
    _pfc_derivatives: Any | None = None
    _sol_derivatives: Any | None = None
    _psi: Any | None = None

    @classmethod
    def from_settings(
        cls,
        *,
        grid: Grid2D,
        pfc: CoilGroup,
        sol: CoilGroup,
        settings: PhysicsSettings,
        batch_size: int,
        angles_rad: np.ndarray,
        limiter_shape: np.ndarray,
        gpu_device: str = "cuda:0",
    ) -> "BatchedGpuTokamakSimulator":
        settings.validate()
        return cls(
            grid=grid,
            pfc=pfc,
            sol=sol,
            settings=settings,
            batch_size=int(batch_size),
            angles_rad=np.asarray(angles_rad, dtype=float).reshape(-1),
            limiter_shape=np.asarray(limiter_shape, dtype=float).reshape(-1, 2),
            gpu_device=str(gpu_device),
        )

    def __post_init__(self) -> None:
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be > 0")
        if self.angles_rad.size <= 0:
            raise ValueError("angles_rad must be non-empty")
        self.torch, self.device = _require_torch(self.gpu_device)
        torch = self.torch
        R, Z = self.grid.mesh()
        with _time_block("plant_step"):
            G_pfc = build_green_for_coils(R, Z, self.pfc.element_positions, self.pfc.element_weights) if self.pfc.n_coils else np.zeros((0, *self.grid.shape))
            G_sol = build_green_for_coils(R, Z, self.sol.element_positions, self.sol.element_weights) if self.sol.n_coils else np.zeros((0, *self.grid.shape))
            G_plasma = build_green_for_plasma_center(R, Z, float(self.settings.R0), float(self.settings.Z0))
            g_pfc = build_green_for_eind(float(self.settings.R0), float(self.settings.Z0), self.pfc.element_positions, self.pfc.element_weights) if self.pfc.n_coils else np.zeros((0,), dtype=float)
            g_sol = build_green_for_eind(float(self.settings.R0), float(self.settings.Z0), self.sol.element_positions, self.sol.element_weights) if self.sol.n_coils else np.zeros((0,), dtype=float)
            if self.settings.ip_coupling_pfc is not None:
                g_pfc = np.asarray(self.settings.ip_coupling_pfc, dtype=float).reshape(self.pfc.n_coils)
            if self.settings.ip_coupling_sol is not None:
                g_sol = np.asarray(self.settings.ip_coupling_sol, dtype=float).reshape(self.sol.n_coils)
            self._G_pfc = self._tensor(G_pfc)
            self._G_sol = self._tensor(G_sol)
            self._G_plasma = self._tensor(G_plasma)
            self._g_pfc = self._tensor(g_pfc)
            self._g_sol = self._tensor(g_sol)
            self._r = self._tensor(self.grid.r.coords())
            self._z = self._tensor(self.grid.z.coords())
            self._angles = self._tensor(self.angles_rad)
            self._dirs = torch.stack((torch.cos(self._angles), torch.sin(self._angles)), dim=1)
            center = (float(self.settings.R0), float(self.settings.Z0))
            self._ray_caps = self._tensor(radii_from_polyline_ray_intersections(self.limiter_shape, center, self.angles_rad))
            self._ray_sample_fracs = torch.linspace(0.0, 1.0, 257, dtype=torch.float64, device=self.device)
            limiter_pts = sample_limiter_points(self.limiter_shape, self.grid)
            self._limiter_points = self._tensor(limiter_pts)
            points = np.column_stack([R.reshape(-1), Z.reshape(-1)])
            self._limiter_mask = torch.as_tensor(points_in_or_on_polygon(points, self.limiter_shape, tol=0.5 * min(abs(float(self.grid.r.step)), abs(float(self.grid.z.step)))).reshape(self.grid.shape), dtype=torch.bool, device=self.device)
            self._pfc_current_limit = self.settings.pfc_current_limit
            self._sol_current_limit = self.settings.sol_current_limit
            self._pfc_deriv_limit = self.settings.pfc_deriv_limit
            self._sol_deriv_limit = self.settings.sol_deriv_limit
            self._ip_coupling_sign = float(getattr(self.settings, "ip_coupling_sign", -1.0))
            self._plasma_psi_sign = float(getattr(self.settings, "plasma_psi_sign", 1.0))

    @property
    def n_active_total(self) -> int:
        return int(self.pfc.n_coils + self.sol.n_coils)

    @property
    def state(self) -> BatchedPlantState:
        return BatchedPlantState(
            step=self._require(self._step, "step"),
            time_s=self._require(self._time_s, "time_s"),
            ip=self._require(self._ip, "ip"),
            pfc_currents=self._require(self._pfc_currents, "pfc_currents"),
            sol_currents=self._require(self._sol_currents, "sol_currents"),
            pfc_derivatives=self._require(self._pfc_derivatives, "pfc_derivatives"),
            sol_derivatives=self._require(self._sol_derivatives, "sol_derivatives"),
            psi=self._require(self._psi, "psi"),
        )

    def reset(self, *, ip: np.ndarray, pfc_currents: np.ndarray, sol_currents: np.ndarray) -> BatchedStepResult:
        torch = self._torch()
        b = int(self.batch_size)
        self._step = torch.zeros((b,), dtype=torch.int64, device=self.device)
        self._time_s = torch.zeros((b,), dtype=torch.float64, device=self.device)
        self._ip = self._tensor(ip).reshape(b)
        self._pfc_currents = self._clip_currents(self._tensor(pfc_currents).reshape(b, self.pfc.n_coils), self._pfc_current_limit)
        self._sol_currents = self._clip_currents(self._tensor(sol_currents).reshape(b, self.sol.n_coils), self._sol_current_limit)
        self._pfc_derivatives = torch.zeros((b, self.pfc.n_coils), dtype=torch.float64, device=self.device)
        self._sol_derivatives = torch.zeros((b, self.sol.n_coils), dtype=torch.float64, device=self.device)
        self._psi = self._compose_psi(self._ip, self._pfc_currents, self._sol_currents)
        return BatchedStepResult(self.state, self.measure_boundary(), self.current_margin(), self.derivative_margin())

    def reset_indices(self, indices, *, ip: np.ndarray, pfc_currents: np.ndarray, sol_currents: np.ndarray) -> BatchedStepResult:
        torch = self._torch()
        if self._step is None:
            raise RuntimeError("cannot reset batch indices before full reset")
        idx = torch.as_tensor(indices, dtype=torch.int64, device=self.device).reshape(-1)
        if idx.numel() <= 0:
            return BatchedStepResult(self.state, self.measure_boundary(), self.current_margin(), self.derivative_margin())
        if bool(torch.any((idx < 0) | (idx >= int(self.batch_size))).detach().cpu()):
            raise IndexError("reset index out of range")
        k = int(idx.numel())
        ip_t = self._tensor(ip).reshape(k)
        pfc_t = self._clip_currents(self._tensor(pfc_currents).reshape(k, self.pfc.n_coils), self._pfc_current_limit)
        sol_t = self._clip_currents(self._tensor(sol_currents).reshape(k, self.sol.n_coils), self._sol_current_limit)
        self._step[idx] = 0
        self._time_s[idx] = 0.0
        self._ip[idx] = ip_t
        self._pfc_currents[idx] = pfc_t
        self._sol_currents[idx] = sol_t
        self._pfc_derivatives[idx] = 0.0
        self._sol_derivatives[idx] = 0.0
        self._psi[idx] = self._compose_psi(ip_t, pfc_t, sol_t)
        return BatchedStepResult(self.state, self.measure_boundary(), self.current_margin(), self.derivative_margin())

    def step(self, active_derivatives) -> BatchedStepResult:
        torch = self._torch()
        with _time_block("step_total"):
            with _time_block("plant_step"):
                b = int(self.batch_size)
                action = active_derivatives.to(device=self.device, dtype=torch.float64).reshape(b, self.n_active_total)
                cmd_pfc = self._clip_derivs(action[:, : self.pfc.n_coils], self._pfc_deriv_limit)
                cmd_sol = self._clip_derivs(action[:, self.pfc.n_coils :], self._sol_deriv_limit)
                alpha = self._actuator_alpha()
                pfc_deriv = self._clip_derivs(alpha * self._pfc_derivatives + (1.0 - alpha) * cmd_pfc, self._pfc_deriv_limit)
                sol_deriv = self._clip_derivs(alpha * self._sol_derivatives + (1.0 - alpha) * cmd_sol, self._sol_deriv_limit)
                dt = float(self.settings.t_step)
                prev_pfc = self._pfc_currents
                prev_sol = self._sol_currents
                next_pfc = self._clip_currents(prev_pfc + dt * pfc_deriv, self._pfc_current_limit)
                next_sol = self._clip_currents(prev_sol + dt * sol_deriv, self._sol_current_limit)
                delta_pfc = next_pfc - prev_pfc
                delta_sol = next_sol - prev_sol
                decay = np.exp(-dt / max(float(self.settings.sigma) * float(self.settings.inductance_L), 1e-30))
                control = torch.zeros((b,), dtype=torch.float64, device=self.device)
                if self.pfc.n_coils:
                    control = control + torch.sum(delta_pfc * self._g_pfc[None, :], dim=1)
                if self.sol.n_coils:
                    control = control + torch.sum(delta_sol * self._g_sol[None, :], dim=1)
                ip_next = self._ip * float(decay) + self._ip_coupling_sign * (float(self.settings.mu0) * float(self.settings.sigma) / float(self.settings.R0)) * control
                psi_next = self._compose_psi(ip_next, next_pfc, next_sol)
                self._step = self._step + 1
                self._time_s = self._time_s + dt
                self._ip = ip_next
                self._pfc_currents = next_pfc
                self._sol_currents = next_sol
                self._pfc_derivatives = pfc_deriv
                self._sol_derivatives = sol_deriv
                self._psi = psi_next
            boundary = self.measure_boundary()
            _PROFILER.step()
            return BatchedStepResult(self.state, boundary, self.current_margin(), self.derivative_margin())

    def measure_boundary(self) -> BatchedBoundaryResult:
        torch = self._torch()
        psi = self._require(self._psi, "psi")
        b = int(self.batch_size)
        with _time_block("boundary_axis"):
            masked = torch.where(self._limiter_mask[None, :, :] & torch.isfinite(psi), psi, torch.full_like(psi, torch.nan))
            flat = masked.reshape(b, -1)
            max_flat = torch.where(torch.isfinite(flat), flat, torch.full_like(flat, -torch.inf))
            min_flat = torch.where(torch.isfinite(flat), flat, torch.full_like(flat, torch.inf))
            max_idx = torch.argmax(max_flat, dim=1)
            min_idx = torch.argmin(min_flat, dim=1)
            nr = int(self.grid.r.size)
            max_j = torch.div(max_idx, nr, rounding_mode="floor")
            max_i = max_idx % nr
            min_j = torch.div(min_idx, nr, rounding_mode="floor")
            min_i = min_idx % nr
            r = self._r
            z = self._z
            center = torch.tensor([float(self.settings.R0), float(self.settings.Z0)], dtype=torch.float64, device=self.device)
            max_points = torch.stack((r[max_i], z[max_j]), dim=1)
            min_points = torch.stack((r[min_i], z[min_j]), dim=1)
            use_max = torch.linalg.norm(max_points - center[None, :], dim=1) <= torch.linalg.norm(min_points - center[None, :], dim=1)
            axis_points = torch.where(use_max[:, None], max_points, min_points)
            axis_levels = torch.where(use_max, flat[torch.arange(b, device=self.device), max_idx], flat[torch.arange(b, device=self.device), min_idx])

        with _time_block("boundary_limiter_levels"):
            limiter_psi = self._sample_points(psi, self._limiter_points)
            levels_sorted = torch.sort(limiter_psi, dim=1, descending=True).values
            levels_sorted_min = torch.sort(limiter_psi, dim=1, descending=False).values

        found = torch.zeros((b,), dtype=torch.bool, device=self.device)
        levels = torch.full((b,), torch.nan, dtype=torch.float64, device=self.device)
        radii = torch.full((b, self.angles_rad.size), torch.nan, dtype=torch.float64, device=self.device)
        points = torch.full((b, self.angles_rad.size, 2), torch.nan, dtype=torch.float64, device=self.device)
        max_candidates = int(levels_sorted.shape[1])
        for candidate_index in range(max_candidates):
            active = ~found
            if not bool(torch.any(active).detach().cpu()):
                break
            candidate_levels = torch.where(use_max, levels_sorted[:, candidate_index], levels_sorted_min[:, candidate_index])
            candidate_levels = torch.where(active, candidate_levels, levels)
            cand_radii, cand_points, cand_valid = self._fixed_angle_boundary_at_levels(candidate_levels)
            accept = active & cand_valid & torch.isfinite(candidate_levels)
            if bool(torch.any(accept).detach().cpu()):
                found = found | accept
                levels = torch.where(accept, candidate_levels, levels)
                radii = torch.where(accept[:, None], cand_radii, radii)
                points = torch.where(accept[:, None, None], cand_points, points)

        status = torch.where(found, torch.ones((b,), dtype=torch.int64, device=self.device), torch.zeros((b,), dtype=torch.int64, device=self.device))
        return BatchedBoundaryResult(found=found, status=status, level=levels, radii=radii, points=points, axis_points=axis_points, axis_kind=torch.where(use_max, torch.ones_like(status), -torch.ones_like(status)))

    def current_margin(self):
        torch = self._torch()
        limits = []
        if self._pfc_current_limit is None or self._sol_current_limit is None:
            return None
        limits.extend([float(self._pfc_current_limit)] * self.pfc.n_coils)
        limits.extend([float(self._sol_current_limit)] * self.sol.n_coils)
        lim = torch.tensor(limits, dtype=torch.float64, device=self.device)
        values = torch.cat((self._pfc_currents, self._sol_currents), dim=1)
        return (lim[None, :] - torch.abs(values)) / torch.clamp(lim[None, :], min=1.0)

    def derivative_margin(self):
        torch = self._torch()
        if self._pfc_deriv_limit is None or self._sol_deriv_limit is None:
            return None
        limits = [float(self._pfc_deriv_limit)] * self.pfc.n_coils + [float(self._sol_deriv_limit)] * self.sol.n_coils
        lim = torch.tensor(limits, dtype=torch.float64, device=self.device)
        values = torch.cat((self._pfc_derivatives, self._sol_derivatives), dim=1)
        return (lim[None, :] - torch.abs(values)) / torch.clamp(lim[None, :], min=1.0)

    def _fixed_angle_boundary_at_levels(self, levels):
        torch = self._torch()
        with _time_block("boundary_fixed_angle"):
            b = int(self.batch_size)
            n_angles = int(self.angles_rad.size)
            fracs = self._ray_sample_fracs
            ray_tol = 2.0 * float(max(abs(float(self.grid.r.step)), abs(float(self.grid.z.step))))
            ray_caps = self._ray_caps + ray_tol
            radii_samples = ray_caps[None, :, None] * fracs[None, None, :]
            center = torch.tensor([float(self.settings.R0), float(self.settings.Z0)], dtype=torch.float64, device=self.device)
            sample_points = center[None, None, None, :] + radii_samples[:, :, :, None] * self._dirs[None, :, None, :]
            psi_samples = self._sample_points(psi=self._psi, points=sample_points.reshape(n_angles * int(fracs.numel()), 2)).reshape(b, n_angles, int(fracs.numel()))
            level = levels[:, None, None]
            finite = torch.isfinite(psi_samples) & torch.isfinite(level)
            v0 = psi_samples[:, :, :-1] - level
            v1 = psi_samples[:, :, 1:] - level
            crosses = finite[:, :, :-1] & finite[:, :, 1:] & ((v0 == 0.0) | (v1 == 0.0) | (v0 * v1 <= 0.0))
            rev = torch.flip(crosses, dims=(2,))
            any_hit = torch.any(rev, dim=2)
            rev_idx = torch.argmax(rev.to(torch.int64), dim=2)
            idx = crosses.shape[2] - 1 - rev_idx
            gather0 = torch.gather(psi_samples, 2, idx[:, :, None]).squeeze(2)
            gather1 = torch.gather(psi_samples, 2, (idx + 1).clamp(max=psi_samples.shape[2] - 1)[:, :, None]).squeeze(2)
            r0 = torch.gather(radii_samples.expand(b, -1, -1), 2, idx[:, :, None]).squeeze(2)
            r1 = torch.gather(radii_samples.expand(b, -1, -1), 2, (idx + 1).clamp(max=psi_samples.shape[2] - 1)[:, :, None]).squeeze(2)
            denom = gather1 - gather0
            frac = torch.where(torch.abs(denom) > 1.0e-30, (levels[:, None] - gather0) / denom, torch.full_like(denom, 0.5))
            frac = torch.clamp(frac, 0.0, 1.0)
            radii = torch.where(any_hit, r0 + frac * (r1 - r0), torch.full_like(r0, torch.nan))
            points = center[None, None, :] + radii[:, :, None] * self._dirs[None, :, :]
            valid = torch.all(any_hit & torch.isfinite(radii) & (radii <= self._ray_caps[None, :] + ray_tol), dim=1)
            return radii, points, valid

    def _sample_points(self, psi, points):
        torch = self._torch()
        pts = points.to(device=self.device, dtype=torch.float64).reshape(-1, 2)
        r = self._r
        z = self._z
        b = int(self.batch_size)
        out = torch.full((b, pts.shape[0]), torch.nan, dtype=torch.float64, device=self.device)
        in_bounds = (pts[:, 0] >= r[0]) & (pts[:, 0] <= r[-1]) & (pts[:, 1] >= z[0]) & (pts[:, 1] <= z[-1])
        if not bool(torch.any(in_bounds).detach().cpu()):
            return out
        idx = torch.nonzero(in_bounds, as_tuple=False).reshape(-1)
        p = pts[idx]
        i0 = torch.searchsorted(r, p[:, 0].contiguous(), right=True) - 1
        j0 = torch.searchsorted(z, p[:, 1].contiguous(), right=True) - 1
        i0 = torch.clamp(i0, 0, self.grid.r.size - 2)
        j0 = torch.clamp(j0, 0, self.grid.z.size - 2)
        r0 = r[i0]
        r1 = r[i0 + 1]
        z0 = z[j0]
        z1 = z[j0 + 1]
        ar = (p[:, 0] - r0) / torch.clamp(r1 - r0, min=1.0e-30)
        az = (p[:, 1] - z0) / torch.clamp(z1 - z0, min=1.0e-30)
        q00 = psi[:, j0, i0]
        q10 = psi[:, j0, i0 + 1]
        q01 = psi[:, j0 + 1, i0]
        q11 = psi[:, j0 + 1, i0 + 1]
        values = (1.0 - ar)[None, :] * (1.0 - az)[None, :] * q00 + ar[None, :] * (1.0 - az)[None, :] * q10 + (1.0 - ar)[None, :] * az[None, :] * q01 + ar[None, :] * az[None, :] * q11
        out[:, idx] = values
        return out

    def _compose_psi(self, ip, pfc_currents, sol_currents):
        torch = self._torch()
        psi = self._plasma_psi_sign * ip[:, None, None] * self._G_plasma[None, :, :]
        if self.pfc.n_coils:
            psi = psi + torch.tensordot(pfc_currents, self._G_pfc, dims=([1], [0]))
        if self.sol.n_coils:
            psi = psi + torch.tensordot(sol_currents, self._G_sol, dims=([1], [0]))
        return float(self.settings.mu0) * psi

    def _actuator_alpha(self) -> float:
        tau = float(self.settings.actuator_tau)
        if tau <= 0.0:
            return 0.0
        return float(np.exp(-float(self.settings.t_step) / tau))

    def _clip_currents(self, values, limit: float | None):
        if limit is None or float(limit) <= 0.0:
            return values
        return self._torch().clamp(values, -float(limit), float(limit))

    def _clip_derivs(self, values, limit: float | None):
        if limit is None or float(limit) < 0.0 or not np.isfinite(float(limit)):
            return values
        return self._torch().clamp(values, -float(limit), float(limit))

    def _tensor(self, value):
        return self._torch().as_tensor(value, dtype=self._torch().float64, device=self.device)

    def _torch(self):
        if self.torch is None:
            raise RuntimeError("torch runtime is not initialized")
        return self.torch

    @staticmethod
    def _require(value, name: str):
        if value is None:
            raise RuntimeError(f"{name} tensor is not initialized")
        return value


def _require_torch(gpu_device: str):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Batched GPU simulator requires torch") from exc
    device = torch.device(str(gpu_device))
    if device.type != "cuda":
        raise RuntimeError(f"Batched GPU simulator requires a CUDA device, got {gpu_device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("Batched GPU simulator requested, but torch.cuda.is_available() is False")
    try:
        torch.empty((1,), device=device)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Batched GPU simulator could not initialize device {gpu_device!r}") from exc
    return torch, device
