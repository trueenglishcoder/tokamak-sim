from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping

import numpy as np

from tokamak_control.bridge.types import CurrentAction, InitialStateOverride, MachineSpec, ReferenceFrame, ResetResult, StepResult, StepSnapshot
from tokamak_control.compute import ComputeBackend, ComputeSettings, compute_runtime_metadata
from tokamak_control.core.coils import CoilGroup
from tokamak_control.config.scenarios import Scenario, ScenarioName, make_scenario
from tokamak_control.core.gpu_plasma_model import GpuPlasmaModel
from tokamak_control.core.plasma_model import PlasmaModel
from tokamak_control.geometry.boundary import BoundaryMode, BoundaryNotFoundError, find_plasma_boundary_with_status
from tokamak_control.geometry.legacy_metrics import legacy_radii_at_angles
from tokamak_control.io.config_io import LoadedConfig, load_config
from tokamak_control.metrics import current_limit_margin, derivative_limit_margin
from tokamak_control.realism import RealismRuntime, RealismSettings


class SimulationSession:
    """Programmatic stepping session around the existing tokamak simulator."""

    def __init__(
        self,
        *,
        config_path: Path,
        initial_currents_path: Path | None,
        scenario_name: ScenarioName,
        scenario_args: Mapping[str, object],
        angles: int,
        steps: int,
        realism_enabled: bool = False,
        realism_settings: RealismSettings | None = None,
        initial_state_override: InitialStateOverride | None = None,
        compute_backend: ComputeBackend | str | None = None,
        gpu_device: str | None = None,
    ) -> None:
        """Создать описание сессии; модель строится при ``reset``."""
        if int(angles) <= 0:
            raise ValueError("angles must be > 0")
        if int(steps) <= 0:
            raise ValueError("steps must be > 0")
        self.config_path = Path(config_path)
        self.initial_currents_path = None if initial_currents_path is None else Path(initial_currents_path)
        self.scenario_name = scenario_name
        self.scenario_args = dict(scenario_args)
        self.angle_count = int(angles)
        self.steps = int(steps)
        self.realism_enabled = bool(realism_enabled)
        self.realism_settings = realism_settings
        self.initial_state_override = initial_state_override
        self.compute_backend = compute_backend
        self.gpu_device = gpu_device

        self._cfg: LoadedConfig | None = None
        self._model: PlasmaModel | GpuPlasmaModel | None = None
        self._machine: MachineSpec | None = None
        self._scenario: Scenario | None = None
        self._realism: RealismRuntime | None = None
        self._angles_rad: np.ndarray | None = None
        self._boundary_poly: np.ndarray | None = None
        self._boundary_level: float | None = None
        self._boundary_status: str | None = None
        self._last_commanded = np.zeros((0,), dtype=float)
        self._last_applied = np.zeros((0,), dtype=float)
        self._terminated = False
        self._termination_reason: str | None = None

    @classmethod
    def from_paths(
        cls,
        config_path: str | Path,
        initial_currents_path: str | Path | None,
        scenario_name: ScenarioName,
        scenario_args: Mapping[str, object] | None,
        angles: int,
        steps: int,
        seed: int | None = None,
        realism_enabled: bool = False,
        realism_settings: RealismSettings | None = None,
        initial_state_override: InitialStateOverride | None = None,
        compute_backend: ComputeBackend | str | None = None,
        gpu_device: str | None = None,
    ) -> SimulationSession:
        """Собрать сессию из путей без запуска шага модели."""
        _ = seed
        return cls(
            config_path=Path(config_path),
            initial_currents_path=None if initial_currents_path is None else Path(initial_currents_path),
            scenario_name=scenario_name,
            scenario_args={} if scenario_args is None else dict(scenario_args),
            angles=int(angles),
            steps=int(steps),
            realism_enabled=bool(realism_enabled),
            realism_settings=realism_settings,
            initial_state_override=initial_state_override,
            compute_backend=compute_backend,
            gpu_device=gpu_device,
        )

    def reset(
        self,
        *,
        initial_currents_path: str | Path | None = None,
        seed: int | None = None,
        realism_settings: RealismSettings | None = None,
        initial_state_override: InitialStateOverride | None = None,
    ) -> ResetResult:
        """Перезагрузить конфигурацию, модель, сценарий и начальную границу."""
        active_initial_path = (
            self.initial_currents_path
            if initial_currents_path is None
            else Path(initial_currents_path)
        )
        active_initial_override = initial_state_override if initial_state_override is not None else self.initial_state_override
        cfg = load_config(self.config_path, initial_currents_path=active_initial_path)
        if self.compute_backend is not None or self.gpu_device is not None:
            cfg = replace(
                cfg,
                compute=ComputeSettings(
                    backend=(cfg.compute.backend if self.compute_backend is None else self.compute_backend),
                    gpu_device=(cfg.compute.gpu_device if self.gpu_device is None else str(self.gpu_device)),
                ),
            )
        cfg.compute.validate(require_available=(cfg.compute.backend == "gpu"))
        compute_meta = compute_runtime_metadata(cfg.compute, validate=(cfg.compute.backend == "gpu"))
        cfg = _apply_initial_state_override(cfg, active_initial_override)
        model = _make_model(cfg)
        active_realism_settings = realism_settings if realism_settings is not None else (self.realism_settings if self.realism_settings is not None else cfg.realism)
        active_realism_settings.validate()
        realism = RealismRuntime(active_realism_settings, seed=seed) if (self.realism_enabled or active_realism_settings.enabled) and RealismRuntime.has_any_effect(active_realism_settings) else None
        angles_rad = np.linspace(-np.pi, np.pi, self.angle_count, endpoint=False, dtype=float)
        center = (model.R0, model.Z0)
        psi0 = _model_compute_psi_for_boundary(model)
        boundary_found = True
        boundary_reason = None
        try:
            boundary_poly, boundary_level, boundary_status = find_plasma_boundary_with_status(
                psi0,
                model.grid,
                center,
                n_levels=80 if cfg.limiter_shape is not None else 10,
                limiter_shape=cfg.limiter_shape,
                boundary_mode=cfg.boundary_mode,
                boundary_base_mode=cfg.boundary_base_mode,
                legacy_precision_index2=cfg.boundary_legacy_precision_index2,
                track_level=cfg.boundary_track_level,
                level_smoothing_alpha=cfg.boundary_level_smoothing_alpha,
                level_search_span_fraction=cfg.boundary_level_search_span_fraction,
                continuity_weight_radii=cfg.boundary_continuity_weight_radii,
                continuity_weight_mean_radius=cfg.boundary_continuity_weight_mean_radius,
                continuity_weight_center=cfg.boundary_continuity_weight_center,
                continuity_weight_area=cfg.boundary_continuity_weight_area,
                continuity_weight_level=cfg.boundary_continuity_weight_level,
                compute_backend=cfg.compute.backend,
                gpu_device=cfg.compute.gpu_device,
            )
            base_radii = legacy_radii_at_angles(boundary_poly, center, angles_rad)
        except BoundaryNotFoundError as exc:
            boundary_found = False
            boundary_reason = str(exc)
            boundary_poly = None
            boundary_level = None
            boundary_status = "not_found"
            base_radii = np.zeros_like(angles_rad, dtype=float)
        scenario = make_scenario(
            self.scenario_name,
            base_radii,
            float(model.Ip0),
            params=self.scenario_args,
            center=center,
        )
        machine = _build_machine_spec(
            cfg=cfg,
            model=model,
            config_path=self.config_path,
            initial_currents_path=active_initial_path,
            initial_state_override=active_initial_override,
            angles_rad=angles_rad,
            base_radii=base_radii,
        )

        self._cfg = cfg
        self._model = model
        self._machine = machine
        self._scenario = scenario
        self._realism = realism
        self._angles_rad = angles_rad
        self._boundary_poly = None if boundary_poly is None else np.asarray(boundary_poly, dtype=float)
        self._boundary_level = None if boundary_level is None else float(boundary_level)
        self._boundary_status = str(boundary_status)
        self._last_commanded = _active_currents(model).copy()
        self._last_applied = self._last_commanded.copy()
        self._terminated = False
        self._termination_reason = None

        snapshot = self._snapshot(
            commanded_active_currents=self._last_commanded,
            commanded_active_derivatives=np.zeros((machine.n_active_total,), dtype=float),
            previous_applied_active_derivatives=np.zeros((machine.n_active_total,), dtype=float),
            boundary_found=boundary_found,
            boundary_reason=boundary_reason,
        )
        return ResetResult(
            observation_snapshot=snapshot,
            machine=machine,
            episode_metadata={
                "scenario_name": str(self.scenario_name),
                "scenario_args": dict(self.scenario_args),
                "boundary_status": self._boundary_status,
                "initial_state_override": _initial_state_override_metadata(active_initial_override),
                "realism_active": realism is not None,
                "realism_settings": _realism_settings_metadata(active_realism_settings),
                "compute": compute_meta,
            },
        )

    def step_currents(self, action: CurrentAction) -> StepResult:
        """Apply absolute next currents for active actuators for one step."""
        if self._terminated:
            raise RuntimeError(f"SimulationSession is terminated: {self._termination_reason}")
        cfg, model, machine, scenario, angles_rad, realism = self._require_ready()
        active_currents_next = np.asarray(action.active_currents_next, dtype=float).reshape(-1)
        if active_currents_next.shape != (machine.n_active_total,):
            raise ValueError(
                f"active_currents_next shape {active_currents_next.shape} != ({machine.n_active_total},)"
            )
        if not np.all(np.isfinite(active_currents_next)):
            raise ValueError("active_currents_next must contain only finite values")

        prev_currents = _active_currents(model).copy()
        prev_applied = _active_derivatives(model).copy()
        pfc_next = active_currents_next[: machine.n_active_pfc]
        sol_next = active_currents_next[machine.n_active_pfc :]
        if realism is not None:
            actuation = realism.apply_actuation(pfc_next, sol_next)
            pfc_next = actuation.pfc_applied
            sol_next = actuation.sol_applied
        state = model.step_currents(pfc_currents_next=pfc_next, sol_currents_next=sol_next)
        psi_true = _model_compute_psi_for_boundary(model)
        if not isinstance(model, GpuPlasmaModel):
            model.state.psi = psi_true

        boundary_found = True
        boundary_reason = None
        try:
            ref_for_boundary = _reference_frame(scenario, angles_rad, float(state.t))
            boundary_poly, boundary_level, boundary_status = find_plasma_boundary_with_status(
                psi_true,
                model.grid,
                (model.R0, model.Z0),
                prev_level=self._boundary_level,
                prev_poly=self._boundary_poly,
                local_n_levels=7,
                local_span_frac=0.02,
                target_mean_radius=float(np.nanmean(ref_for_boundary.radii_ref)),
                limiter_shape=cfg.limiter_shape,
                boundary_mode=cfg.boundary_mode,
                boundary_base_mode=cfg.boundary_base_mode,
                legacy_precision_index2=cfg.boundary_legacy_precision_index2,
                track_level=cfg.boundary_track_level,
                level_smoothing_alpha=cfg.boundary_level_smoothing_alpha,
                level_search_span_fraction=cfg.boundary_level_search_span_fraction,
                continuity_weight_radii=cfg.boundary_continuity_weight_radii,
                continuity_weight_mean_radius=cfg.boundary_continuity_weight_mean_radius,
                continuity_weight_center=cfg.boundary_continuity_weight_center,
                continuity_weight_area=cfg.boundary_continuity_weight_area,
                continuity_weight_level=cfg.boundary_continuity_weight_level,
                compute_backend=cfg.compute.backend,
                gpu_device=cfg.compute.gpu_device,
            )
            self._boundary_poly = np.asarray(boundary_poly, dtype=float)
            self._boundary_level = float(boundary_level)
            self._boundary_status = str(boundary_status)
        except BoundaryNotFoundError as exc:
            boundary_found = False
            boundary_reason = str(exc)
            self._boundary_poly = None
            self._boundary_level = None
            self._boundary_status = None

        applied = _active_derivatives(model).copy()
        self._last_commanded = active_currents_next.copy()
        self._last_applied = _active_currents(model).copy()
        truncated = bool((not self._terminated) and int(state.step) >= self.steps)
        commanded_derivs = (active_currents_next - prev_currents) / max(float(model.t_step), 1.0e-12)
        snapshot = self._snapshot(
            commanded_active_currents=active_currents_next,
            commanded_active_derivatives=commanded_derivs,
            previous_applied_active_derivatives=prev_applied,
            boundary_found=boundary_found,
            boundary_reason=boundary_reason,
        )
        return StepResult(
            snapshot=snapshot,
            terminated=bool(self._terminated),
            truncated=truncated,
            termination_reason=self._termination_reason,
        )

    def reference_at_time(self, time_s: float) -> ReferenceFrame:
        """Return the active scenario reference at a requested time without stepping."""
        _cfg, _model, _machine, scenario, angles_rad, _realism = self._require_ready()
        return _reference_frame(scenario, angles_rad, float(time_s))

    def close(self) -> None:
        """Освободить ссылки на runtime-состояние сессии."""
        self._cfg = None
        self._model = None
        self._machine = None
        self._scenario = None
        self._realism = None
        self._angles_rad = None
        self._boundary_poly = None
        self._boundary_level = None
        self._boundary_status = None
        self._terminated = False
        self._termination_reason = None

    def _snapshot(
        self,
        *,
        commanded_active_currents: np.ndarray,
        commanded_active_derivatives: np.ndarray,
        previous_applied_active_derivatives: np.ndarray,
        boundary_found: bool,
        boundary_reason: str | None,
    ) -> StepSnapshot:
        """Собрать публичный снимок текущего состояния модели."""
        cfg, model, machine, scenario, angles_rad, realism = self._require_ready()
        state = model.snapshot_state()
        reference = _reference_frame(scenario, angles_rad, float(state.t))
        currents = _active_currents(model)
        applied = _active_derivatives(model)
        boundary_poly = None if self._boundary_poly is None else np.asarray(self._boundary_poly, dtype=float).copy()
        radii = None
        if boundary_poly is not None:
            radii = legacy_radii_at_angles(boundary_poly, (model.R0, model.Z0), angles_rad)
        if realism is None:
            measured_ip = float(state.Ip)
            measured_currents = currents.copy()
            measured_boundary_poly = None if boundary_poly is None else boundary_poly.copy()
            measured_radii = None if radii is None else np.asarray(radii, dtype=float).copy()
        else:
            sensors = realism.measure(
                true_ip=float(state.Ip),
                true_active_currents=currents,
                true_boundary_poly=boundary_poly,
                true_radii=radii,
                true_psi=np.asarray(state.psi, dtype=float),
                model=model,
                center=(model.R0, model.Z0),
                angles_rad=angles_rad,
                limiter_shape=cfg.limiter_shape,
                boundary_mode=cfg.boundary_mode,
                boundary_base_mode=cfg.boundary_base_mode,
                legacy_precision_index2=cfg.boundary_legacy_precision_index2,
                track_level=cfg.boundary_track_level,
                level_smoothing_alpha=cfg.boundary_level_smoothing_alpha,
                level_search_span_fraction=cfg.boundary_level_search_span_fraction,
                continuity_weight_radii=cfg.boundary_continuity_weight_radii,
                continuity_weight_mean_radius=cfg.boundary_continuity_weight_mean_radius,
                continuity_weight_center=cfg.boundary_continuity_weight_center,
                continuity_weight_area=cfg.boundary_continuity_weight_area,
                continuity_weight_level=cfg.boundary_continuity_weight_level,
                compute_backend=cfg.compute.backend,
                gpu_device=cfg.compute.gpu_device,
            )
            measured_ip = sensors.measured_ip
            measured_currents = sensors.measured_active_currents
            measured_boundary_poly = sensors.measured_boundary_poly
            measured_radii = sensors.measured_radii
        current_margin = _safe_current_margin(currents, machine.current_limits)
        derivative_margin = _safe_derivative_margin(applied, machine.derivative_limits)
        return StepSnapshot(
            step_index=int(state.step),
            time_s=float(state.t),
            reference=reference,
            true_ip=float(state.Ip),
            measured_ip=float(measured_ip),
            true_active_currents=currents,
            measured_active_currents=np.asarray(measured_currents, dtype=float).copy(),
            commanded_active_currents=np.asarray(commanded_active_currents, dtype=float).copy(),
            applied_active_currents=currents.copy(),
            commanded_active_derivatives=np.asarray(commanded_active_derivatives, dtype=float).copy(),
            applied_active_derivatives=applied,
            previous_applied_active_derivatives=np.asarray(previous_applied_active_derivatives, dtype=float).copy(),
            true_boundary_poly=boundary_poly,
            measured_boundary_poly=None if measured_boundary_poly is None else np.asarray(measured_boundary_poly, dtype=float).copy(),
            true_radii=None if radii is None else np.asarray(radii, dtype=float).copy(),
            measured_radii=None if measured_radii is None else np.asarray(measured_radii, dtype=float).copy(),
            psi_boundary_value=None if self._boundary_level is None else float(self._boundary_level),
            boundary_found=bool(boundary_found),
            boundary_reason=boundary_reason,
            current_limit_margin=current_margin,
            derivative_limit_margin=derivative_margin,
        )

    def _require_ready(self) -> tuple[LoadedConfig, PlasmaModel | GpuPlasmaModel, MachineSpec, Scenario, np.ndarray, RealismRuntime | None]:
        """Вернуть runtime-объекты после успешного reset."""
        if self._cfg is None or self._model is None or self._machine is None or self._scenario is None or self._angles_rad is None:
            raise RuntimeError("SimulationSession.reset() must be called before stepping")
        return self._cfg, self._model, self._machine, self._scenario, self._angles_rad, self._realism


def _make_model(cfg: LoadedConfig) -> PlasmaModel | GpuPlasmaModel:
    if cfg.compute.backend == "gpu":
        return GpuPlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics, gpu_device=cfg.compute.gpu_device)
    return PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics)


def _model_compute_psi_for_boundary(model: PlasmaModel | GpuPlasmaModel):
    if isinstance(model, GpuPlasmaModel):
        return model.compute_psi_tensor()
    return model.compute_psi()


def _apply_initial_state_override(cfg: LoadedConfig, override: InitialStateOverride | None) -> LoadedConfig:
    """Apply explicit bridge-only initial Ip/current overrides to a loaded config."""
    if override is None:
        return cfg
    physics = cfg.physics if override.ip is None else replace(cfg.physics, Ip0=float(override.ip))
    pfc = cfg.pfc
    sol = cfg.sol
    if override.coil_currents == "zero":
        pfc = CoilGroup(name=cfg.pfc.name, coils=list(cfg.pfc.coils), currents=np.zeros((cfg.pfc.n_coils,), dtype=float))
        sol = CoilGroup(name=cfg.sol.name, coils=list(cfg.sol.coils), currents=np.zeros((cfg.sol.n_coils,), dtype=float))
    elif override.coil_currents == "explicit":
        pfc_values = np.asarray(override.pfc_currents, dtype=float).reshape(-1)
        sol_values = np.asarray(override.sol_currents, dtype=float).reshape(-1)
        if pfc_values.shape != (cfg.pfc.n_coils,):
            raise ValueError(f"initial override pfc_currents shape {pfc_values.shape} != ({cfg.pfc.n_coils},)")
        if sol_values.shape != (cfg.sol.n_coils,):
            raise ValueError(f"initial override sol_currents shape {sol_values.shape} != ({cfg.sol.n_coils},)")
        pfc = CoilGroup(name=cfg.pfc.name, coils=list(cfg.pfc.coils), currents=pfc_values.copy())
        sol = CoilGroup(name=cfg.sol.name, coils=list(cfg.sol.coils), currents=sol_values.copy())
    return replace(cfg, physics=physics, pfc=pfc, sol=sol)


def _initial_state_override_metadata(override: InitialStateOverride | None) -> dict[str, object]:
    if override is None:
        return {"enabled": False, "ip": None, "coil_currents": "config", "ip_scale": None}
    data: dict[str, object] = {
        "enabled": bool(override.ip is not None or override.coil_currents != "config" or override.ip_scale is not None),
        "ip": None if override.ip is None else float(override.ip),
        "coil_currents": str(override.coil_currents),
        "ip_scale": None if override.ip_scale is None else float(override.ip_scale),
    }
    if override.pfc_currents is not None:
        data["pfc_currents"] = np.asarray(override.pfc_currents, dtype=float).tolist()
    if override.sol_currents is not None:
        data["sol_currents"] = np.asarray(override.sol_currents, dtype=float).tolist()
    return data


def _build_machine_spec(
    *,
    cfg: LoadedConfig,
    model: PlasmaModel,
    config_path: Path,
    initial_currents_path: Path | None,
    initial_state_override: InitialStateOverride | None,
    angles_rad: np.ndarray,
    base_radii: np.ndarray,
) -> MachineSpec:
    """Собрать MachineSpec из загруженной конфигурации и начальной границы."""
    n_pfc = int(model.pfc.n_coils)
    n_sol = int(model.sol.n_coils)
    current_limits = np.concatenate([
        _bank_limit_vector(cfg.physics.pfc_current_limit, n_pfc),
        _bank_limit_vector(cfg.physics.sol_current_limit, n_sol),
    ])
    derivative_limits = np.concatenate([
        _bank_limit_vector(cfg.physics.pfc_deriv_limit, n_pfc),
        _bank_limit_vector(cfg.physics.sol_deriv_limit, n_sol),
    ])
    currents = _active_currents(model)
    radius_scale = _radius_scale(cfg=cfg, model=model, base_radii=base_radii)
    return MachineSpec(
        config_path=Path(config_path),
        initial_currents_path=None if initial_currents_path is None else Path(initial_currents_path),
        boundary_mode=str(cfg.boundary_mode),
        compute_backend=str(cfg.compute.backend),
        gpu_device=str(cfg.compute.gpu_device),
        limiter_name=cfg.limiter_name,
        t_step=float(model.t_step),
        n_active_pfc=n_pfc,
        n_active_sol=n_sol,
        n_active_total=n_pfc + n_sol,
        active_order=tuple([f"pfc_{i}" for i in range(n_pfc)] + [f"sol_{i}" for i in range(n_sol)]),
        center=(float(model.R0), float(model.Z0)),
        pfc_active_mask=np.asarray(cfg.pfc_active_mask, dtype=bool).copy() if cfg.pfc_active_mask is not None else np.ones((n_pfc,), dtype=bool),
        sol_active_mask=np.asarray(cfg.sol_active_mask, dtype=bool).copy() if cfg.sol_active_mask is not None else np.ones((n_sol,), dtype=bool),
        current_limits=current_limits,
        derivative_limits=derivative_limits,
        angles_rad=np.asarray(angles_rad, dtype=float).copy(),
        radius_scale=radius_scale,
        ip_scale=_ip_scale(model=model, initial_state_override=initial_state_override),
        current_scale=_scale_from_limits_or_values(current_limits, currents),
        derivative_scale=_scale_from_limits_or_values(derivative_limits, np.ones((n_pfc + n_sol,), dtype=float)),
    )


def _ip_scale(*, model: PlasmaModel, initial_state_override: InitialStateOverride | None) -> float:
    if initial_state_override is not None and initial_state_override.ip_scale is not None:
        return float(initial_state_override.ip_scale)
    return max(abs(float(model.Ip0)), 1.0)


def _realism_settings_metadata(settings: RealismSettings) -> dict[str, object]:
    data = asdict(settings)
    return {
        "enabled": bool(data["enabled"]),
        "seed": data["seed"],
        "actuators": dict(data["actuators"]),
        "sensors": dict(data["sensors"]),
    }


def _bank_limit_vector(limit: float | None, size: int) -> np.ndarray:
    """Развернуть bank-level limit в active-actuator vector."""
    if int(size) <= 0:
        return np.zeros((0,), dtype=float)
    if limit is None:
        return np.full((int(size),), np.nan, dtype=float)
    value = float(limit)
    return np.full((int(size),), value if value > 0.0 and np.isfinite(value) else np.nan, dtype=float)


def _scale_from_limits_or_values(limits: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Выбрать положительный scale из limit vector или runtime values."""
    lim = np.asarray(limits, dtype=float).reshape(-1)
    val = np.asarray(values, dtype=float).reshape(-1)
    out = np.where(np.isfinite(lim) & (lim > 0.0), lim, np.maximum(np.abs(val), 1.0))
    return np.asarray(out, dtype=float)


def _radius_scale(*, cfg: LoadedConfig, model: PlasmaModel, base_radii: np.ndarray) -> float:
    """Оценить радиальный масштаб для внешней нормировки diagnostics."""
    center = np.array([float(model.R0), float(model.Z0)], dtype=float)
    if cfg.limiter_shape is not None and np.asarray(cfg.limiter_shape).size:
        pts = np.asarray(cfg.limiter_shape, dtype=float).reshape(-1, 2)
        return max(float(np.max(np.linalg.norm(pts - center, axis=1))), 1.0)
    radii = np.asarray(base_radii, dtype=float).reshape(-1)
    finite = radii[np.isfinite(radii)]
    if finite.size == 0:
        return 1.0
    return max(float(np.max(np.abs(finite))), 1.0)


def _reference_frame(scenario: Scenario, angles_rad: np.ndarray, time_s: float) -> ReferenceFrame:
    """Вычислить reference frame сценария на заданном времени."""
    radii_ref = np.asarray(scenario.ref_radii(angles_rad, float(time_s)), dtype=float).reshape(-1)
    if radii_ref.shape != np.asarray(angles_rad).reshape(-1).shape:
        raise ValueError("scenario.ref_radii returned a vector with unexpected shape")
    return ReferenceFrame(
        time_s=float(time_s),
        ip_ref=float(scenario.Ip_ref(float(time_s))),
        radii_ref=radii_ref,
        metadata={"scenario_name": scenario.name},
    )


def _active_currents(model: PlasmaModel) -> np.ndarray:
    """Вернуть токи активных PFC/SOL актуаторов в bridge order."""
    state = model.snapshot_state()
    return np.concatenate([
        np.asarray(state.pfc_currents, dtype=float).reshape(-1),
        np.asarray(state.sol_currents, dtype=float).reshape(-1),
    ])


def _active_derivatives(model: PlasmaModel) -> np.ndarray:
    """Вернуть примененные производные токов активных PFC/SOL актуаторов."""
    state = model.snapshot_state()
    return np.concatenate([
        np.asarray(state.pfc_current_derivs, dtype=float).reshape(-1),
        np.asarray(state.sol_current_derivs, dtype=float).reshape(-1),
    ])


def _safe_current_margin(currents: np.ndarray, limits: np.ndarray) -> np.ndarray | None:
    """Вернуть current margins, если все пределы заданы конечно."""
    if not np.all(np.isfinite(limits)):
        return None
    return current_limit_margin(currents, limits)


def _safe_derivative_margin(derivatives: np.ndarray, limits: np.ndarray) -> np.ndarray | None:
    """Вернуть derivative margins, если все пределы заданы конечно."""
    if not np.all(np.isfinite(limits)):
        return None
    return derivative_limit_margin(derivatives, limits)
