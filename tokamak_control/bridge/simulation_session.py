from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from tokamak_control.bridge.types import DerivativeAction, MachineSpec, ReferenceFrame, ResetResult, StepResult, StepSnapshot
from tokamak_control.config.scenarios import Scenario, ScenarioName, make_scenario
from tokamak_control.core.plasma_model import PlasmaModel
from tokamak_control.geometry.boundary import BoundaryMode, BoundaryNotFoundError, find_plasma_boundary_with_status
from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections
from tokamak_control.io.config_io import LoadedConfig, load_config
from tokamak_control.metrics import current_limit_margin, derivative_limit_margin


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

        self._cfg: LoadedConfig | None = None
        self._model: PlasmaModel | None = None
        self._machine: MachineSpec | None = None
        self._scenario: Scenario | None = None
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
        )

    def reset(
        self,
        *,
        initial_currents_path: str | Path | None = None,
        seed: int | None = None,
    ) -> ResetResult:
        """Перезагрузить конфигурацию, модель, сценарий и начальную границу."""
        _ = seed
        active_initial_path = (
            self.initial_currents_path
            if initial_currents_path is None
            else Path(initial_currents_path)
        )
        cfg = load_config(self.config_path, initial_currents_path=active_initial_path)
        model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics)
        angles_rad = np.linspace(-np.pi, np.pi, self.angle_count, endpoint=False, dtype=float)
        center = (model.R0, model.Z0)
        psi0 = model.compute_psi()
        boundary_poly, boundary_level, boundary_status = find_plasma_boundary_with_status(
            psi0,
            model.grid,
            center,
            n_levels=80 if cfg.limiter_shape is not None else 10,
            limiter_shape=cfg.limiter_shape,
            boundary_mode=cfg.boundary_mode,
        )
        base_radii = radii_from_polyline_ray_intersections(boundary_poly, center, angles_rad)
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
            angles_rad=angles_rad,
            base_radii=base_radii,
        )

        self._cfg = cfg
        self._model = model
        self._machine = machine
        self._scenario = scenario
        self._angles_rad = angles_rad
        self._boundary_poly = np.asarray(boundary_poly, dtype=float)
        self._boundary_level = float(boundary_level)
        self._boundary_status = str(boundary_status)
        self._last_commanded = np.zeros((machine.n_active_total,), dtype=float)
        self._last_applied = np.zeros((machine.n_active_total,), dtype=float)
        self._terminated = False
        self._termination_reason = None

        snapshot = self._snapshot(
            commanded_active_derivatives=self._last_commanded,
            previous_applied_active_derivatives=self._last_applied,
            boundary_found=True,
            boundary_reason=None,
        )
        return ResetResult(
            observation_snapshot=snapshot,
            machine=machine,
            episode_metadata={
                "scenario_name": str(self.scenario_name),
                "scenario_args": dict(self.scenario_args),
                "boundary_status": self._boundary_status,
            },
        )

    def step_derivatives(self, action: DerivativeAction) -> StepResult:
        """Применить физические производные токов активных актуаторов на один шаг."""
        if self._terminated:
            raise RuntimeError(f"SimulationSession is terminated: {self._termination_reason}")
        cfg, model, machine, scenario, angles_rad = self._require_ready()
        active_derivs = np.asarray(action.active_current_derivatives, dtype=float).reshape(-1)
        if active_derivs.shape != (machine.n_active_total,):
            raise ValueError(
                f"active_current_derivatives shape {active_derivs.shape} != ({machine.n_active_total},)"
            )
        if not np.all(np.isfinite(active_derivs)):
            raise ValueError("active_current_derivatives must contain only finite values")

        prev_applied = _active_derivatives(model).copy()
        pfc_derivs = active_derivs[: machine.n_active_pfc]
        sol_derivs = active_derivs[machine.n_active_pfc :]
        state = model.step(pfc_current_derivs=pfc_derivs, sol_current_derivs=sol_derivs)
        psi_true = model.compute_psi()
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
            self._terminated = True
            self._termination_reason = boundary_reason

        applied = _active_derivatives(model).copy()
        self._last_commanded = active_derivs.copy()
        self._last_applied = applied.copy()
        truncated = bool((not self._terminated) and int(state.step) >= self.steps)
        snapshot = self._snapshot(
            commanded_active_derivatives=active_derivs,
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

    def close(self) -> None:
        """Освободить ссылки на runtime-состояние сессии."""
        self._cfg = None
        self._model = None
        self._machine = None
        self._scenario = None
        self._angles_rad = None
        self._boundary_poly = None
        self._boundary_level = None
        self._boundary_status = None
        self._terminated = False
        self._termination_reason = None

    def _snapshot(
        self,
        *,
        commanded_active_derivatives: np.ndarray,
        previous_applied_active_derivatives: np.ndarray,
        boundary_found: bool,
        boundary_reason: str | None,
    ) -> StepSnapshot:
        """Собрать публичный снимок текущего состояния модели."""
        _cfg, model, machine, scenario, angles_rad = self._require_ready()
        state = model.snapshot_state()
        reference = _reference_frame(scenario, angles_rad, float(state.t))
        currents = _active_currents(model)
        applied = _active_derivatives(model)
        boundary_poly = None if self._boundary_poly is None else np.asarray(self._boundary_poly, dtype=float).copy()
        radii = None
        if boundary_poly is not None:
            radii = radii_from_polyline_ray_intersections(boundary_poly, (model.R0, model.Z0), angles_rad)
        current_margin = _safe_current_margin(currents, machine.current_limits)
        derivative_margin = _safe_derivative_margin(applied, machine.derivative_limits)
        return StepSnapshot(
            step_index=int(state.step),
            time_s=float(state.t),
            reference=reference,
            true_ip=float(state.Ip),
            true_active_currents=currents,
            commanded_active_derivatives=np.asarray(commanded_active_derivatives, dtype=float).copy(),
            applied_active_derivatives=applied,
            previous_applied_active_derivatives=np.asarray(previous_applied_active_derivatives, dtype=float).copy(),
            true_boundary_poly=boundary_poly,
            true_radii=None if radii is None else np.asarray(radii, dtype=float).copy(),
            psi_boundary_value=None if self._boundary_level is None else float(self._boundary_level),
            boundary_found=bool(boundary_found),
            boundary_reason=boundary_reason,
            current_limit_margin=current_margin,
            derivative_limit_margin=derivative_margin,
        )

    def _require_ready(self) -> tuple[LoadedConfig, PlasmaModel, MachineSpec, Scenario, np.ndarray]:
        """Вернуть runtime-объекты после успешного reset."""
        if self._cfg is None or self._model is None or self._machine is None or self._scenario is None or self._angles_rad is None:
            raise RuntimeError("SimulationSession.reset() must be called before stepping")
        return self._cfg, self._model, self._machine, self._scenario, self._angles_rad


def _build_machine_spec(
    *,
    cfg: LoadedConfig,
    model: PlasmaModel,
    config_path: Path,
    initial_currents_path: Path | None,
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
        limiter_name=cfg.limiter_name,
        t_step=float(model.t_step),
        n_active_pfc=n_pfc,
        n_active_sol=n_sol,
        n_active_total=n_pfc + n_sol,
        active_order=tuple([f"pfc_{i}" for i in range(n_pfc)] + [f"sol_{i}" for i in range(n_sol)]),
        pfc_active_mask=np.asarray(cfg.pfc_active_mask, dtype=bool).copy() if cfg.pfc_active_mask is not None else np.ones((n_pfc,), dtype=bool),
        sol_active_mask=np.asarray(cfg.sol_active_mask, dtype=bool).copy() if cfg.sol_active_mask is not None else np.ones((n_sol,), dtype=bool),
        current_limits=current_limits,
        derivative_limits=derivative_limits,
        angles_rad=np.asarray(angles_rad, dtype=float).copy(),
        radius_scale=radius_scale,
        ip_scale=max(abs(float(model.Ip0)), 1.0),
        current_scale=_scale_from_limits_or_values(current_limits, currents),
        derivative_scale=_scale_from_limits_or_values(derivative_limits, np.ones((n_pfc + n_sol,), dtype=float)),
    )


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
