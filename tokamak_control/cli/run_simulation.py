"""Canonical single-run tokamak simulation orchestration API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import time
from typing import Literal

import numpy as np

from tokamak_control.compute import ComputeBackend, ComputeSettings, compute_runtime_metadata, normalize_compute_backend
from tokamak_control.config.scenarios import Scenario, ScenarioName, make_scenario
from tokamak_control.control.base import ControlAction, Controller
from tokamak_control.control.registry import (
    build_controller_runtime_call,
    make_controller,
    normalize_controller_launch,
)
from tokamak_control.core.batched_gpu_simulator import BatchedGpuTokamakSimulator, BatchedGpuSimulatorResult
from tokamak_control.core.gpu_plasma_model import GpuPlasmaModel
from tokamak_control.core.plasma_model import (
    PlasmaModel,
    configure_plasma_model_profiling,
    log_plasma_model_profiling_summary,
    plasma_model_profiling_snapshot,
)
from tokamak_control.core.plasma_state import PlasmaState
from tokamak_control.experiments.disturbances import (
    Disturbance,
    apply_prepared_disturbances,
    prepare_disturbances,
    IpCrash,
)
from tokamak_control.geometry.boundary import (
    BoundaryMode,
    BoundaryNotFoundError,
    configure_boundary_profiling,
    find_plasma_boundary_with_status,
    log_boundary_profiling_summary,
    boundary_profiling_snapshot,
)
from tokamak_control.geometry.legacy_metrics import legacy_measurement_angles_from_actuators, legacy_radii_at_angles
from tokamak_control.io.config_io import LoadedConfig, load_config
from tokamak_control.io.data_io import RunWriter
from tokamak_control.io.logger import configure_logging, get_logger
from tokamak_control.io.profiling import Profiler
from tokamak_control.realism import RealismRuntime, SensorRealismResult

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


LaunchScenarioName = Literal[
    "nominal",
    "boundary_step",
    "ip_ramp",
    "ip_flat_top",
    "ip_jet_like",
    "boundary_pulse",
    "joint_disturbance",
    "shot_follow",
    "ip_table",
    "t15_replay_reference",
    "ip_follow",
    "t15_synthetic_follow",
    "ip_crash",
]


@dataclass(frozen=True, slots=True)
class RunResult:
    """Пути и основные metadata завершенного или остановленного расчета."""

    run_dir: Path
    manifest_path: Path
    npz_path: Path
    events_path: Path
    angles: np.ndarray
    last_ref_radii: np.ndarray
    completed_steps: int = 0
    stop_reason: str | None = None
    boundary_missing_step: int | None = None


@dataclass(slots=True)
class _BoundaryTracker:
    """Хранит текущий физически найденный контур плазмы во время расчета."""

    poly: np.ndarray | None
    level: float
    status: str
    fail_reason: str | None
    found: bool = True


@dataclass(frozen=True, slots=True)
class _RunPaths:
    """Пути основных артефактов одного запуска."""

    run_dir: Path
    manifest_path: Path
    profile_path: Path
    run_id: int


@dataclass(frozen=True, slots=True)
class _StepRefs:
    """Опорные сигналы сценария для одного шага."""

    ref_radii: np.ndarray
    ip_ref: float
    target_mean_radius: float


@dataclass(frozen=True, slots=True)
class _StepCommands:
    """Абсолютные команды следующего тока до и после слоя реализма."""

    pfc_cmd: np.ndarray
    sol_cmd: np.ndarray
    pfc_eff: np.ndarray
    sol_eff: np.ndarray


@dataclass(frozen=True, slots=True)
class _StepRecord:
    """Данные, которые нужно записать после одного шага модели."""

    state: PlasmaState
    commands: _StepCommands
    refs: _StepRefs
    ref_radii_log: np.ndarray
    ip_ref_log: float
    sensors: SensorRealismResult
    disturbances_applied: list[str]
    controller_diagnostics: dict[str, object]


class _BatchedGpuModelView:
    """Single-lane model facade passed to exported learned controllers.

    Learned policies are trained against the batched GPU environment, whose
    boundary signal is fixed-angle radii rather than a full CPU contour. This
    view exposes the minimal ``PlasmaModel``-like surface the exported
    controller expects while the actual plant state lives in
    ``BatchedGpuTokamakSimulator``.
    """

    def __init__(self, *, simulator: BatchedGpuTokamakSimulator, state: PlasmaState) -> None:
        self._simulator = simulator
        self.grid = simulator.grid
        self.pfc = simulator.pfc
        self.sol = simulator.sol
        self.R0 = float(simulator.settings.R0)
        self.Z0 = float(simulator.settings.Z0)
        self.t_step = float(simulator.settings.t_step)
        self.actuator_tau = float(simulator.settings.actuator_tau)
        self.state = state

    def snapshot_state(self) -> PlasmaState:
        return self.state.copied()



def _as_numpy_psi(psi: object) -> np.ndarray:
    """Return a CPU NumPy psi array from either NumPy or Torch input."""
    if hasattr(psi, "detach"):
        return np.asarray(psi.detach().cpu().numpy(), dtype=float)
    return np.asarray(psi, dtype=float)


def _tensor_lane_to_numpy(value: object, lane: int = 0, *, dtype=float) -> np.ndarray:
    """Return one lane from a tensor/array batch as a NumPy array."""
    item = value[lane]  # type: ignore[index]
    if hasattr(item, "detach"):
        item = item.detach()
    if hasattr(item, "cpu"):
        item = item.cpu()
    return np.asarray(item, dtype=dtype)


def _tensor_lane_scalar(value: object, lane: int = 0, *, cast=float) -> object:
    """Return one scalar lane from a tensor/array batch."""
    item = value[lane]  # type: ignore[index]
    if hasattr(item, "detach"):
        item = item.detach()
    if hasattr(item, "cpu"):
        item = item.cpu()
    if hasattr(item, "item"):
        item = item.item()
    return cast(item)


def _state_from_batched_result(result: BatchedGpuSimulatorResult, simulator: BatchedGpuTokamakSimulator) -> PlasmaState:
    """Convert the single-lane batched GPU state to a normal artifact state."""
    return PlasmaState(
        t=float(_tensor_lane_scalar(result.state.t, cast=float)),
        step=int(_tensor_lane_scalar(result.state.step, cast=int)),
        Ip=float(_tensor_lane_scalar(result.state.Ip, cast=float)),
        Ip0=float(_tensor_lane_scalar(simulator.Ip0, cast=float)),
        psi=_tensor_lane_to_numpy(result.state.psi, dtype=float),
        pfc_currents=_tensor_lane_to_numpy(result.state.pfc_currents, dtype=float).reshape(simulator.pfc.n_coils),
        pfc_current_derivs=_tensor_lane_to_numpy(result.state.pfc_current_derivs, dtype=float).reshape(simulator.pfc.n_coils),
        sol_currents=_tensor_lane_to_numpy(result.state.sol_currents, dtype=float).reshape(simulator.sol.n_coils),
        sol_current_derivs=_tensor_lane_to_numpy(result.state.sol_current_derivs, dtype=float).reshape(simulator.sol.n_coils),
    )


def _fixed_angle_radii_from_result(result: BatchedGpuSimulatorResult) -> np.ndarray:
    """Return the single-lane fixed-angle boundary radii from a batched GPU result."""
    return _tensor_lane_to_numpy(result.boundary.radii, dtype=float).reshape(-1)


def _fixed_angle_found_from_result(result: BatchedGpuSimulatorResult) -> bool:
    """Return whether the single-lane fixed-angle boundary was found."""
    return bool(_tensor_lane_scalar(result.boundary.found, cast=bool))


def _fixed_angle_polyline(
    *,
    center: tuple[float, float],
    angles: np.ndarray,
    radii: np.ndarray,
    found: bool,
) -> np.ndarray | None:
    """Build a closed display polyline from fixed-angle boundary radii."""
    del found
    r = np.asarray(radii, dtype=float).reshape(-1)
    a = np.asarray(angles, dtype=float).reshape(-1)
    finite = np.isfinite(r) & (r > 0.0)
    if r.shape != a.shape or int(np.count_nonzero(finite)) < 3:
        return None
    R0, Z0 = center
    points = np.stack([R0 + r[finite] * np.cos(a[finite]), Z0 + r[finite] * np.sin(a[finite])], axis=1)
    if points.shape[0] >= 2 and not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[:1]])
    return points.astype(float, copy=False)


def _coerce_value(token: str) -> object:
    s = token.strip()
    low = s.lower()

    if low == "true":
        return True
    if low == "false":
        return False
    if low == "none":
        return None

    if re.fullmatch(r"[+-]?\d+", s):
        try:
            return int(s)
        except ValueError:
            pass

    try:
        val = float(s)
        if np.isfinite(val):
            return val
    except ValueError:
        pass

    return s


def parse_key_value_args(items: Sequence[str] | None) -> dict[str, object]:
    out: dict[str, object] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"Expected key=value argument, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Expected non-empty key in {item!r}")
        out[key] = _coerce_value(value)
    return out


def resolve_runtime_scenario(
    *,
    scenario_name: LaunchScenarioName,
    steps: int,
    scenario_params: Mapping[str, object] | None = None,
) -> tuple[ScenarioName, dict[str, object], list[Disturbance]]:
    params = {} if scenario_params is None else dict(scenario_params)

    if scenario_name == "ip_crash":
        return "nominal", params, [IpCrash.default_for_run(int(steps))]

    return scenario_name, params, []


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-") or "run"


def _make_run_stem(
    *,
    controller_name: str,
    scenario_name: str,
    steps: int,
    realism_enabled: bool,
    disturbances: Sequence[Disturbance] | None,
) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    realism_tag = "realism-on" if realism_enabled else "realism-off"
    disturbance_tag = "dist-none"
    if disturbances:
        names = "-".join(sorted({d.__class__.__name__.lower() for d in disturbances}))
        disturbance_tag = f"dist-{_slug(names)}"
    return _slug(
        f"{ts}_{controller_name}_{scenario_name}_{realism_tag}_{disturbance_tag}_steps-{steps}"
    )


def _allocate_run_dir(root: Path, stem: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / stem
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    i = 2
    while True:
        candidate = root / f"{stem}_{i}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        i += 1


def _serialize_disturbance_template(d: Disturbance) -> dict[str, object]:
    params = {k: v for k, v in vars(d).items() if not k.startswith("_")}
    return {
        "type": d.__class__.__name__,
        "params": params,
    }


def _build_run_metadata(
    *,
    run_id: int,
    cfg: LoadedConfig,
    config_source: str,
    controller_name: str,
    controller_params: Mapping[str, object],
    scenario_name: str,
    scenario_params: Mapping[str, object],
    disturbances: Sequence[Disturbance] | None,
    realism_enabled: bool,
    steps: int,
    profiling_enabled: bool,
    runtime_overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "run_id": int(run_id),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_source": config_source,
        "initial_currents_source": cfg.initial_currents_source,
        "steps": int(steps),
        "controller": {
            "name": controller_name,
            "params": dict(controller_params),
        },
        "scenario": {
            "name": scenario_name,
            "params": dict(scenario_params),
        },
        "disturbances": [
            _serialize_disturbance_template(d) for d in (disturbances or ())
        ],
        "realism_enabled": bool(realism_enabled),
        "realism": _json_safe(cfg.realism),
        "profiling_enabled": bool(profiling_enabled),
        "runtime_overrides": dict(runtime_overrides or {}),
        "grid": {
            "r_start": float(cfg.grid.r.start),
            "r_step": float(cfg.grid.r.step),
            "r_size": int(cfg.grid.r.size),
            "r_center": float(cfg.grid.r.center),
            "z_start": float(cfg.grid.z.start),
            "z_step": float(cfg.grid.z.step),
            "z_size": int(cfg.grid.z.size),
            "z_center": float(cfg.grid.z.center),
        },
        "center": {
            "R0": float(cfg.physics.R0),
            "Z0": float(cfg.physics.Z0),
        },
        "physics": {
            "Ip0": float(cfg.physics.Ip0),
            "t_step": float(cfg.physics.t_step),
            "sigma": float(cfg.physics.sigma),
            "inductance_L": float(cfg.physics.inductance_L),
            "ip_coupling_pfc": None if cfg.physics.ip_coupling_pfc is None else list(cfg.physics.ip_coupling_pfc),
            "ip_coupling_sol": None if cfg.physics.ip_coupling_sol is None else list(cfg.physics.ip_coupling_sol),
            "plasma_psi_sign": float(cfg.physics.plasma_psi_sign),
        },
        "coil_positions": {
            "pfc": np.asarray(cfg.pfc.positions, dtype=float).tolist(),
            "sol": np.asarray(cfg.sol.positions, dtype=float).tolist(),
        },
        "active_coils": {
            "pfc": None if cfg.pfc_active_mask is None else np.asarray(cfg.pfc_active_mask, dtype=bool).tolist(),
            "sol": None if cfg.sol_active_mask is None else np.asarray(cfg.sol_active_mask, dtype=bool).tolist(),
        },
        "boundary": {
            "mode": cfg.boundary_mode,
            "base_mode": cfg.boundary_base_mode,
            "legacy_precision_index2": cfg.boundary_legacy_precision_index2,
            "track_level": cfg.boundary_track_level,
            "smooth_selected_level": cfg.boundary_smooth_selected_level,
            "soft_level_selection": cfg.boundary_soft_level_selection,
            "soft_level_candidates": cfg.boundary_soft_level_candidates,
            "soft_level_temperature": cfg.boundary_soft_level_temperature,
            "soft_level_radius_weight": cfg.boundary_soft_level_radius_weight,
            "soft_level_missing_penalty": cfg.boundary_soft_level_missing_penalty,
            "soft_level_roughness_penalty": cfg.boundary_soft_level_roughness_penalty,
            "level_smoothing_alpha": cfg.boundary_level_smoothing_alpha,
            "level_search_span_fraction": cfg.boundary_level_search_span_fraction,
            "continuity_weight_radii": cfg.boundary_continuity_weight_radii,
            "continuity_weight_mean_radius": cfg.boundary_continuity_weight_mean_radius,
            "continuity_weight_center": cfg.boundary_continuity_weight_center,
            "continuity_weight_area": cfg.boundary_continuity_weight_area,
            "continuity_weight_level": cfg.boundary_continuity_weight_level,
        },
        "compute": compute_runtime_metadata(cfg.compute, validate=False),
        "limiter": {
            "name": cfg.limiter_name,
            "shape": None if cfg.limiter_shape is None else np.asarray(cfg.limiter_shape, dtype=float).tolist(),
        },
    }


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value




def _effective_config_for_controller(
    cfg: LoadedConfig,
    *,
    controller_name: str,
) -> tuple[LoadedConfig, dict[str, object]]:
    if controller_name != "t15md_replay":
        return cfg, {}

    old = cfg.physics
    new_physics = replace(
        old,
        actuator_tau=0.0,
        pfc_current_limit=None,
        sol_current_limit=None,
        pfc_deriv_limit=None,
        sol_deriv_limit=None,
    )
    out = replace(cfg, physics=new_physics)
    overrides = {
        "t15md_replay_exact_applied_current_replay": True,
        "physics.actuator_tau": {"from": old.actuator_tau, "to": new_physics.actuator_tau},
        "physics.pfc_current_limit": {"from": old.pfc_current_limit, "to": new_physics.pfc_current_limit},
        "physics.sol_current_limit": {"from": old.sol_current_limit, "to": new_physics.sol_current_limit},
        "physics.pfc_deriv_limit": {"from": old.pfc_deriv_limit, "to": new_physics.pfc_deriv_limit},
        "physics.sol_deriv_limit": {"from": old.sol_deriv_limit, "to": new_physics.sol_deriv_limit},
    }
    return out, overrides


def _model_compute_psi_for_boundary(model: PlasmaModel | GpuPlasmaModel):
    """Return psi in the backend-native form used by boundary dispatch."""
    if isinstance(model, GpuPlasmaModel):
        return model.compute_psi_tensor()
    return model.compute_psi()


def _legacy_radii(boundary_poly: np.ndarray | None, center: tuple[float, float], angles: np.ndarray) -> np.ndarray:
    """Измерить радиусы контура правилом угловой интерполяции tokamak-sim-0."""
    if boundary_poly is None:
        return np.full((angles.shape[0],), np.nan, dtype=float)
    return legacy_radii_at_angles(boundary_poly, center, angles)


def _prepare(
    cfg: LoadedConfig,
    *,
    M_angles: int,
    scenario_name: ScenarioName,
    scenario_params: Mapping[str, object] | None,
    controller_name: str = "",
) -> tuple[PlasmaModel | GpuPlasmaModel, np.ndarray, Scenario, np.ndarray]:
    """Подготовить модель, опорные углы и сценарий для выбранного контроллера."""
    model: PlasmaModel | GpuPlasmaModel
    if cfg.compute.backend == "gpu":
        model = GpuPlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics, gpu_device=cfg.compute.gpu_device)
    else:
        model = PlasmaModel.from_settings(
            grid=cfg.grid,
            pfc=cfg.pfc,
            sol=cfg.sol,
            settings=cfg.physics,
        )

    angles = np.linspace(-np.pi, np.pi, M_angles, endpoint=False, dtype=float)

    psi0 = _model_compute_psi_for_boundary(model)
    center = (model.R0, model.Z0)

    try:
        boundary0, _level0, _status0 = find_plasma_boundary_with_status(
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
        if (
            str(controller_name).lower() == "lqr_t15_zaitsev"
            and int(getattr(model.pfc, "n_coils", 0)) > 0
        ):
            angles, base_radii = legacy_measurement_angles_from_actuators(boundary0, center, model.pfc.positions)
        else:
            base_radii = _legacy_radii(boundary0, center, angles)
    except BoundaryNotFoundError:
        base_radii = np.full((angles.shape[0],), np.nan, dtype=float)

    scenario = make_scenario(
        scenario_name,
        base_radii,
        model.Ip0,
        params=scenario_params,
        center=center,
    )
    return model, angles, scenario, base_radii


def _controller_diagnostics(controller: Controller) -> dict[str, object]:
    """Собрать необязательные диагностические поля аналитического контроллера."""
    fields = (
        "last_derivative_clipping_fraction",
        "last_gain_fallback_used",
        "last_gain_failure",
        "last_gain_recomputed",
        "last_gain_recompute_reason",
        "last_gain_age_steps",
        "gain_fallback_count",
        "gain_recompute_count",
        "last_boundary_sensitivity_max",
        "last_ip_sensitivity_max",
        "last_response_condition",
        "last_h_size",
        "last_n_boundary",
    )
    out: dict[str, object] = {}
    for name in fields:
        if hasattr(controller, name):
            value = getattr(controller, name)
            if isinstance(value, np.generic):
                value = value.item()
            out[f"controller_{name}"] = value
    for source_name, prefix in (("last_q_diag_summary", "controller_q"), ("last_r_diag_summary", "controller_r")):
        if hasattr(controller, source_name):
            summary = getattr(controller, source_name)
            if isinstance(summary, Mapping):
                for key, value in summary.items():
                    if isinstance(value, np.generic):
                        value = value.item()
                    out[f"{prefix}_{key}"] = value
    for source_name, prefix in (
        ("last_delta_jdot_raw", "controller_delta_jdot_raw"),
        ("last_delta_jdot_applied", "controller_delta_jdot_applied"),
        ("last_jdot_command", "controller_jdot_command"),
    ):
        if hasattr(controller, source_name):
            arr = getattr(controller, source_name)
            if arr is None:
                continue
            vec = np.asarray(arr, dtype=float).reshape(-1)
            if vec.size:
                out[f"{prefix}_max_abs"] = float(np.max(np.abs(vec)))
                out[f"{prefix}_rms"] = float(np.sqrt(np.mean(vec * vec)))
    if hasattr(controller, "last_gain"):
        gain = getattr(controller, "last_gain")
        if gain is not None:
            K = np.asarray(gain, dtype=float)
            out["controller_gain_max_abs"] = float(np.max(np.abs(K))) if K.size else 0.0
            out["controller_gain_fro_norm"] = float(np.linalg.norm(K)) if K.size else 0.0
    if hasattr(controller, "last_state"):
        state = getattr(controller, "last_state")
        if state is not None:
            out["controller_state_max_abs"] = float(np.max(np.abs(np.asarray(state, dtype=float)))) if np.asarray(state).size else 0.0
    return out


def _make_progress(*, enabled: bool, total: int, desc: str):
    if enabled and tqdm is not None:
        return tqdm(total=total, desc=desc, unit="step", dynamic_ncols=True)
    return None


def _step_profile_keys() -> tuple[str, ...]:
    """Вернуть имена профилируемых блоков одного шага симуляции."""
    return (
        "step_scenario",
        "step_measurements_pre",
        "step_controller",
        "step_actuation_realism",
        "step_model",
        "step_disturbances",
        "step_compute_psi_post",
        "step_boundary_post",
        "step_radii_true",
        "step_measurements_post",
        "step_radii_meas",
        "step_log_refs",
        "step_writer",
    )


def _configure_run_logging(
    *,
    verbose: bool,
    profile: bool,
    profile_summary_every: int,
) -> tuple[logging.Logger, Profiler]:
    """Настроить обычное логирование и профилирование запуска."""
    log_level = logging.DEBUG if verbose else logging.INFO
    configure_logging(level=log_level)
    logger = get_logger("cli.run_simulation")
    run_profiler = Profiler(
        enabled=bool(profile),
        summary_every=int(profile_summary_every),
        logger=get_logger("profiling.run"),
    )
    configure_plasma_model_profiling(enabled=bool(profile), summary_every=int(profile_summary_every))
    configure_boundary_profiling(enabled=bool(profile), summary_every=int(profile_summary_every))
    return logger, run_profiler


def _load_config_for_run(
    config: str | Path | LoadedConfig,
    *,
    initial_currents_path: str | Path | None,
    run_profiler: Profiler,
) -> tuple[LoadedConfig, str]:
    """Загрузить TOML-конфигурацию или принять уже разобранный объект."""
    with run_profiler.time_block("load_config"):
        cfg = load_config(config, initial_currents_path=initial_currents_path) if not isinstance(config, LoadedConfig) else config
        source = str(config) if not isinstance(config, LoadedConfig) else "<LoadedConfig>"
    return cfg, source


def _normalize_controller_for_run(
    controller_name: str,
    controller_params: Mapping[str, object] | None,
    *,
    run_profiler: Profiler,
) -> tuple[str, dict[str, object], dict[str, object] | None]:
    """Нормализовать имя и параметры регулятора через реестр."""
    with run_profiler.time_block("normalize_controller"):
        return normalize_controller_launch(controller_name, controller_params)


def _allocate_paths_for_run(
    *,
    output_dir: str | Path | None,
    controller_name: str,
    scenario_name: str,
    steps: int,
    realism_active: bool,
    disturbances: Sequence[Disturbance] | None,
    run_profiler: Profiler,
) -> _RunPaths:
    """Создать директорию запуска и базовые пути артефактов."""
    output_root = Path("./runs") if output_dir is None else Path(output_dir)
    run_stem = _make_run_stem(
        controller_name=controller_name,
        scenario_name=scenario_name,
        steps=steps,
        realism_enabled=realism_active,
        disturbances=disturbances,
    )
    with run_profiler.time_block("allocate_run_dir"):
        run_dir = _allocate_run_dir(output_root, run_stem)
    run_id = int(time.time_ns())
    return _RunPaths(
        run_dir=run_dir,
        manifest_path=run_dir / f"manifest{run_id}.json",
        profile_path=run_dir / f"profile_summary{run_id}.json",
        run_id=run_id,
    )


def _write_manifest(path: Path, metadata: Mapping[str, object]) -> None:
    """Записать JSON manifest запуска."""
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_writer(
    *,
    paths: _RunPaths,
    cfg: LoadedConfig,
    snapshot_every: int,
    metadata: Mapping[str, object],
) -> RunWriter:
    """Создать накопитель артефактов запуска."""
    return RunWriter(
        output_dir=paths.run_dir,
        snapshot_every=snapshot_every,
        grid_shape=cfg.grid.shape,
        metadata=dict(metadata),
        artifact_suffix=str(paths.run_id),
    )


def _construct_controller(
    *,
    controller_name: str,
    ctor_kwargs: Mapping[str, object] | None,
    run_profiler: Profiler,
) -> Controller:
    """Создать и сбросить выбранный регулятор."""
    with run_profiler.time_block("construct_controller"):
        controller = make_controller(controller_name, config=ctor_kwargs)
        controller.reset()
    return controller


def _initial_boundary_tracker(
    *,
    model: PlasmaModel | GpuPlasmaModel,
    psi_true: object,
    center: tuple[float, float],
    target_mean_radius: float | None,
    limiter_shape: np.ndarray | None,
    boundary_mode: BoundaryMode,
    boundary_base_mode: BoundaryMode,
    legacy_precision_index2: float,
    track_level: bool,
    level_smoothing_alpha: float,
    level_search_span_fraction: float,
    continuity_weight_radii: float,
    continuity_weight_mean_radius: float,
    continuity_weight_center: float,
    continuity_weight_area: float,
    continuity_weight_level: float,
    compute_backend: str,
    gpu_device: str,
    logger: logging.Logger,
    run_profiler: Profiler,
) -> _BoundaryTracker:
    """Найти начальный физический контур плазмы."""
    with run_profiler.time_block("initial_boundary"):
        _ = logger
        try:
            poly, level, status = find_plasma_boundary_with_status(
                psi_true,
                model.grid,
                center,
                n_levels=80 if limiter_shape is not None else 10,
                target_mean_radius=target_mean_radius,
                limiter_shape=limiter_shape,
                boundary_mode=boundary_mode,
                boundary_base_mode=boundary_base_mode,
                legacy_precision_index2=legacy_precision_index2,
                track_level=track_level,
                level_smoothing_alpha=level_smoothing_alpha,
                level_search_span_fraction=level_search_span_fraction,
                continuity_weight_radii=continuity_weight_radii,
                continuity_weight_mean_radius=continuity_weight_mean_radius,
                continuity_weight_center=continuity_weight_center,
                continuity_weight_area=continuity_weight_area,
                continuity_weight_level=continuity_weight_level,
                compute_backend=compute_backend,
                gpu_device=gpu_device,
            )
            return _BoundaryTracker(
                poly=poly,
                level=float(level),
                status=status,
                fail_reason=None,
            )
        except BoundaryNotFoundError as exc:
            return _BoundaryTracker(
                poly=None,
                level=float("nan"),
                status="not_found",
                fail_reason=str(exc),
                found=False,
            )


def _scenario_refs(scenario: Scenario, angles: np.ndarray, t_now: float) -> _StepRefs:
    """Вычислить опорные радиусы и ток плазмы на текущем времени."""
    ref_radii = np.asarray(scenario.ref_radii(angles, t_now), dtype=float)
    ip_ref = float(scenario.Ip_ref(t_now))
    return _StepRefs(
        ref_radii=ref_radii,
        ip_ref=ip_ref,
        target_mean_radius=float(np.nanmean(ref_radii)),
    )


def _sensor_measurements(
    *,
    realism: RealismRuntime | None,
    model: PlasmaModel | GpuPlasmaModel,
    psi_true: object,
    boundary_poly: np.ndarray | None,
    radii_true: np.ndarray,
    center: tuple[float, float],
    angles: np.ndarray,
    limiter_shape: np.ndarray | None,
    boundary_mode: BoundaryMode,
    boundary_base_mode: BoundaryMode,
    legacy_precision_index2: float,
    track_level: bool,
    level_smoothing_alpha: float,
    level_search_span_fraction: float,
    continuity_weight_radii: float,
    continuity_weight_mean_radius: float,
    continuity_weight_center: float,
    continuity_weight_area: float,
    continuity_weight_level: float,
    compute_backend: str,
    gpu_device: str,
    run_profiler: Profiler,
    profile_key: str,
) -> SensorRealismResult:
    """Подготовить true/measured channels для регулятора или записи."""
    state = model.snapshot_state()
    true_currents = np.concatenate([
        np.asarray(state.pfc_currents, dtype=float).reshape(-1),
        np.asarray(state.sol_currents, dtype=float).reshape(-1),
    ])
    if realism is None:
        true_boundary_poly = None if boundary_poly is None else np.asarray(boundary_poly, dtype=float).copy()
        return SensorRealismResult(
            true_ip=float(state.Ip),
            measured_ip=float(state.Ip),
            true_active_currents=true_currents,
            measured_active_currents=true_currents.copy(),
            true_boundary_poly=true_boundary_poly,
            measured_boundary_poly=None if true_boundary_poly is None else true_boundary_poly.copy(),
            true_radii=np.asarray(radii_true, dtype=float).reshape(-1).copy(),
            measured_radii=np.asarray(radii_true, dtype=float).reshape(-1).copy(),
            true_psi=_as_numpy_psi(psi_true).copy(),
            measured_psi=_as_numpy_psi(psi_true).copy(),
        )
    with run_profiler.time_block(profile_key):
        return realism.measure(
            true_ip=float(state.Ip),
            true_active_currents=true_currents,
            true_boundary_poly=boundary_poly,
            true_radii=radii_true,
            true_psi=_as_numpy_psi(psi_true),
            model=model,
            center=center,
            angles_rad=angles,
            limiter_shape=limiter_shape,
            boundary_mode=boundary_mode,
            boundary_base_mode=boundary_base_mode,
            legacy_precision_index2=legacy_precision_index2,
            track_level=track_level,
            level_smoothing_alpha=level_smoothing_alpha,
            level_search_span_fraction=level_search_span_fraction,
            continuity_weight_radii=continuity_weight_radii,
            continuity_weight_mean_radius=continuity_weight_mean_radius,
            continuity_weight_center=continuity_weight_center,
            continuity_weight_area=continuity_weight_area,
            continuity_weight_level=continuity_weight_level,
            compute_backend=compute_backend,
            gpu_device=gpu_device,
        )


def _compute_controller_commands(
    *,
    controller: Controller,
    controller_name: str,
    model: PlasmaModel | GpuPlasmaModel,
    psi_meas: np.ndarray,
    boundary_meas: np.ndarray | None,
    center: tuple[float, float],
    angles: np.ndarray,
    refs: _StepRefs,
    scenario: Scenario,
    max_episode_steps: int,
    sensors: SensorRealismResult,
    limiter_shape: np.ndarray | None,
    boundary_mode: BoundaryMode,
    legacy_precision_index2: float,
    run_profiler: Profiler,
) -> tuple[np.ndarray, np.ndarray]:
    """Вызвать регулятор и вернуть абсолютные команды следующего тока."""
    runtime_context = {
        "model": model,
        "psi": psi_meas,
        "boundary_poly": boundary_meas,
        "center": center,
        "measure_angles": angles,
        "ref_radii": refs.ref_radii,
        "Ip_ref": refs.ip_ref,
        "scenario": scenario,
        "max_episode_steps": int(max_episode_steps),
        "measured_ip": float(sensors.measured_ip),
        "measured_active_currents": np.asarray(sensors.measured_active_currents, dtype=float),
        "measured_radii": None if sensors.measured_radii is None else np.asarray(sensors.measured_radii, dtype=float),
        "boundary_found": bool(sensors.measured_boundary_poly is not None and sensors.measured_radii is not None),
        "limiter_shape": limiter_shape,
        "boundary_mode": boundary_mode,
        "legacy_precision_index2": float(legacy_precision_index2),
    }
    with run_profiler.time_block("step_controller"):
        ctrl_kwargs = build_controller_runtime_call(controller_name, runtime_context)
        action = controller.compute_control(**ctrl_kwargs)
    if not isinstance(action, ControlAction):
        raise TypeError("Controller must return ControlAction")
    return np.asarray(action.pfc_currents_next, dtype=float), np.asarray(action.sol_currents_next, dtype=float)


def _effective_commands(
    *,
    realism: RealismRuntime | None,
    pfc_cmd: np.ndarray,
    sol_cmd: np.ndarray,
    run_profiler: Profiler,
) -> _StepCommands:
    """Применить слой реализма к абсолютным командам следующего тока."""
    if realism is None:
        return _StepCommands(pfc_cmd=pfc_cmd, sol_cmd=sol_cmd, pfc_eff=pfc_cmd, sol_eff=sol_cmd)
    with run_profiler.time_block("step_actuation_realism"):
        result = realism.apply_actuation(pfc_cmd, sol_cmd)
    return _StepCommands(pfc_cmd=result.pfc_commanded, sol_cmd=result.sol_commanded, pfc_eff=result.pfc_applied, sol_eff=result.sol_applied)


def _advance_model_state(
    *,
    model: PlasmaModel | GpuPlasmaModel,
    commands: _StepCommands,
    step_index: int,
    steps: int,
    scenario_name: ScenarioName,
    active_disturbances: Sequence[Disturbance],
    run_profiler: Profiler,
) -> tuple[PlasmaState, list[str], np.ndarray]:
    """Продвинуть модель на шаг и вернуть обновленное состояние и psi."""
    with run_profiler.time_block("step_model"):
        state = model.step_currents(
            pfc_currents_next=commands.pfc_eff,
            sol_currents_next=commands.sol_eff,
        )
    with run_profiler.time_block("step_disturbances"):
        state, disturbances_applied = apply_prepared_disturbances(
            state=state,
            step_index=int(step_index),
            total_steps=int(steps),
            scenario_name=str(scenario_name),
            disturbances=active_disturbances,
        )
        model.state = state
    with run_profiler.time_block("step_compute_psi_post"):
        psi_true = _model_compute_psi_for_boundary(model)
        if not isinstance(model, GpuPlasmaModel):
            model.state.psi = psi_true
    return state, disturbances_applied, psi_true


def _update_boundary_tracker(
    *,
    tracker: _BoundaryTracker,
    model: PlasmaModel | GpuPlasmaModel,
    psi_true: object,
    center: tuple[float, float],
    refs: _StepRefs,
    limiter_shape: np.ndarray | None,
    boundary_mode: BoundaryMode,
    boundary_base_mode: BoundaryMode,
    legacy_precision_index2: float,
    track_level: bool,
    level_smoothing_alpha: float,
    level_search_span_fraction: float,
    continuity_weight_radii: float,
    continuity_weight_mean_radius: float,
    continuity_weight_center: float,
    continuity_weight_area: float,
    continuity_weight_level: float,
    compute_backend: str,
    gpu_device: str,
    logger: logging.Logger,
    verbose: bool,
    step_index: int,
    run_profiler: Profiler,
) -> None:
    """Обновить найденный контур плазмы после шага модели."""
    with run_profiler.time_block("step_boundary_post"):
        _ = (logger, verbose, step_index)
        try:
            poly, level, status = find_plasma_boundary_with_status(
                psi_true,
                model.grid,
                center,
                prev_level=tracker.level if np.isfinite(tracker.level) else None,
                prev_poly=tracker.poly,
                local_n_levels=7,
                local_span_frac=0.02,
                target_mean_radius=refs.target_mean_radius,
                limiter_shape=limiter_shape,
                boundary_mode=boundary_mode,
                boundary_base_mode=boundary_base_mode,
                legacy_precision_index2=legacy_precision_index2,
                track_level=track_level,
                level_smoothing_alpha=level_smoothing_alpha,
                level_search_span_fraction=level_search_span_fraction,
                continuity_weight_radii=continuity_weight_radii,
                continuity_weight_mean_radius=continuity_weight_mean_radius,
                continuity_weight_center=continuity_weight_center,
                continuity_weight_area=continuity_weight_area,
                continuity_weight_level=continuity_weight_level,
                compute_backend=compute_backend,
                gpu_device=gpu_device,
            )
            tracker.poly = poly
            tracker.level = float(level)
            tracker.status = status
            tracker.fail_reason = None
            tracker.found = True
        except BoundaryNotFoundError as exc:
            tracker.poly = None
            tracker.level = float("nan")
            tracker.status = "not_found"
            tracker.fail_reason = str(exc)
            tracker.found = False


def _build_step_record(
    *,
    state: PlasmaState,
    commands: _StepCommands,
    refs: _StepRefs,
    scenario: Scenario,
    angles: np.ndarray,
    center: tuple[float, float],
    tracker: _BoundaryTracker,
    sensors: SensorRealismResult,
    disturbances_applied: list[str],
    controller_diagnostics: dict[str, object],
    run_profiler: Profiler,
) -> _StepRecord:
    """Собрать данные шага перед записью в RunWriter."""
    with run_profiler.time_block("step_radii_true"):
        radii_true = _legacy_radii(tracker.poly, center, angles)
    if sensors.true_radii is None or not np.allclose(np.asarray(sensors.true_radii, dtype=float), radii_true, equal_nan=True):
        sensors = SensorRealismResult(
            true_ip=sensors.true_ip,
            measured_ip=sensors.measured_ip,
            true_active_currents=sensors.true_active_currents,
            measured_active_currents=sensors.measured_active_currents,
            true_boundary_poly=sensors.true_boundary_poly,
            measured_boundary_poly=sensors.measured_boundary_poly,
            true_radii=radii_true,
            measured_radii=sensors.measured_radii,
            true_psi=sensors.true_psi,
            measured_psi=sensors.measured_psi,
        )
    with run_profiler.time_block("step_log_refs"):
        t_log = float(state.t)
        ref_radii_log = np.asarray(scenario.ref_radii(angles, t_log), dtype=float)
        ip_ref_log = float(scenario.Ip_ref(t_log))
    return _StepRecord(
        state=state,
        commands=commands,
        refs=refs,
        ref_radii_log=ref_radii_log,
        ip_ref_log=ip_ref_log,
        sensors=sensors,
        disturbances_applied=disturbances_applied,
        controller_diagnostics=dict(controller_diagnostics),
    )


def _write_step_record(
    *,
    writer: RunWriter,
    record: _StepRecord,
    tracker: _BoundaryTracker,
    psi_true: object,
    snapshot_every: int,
    step_index: int,
    run_profiler: Profiler,
) -> None:
    """Записать один шаг временных рядов и событий."""
    state = record.state
    commands = record.commands
    sensors = record.sensors
    n_pfc = np.asarray(state.pfc_currents, dtype=float).reshape(-1).shape[0]
    measured_currents = np.asarray(sensors.measured_active_currents, dtype=float).reshape(-1)
    with run_profiler.time_block("step_writer"):
        writer.append(
            t=float(state.t),
            Ip=float(state.Ip),
            pfc_currents=np.asarray(state.pfc_currents, dtype=float),
            pfc_derivs=np.asarray(state.pfc_current_derivs, dtype=float),
            sol_currents=np.asarray(state.sol_currents, dtype=float),
            sol_derivs=np.asarray(state.sol_current_derivs, dtype=float),
            psi=(_as_numpy_psi(psi_true) if snapshot_every > 0 and ((step_index + 1) % snapshot_every == 0) else None),
            pfc_currents_cmd=commands.pfc_cmd,
            sol_currents_cmd=commands.sol_cmd,
            pfc_currents_eff=commands.pfc_eff,
            sol_currents_eff=commands.sol_eff,
            psi_latest=_as_numpy_psi(psi_true),
            step=int(state.step),
            Ip_ref=record.ip_ref_log,
            radii_ref=record.ref_radii_log,
            Ip_meas=float(sensors.measured_ip),
            pfc_currents_meas=measured_currents[:n_pfc],
            sol_currents_meas=measured_currents[n_pfc:],
            radii_true=sensors.true_radii,
            radii_meas=sensors.measured_radii,
            boundary_poly_true=tracker.poly,
            boundary_poly_meas=sensors.measured_boundary_poly,
        )
        event_payload = {
            "type": "step",
            "k": int(state.step),
            "t": float(state.t),
            "Ip": float(state.Ip),
            "Ip_ref": float(record.ip_ref_log),
            "target_mean_radius": record.refs.target_mean_radius,
            "boundary_status": tracker.status,
            "boundary_found": bool(tracker.found),
            "boundary_fail_reason": tracker.fail_reason,
            "norm_u_pfc": float(np.linalg.norm(commands.pfc_cmd)),
            "norm_u_sol": float(np.linalg.norm(commands.sol_cmd)),
            "disturbances_applied": record.disturbances_applied,
        }
        event_payload.update(record.controller_diagnostics)
        writer.log_event(
            event_payload
        )


def _run_single_step(
    *,
    step_index: int,
    steps: int,
    model: PlasmaModel | GpuPlasmaModel,
    scenario: Scenario,
    scenario_name: ScenarioName,
    controller: Controller,
    controller_name: str,
    realism: RealismRuntime | None,
    tracker: _BoundaryTracker,
    psi_true: object,
    center: tuple[float, float],
    limiter_shape: np.ndarray | None,
    boundary_mode: BoundaryMode,
    boundary_base_mode: BoundaryMode,
    legacy_precision_index2: float,
    track_level: bool,
    level_smoothing_alpha: float,
    level_search_span_fraction: float,
    continuity_weight_radii: float,
    continuity_weight_mean_radius: float,
    continuity_weight_center: float,
    continuity_weight_area: float,
    continuity_weight_level: float,
    compute_backend: str,
    gpu_device: str,
    angles: np.ndarray,
    active_disturbances: Sequence[Disturbance],
    writer: RunWriter,
    snapshot_every: int,
    logger: logging.Logger,
    verbose: bool,
    run_profiler: Profiler,
) -> tuple[np.ndarray, _StepRecord]:
    """Выполнить один полный шаг замкнутой симуляции."""
    with run_profiler.time_block("step_scenario"):
        refs = _scenario_refs(scenario, angles, float(model.state.t))
    radii_pre = _legacy_radii(tracker.poly, center, angles)
    sensors_pre = _sensor_measurements(
        realism=realism,
        model=model,
        psi_true=psi_true,
        boundary_poly=tracker.poly,
        radii_true=radii_pre,
        center=center,
        angles=angles,
        limiter_shape=limiter_shape,
        boundary_mode=boundary_mode,
        boundary_base_mode=boundary_base_mode,
        legacy_precision_index2=legacy_precision_index2,
        track_level=track_level,
        level_smoothing_alpha=level_smoothing_alpha,
        level_search_span_fraction=level_search_span_fraction,
        continuity_weight_radii=continuity_weight_radii,
        continuity_weight_mean_radius=continuity_weight_mean_radius,
        continuity_weight_center=continuity_weight_center,
        continuity_weight_area=continuity_weight_area,
        continuity_weight_level=continuity_weight_level,
        compute_backend=compute_backend,
        gpu_device=gpu_device,
        run_profiler=run_profiler,
        profile_key="step_measurements_pre",
    )
    boundary_meas = None if sensors_pre.measured_boundary_poly is None else np.asarray(sensors_pre.measured_boundary_poly, dtype=float)
    pfc_cmd, sol_cmd = _compute_controller_commands(
        controller=controller,
        controller_name=controller_name,
        model=model,
        psi_meas=np.asarray(sensors_pre.measured_psi if sensors_pre.measured_psi is not None else _as_numpy_psi(psi_true), dtype=float),
        boundary_meas=boundary_meas,
        center=center,
        angles=angles,
        refs=refs,
        scenario=scenario,
        max_episode_steps=steps,
        sensors=sensors_pre,
        limiter_shape=limiter_shape,
        boundary_mode=boundary_mode,
        legacy_precision_index2=legacy_precision_index2,
        run_profiler=run_profiler,
    )
    commands = _effective_commands(
        realism=realism,
        pfc_cmd=pfc_cmd,
        sol_cmd=sol_cmd,
        run_profiler=run_profiler,
    )
    state, disturbances_applied, psi_next = _advance_model_state(
        model=model,
        commands=commands,
        step_index=step_index,
        steps=steps,
        scenario_name=scenario_name,
        active_disturbances=active_disturbances,
        run_profiler=run_profiler,
    )
    _update_boundary_tracker(
        tracker=tracker,
        model=model,
        psi_true=psi_next,
        center=center,
        refs=refs,
        limiter_shape=limiter_shape,
        boundary_mode=boundary_mode,
        boundary_base_mode=boundary_base_mode,
        legacy_precision_index2=legacy_precision_index2,
        track_level=track_level,
        level_smoothing_alpha=level_smoothing_alpha,
        level_search_span_fraction=level_search_span_fraction,
        continuity_weight_radii=continuity_weight_radii,
        continuity_weight_mean_radius=continuity_weight_mean_radius,
        continuity_weight_center=continuity_weight_center,
        continuity_weight_area=continuity_weight_area,
        continuity_weight_level=continuity_weight_level,
        compute_backend=compute_backend,
        gpu_device=gpu_device,
        logger=logger,
        verbose=verbose,
        step_index=step_index,
        run_profiler=run_profiler,
    )
    radii_post = _legacy_radii(tracker.poly, center, angles)
    sensors_post = _sensor_measurements(
        realism=realism,
        model=model,
        psi_true=psi_next,
        boundary_poly=tracker.poly,
        radii_true=radii_post,
        center=center,
        angles=angles,
        limiter_shape=limiter_shape,
        boundary_mode=boundary_mode,
        boundary_base_mode=boundary_base_mode,
        legacy_precision_index2=legacy_precision_index2,
        track_level=track_level,
        level_smoothing_alpha=level_smoothing_alpha,
        level_search_span_fraction=level_search_span_fraction,
        continuity_weight_radii=continuity_weight_radii,
        continuity_weight_mean_radius=continuity_weight_mean_radius,
        continuity_weight_center=continuity_weight_center,
        continuity_weight_area=continuity_weight_area,
        continuity_weight_level=continuity_weight_level,
        compute_backend=compute_backend,
        gpu_device=gpu_device,
        run_profiler=run_profiler,
        profile_key="step_measurements_post",
    )
    record = _build_step_record(
        state=state,
        commands=commands,
        refs=refs,
        scenario=scenario,
        angles=angles,
        center=center,
        tracker=tracker,
        sensors=sensors_post,
        disturbances_applied=disturbances_applied,
        controller_diagnostics=_controller_diagnostics(controller),
        run_profiler=run_profiler,
    )
    _write_step_record(
        writer=writer,
        record=record,
        tracker=tracker,
        psi_true=psi_next,
        snapshot_every=snapshot_every,
        step_index=step_index,
        run_profiler=run_profiler,
    )
    return psi_next, record


def _log_step_profile(run_profiler: Profiler) -> None:
    """Записать периодическую сводку профилирования шага."""
    run_profiler.step()
    run_profiler.log_summary(total_key="step_total", keys=_step_profile_keys(), title="run")
    log_plasma_model_profiling_summary()
    log_boundary_profiling_summary()


def _update_progress(progress, record: _StepRecord) -> None:
    """Обновить индикатор прогресса после шага."""
    if progress is None:
        return
    progress.update(1)
    progress.set_postfix(
        step=int(record.state.step),
        t=f"{float(record.state.t):.4f}",
        Ip=f"{float(record.state.Ip):.4g}",
        refresh=False,
    )


def _log_verbose_step(
    *,
    logger: logging.Logger,
    record: _StepRecord,
    step_index: int,
    steps: int,
) -> None:
    """Записать подробный лог шага при включенном verbose."""
    report_every = max(1, min(100, int(steps) // 20 or 1))
    if (step_index + 1) % report_every != 0 and (step_index + 1) != int(steps):
        return
    logger.info("Step %d/%d t=%.6f Ip=%.6g ||Jnext_pfc||=%.3e ||Jnext_sol||=%.3e", step_index + 1, int(steps), float(record.state.t), float(record.state.Ip), float(np.linalg.norm(record.commands.pfc_cmd)), float(np.linalg.norm(record.commands.sol_cmd)))


def _write_profile_summary(
    *,
    path: Path,
    run_profiler: Profiler,
    logger: logging.Logger,
) -> None:
    """Записать итоговую JSON-сводку профилирования."""
    profile_payload = {
        "run": run_profiler.summary_dict(
            total_key="step_total",
            keys=_step_profile_keys(),
            title="run",
        ),
        "plasma_model": plasma_model_profiling_snapshot(),
        "boundary": boundary_profiling_snapshot(),
    }
    path.write_text(json.dumps(profile_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Profiling summary written: %s", path)


def _run_batched_gpu_fixed_angle_artifacts(
    *,
    cfg: LoadedConfig,
    config_source: str,
    paths: _RunPaths,
    run_meta: Mapping[str, object],
    controller_name: str,
    ctor_kwargs: Mapping[str, object] | None,
    scenario_name: ScenarioName,
    scenario_params: Mapping[str, object] | None,
    steps: int,
    M_angles: int,
    snapshot_every: int,
    show_progress: bool,
    verbose: bool,
    profile: bool,
    logger: logging.Logger,
    run_profiler: Profiler,
) -> RunResult:
    """Run a clean controller through the batched-GPU fixed-angle boundary path."""
    if cfg.compute.backend != "gpu":
        raise ValueError("batched-GPU fixed-angle artifact mode requires compute.backend='gpu'")

    logger.info("Preparing batched GPU fixed-angle simulator and scenario")
    angles = np.linspace(-np.pi, np.pi, int(M_angles), endpoint=False, dtype=float)
    limiter_shape = (
        np.asarray(cfg.limiter_shape, dtype=float).reshape(-1, 2)
        if cfg.limiter_shape is not None
        else np.zeros((0, 2), dtype=float)
    )
    simulator = BatchedGpuTokamakSimulator(
        grid=cfg.grid,
        pfc=cfg.pfc,
        sol=cfg.sol,
        settings=cfg.physics,
        batch_size=1,
        angles_rad=angles,
        limiter_shape=limiter_shape,
        boundary_mode=cfg.boundary_mode,
        boundary_base_mode=cfg.boundary_base_mode,
        boundary_legacy_precision_index2=cfg.boundary_legacy_precision_index2,
        boundary_smooth_selected_level=cfg.boundary_smooth_selected_level,
        boundary_soft_level_selection=cfg.boundary_soft_level_selection,
        boundary_soft_level_candidates=cfg.boundary_soft_level_candidates,
        boundary_soft_level_temperature=cfg.boundary_soft_level_temperature,
        boundary_soft_level_radius_weight=cfg.boundary_soft_level_radius_weight,
        boundary_soft_level_missing_penalty=cfg.boundary_soft_level_missing_penalty,
        boundary_soft_level_roughness_penalty=cfg.boundary_soft_level_roughness_penalty,
        boundary_level_smoothing_alpha=cfg.boundary_level_smoothing_alpha,
        boundary_level_search_span_fraction=cfg.boundary_level_search_span_fraction,
        boundary_continuity_weight_radii=cfg.boundary_continuity_weight_radii,
        boundary_continuity_weight_mean_radius=cfg.boundary_continuity_weight_mean_radius,
        boundary_continuity_weight_center=cfg.boundary_continuity_weight_center,
        boundary_continuity_weight_area=cfg.boundary_continuity_weight_area,
        boundary_continuity_weight_level=cfg.boundary_continuity_weight_level,
        gpu_device=cfg.compute.gpu_device,
    )
    result = simulator.reset(
        ip=np.asarray([cfg.physics.Ip0], dtype=float),
        pfc_currents=np.asarray(cfg.pfc.initial_currents, dtype=float).reshape(1, cfg.pfc.n_coils),
        sol_currents=np.asarray(cfg.sol.initial_currents, dtype=float).reshape(1, cfg.sol.n_coils),
    )
    state = _state_from_batched_result(result, simulator)
    center = (float(cfg.physics.R0), float(cfg.physics.Z0))
    base_radii = _fixed_angle_radii_from_result(result)
    if not _fixed_angle_found_from_result(result):
        base_radii = np.full((angles.shape[0],), np.nan, dtype=float)
    scenario = make_scenario(
        scenario_name,
        base_radii,
        float(state.Ip),
        params=scenario_params,
        center=center,
    )
    model_view = _BatchedGpuModelView(simulator=simulator, state=state)

    writer = _make_writer(
        paths=paths,
        cfg=cfg,
        snapshot_every=snapshot_every,
        metadata=dict(run_meta),
    )
    logger.info("Constructing controller: %s", controller_name)
    controller = _construct_controller(
        controller_name=controller_name,
        ctor_kwargs=ctor_kwargs,
        run_profiler=run_profiler,
    )

    progress = _make_progress(
        enabled=show_progress,
        total=int(steps),
        desc=f"{controller_name}:{scenario_name}:batched-gpu-fixed-angle",
    )
    last_ref_radii = np.zeros_like(angles, dtype=float)
    completed_steps = 0

    try:
        for k in range(int(steps)):
            with run_profiler.time_block("step_total"):
                state_pre = _state_from_batched_result(result, simulator)
                model_view.state = state_pre
                radii_pre = _fixed_angle_radii_from_result(result)
                found_pre = _fixed_angle_found_from_result(result)
                poly_pre = _fixed_angle_polyline(center=center, angles=angles, radii=radii_pre, found=found_pre)
                refs = _scenario_refs(scenario, angles, float(state_pre.t))
                measured_currents_pre = np.concatenate([
                    np.asarray(state_pre.pfc_currents, dtype=float).reshape(-1),
                    np.asarray(state_pre.sol_currents, dtype=float).reshape(-1),
                ])
                sensors_pre = SensorRealismResult(
                    true_ip=float(state_pre.Ip),
                    measured_ip=float(state_pre.Ip),
                    true_active_currents=measured_currents_pre,
                    measured_active_currents=measured_currents_pre.copy(),
                    true_boundary_poly=poly_pre,
                    measured_boundary_poly=(poly_pre.copy() if found_pre and poly_pre is not None else None),
                    true_radii=np.asarray(radii_pre, dtype=float).reshape(-1),
                    measured_radii=np.asarray(radii_pre, dtype=float).reshape(-1),
                    true_psi=np.asarray(state_pre.psi, dtype=float).copy(),
                    measured_psi=np.asarray(state_pre.psi, dtype=float).copy(),
                )
                pfc_cmd, sol_cmd = _compute_controller_commands(
                    controller=controller,
                    controller_name=controller_name,
                    model=model_view,
                    psi_meas=np.asarray(state_pre.psi, dtype=float),
                    boundary_meas=None if poly_pre is None else np.asarray(poly_pre, dtype=float),
                    center=center,
                    angles=angles,
                    refs=refs,
                    scenario=scenario,
                    max_episode_steps=int(steps),
                    sensors=sensors_pre,
                    limiter_shape=cfg.limiter_shape,
                    boundary_mode=cfg.boundary_mode,
                    legacy_precision_index2=cfg.boundary_legacy_precision_index2,
                    run_profiler=run_profiler,
                )
                commands = _StepCommands(pfc_cmd=pfc_cmd, sol_cmd=sol_cmd, pfc_eff=pfc_cmd, sol_eff=sol_cmd)
                active_next = np.concatenate([commands.pfc_eff, commands.sol_eff], axis=0).reshape(1, -1)
                with run_profiler.time_block("step_model"):
                    result = simulator.step_currents(active_next)
                state_post = _state_from_batched_result(result, simulator)
                model_view.state = state_post
                radii_post = _fixed_angle_radii_from_result(result)
                found_post = _fixed_angle_found_from_result(result)
                poly_post = _fixed_angle_polyline(center=center, angles=angles, radii=radii_post, found=found_post)
                t_log = float(state_post.t)
                ref_radii_log = np.asarray(scenario.ref_radii(angles, t_log), dtype=float)
                ip_ref_log = float(scenario.Ip_ref(t_log))
                last_ref_radii = ref_radii_log.copy()
                pfc_cmd_deriv = (commands.pfc_cmd - state_pre.pfc_currents) / float(cfg.physics.t_step)
                sol_cmd_deriv = (commands.sol_cmd - state_pre.sol_currents) / float(cfg.physics.t_step)
                measured_currents_post = np.concatenate([
                    np.asarray(state_post.pfc_currents, dtype=float).reshape(-1),
                    np.asarray(state_post.sol_currents, dtype=float).reshape(-1),
                ])
                with run_profiler.time_block("step_writer"):
                    writer.append(
                        t=float(state_post.t),
                        Ip=float(state_post.Ip),
                        pfc_currents=np.asarray(state_post.pfc_currents, dtype=float),
                        pfc_derivs=np.asarray(state_post.pfc_current_derivs, dtype=float),
                        sol_currents=np.asarray(state_post.sol_currents, dtype=float),
                        sol_derivs=np.asarray(state_post.sol_current_derivs, dtype=float),
                        psi=(
                            np.asarray(state_post.psi, dtype=float)
                            if snapshot_every > 0 and ((k + 1) % snapshot_every == 0)
                            else None
                        ),
                        pfc_derivs_cmd=pfc_cmd_deriv,
                        sol_derivs_cmd=sol_cmd_deriv,
                        pfc_derivs_eff=np.asarray(state_post.pfc_current_derivs, dtype=float),
                        sol_derivs_eff=np.asarray(state_post.sol_current_derivs, dtype=float),
                        pfc_currents_cmd=commands.pfc_cmd,
                        sol_currents_cmd=commands.sol_cmd,
                        pfc_currents_eff=commands.pfc_eff,
                        sol_currents_eff=commands.sol_eff,
                        psi_latest=np.asarray(state_post.psi, dtype=float),
                        step=int(state_post.step),
                        Ip_ref=ip_ref_log,
                        radii_ref=ref_radii_log,
                        Ip_meas=float(state_post.Ip),
                        pfc_currents_meas=measured_currents_post[: cfg.pfc.n_coils],
                        sol_currents_meas=measured_currents_post[cfg.pfc.n_coils :],
                        radii_true=np.asarray(radii_post, dtype=float).reshape(-1),
                        radii_meas=np.asarray(radii_post, dtype=float).reshape(-1),
                        boundary_poly_true=poly_post,
                        boundary_poly_meas=poly_post,
                    )
                    event_payload = {
                        "type": "step",
                        "k": int(state_post.step),
                        "t": float(state_post.t),
                        "Ip": float(state_post.Ip),
                        "Ip_ref": ip_ref_log,
                        "target_mean_radius": float(np.nanmean(refs.ref_radii)),
                        "boundary_status": "fixed_angle_gpu_found" if found_post else "fixed_angle_gpu_not_found",
                        "boundary_mode": str(cfg.boundary_mode),
                        "boundary_base_mode": str(cfg.boundary_base_mode),
                        "boundary_gpu_status_code": int(np.asarray(result.boundary.status_code.detach().cpu())[0]),
                        "boundary_gpu_level": float(np.asarray(result.boundary.level.detach().cpu())[0]),
                        "boundary_found": bool(found_post),
                        "boundary_fail_reason": None if found_post else "fixed-angle GPU boundary was not fully found",
                        "norm_u_pfc": float(np.linalg.norm(commands.pfc_cmd)),
                        "norm_u_sol": float(np.linalg.norm(commands.sol_cmd)),
                        "disturbances_applied": [],
                    }
                    event_payload.update(_controller_diagnostics(controller))
                    writer.log_event(event_payload)
                completed_steps += 1
                run_profiler.step()
                _update_progress(
                    progress,
                    _StepRecord(
                        state=state_post,
                        commands=commands,
                        refs=refs,
                        ref_radii_log=ref_radii_log,
                        ip_ref_log=ip_ref_log,
                        sensors=SensorRealismResult(
                            true_ip=float(state_post.Ip),
                            measured_ip=float(state_post.Ip),
                            true_active_currents=measured_currents_post,
                            measured_active_currents=measured_currents_post.copy(),
                            true_boundary_poly=poly_post,
                            measured_boundary_poly=None if poly_post is None else poly_post.copy(),
                            true_radii=np.asarray(radii_post, dtype=float).reshape(-1),
                            measured_radii=np.asarray(radii_post, dtype=float).reshape(-1),
                            true_psi=np.asarray(state_post.psi, dtype=float),
                            measured_psi=np.asarray(state_post.psi, dtype=float),
                        ),
                        disturbances_applied=[],
                        controller_diagnostics={},
                    ),
                )
                if verbose:
                    logger.info(
                        "Step %d/%d t=%.6f Ip=%.6g boundary_found=%s",
                        k + 1,
                        int(steps),
                        float(state_post.t),
                        float(state_post.Ip),
                        bool(found_post),
                    )
    finally:
        if progress is not None:
            progress.close()

    logger.info("Finalizing run artifacts")
    with run_profiler.time_block("finalize_writer"):
        npz_path = writer.finalize()
    events_path = writer.events_csv_path
    if profile:
        _write_profile_summary(path=paths.profile_path, run_profiler=run_profiler, logger=logger)
    logger.info("Run finished: %s", npz_path)
    return RunResult(
        run_dir=paths.run_dir,
        manifest_path=paths.manifest_path,
        npz_path=npz_path,
        events_path=events_path,
        angles=angles,
        last_ref_radii=np.asarray(last_ref_radii, dtype=float),
        completed_steps=int(completed_steps),
        stop_reason=None,
        boundary_missing_step=None,
    )


def run(
    config: str | Path | LoadedConfig,
    *,
    initial_currents_path: str | Path | None = None,
    steps: int,
    output_dir: str | Path | None,
    controller_name: str = "lqr_t15_zaitsev",
    controller_params: Mapping[str, object] | None = None,
    M_angles: int = 16,
    scenario_name: ScenarioName = "nominal",
    scenario_params: Mapping[str, object] | None = None,
    snapshot_every: int = 0,
    disturbances: Sequence[Disturbance] | None = None,
    realism_enabled: bool = False,
    verbose: bool = False,
    show_progress: bool = False,
    profile: bool = False,
    profile_summary_every: int = 0,
    compute_backend: ComputeBackend | str | None = None,
    gpu_device: str | None = None,
) -> RunResult:
    """Запустить одну замкнутую симуляцию и вернуть пути созданных артефактов."""
    logger, run_profiler = _configure_run_logging(
        verbose=verbose,
        profile=profile,
        profile_summary_every=profile_summary_every,
    )

    logger.info("Loading configuration")
    cfg, config_source = _load_config_for_run(config, initial_currents_path=initial_currents_path, run_profiler=run_profiler)
    logger.info("Normalizing controller launch: %s", controller_name)
    canonical_controller_name, normalized_controller_params, ctor_kwargs = _normalize_controller_for_run(
        controller_name,
        controller_params,
        run_profiler=run_profiler,
    )
    cfg_runtime, runtime_overrides = _effective_config_for_controller(
        cfg,
        controller_name=canonical_controller_name,
    )
    if compute_backend is not None or gpu_device is not None:
        cfg_runtime = replace(
            cfg_runtime,
            compute=ComputeSettings(
                backend=cfg_runtime.compute.backend if compute_backend is None else normalize_compute_backend(compute_backend),
                gpu_device=cfg_runtime.compute.gpu_device if gpu_device is None else str(gpu_device),
            ),
        )
    cfg_runtime.compute.validate(require_available=(cfg_runtime.compute.backend == "gpu"))

    if runtime_overrides:
        logger.info("Applying runtime overrides for %s", canonical_controller_name)

    realism_active = bool(
        (realism_enabled or cfg_runtime.realism.enabled)
        and canonical_controller_name != "t15md_replay"
        and RealismRuntime.has_any_effect(cfg_runtime.realism)
    )
    gpu_fixed_angle_controllers = {"learned_magnetic_controller", "t15md_replay"}
    use_batched_gpu_fixed_angle = (
        cfg_runtime.compute.backend == "gpu"
        and canonical_controller_name in gpu_fixed_angle_controllers
    )

    if canonical_controller_name == "learned_magnetic_controller":
        if cfg_runtime.compute.backend != "gpu":
            raise ValueError(
                "learned_magnetic_controller artifact runs require --compute-backend gpu "
                "so deployment uses the same fixed-angle boundary path as RL training"
            )
    if use_batched_gpu_fixed_angle:
        if realism_active or bool(disturbances):
            raise ValueError(
                "batched GPU fixed-angle artifact runs currently support the clean plant path only; "
                "disable realism/disturbances for this artifact mode"
            )
    paths = _allocate_paths_for_run(
        output_dir=output_dir,
        controller_name=canonical_controller_name,
        scenario_name=scenario_name,
        steps=steps,
        realism_active=realism_active,
        disturbances=disturbances,
        run_profiler=run_profiler,
    )
    run_meta = _build_run_metadata(
        run_id=paths.run_id,
        cfg=cfg_runtime,
        config_source=config_source,
        controller_name=canonical_controller_name,
        controller_params=normalized_controller_params,
        scenario_name=scenario_name,
        scenario_params=({} if scenario_params is None else dict(scenario_params)),
        disturbances=disturbances,
        realism_enabled=realism_active,
        steps=steps,
        profiling_enabled=bool(profile),
        runtime_overrides=runtime_overrides,
    )
    run_meta = _json_safe(run_meta)
    if use_batched_gpu_fixed_angle:
        meta_overrides = dict(run_meta.get("runtime_overrides", {})) if isinstance(run_meta, Mapping) else {}
        meta_overrides["controller_artifact_path"] = "batched_gpu_fixed_angle"
        meta_overrides["controller_boundary_signal"] = "fixed_angle_radii"
        if canonical_controller_name == "learned_magnetic_controller":
            meta_overrides["learned_controller_artifact_path"] = "batched_gpu_fixed_angle"
            meta_overrides["learned_controller_boundary_signal"] = "fixed_angle_radii"
        run_meta["runtime_overrides"] = meta_overrides
    _write_manifest(paths.manifest_path, run_meta)
    logger.info("Allocated run directory: %s", paths.run_dir)

    if use_batched_gpu_fixed_angle:
        logger.info("Using batched GPU fixed-angle artifact path")
        return _run_batched_gpu_fixed_angle_artifacts(
            cfg=cfg_runtime,
            config_source=config_source,
            paths=paths,
            run_meta=run_meta,
            controller_name=canonical_controller_name,
            ctor_kwargs=ctor_kwargs,
            scenario_name=scenario_name,
            scenario_params=scenario_params,
            steps=int(steps),
            M_angles=int(M_angles),
            snapshot_every=int(snapshot_every),
            show_progress=bool(show_progress),
            verbose=bool(verbose),
            profile=bool(profile),
            logger=logger,
            run_profiler=run_profiler,
        )

    logger.info("Preparing model and scenario")
    with run_profiler.time_block("prepare"):
        model, angles, scenario, _base_radii = _prepare(
            cfg_runtime,
            M_angles=M_angles,
            scenario_name=scenario_name,
            scenario_params=scenario_params,
            controller_name=canonical_controller_name,
        )

    with run_profiler.time_block("prepare_disturbances"):
        active_disturbances = prepare_disturbances(disturbances, int(steps))
    with run_profiler.time_block("setup_realism"):
        realism = RealismRuntime(cfg_runtime.realism) if realism_active else None

    if realism_active:
        logger.info("Realism is active")
    if active_disturbances:
        logger.info("Prepared disturbances: %s", ", ".join(d.__class__.__name__ for d in active_disturbances))

    writer = _make_writer(
        paths=paths,
        cfg=cfg_runtime,
        snapshot_every=snapshot_every,
        metadata=run_meta,
    )
    logger.info("Constructing controller: %s", canonical_controller_name)
    controller = _construct_controller(
        controller_name=canonical_controller_name,
        ctor_kwargs=ctor_kwargs,
        run_profiler=run_profiler,
    )

    center = (model.R0, model.Z0)
    with run_profiler.time_block("initial_compute_psi"):
        psi_true = _model_compute_psi_for_boundary(model)
    initial_refs = _scenario_refs(scenario, angles, float(model.state.t))
    tracker = _initial_boundary_tracker(
        model=model,
        psi_true=psi_true,
        center=center,
        target_mean_radius=initial_refs.target_mean_radius,
        limiter_shape=cfg_runtime.limiter_shape,
        boundary_mode=cfg_runtime.boundary_mode,
        boundary_base_mode=cfg_runtime.boundary_base_mode,
        legacy_precision_index2=cfg_runtime.boundary_legacy_precision_index2,
        track_level=cfg_runtime.boundary_track_level,
        level_smoothing_alpha=cfg_runtime.boundary_level_smoothing_alpha,
        level_search_span_fraction=cfg_runtime.boundary_level_search_span_fraction,
        continuity_weight_radii=cfg_runtime.boundary_continuity_weight_radii,
        continuity_weight_mean_radius=cfg_runtime.boundary_continuity_weight_mean_radius,
        continuity_weight_center=cfg_runtime.boundary_continuity_weight_center,
        continuity_weight_area=cfg_runtime.boundary_continuity_weight_area,
        continuity_weight_level=cfg_runtime.boundary_continuity_weight_level,
        compute_backend=cfg_runtime.compute.backend,
        gpu_device=cfg_runtime.compute.gpu_device,
        logger=logger,
        run_profiler=run_profiler,
    )

    logger.info("Starting simulation loop for %d steps", int(steps))
    last_ref_radii = np.zeros_like(angles, dtype=float)
    completed_steps = 0
    stop_reason: str | None = None
    boundary_missing_step: int | None = None
    progress = _make_progress(
        enabled=show_progress,
        total=int(steps),
        desc=f"{canonical_controller_name}:{scenario_name}",
    )

    try:
        for k in range(int(steps)):
            try:
                with run_profiler.time_block("step_total"):
                    psi_true, record = _run_single_step(
                        step_index=k,
                        steps=int(steps),
                        model=model,
                        scenario=scenario,
                        scenario_name=scenario_name,
                        controller=controller,
                        controller_name=canonical_controller_name,
                        realism=realism,
                        tracker=tracker,
                        psi_true=psi_true,
                        center=center,
                        limiter_shape=cfg_runtime.limiter_shape,
                        boundary_mode=cfg_runtime.boundary_mode,
                        boundary_base_mode=cfg_runtime.boundary_base_mode,
                        legacy_precision_index2=cfg_runtime.boundary_legacy_precision_index2,
                        track_level=cfg_runtime.boundary_track_level,
                        level_smoothing_alpha=cfg_runtime.boundary_level_smoothing_alpha,
                        level_search_span_fraction=cfg_runtime.boundary_level_search_span_fraction,
                        continuity_weight_radii=cfg_runtime.boundary_continuity_weight_radii,
                        continuity_weight_mean_radius=cfg_runtime.boundary_continuity_weight_mean_radius,
                        continuity_weight_center=cfg_runtime.boundary_continuity_weight_center,
                        continuity_weight_area=cfg_runtime.boundary_continuity_weight_area,
                        continuity_weight_level=cfg_runtime.boundary_continuity_weight_level,
                        compute_backend=cfg_runtime.compute.backend,
                        gpu_device=cfg_runtime.compute.gpu_device,
                        angles=angles,
                        active_disturbances=active_disturbances,
                        writer=writer,
                        snapshot_every=snapshot_every,
                        logger=logger,
                        verbose=verbose,
                        run_profiler=run_profiler,
                    )
                    last_ref_radii = record.ref_radii_log.copy()
            except BoundaryNotFoundError as exc:
                state = model.state
                boundary_missing_step = int(k + 1)
                stop_reason = str(exc)
                writer.log_event(
                    {
                        "type": "boundary_missing",
                        "k": boundary_missing_step,
                        "t": None if state is None else float(state.t),
                        "Ip": None if state is None else float(state.Ip),
                        "boundary_mode": cfg_runtime.boundary_mode,
                        "reason": stop_reason,
                    }
                )
                logger.warning("No usable plasma boundary at step=%d/%d reason=%s", boundary_missing_step, int(steps), stop_reason)
                break
            completed_steps += 1
            _log_step_profile(run_profiler)
            _update_progress(progress, record)
            if verbose:
                _log_verbose_step(logger=logger, record=record, step_index=k, steps=int(steps))
    finally:
        if progress is not None:
            progress.close()

    logger.info("Finalizing run artifacts")
    with run_profiler.time_block("finalize_writer"):
        npz_path = writer.finalize()
    events_path = writer.events_csv_path

    if profile:
        _write_profile_summary(path=paths.profile_path, run_profiler=run_profiler, logger=logger)

    if stop_reason is None:
        logger.info("Run finished: %s", npz_path)
    else:
        logger.warning("Run stopped early after %d/%d valid steps: %s", completed_steps, int(steps), stop_reason)
        logger.info("Partial run artifacts written: %s", npz_path)

    return RunResult(
        run_dir=paths.run_dir,
        manifest_path=paths.manifest_path,
        npz_path=npz_path,
        events_path=events_path,
        angles=angles,
        last_ref_radii=np.asarray(last_ref_radii, dtype=float),
        completed_steps=int(completed_steps),
        stop_reason=stop_reason,
        boundary_missing_step=boundary_missing_step,
    )
