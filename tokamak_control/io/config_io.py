from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast
import tomllib

import numpy as np

import tomli_w

from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.core.coils import Coil, CoilActuator, CoilGroup
from tokamak_control.compute import ComputeSettings
from tokamak_control.config.settings import PhysicsSettings
from tokamak_control.geometry.boundary import BoundaryMode
from tokamak_control.geometry.limiters import get_limiter_shape
from tokamak_control.realism import ActuatorRealismSettings, RealismSettings, SensorRealismSettings


@dataclass(frozen=True, slots=True)
class LoadedConfig:
    """Разобранная конфигурация запуска."""

    grid: Grid2D
    pfc: CoilGroup
    sol: CoilGroup
    physics: PhysicsSettings
    compute: ComputeSettings = ComputeSettings()
    realism: RealismSettings = RealismSettings()
    boundary_mode: BoundaryMode = "legacy_contour"
    boundary_base_mode: BoundaryMode = "legacy_contour_limited"
    boundary_legacy_precision_index2: float = 1.0e-3
    boundary_track_level: bool = False
    boundary_smooth_selected_level: bool = False
    boundary_soft_level_selection: bool = False
    boundary_soft_level_candidates: int = 64
    boundary_soft_level_temperature: float = 0.05
    boundary_soft_level_radius_weight: float = 1.0
    boundary_soft_level_missing_penalty: float = 4.0
    boundary_soft_level_roughness_penalty: float = 0.2
    boundary_level_smoothing_alpha: float = 1.0
    boundary_level_search_span_fraction: float = 0.02
    boundary_continuity_weight_radii: float = 1.0
    boundary_continuity_weight_mean_radius: float = 0.3
    boundary_continuity_weight_center: float = 0.2
    boundary_continuity_weight_area: float = 0.2
    boundary_continuity_weight_level: float = 0.1
    limiter_name: str | None = None
    limiter_shape: np.ndarray | None = None
    pfc_active_mask: np.ndarray | None = None
    sol_active_mask: np.ndarray | None = None
    initial_state_source: str | None = None
    initial_ip0: float | None = None


@dataclass(frozen=True, slots=True)
class InitialState:
    """Explicit runtime initial state for one simulation reset."""

    ip0: float
    pfc_currents: np.ndarray
    sol_currents: np.ndarray
    source: str | None = None


def _require_mapping(obj: object, name: str) -> dict:
    if not isinstance(obj, dict):
        raise ValueError(f"{name} must be a TOML table")
    return obj


def _require_key(node: dict, key: str, name: str) -> object:
    if key not in node:
        raise ValueError(f"Missing required key {name}.{key}")
    return node[key]


def _coerce_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _coerce_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if not np.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name} must be an integer")
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    raise ValueError(f"{name} must be an integer")


def _coerce_bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "1", "yes", "on"}:
            return True
        if low in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def _coerce_boundary_mode(value: object, name: str) -> BoundaryMode:
    """Прочитать режим физического определения границы плазмы."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    mode = value.strip().lower()
    if mode not in {"legacy_contour", "legacy_contour_limited", "tracked_flux_contour"}:
        raise ValueError(
            f"{name} must be 'legacy_contour', 'legacy_contour_limited', or 'tracked_flux_contour', got {value!r}"
        )
    return cast(BoundaryMode, mode)


def _coerce_optional_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _coerce_float(value, name)


def _coerce_optional_tuple_floats(value: object, name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list/tuple of numbers or null")
    out = tuple(_coerce_float(x, f"{name}[]") for x in value)
    if len(out) == 0:
        raise ValueError(f"{name} cannot be empty when provided")
    return out


def _coerce_optional_bool_vector(value: object, name: str, *, size: int) -> np.ndarray:
    """Прочитать необязательную маску активности катушек."""
    if value is None:
        return np.ones((int(size),), dtype=bool)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of booleans")
    out = np.asarray(value, dtype=bool)
    if out.shape != (int(size),):
        raise ValueError(f"{name} must have shape ({int(size)},)")
    return out


def _load_grid_axis(node: dict, name: str) -> Grid1D:
    """Собрать ось сетки из диапазона TOML и вычислить шаг."""
    start = _coerce_float(_require_key(node, "start", name), f"{name}.start")
    size = _coerce_int(_require_key(node, "size", name), f"{name}.size")
    center = _coerce_float(_require_key(node, "center", name), f"{name}.center")

    has_end = "end" in node
    has_step = "step" in node
    if has_end == has_step:
        raise ValueError(f"{name} must define exactly one of 'end' or 'step'")
    if has_end:
        end = _coerce_float(node["end"], f"{name}.end")
        if size < 2:
            raise ValueError(f"{name}.size must be >= 2")
        step = (end - start) / float(size - 1)
    else:
        step = _coerce_float(node["step"], f"{name}.step")

    return Grid1D(start=start, step=step, size=size, center=center)


def _dump_grid_axis(axis: Grid1D) -> dict:
    """Записать ось сетки через диапазон, без дублирования шага."""
    return {
        "start": axis.start,
        "end": axis.start + axis.step * float(axis.size - 1),
        "size": axis.size,
        "center": axis.center,
    }


def _drop_none_values(value: object) -> object:
    """Удалить значения None перед записью TOML, где нет null-типа."""
    if isinstance(value, dict):
        return {key: _drop_none_values(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none_values(item) for item in value]
    return value


def _coerce_elements(value: object, name: str) -> list[np.ndarray]:
    """Прочитать группы точечных элементов катушек из TOML."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of actuator element lists")
    groups: list[np.ndarray] = []
    for i, group in enumerate(value):
        arr = np.asarray(group, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"{name}[{i}] must have shape (n_elements, 2)")
        if arr.shape[0] == 0:
            raise ValueError(f"{name}[{i}] cannot be empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}[{i}] must contain only finite values")
        groups.append(arr)
    return groups


def _coerce_optional_element_weights(
    value: object,
    name: str,
    *,
    groups: list[np.ndarray],
) -> list[np.ndarray] | None:
    """Прочитать необязательные веса split-элементов по актуаторам."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of per-actuator weight lists")
    if len(value) != len(groups):
        raise ValueError(f"{name} must contain {len(groups)} actuator entries")

    out: list[np.ndarray] = []
    for i, (group, weights_raw) in enumerate(zip(groups, value, strict=True)):
        weights = np.asarray(weights_raw, dtype=float).reshape(-1)
        if weights.shape != (group.shape[0],):
            raise ValueError(f"{name}[{i}] must have shape ({group.shape[0]},)")
        if not np.all(np.isfinite(weights)):
            raise ValueError(f"{name}[{i}] must contain only finite values")
        if np.any(weights < 0.0):
            raise ValueError(f"{name}[{i}] must be >= 0")
        if not float(np.sum(weights)) > 0.0:
            raise ValueError(f"{name}[{i}] must contain a positive total weight")
        out.append(weights.copy())
    return out


def _coerce_actuator_elements(node: dict, name: str) -> list[np.ndarray]:
    """Прочитать геометрию актуаторов из нового elements или старого positions."""
    if "elements" in node:
        return _coerce_elements(_require_key(node, "elements", name), f"{name}.elements")
    positions = np.asarray(_require_key(node, "positions", name), dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"{name}.positions must have shape (n_actuators, 2)")
    if positions.shape[0] == 0:
        raise ValueError(f"{name}.positions cannot be empty")
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"{name}.positions must contain only finite values")
    return [positions[i : i + 1].copy() for i in range(positions.shape[0])]



def _load_compute_settings(cfg: dict) -> ComputeSettings:
    defaults = ComputeSettings()
    node = _require_mapping(cfg.get("compute", {}), "compute")
    return ComputeSettings(
        backend=str(node.get("backend", defaults.backend)),
        gpu_device=str(node.get("gpu_device", defaults.gpu_device)),
    )

def _load_realism_settings(cfg: dict) -> RealismSettings:
    """Read neutral realism settings from the top-level realism table."""
    defaults = RealismSettings()
    node = _require_mapping(cfg.get("realism", {}), "realism")
    actuators_node = _require_mapping(node.get("actuators", {}), "realism.actuators")
    sensors_node = _require_mapping(node.get("sensors", {}), "realism.sensors")

    actuators = ActuatorRealismSettings(
        pfc_delay_steps=_coerce_int(actuators_node.get("pfc_delay_steps", defaults.actuators.pfc_delay_steps), "realism.actuators.pfc_delay_steps"),
        sol_delay_steps=_coerce_int(actuators_node.get("sol_delay_steps", defaults.actuators.sol_delay_steps), "realism.actuators.sol_delay_steps"),
        pfc_gain_sigma=_coerce_float(actuators_node.get("pfc_gain_sigma", defaults.actuators.pfc_gain_sigma), "realism.actuators.pfc_gain_sigma"),
        sol_gain_sigma=_coerce_float(actuators_node.get("sol_gain_sigma", defaults.actuators.sol_gain_sigma), "realism.actuators.sol_gain_sigma"),
        pfc_bias_sigma=_coerce_float(actuators_node.get("pfc_bias_sigma", defaults.actuators.pfc_bias_sigma), "realism.actuators.pfc_bias_sigma"),
        sol_bias_sigma=_coerce_float(actuators_node.get("sol_bias_sigma", defaults.actuators.sol_bias_sigma), "realism.actuators.sol_bias_sigma"),
        pfc_command_noise_sigma=_coerce_float(actuators_node.get("pfc_command_noise_sigma", defaults.actuators.pfc_command_noise_sigma), "realism.actuators.pfc_command_noise_sigma"),
        sol_command_noise_sigma=_coerce_float(actuators_node.get("sol_command_noise_sigma", defaults.actuators.sol_command_noise_sigma), "realism.actuators.sol_command_noise_sigma"),
    )
    sensors = SensorRealismSettings(
        ip_noise_sigma=_coerce_float(sensors_node.get("ip_noise_sigma", defaults.sensors.ip_noise_sigma), "realism.sensors.ip_noise_sigma"),
        ip_bias=_coerce_float(sensors_node.get("ip_bias", defaults.sensors.ip_bias), "realism.sensors.ip_bias"),
        ip_bias_sigma=_coerce_float(sensors_node.get("ip_bias_sigma", defaults.sensors.ip_bias_sigma), "realism.sensors.ip_bias_sigma"),
        ip_delay_steps=_coerce_int(sensors_node.get("ip_delay_steps", defaults.sensors.ip_delay_steps), "realism.sensors.ip_delay_steps"),
        active_current_noise_sigma=_coerce_float(sensors_node.get("active_current_noise_sigma", defaults.sensors.active_current_noise_sigma), "realism.sensors.active_current_noise_sigma"),
        active_current_bias_sigma=_coerce_float(sensors_node.get("active_current_bias_sigma", defaults.sensors.active_current_bias_sigma), "realism.sensors.active_current_bias_sigma"),
        active_current_delay_steps=_coerce_int(sensors_node.get("active_current_delay_steps", defaults.sensors.active_current_delay_steps), "realism.sensors.active_current_delay_steps"),
        radii_noise_sigma=_coerce_float(sensors_node.get("radii_noise_sigma", defaults.sensors.radii_noise_sigma), "realism.sensors.radii_noise_sigma"),
        radii_bias_sigma=_coerce_float(sensors_node.get("radii_bias_sigma", defaults.sensors.radii_bias_sigma), "realism.sensors.radii_bias_sigma"),
        radii_delay_steps=_coerce_int(sensors_node.get("radii_delay_steps", defaults.sensors.radii_delay_steps), "realism.sensors.radii_delay_steps"),
        boundary_xy_noise_sigma=_coerce_float(sensors_node.get("boundary_xy_noise_sigma", defaults.sensors.boundary_xy_noise_sigma), "realism.sensors.boundary_xy_noise_sigma"),
        boundary_delay_steps=_coerce_int(sensors_node.get("boundary_delay_steps", defaults.sensors.boundary_delay_steps), "realism.sensors.boundary_delay_steps"),
        psi_noise_sigma=_coerce_float(sensors_node.get("psi_noise_sigma", defaults.sensors.psi_noise_sigma), "realism.sensors.psi_noise_sigma"),
    )
    return RealismSettings(
        enabled=_coerce_bool(node.get("enabled", defaults.enabled), "realism.enabled"),
        seed=(None if node.get("seed", defaults.seed) is None else _coerce_int(node.get("seed", defaults.seed), "realism.seed")),
        actuators=actuators,
        sensors=sensors,
    )



def load_config(path: str | Path) -> LoadedConfig:
    """Load a machine TOML configuration and construct domain objects.

    Machine configs intentionally do not contain runtime initial conditions.
    Use :func:`load_initial_state` and :func:`apply_initial_state` for Ip and
    starting coil currents.
    """
    path = Path(path)
    with path.open("rb") as f:
        cfg = tomllib.load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Top-level TOML content must be a table")

    version = _coerce_int(cfg.get("version", 1), "version")
    if version != 1:
        raise ValueError(f"Unsupported config version: {version}")

    g = _require_mapping(cfg.get("grid", {}), "grid")
    r_cfg = _require_mapping(_require_key(g, "r", "grid"), "grid.r")
    z_cfg = _require_mapping(_require_key(g, "z", "grid"), "grid.z")

    r = _load_grid_axis(r_cfg, "grid.r")
    z = _load_grid_axis(z_cfg, "grid.z")
    grid = Grid2D(r=r, z=z)

    p = _require_mapping(cfg.get("physics", {}), "physics")
    if "Ip0" in p:
        raise ValueError("physics.Ip0 is no longer allowed in machine configs; use an initial-state TOML")
    defaults = PhysicsSettings()
    physics = PhysicsSettings(
        mu0=_coerce_float(p.get("mu0", defaults.mu0), "physics.mu0"),
        sigma=_coerce_float(p.get("sigma", defaults.sigma), "physics.sigma"),
        inductance_L=_coerce_float(p.get("inductance_L", defaults.inductance_L), "physics.inductance_L"),
        ip_coupling_sign=_coerce_float(p.get("ip_coupling_sign", defaults.ip_coupling_sign), "physics.ip_coupling_sign"),
        plasma_psi_sign=_coerce_float(p.get("plasma_psi_sign", defaults.plasma_psi_sign), "physics.plasma_psi_sign"),
        t_step=_coerce_float(p.get("t_step", defaults.t_step), "physics.t_step"),
        actuator_tau=_coerce_float(p.get("actuator_tau", defaults.actuator_tau), "physics.actuator_tau"),
        R0=_coerce_float(p.get("R0", defaults.R0), "physics.R0"),
        Z0=_coerce_float(p.get("Z0", defaults.Z0), "physics.Z0"),
        pfc_current_limit=_coerce_optional_float(p.get("pfc_current_limit", defaults.pfc_current_limit), "physics.pfc_current_limit"),
        sol_current_limit=_coerce_optional_float(p.get("sol_current_limit", defaults.sol_current_limit), "physics.sol_current_limit"),
        ip_coupling_pfc=_coerce_optional_tuple_floats(p.get("ip_coupling_pfc", defaults.ip_coupling_pfc), "physics.ip_coupling_pfc"),
        ip_coupling_sol=_coerce_optional_tuple_floats(p.get("ip_coupling_sol", defaults.ip_coupling_sol), "physics.ip_coupling_sol"),
        pfc_deriv_limit=_coerce_optional_float(p.get("pfc_deriv_limit", defaults.pfc_deriv_limit), "physics.pfc_deriv_limit"),
        sol_deriv_limit=_coerce_optional_float(p.get("sol_deriv_limit", defaults.sol_deriv_limit), "physics.sol_deriv_limit"),
    )
    physics.validate()

    compute = _load_compute_settings(cfg)
    compute.validate(require_available=False)
    realism = _load_realism_settings(cfg)
    realism.validate()

    c = _require_mapping(cfg.get("coils", {}), "coils")
    pfc_cfg = _require_mapping(_require_key(c, "pfc", "coils"), "coils.pfc")
    sol_cfg = _require_mapping(_require_key(c, "sol", "coils"), "coils.sol")

    def _mk_group(name: str, node: dict) -> tuple[CoilGroup, np.ndarray]:
        """Собрать банк катушек с учетом активной маски."""
        elements = _coerce_actuator_elements(node, name)
        element_weights = _coerce_optional_element_weights(node.get("element_weights"), f"{name}.element_weights", groups=elements)
        if "currents" in node:
            raise ValueError(f"{name}.currents is no longer allowed in machine configs; use an initial-state TOML")
        active_mask = _coerce_optional_bool_vector(node.get("active"), f"{name}.active", size=len(elements))
        if not np.any(active_mask):
            raise ValueError(f"{name}.active must contain at least one active actuator")
        active_elements = [group for group, is_active in zip(elements, active_mask, strict=True) if bool(is_active)]
        active_weights = None if element_weights is None else [weights for weights, is_active in zip(element_weights, active_mask, strict=True) if bool(is_active)]
        actuators = [
            CoilActuator(
                elements=[Coil(R=float(R), Z=float(Z)) for R, Z in group],
                element_weights=(None if active_weights is None else active_weights[i]),
            )
            for i, group in enumerate(active_elements)
        ]
        return CoilGroup(name=str(node.get("name", name)), coils=actuators, currents=np.zeros((len(actuators),), dtype=float)), active_mask

    pfc, pfc_active_mask = _mk_group("coils.pfc", pfc_cfg)
    sol, sol_active_mask = _mk_group("coils.sol", sol_cfg)

    def _active_couplings(values: tuple[float, ...] | None, mask: np.ndarray, active_count: int, name: str) -> tuple[float, ...] | None:
        """Согласовать вектор связи Ip с активными катушками."""
        if values is None:
            return None
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.shape == (int(active_count),):
            return tuple(float(x) for x in arr)
        if arr.shape == mask.shape:
            return tuple(float(x) for x in arr[mask])
        raise ValueError(f"{name} length {arr.size} must match all actuators {mask.size} or active actuators {int(active_count)}")

    physics = replace(
        physics,
        ip_coupling_pfc=_active_couplings(physics.ip_coupling_pfc, pfc_active_mask, pfc.n_coils, "physics.ip_coupling_pfc"),
        ip_coupling_sol=_active_couplings(physics.ip_coupling_sol, sol_active_mask, sol.n_coils, "physics.ip_coupling_sol"),
    )
    physics.validate()

    if physics.ip_coupling_pfc is not None and len(physics.ip_coupling_pfc) != pfc.n_coils:
        raise ValueError(
            f"physics.ip_coupling_pfc length {len(physics.ip_coupling_pfc)} != number of PFC actuators {pfc.n_coils}"
        )
    if physics.ip_coupling_sol is not None and len(physics.ip_coupling_sol) != sol.n_coils:
        raise ValueError(
            f"physics.ip_coupling_sol length {len(physics.ip_coupling_sol)} != number of SOL actuators {sol.n_coils}"
        )
    boundary_node = _require_mapping(cfg.get("boundary", {}), "boundary")
    boundary_mode = _coerce_boundary_mode(boundary_node.get("mode", "legacy_contour"), "boundary.mode")
    boundary_base_mode = _coerce_boundary_mode(
        boundary_node.get("base_mode", "legacy_contour_limited"),
        "boundary.base_mode",
    )
    if boundary_mode == "tracked_flux_contour" and boundary_base_mode == "tracked_flux_contour":
        raise ValueError("boundary.base_mode must be a strict legacy mode when boundary.mode='tracked_flux_contour'")
    boundary_legacy_precision_index2 = _coerce_float(
        boundary_node.get("legacy_precision_index2", 1.0e-3),
        "boundary.legacy_precision_index2",
    )
    if boundary_legacy_precision_index2 <= 0.0:
        raise ValueError("boundary.legacy_precision_index2 must be > 0")
    boundary_track_level = _coerce_bool(boundary_node.get("track_level", False), "boundary.track_level")
    boundary_smooth_selected_level = _coerce_bool(
        boundary_node.get("smooth_selected_level", False),
        "boundary.smooth_selected_level",
    )
    boundary_soft_level_selection = _coerce_bool(
        boundary_node.get("soft_level_selection", False),
        "boundary.soft_level_selection",
    )
    if boundary_soft_level_selection and boundary_smooth_selected_level:
        raise ValueError("boundary.soft_level_selection and boundary.smooth_selected_level are mutually exclusive")
    boundary_soft_level_candidates = _coerce_int(
        boundary_node.get("soft_level_candidates", 64),
        "boundary.soft_level_candidates",
    )
    if boundary_soft_level_candidates < 3:
        raise ValueError("boundary.soft_level_candidates must be >= 3")
    boundary_soft_level_temperature = _coerce_float(
        boundary_node.get("soft_level_temperature", 0.05),
        "boundary.soft_level_temperature",
    )
    if boundary_soft_level_temperature <= 0.0:
        raise ValueError("boundary.soft_level_temperature must be > 0")
    boundary_soft_level_radius_weight = _coerce_float(
        boundary_node.get("soft_level_radius_weight", 1.0),
        "boundary.soft_level_radius_weight",
    )
    boundary_soft_level_missing_penalty = _coerce_float(
        boundary_node.get("soft_level_missing_penalty", 4.0),
        "boundary.soft_level_missing_penalty",
    )
    boundary_soft_level_roughness_penalty = _coerce_float(
        boundary_node.get("soft_level_roughness_penalty", 0.2),
        "boundary.soft_level_roughness_penalty",
    )
    boundary_level_smoothing_alpha = _coerce_float(
        boundary_node.get("level_smoothing_alpha", 1.0),
        "boundary.level_smoothing_alpha",
    )
    if boundary_level_smoothing_alpha < 0.0 or boundary_level_smoothing_alpha > 1.0:
        raise ValueError("boundary.level_smoothing_alpha must be in [0, 1]")
    boundary_level_search_span_fraction = _coerce_float(
        boundary_node.get("level_search_span_fraction", 0.02),
        "boundary.level_search_span_fraction",
    )
    if boundary_level_search_span_fraction < 0.0:
        raise ValueError("boundary.level_search_span_fraction must be >= 0")
    boundary_continuity_weight_radii = _coerce_float(boundary_node.get("continuity_weight_radii", 1.0), "boundary.continuity_weight_radii")
    boundary_continuity_weight_mean_radius = _coerce_float(boundary_node.get("continuity_weight_mean_radius", 0.3), "boundary.continuity_weight_mean_radius")
    boundary_continuity_weight_center = _coerce_float(boundary_node.get("continuity_weight_center", 0.2), "boundary.continuity_weight_center")
    boundary_continuity_weight_area = _coerce_float(boundary_node.get("continuity_weight_area", 0.2), "boundary.continuity_weight_area")
    boundary_continuity_weight_level = _coerce_float(boundary_node.get("continuity_weight_level", 0.1), "boundary.continuity_weight_level")
    for name, value in {
        "boundary.continuity_weight_radii": boundary_continuity_weight_radii,
        "boundary.continuity_weight_mean_radius": boundary_continuity_weight_mean_radius,
        "boundary.continuity_weight_center": boundary_continuity_weight_center,
        "boundary.continuity_weight_area": boundary_continuity_weight_area,
        "boundary.continuity_weight_level": boundary_continuity_weight_level,
        "boundary.soft_level_radius_weight": boundary_soft_level_radius_weight,
        "boundary.soft_level_missing_penalty": boundary_soft_level_missing_penalty,
        "boundary.soft_level_roughness_penalty": boundary_soft_level_roughness_penalty,
    }.items():
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0")

    limiter_name: str | None = None
    limiter_shape: np.ndarray | None = None
    if "limiter" in cfg:
        v = _require_mapping(cfg.get("limiter", {}), "limiter")
        raw_name = _require_key(v, "name", "limiter")
        if not isinstance(raw_name, str) or raw_name.strip() == "":
            raise ValueError("limiter.name must be a non-empty string")
        limiter_name = raw_name.strip()
        limiter_shape = get_limiter_shape(limiter_name)

    return LoadedConfig(
        grid=grid,
        pfc=pfc,
        sol=sol,
        physics=physics,
        compute=compute,
        realism=realism,
        boundary_mode=boundary_mode,
        boundary_base_mode=boundary_base_mode,
        boundary_legacy_precision_index2=boundary_legacy_precision_index2,
        boundary_track_level=boundary_track_level,
        boundary_smooth_selected_level=boundary_smooth_selected_level,
        boundary_soft_level_selection=boundary_soft_level_selection,
        boundary_soft_level_candidates=boundary_soft_level_candidates,
        boundary_soft_level_temperature=boundary_soft_level_temperature,
        boundary_soft_level_radius_weight=boundary_soft_level_radius_weight,
        boundary_soft_level_missing_penalty=boundary_soft_level_missing_penalty,
        boundary_soft_level_roughness_penalty=boundary_soft_level_roughness_penalty,
        boundary_level_smoothing_alpha=boundary_level_smoothing_alpha,
        boundary_level_search_span_fraction=boundary_level_search_span_fraction,
        boundary_continuity_weight_radii=boundary_continuity_weight_radii,
        boundary_continuity_weight_mean_radius=boundary_continuity_weight_mean_radius,
        boundary_continuity_weight_center=boundary_continuity_weight_center,
        boundary_continuity_weight_area=boundary_continuity_weight_area,
        boundary_continuity_weight_level=boundary_continuity_weight_level,
        limiter_name=limiter_name,
        limiter_shape=limiter_shape,
        pfc_active_mask=pfc_active_mask.copy(),
        sol_active_mask=sol_active_mask.copy(),
    )


def load_initial_state(machine_cfg: LoadedConfig, path: str | Path) -> InitialState:
    """Load explicit runtime initial Ip and coil currents for ``machine_cfg``."""
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Initial-state TOML content must be a table")
    version = _coerce_int(raw.get("version", 1), "version")
    if version != 1:
        raise ValueError(f"Unsupported initial-state version: {version}")
    plasma = _require_mapping(raw.get("plasma", {}), "plasma")
    ip0 = _coerce_float(_require_key(plasma, "Ip0", "plasma"), "plasma.Ip0")
    coils = _require_mapping(raw.get("coils", {}), "coils")
    pfc_node = _require_mapping(_require_key(coils, "pfc", "coils"), "coils.pfc")
    sol_node = _require_mapping(_require_key(coils, "sol", "coils"), "coils.sol")
    for name, node in (("coils.pfc", pfc_node), ("coils.sol", sol_node)):
        if "active" in node:
            raise ValueError(f"{name}.active is machine topology and is not allowed in initial-state TOMLs")

    def _currents(node: dict, name: str, count: int) -> np.ndarray:
        arr = np.asarray(_require_key(node, "currents", name), dtype=float)
        if arr.shape != (int(count),):
            raise ValueError(f"{name}.currents must have shape ({int(count)},)")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}.currents must contain only finite values")
        return arr.copy()

    return InitialState(
        ip0=float(ip0),
        pfc_currents=_currents(pfc_node, "coils.pfc", machine_cfg.pfc.n_coils),
        sol_currents=_currents(sol_node, "coils.sol", machine_cfg.sol.n_coils),
        source=str(path),
    )


def apply_initial_state(machine_cfg: LoadedConfig, initial_state: InitialState) -> LoadedConfig:
    """Attach an explicit runtime initial state to a loaded machine config."""
    pfc_currents = np.asarray(initial_state.pfc_currents, dtype=float).reshape(-1)
    sol_currents = np.asarray(initial_state.sol_currents, dtype=float).reshape(-1)
    if pfc_currents.shape != (machine_cfg.pfc.n_coils,):
        raise ValueError(f"initial_state.pfc_currents shape {pfc_currents.shape} != ({machine_cfg.pfc.n_coils},)")
    if sol_currents.shape != (machine_cfg.sol.n_coils,):
        raise ValueError(f"initial_state.sol_currents shape {sol_currents.shape} != ({machine_cfg.sol.n_coils},)")
    if not np.isfinite(float(initial_state.ip0)):
        raise ValueError("initial_state.ip0 must be finite")
    pfc = CoilGroup(name=machine_cfg.pfc.name, coils=list(machine_cfg.pfc.coils), currents=pfc_currents.copy())
    sol = CoilGroup(name=machine_cfg.sol.name, coils=list(machine_cfg.sol.coils), currents=sol_currents.copy())
    return replace(
        machine_cfg,
        pfc=pfc,
        sol=sol,
        initial_state_source=initial_state.source,
        initial_ip0=float(initial_state.ip0),
    )


def require_initial_state(cfg: LoadedConfig) -> InitialState:
    """Return the attached initial state or fail with the new contract error."""
    if cfg.initial_ip0 is None:
        raise ValueError("An explicit initial state is required; pass --initial-state or attach a reset payload")
    return InitialState(
        ip0=float(cfg.initial_ip0),
        pfc_currents=np.asarray(cfg.pfc.initial_currents, dtype=float).copy(),
        sol_currents=np.asarray(cfg.sol.initial_currents, dtype=float).copy(),
        source=cfg.initial_state_source,
    )


def dump_config(
    path: str | Path,
    grid: Grid2D,
    pfc: CoilGroup,
    sol: CoilGroup,
    physics: PhysicsSettings,
    compute: ComputeSettings | None = None,
    realism: RealismSettings | None = None,
    limiter_name: str | None = None,
    boundary_mode: BoundaryMode = "legacy_contour",
    boundary_base_mode: BoundaryMode = "legacy_contour_limited",
    boundary_legacy_precision_index2: float = 1.0e-3,
    boundary_track_level: bool = False,
    boundary_smooth_selected_level: bool = False,
    boundary_soft_level_selection: bool = False,
    boundary_soft_level_candidates: int = 64,
    boundary_soft_level_temperature: float = 0.05,
    boundary_soft_level_radius_weight: float = 1.0,
    boundary_soft_level_missing_penalty: float = 4.0,
    boundary_soft_level_roughness_penalty: float = 0.2,
    boundary_level_smoothing_alpha: float = 1.0,
    boundary_level_search_span_fraction: float = 0.02,
    boundary_continuity_weight_radii: float = 1.0,
    boundary_continuity_weight_mean_radius: float = 0.3,
    boundary_continuity_weight_center: float = 0.2,
    boundary_continuity_weight_area: float = 0.2,
    boundary_continuity_weight_level: float = 0.1,
) -> None:
    """Записать TOML-конфигурацию текущей расчетной схемы."""
    path = Path(path)
    physics.validate()
    compute = ComputeSettings() if compute is None else compute
    compute.validate(require_available=False)
    realism = RealismSettings() if realism is None else realism
    realism.validate()

    data = {
        "version": 1,
        "grid": {
            "r": _dump_grid_axis(grid.r),
            "z": _dump_grid_axis(grid.z),
        },
        "physics": {
            "mu0": physics.mu0,
            "sigma": physics.sigma,
            "inductance_L": physics.inductance_L,
            "ip_coupling_sign": physics.ip_coupling_sign,
            "plasma_psi_sign": physics.plasma_psi_sign,
            "t_step": physics.t_step,
            "actuator_tau": physics.actuator_tau,
            "R0": physics.R0,
            "Z0": physics.Z0,
            "pfc_current_limit": physics.pfc_current_limit,
            "sol_current_limit": physics.sol_current_limit,
            "ip_coupling_pfc": list(physics.ip_coupling_pfc) if physics.ip_coupling_pfc is not None else None,
            "ip_coupling_sol": list(physics.ip_coupling_sol) if physics.ip_coupling_sol is not None else None,
            "pfc_deriv_limit": physics.pfc_deriv_limit,
            "sol_deriv_limit": physics.sol_deriv_limit,
        },
        "compute": {
            "backend": compute.backend,
            "gpu_device": compute.gpu_device,
        },
        "realism": {
            "enabled": realism.enabled,
            "seed": realism.seed,
            "actuators": {
                "pfc_delay_steps": realism.actuators.pfc_delay_steps,
                "sol_delay_steps": realism.actuators.sol_delay_steps,
                "pfc_gain_sigma": realism.actuators.pfc_gain_sigma,
                "sol_gain_sigma": realism.actuators.sol_gain_sigma,
                "pfc_bias_sigma": realism.actuators.pfc_bias_sigma,
                "sol_bias_sigma": realism.actuators.sol_bias_sigma,
                "pfc_command_noise_sigma": realism.actuators.pfc_command_noise_sigma,
                "sol_command_noise_sigma": realism.actuators.sol_command_noise_sigma,
            },
            "sensors": {
                "ip_noise_sigma": realism.sensors.ip_noise_sigma,
                "ip_bias": realism.sensors.ip_bias,
                "ip_bias_sigma": realism.sensors.ip_bias_sigma,
                "ip_delay_steps": realism.sensors.ip_delay_steps,
                "active_current_noise_sigma": realism.sensors.active_current_noise_sigma,
                "active_current_bias_sigma": realism.sensors.active_current_bias_sigma,
                "active_current_delay_steps": realism.sensors.active_current_delay_steps,
                "radii_noise_sigma": realism.sensors.radii_noise_sigma,
                "radii_bias_sigma": realism.sensors.radii_bias_sigma,
                "radii_delay_steps": realism.sensors.radii_delay_steps,
                "boundary_xy_noise_sigma": realism.sensors.boundary_xy_noise_sigma,
                "boundary_delay_steps": realism.sensors.boundary_delay_steps,
                "psi_noise_sigma": realism.sensors.psi_noise_sigma,
            },
        },
        "boundary": {
            "mode": str(boundary_mode),
            "base_mode": str(boundary_base_mode),
            "legacy_precision_index2": float(boundary_legacy_precision_index2),
            "track_level": bool(boundary_track_level),
            "smooth_selected_level": bool(boundary_smooth_selected_level),
            "soft_level_selection": bool(boundary_soft_level_selection),
            "soft_level_candidates": int(boundary_soft_level_candidates),
            "soft_level_temperature": float(boundary_soft_level_temperature),
            "soft_level_radius_weight": float(boundary_soft_level_radius_weight),
            "soft_level_missing_penalty": float(boundary_soft_level_missing_penalty),
            "soft_level_roughness_penalty": float(boundary_soft_level_roughness_penalty),
            "level_smoothing_alpha": float(boundary_level_smoothing_alpha),
            "level_search_span_fraction": float(boundary_level_search_span_fraction),
            "continuity_weight_radii": float(boundary_continuity_weight_radii),
            "continuity_weight_mean_radius": float(boundary_continuity_weight_mean_radius),
            "continuity_weight_center": float(boundary_continuity_weight_center),
            "continuity_weight_area": float(boundary_continuity_weight_area),
            "continuity_weight_level": float(boundary_continuity_weight_level),
        },
        "coils": {
            "pfc": {
                "name": pfc.name,
                "elements": [arr.tolist() for arr in pfc.element_positions],
                "element_weights": [arr.tolist() for arr in pfc.element_weights] if any(not np.allclose(arr, np.ones_like(arr)) for arr in pfc.element_weights) else None,
            },
            "sol": {
                "name": sol.name,
                "elements": [arr.tolist() for arr in sol.element_positions],
                "element_weights": [arr.tolist() for arr in sol.element_weights] if any(not np.allclose(arr, np.ones_like(arr)) for arr in sol.element_weights) else None,
            },
        },
    }
    if limiter_name is not None:
        data["limiter"] = {"name": str(limiter_name)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        tomli_w.dump(_drop_none_values(data), f)
