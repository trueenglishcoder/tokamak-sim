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
    boundary_mode: BoundaryMode = "limited"
    limiter_name: str | None = None
    limiter_shape: np.ndarray | None = None
    pfc_active_mask: np.ndarray | None = None
    sol_active_mask: np.ndarray | None = None
    initial_currents_source: str | None = None


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
    if mode not in {"limited", "diverted"}:
        raise ValueError(f"{name} must be 'limited' or 'diverted', got {value!r}")
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


def _load_compute_settings(cfg: dict) -> ComputeSettings:
    """Read runtime compute backend settings from the top-level compute table."""
    defaults = ComputeSettings()
    node = _require_mapping(cfg.get("compute", {}), "compute")
    return ComputeSettings(
        backend=str(node.get("backend", defaults.backend)),
        gpu_device=str(node.get("gpu_device", defaults.gpu_device)),
        boundary_equivalence_mode=str(node.get("boundary_equivalence_mode", defaults.boundary_equivalence_mode)),
    )


def load_config(path: str | Path, initial_currents_path: str | Path | None = None) -> LoadedConfig:
    """Load a TOML configuration and construct domain objects."""
    path = Path(path)
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    initial_cfg: dict[str, object] | None = None
    if initial_currents_path is not None:
        initial_path = Path(initial_currents_path)
        with initial_path.open("rb") as f:
            loaded_initial = tomllib.load(f)
        if not isinstance(loaded_initial, dict):
            raise ValueError("Initial-current TOML content must be a table")
        initial_cfg = loaded_initial
    else:
        initial_path = None

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
    defaults = PhysicsSettings()
    physics = PhysicsSettings(
        mu0=_coerce_float(p.get("mu0", defaults.mu0), "physics.mu0"),
        sigma=_coerce_float(p.get("sigma", defaults.sigma), "physics.sigma"),
        inductance_L=_coerce_float(p.get("inductance_L", defaults.inductance_L), "physics.inductance_L"),
        ip_coupling_sign=_coerce_float(p.get("ip_coupling_sign", defaults.ip_coupling_sign), "physics.ip_coupling_sign"),
        plasma_psi_sign=_coerce_float(p.get("plasma_psi_sign", defaults.plasma_psi_sign), "physics.plasma_psi_sign"),
        t_step=_coerce_float(p.get("t_step", defaults.t_step), "physics.t_step"),
        actuator_tau=_coerce_float(p.get("actuator_tau", defaults.actuator_tau), "physics.actuator_tau"),
        Ip0=_coerce_float(p.get("Ip0", defaults.Ip0), "physics.Ip0"),
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
    realism = _load_realism_settings(cfg)
    realism.validate()

    c = _require_mapping(cfg.get("coils", {}), "coils")
    pfc_cfg = _require_mapping(_require_key(c, "pfc", "coils"), "coils.pfc")
    sol_cfg = _require_mapping(_require_key(c, "sol", "coils"), "coils.sol")

    initial_coils_cfg: dict[str, object] | None = None
    if initial_cfg is not None:
        initial_coils_cfg = _require_mapping(initial_cfg.get("coils", {}), "coils")

    def _initial_bank_node(bank: str) -> dict:
        """Вернуть TOML-таблицу начального состояния для банка."""
        if initial_coils_cfg is None:
            return {}
        return _require_mapping(_require_key(initial_coils_cfg, bank, "coils"), f"coils.{bank}")

    def _mk_group(name: str, node: dict, initial_node: dict | None = None) -> tuple[CoilGroup, np.ndarray]:
        """Собрать банк катушек с учетом активной маски."""
        elements = _coerce_actuator_elements(node, name)
        element_weights = _coerce_optional_element_weights(node.get("element_weights"), f"{name}.element_weights", groups=elements)
        state_node = node if initial_node is None or len(initial_node) == 0 else initial_node

        if "currents" in state_node:
            currents = np.asarray(state_node["currents"], dtype=float)
        else:
            currents = np.zeros((len(elements),), dtype=float)
        if currents.shape != (len(elements),):
            raise ValueError(f"{name}.currents must have shape ({len(elements)},)")
        if not np.all(np.isfinite(currents)):
            raise ValueError(f"{name}.currents must contain only finite values")
        active_mask = _coerce_optional_bool_vector(state_node.get("active"), f"{name}.active", size=len(elements))
        if not np.any(active_mask):
            raise ValueError(f"{name}.active must contain at least one active actuator")
        active_elements = [group for group, is_active in zip(elements, active_mask, strict=True) if bool(is_active)]
        active_weights = None if element_weights is None else [weights for weights, is_active in zip(element_weights, active_mask, strict=True) if bool(is_active)]
        active_currents = currents[active_mask]
        actuators = [
            CoilActuator(
                elements=[Coil(R=float(R), Z=float(Z)) for R, Z in group],
                element_weights=(None if active_weights is None else active_weights[i]),
            )
            for i, group in enumerate(active_elements)
        ]
        return CoilGroup(name=str(node.get("name", name)), coils=actuators, currents=active_currents), active_mask

    pfc, pfc_active_mask = _mk_group("coils.pfc", pfc_cfg, _initial_bank_node("pfc"))
    sol, sol_active_mask = _mk_group("coils.sol", sol_cfg, _initial_bank_node("sol"))

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
    boundary_mode = _coerce_boundary_mode(boundary_node.get("mode", "limited"), "boundary.mode")

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
        limiter_name=limiter_name,
        limiter_shape=limiter_shape,
        pfc_active_mask=pfc_active_mask.copy(),
        sol_active_mask=sol_active_mask.copy(),
        initial_currents_source=None if initial_path is None else str(initial_path),
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
    boundary_mode: BoundaryMode = "limited",
) -> None:
    """Записать TOML-конфигурацию текущей расчетной схемы."""
    path = Path(path)
    physics.validate()
    compute = ComputeSettings() if compute is None else compute
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
            "Ip0": physics.Ip0,
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
            "boundary_equivalence_mode": compute.boundary_equivalence_mode,
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
        },
        "coils": {
            "pfc": {
                "name": pfc.name,
                "elements": [arr.tolist() for arr in pfc.element_positions],
                "element_weights": [arr.tolist() for arr in pfc.element_weights] if any(not np.allclose(arr, np.ones_like(arr)) for arr in pfc.element_weights) else None,
                "currents": pfc.initial_currents.tolist(),
            },
            "sol": {
                "name": sol.name,
                "elements": [arr.tolist() for arr in sol.element_positions],
                "element_weights": [arr.tolist() for arr in sol.element_weights] if any(not np.allclose(arr, np.ones_like(arr)) for arr in sol.element_weights) else None,
                "currents": sol.initial_currents.tolist(),
            },
        },
    }
    if limiter_name is not None:
        data["limiter"] = {"name": str(limiter_name)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        tomli_w.dump(_drop_none_values(data), f)
