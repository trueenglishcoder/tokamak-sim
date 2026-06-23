from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import types
from typing import Any, Callable, Mapping, Union, get_args, get_origin

import numpy as np

from tokamak_control.control.base import Controller
from tokamak_control.control.coil_replay import CoilReplayController
from tokamak_control.control.learned_magnetic_controller import LearnedMagneticController
from tokamak_control.control.lqr_t15_zaitsev import LQRT15ZaitsevController
from tokamak_control.control.t15md_replay import T15MDReplayController


@dataclass(frozen=True, slots=True)
class ControllerLaunchParam:
    """Launch-time schema entry for one controller parameter."""

    name: str
    annotation: object
    required: bool
    default: object | None
    validator: Callable[[object], None] | None = None


@dataclass(frozen=True, slots=True)
class ControllerSpec:
    """Registry entry describing one supported controller."""

    name: str
    family: str
    controller_cls: type[Controller]
    launch_params: tuple[ControllerLaunchParam, ...] = ()
    runtime_inputs: tuple[str, ...] = ()


def _validate_finite_positive(name: str) -> Callable[[object], None]:
    def _inner(value: object) -> None:
        v = float(value)
        if not (v > 0.0):
            raise ValueError(f"{name} must be > 0")
    return _inner


def _validate_finite_nonnegative(name: str) -> Callable[[object], None]:
    def _inner(value: object) -> None:
        v = float(value)
        if not (v >= 0.0):
            raise ValueError(f"{name} must be >= 0")
    return _inner


def _validate_positive_int(name: str) -> Callable[[object], None]:
    def _inner(value: object) -> None:
        v = int(value)
        if v <= 0:
            raise ValueError(f"{name} must be > 0")
    return _inner


def _validate_existing_path(name: str) -> Callable[[object], None]:
    def _inner(value: object) -> None:
        p = Path(value)
        if str(p).strip() == "":
            raise ValueError(f"{name} must be a non-empty path")
    return _inner


_ANALYTIC_PARAMS: dict[str, tuple[ControllerLaunchParam, ...]] = {
    "lqr_t15_zaitsev": (
        ControllerLaunchParam("boundary_weight", float, False, 1.0, _validate_finite_nonnegative("boundary_weight")),
        ControllerLaunchParam("ip_weight", float, False, 1.0, _validate_finite_nonnegative("ip_weight")),
        ControllerLaunchParam("derivative_weight", float, False, 0.0, _validate_finite_nonnegative("derivative_weight")),
        ControllerLaunchParam("delta_derivative_weight", float, False, 1.0, _validate_finite_positive("delta_derivative_weight")),
        ControllerLaunchParam("drift_weight_fraction", float, False, 0.0, _validate_finite_nonnegative("drift_weight_fraction")),
        ControllerLaunchParam("current_weight", float, False, 0.0, _validate_finite_nonnegative("current_weight")),
        ControllerLaunchParam("derivative_scale_aps", float, False, 1.0e6, _validate_finite_positive("derivative_scale_aps")),
        ControllerLaunchParam("delta_derivative_scale_aps", float, False, 5.0e5, _validate_finite_positive("delta_derivative_scale_aps")),
        ControllerLaunchParam("boundary_scale_m", float, False, 0.03, _validate_finite_positive("boundary_scale_m")),
        ControllerLaunchParam("ip_scale_a", float, False, 25000.0, _validate_finite_positive("ip_scale_a")),
        ControllerLaunchParam("finite_difference_current_delta_a", float, False, 10.0, _validate_finite_positive("finite_difference_current_delta_a")),
        ControllerLaunchParam("gain_recompute_interval_steps", int, False, 25, _validate_positive_int("gain_recompute_interval_steps")),
        ControllerLaunchParam("gain_recompute_on_boundary_loss", bool, False, True),
    ),
}


_REPLAY_PARAMS: tuple[ControllerLaunchParam, ...] = (
    ControllerLaunchParam("replay_path", Path, True, None, _validate_existing_path("replay_path")),
    ControllerLaunchParam("bank_order", str, False, "pfc,sol"),
    ControllerLaunchParam("time_offset", float | None, False, None),
    ControllerLaunchParam("u_clip", float | None, False, None, _validate_finite_positive("u_clip")),
)

_T15MD_REPLAY_PARAMS: tuple[ControllerLaunchParam, ...] = (
    ControllerLaunchParam("replay_path", Path, True, None, _validate_existing_path("replay_path")),
)

_LEARNED_MAGNETIC_CONTROLLER_PARAMS: tuple[ControllerLaunchParam, ...] = (
    ControllerLaunchParam("export_dir", Path, True, None, _validate_existing_path("export_dir")),
    ControllerLaunchParam("target_preview_stride", int | None, False, None, _validate_positive_int("target_preview_stride")),
    ControllerLaunchParam("action_clip", float, False, 1.0, _validate_finite_positive("action_clip")),
    ControllerLaunchParam("episode_norm_steps", int | None, False, None, _validate_positive_int("episode_norm_steps")),
    ControllerLaunchParam("rolling_episode_norm", bool, False, False),
)



_RUNTIME_INPUTS: dict[str, tuple[str, ...]] = {
    "coil_replay": ("model",),
    "t15md_replay": ("model",),
    "lqr_t15_zaitsev": (
        "model",
        "psi",
        "boundary_poly",
        "center",
        "measure_angles",
        "ref_radii",
        "Ip_ref",
        "scenario",
        "limiter_shape",
        "boundary_mode",
        "legacy_precision_index2",
    ),
    "learned_magnetic_controller": (
        "model",
        "psi",
        "boundary_poly",
        "measured_ip",
        "measured_active_currents",
        "measured_radii",
        "boundary_found",
        "center",
        "measure_angles",
        "ref_radii",
        "Ip_ref",
        "scenario",
        "max_episode_steps",
    ),
}

_SPECS: dict[str, ControllerSpec] = {
    "coil_replay": ControllerSpec(
        name="coil_replay",
        family="current",
        controller_cls=CoilReplayController,
        launch_params=_REPLAY_PARAMS,
        runtime_inputs=_RUNTIME_INPUTS["coil_replay"],
    ),
    "t15md_replay": ControllerSpec(
        name="t15md_replay",
        family="current",
        controller_cls=T15MDReplayController,
        launch_params=_T15MD_REPLAY_PARAMS,
        runtime_inputs=_RUNTIME_INPUTS["t15md_replay"],
    ),
    "lqr_t15_zaitsev": ControllerSpec(
        name="lqr_t15_zaitsev",
        family="joint",
        controller_cls=LQRT15ZaitsevController,
        launch_params=_ANALYTIC_PARAMS["lqr_t15_zaitsev"],
        runtime_inputs=_RUNTIME_INPUTS["lqr_t15_zaitsev"],
    ),
    "learned_magnetic_controller": ControllerSpec(
        name="learned_magnetic_controller",
        family="learned_magnetic",
        controller_cls=LearnedMagneticController,
        launch_params=_LEARNED_MAGNETIC_CONTROLLER_PARAMS,
        runtime_inputs=_RUNTIME_INPUTS["learned_magnetic_controller"],
    ),
}


def controller_names() -> tuple[str, ...]:
    """Return the supported controller names in registry order."""
    return tuple(_SPECS.keys())


def get_controller_spec(name: str) -> ControllerSpec:
    """Return the registry spec for a supported controller name."""
    key = name.lower()
    try:
        return _SPECS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown or disabled controller: {name}") from exc


def controller_launch_params(name: str) -> tuple[ControllerLaunchParam, ...]:
    """Return the launch-time parameter schema for a controller."""
    return get_controller_spec(name).launch_params


def controller_runtime_inputs(name: str) -> tuple[str, ...]:
    """Return the runtime input names consumed by a controller."""
    return get_controller_spec(name).runtime_inputs


def normalize_controller_launch(
    name: str,
    params: Mapping[str, object] | None = None,
) -> tuple[str, dict[str, object], dict[str, object] | None]:
    """
    Normalize launch-time controller parameters using the registry schema.

    Returns canonical controller name, normalized launch parameter dictionary,
    and constructor keyword arguments for the selected controller.
    """
    spec = get_controller_spec(name)
    raw = {} if params is None else dict(params)
    schema = {p.name: p for p in controller_launch_params(spec.name)}

    unknown = sorted(set(raw) - set(schema))
    if unknown:
        raise ValueError(
            f'Controller "{spec.name}" does not accept launch parameters: {", ".join(unknown)}'
        )

    normalized: dict[str, object] = {}
    for param_name, param in schema.items():
        if param_name in raw:
            value = _coerce_launch_value(
                raw[param_name],
                param.annotation,
                f'{spec.name}.{param_name}',
            )
        elif param.required:
            raise ValueError(
                f'Controller "{spec.name}" requires controller parameter "{param_name}"'
            )
        else:
            value = param.default

        if value is not None and param.validator is not None:
            param.validator(value)
        normalized[param_name] = value

    if not schema and raw:
        raise ValueError(
            f'Controller "{spec.name}" does not accept launch parameters'
        )

    return spec.name, normalized, normalized


def build_controller_runtime_call(
    name: str,
    runtime_context: Mapping[str, object],
) -> dict[str, object]:
    """
    Filter the superset runtime context down to the inputs consumed by a controller.
    """
    spec = get_controller_spec(name)
    missing = [key for key in spec.runtime_inputs if key not in runtime_context]
    if missing:
        raise KeyError(
            f'Runtime context for controller "{spec.name}" is missing keys: {", ".join(missing)}'
        )
    return {key: runtime_context[key] for key in spec.runtime_inputs}


def make_controller(name: str, **kwargs: Any) -> Controller:
    """
    Construct a controller by name using the registry spec.

    `config` is expected to be a mapping of constructor keyword arguments
    produced by `normalize_controller_launch(...)`.
    """
    spec = get_controller_spec(name)
    cfg = kwargs.get("config")

    if cfg is None:
        return spec.controller_cls()
    if not isinstance(cfg, Mapping):
        raise TypeError(
            f'Controller "{spec.name}" expects launch parameters as a mapping; '
            f"got {type(cfg).__name__}"
        )
    return spec.controller_cls(**dict(cfg))


def _coerce_launch_value(value: object, annotation: object, param_name: str) -> object:
    if annotation in (Any, object):
        return value

    origin = get_origin(annotation)
    if origin in (types.UnionType, Union):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if value is None:
            if len(args) != len(get_args(annotation)):
                return None
            raise ValueError(f"{param_name} does not allow null")
        last_error: Exception | None = None
        for arg in args:
            try:
                return _coerce_launch_value(value, arg, param_name)
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Could not coerce {param_name}={value!r}") from last_error

    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in {"1", "true", "yes", "on"}:
                return True
            if lower in {"0", "false", "no", "off"}:
                return False
        raise ValueError(f"{param_name} must be a boolean")

    if annotation is int:
        if isinstance(value, bool):
            raise ValueError(f"{param_name} must be an int, not bool")
        if isinstance(value, (int,)):
            return int(value)
        if isinstance(value, float):
            if not np.isfinite(value) or not value.is_integer():
                raise ValueError(f"{param_name} must be an int")
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except Exception as exc:
                raise ValueError(f"{param_name} must be an int") from exc
        raise ValueError(f"{param_name} must be an int")

    if annotation is float:
        if isinstance(value, bool):
            raise ValueError(f"{param_name} must be a float, not bool")
        try:
            out = float(value)
        except Exception as exc:
            raise ValueError(f"{param_name} must be a float") from exc
        if not np.isfinite(out):
            raise ValueError(f"{param_name} must be finite")
        return out

    if annotation is str:
        return str(value)

    if annotation is Path:
        if isinstance(value, Path):
            return value
        return Path(str(value))

    if origin is tuple:
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            inner = args[0]
            if isinstance(value, str):
                items = [] if value == "" else [item.strip() for item in value.split(",")]
            elif isinstance(value, tuple):
                items = list(value)
            elif isinstance(value, list):
                items = value
            else:
                raise ValueError(f"{param_name} must be a tuple-like value")
            return tuple(_coerce_launch_value(item, inner, param_name) for item in items)

    return value
