from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Literal

import numpy as np

from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections


ScenarioName = Literal[
    "nominal",
    "boundary_step",
    "ip_ramp",
    "ip_flat_top",
    "ip_jet_like",
    "boundary_pulse",
    "joint_disturbance",
    "shot_follow",
    "ip_table",
    "ip_follow",
]


_IP_FOLLOW_HIGH_RADII_32 = np.asarray(
    [
        1.018318847924856,
        1.024075461450218,
        1.0418022177227833,
        1.0727559092605146,
        1.1189267823782143,
        1.1827041020900368,
        1.2663763235525245,
        1.3695675632742756,
        1.4818834088702255,
        1.5627602484015104,
        1.5201101326934905,
        1.331340357620933,
        1.1330847923588183,
        0.9924403054444496,
        0.9045276918068736,
        0.856791094714635,
        0.8416641056372222,
        0.8567910985378567,
        0.904527693018612,
        0.992440308149898,
        1.133084778411449,
        1.3313403359700646,
        1.520110151155068,
        1.562760255351232,
        1.4818834031851065,
        1.369567560788898,
        1.266376316039964,
        1.1827041078328442,
        1.1189267945252548,
        1.0727559070805186,
        1.0418022129729416,
        1.0240754736216815,
    ],
    dtype=float,
)

_IP_FOLLOW_T15_LINEAR_INTERCEPT_32 = np.asarray(
    [
        0.6658807448552658,
        0.6703106609922166,
        0.6828203727375205,
        0.7026708241078304,
        0.7274045146003496,
        0.7490463865833679,
        0.7348604184483588,
        0.6495192907273201,
        0.6067774896545979,
        0.5738611665937543,
        0.5443613568955203,
        0.5194483994642513,
        0.49923978615271203,
        0.4838653899276846,
        0.47315953422019585,
        0.46633202713385735,
        0.4639348229454704,
        0.4644708923229907,
        0.46959681550418025,
        0.478573518569924,
        0.49266688000120723,
        0.5115813059263233,
        0.5361571936402556,
        0.5664456298399532,
        0.6042663917919125,
        0.653019930224456,
        0.7114268316649008,
        0.7337222613679713,
        0.7195715521723269,
        0.6981948531565905,
        0.680374244574667,
        0.6692787470560265,
    ],
    dtype=float,
)

_IP_FOLLOW_T15_LINEAR_SLOPE_32 = np.asarray(
    [
        1.1136715085657283e-08,
        1.6368107049087094e-08,
        3.3604546091523256e-08,
        6.998337303038462e-08,
        1.3627268748334725e-07,
        2.5207463537709397e-07,
        4.4616859032880337e-07,
        6.078928868373626e-07,
        5.485112416631507e-07,
        4.712740333723755e-07,
        4.204601274504942e-07,
        3.907035444770126e-07,
        3.7669344278644623e-07,
        3.730793422251813e-07,
        3.763929665507843e-07,
        3.8519231681175493e-07,
        3.952758194058193e-07,
        4.068984667368114e-07,
        4.17213806662644e-07,
        4.30116465620165e-07,
        4.4421845630439315e-07,
        4.6376117885936766e-07,
        4.885188963419453e-07,
        5.201374169033318e-07,
        5.473834838318301e-07,
        5.301459666376297e-07,
        3.9827786858933327e-07,
        2.3573985825917623e-07,
        1.303744769259839e-07,
        6.853860660768538e-08,
        3.346551934663502e-08,
        1.6316805132700745e-08,
    ],
    dtype=float,
)


@dataclass(frozen=True, slots=True)
class Scenario:
    """
    Reference signals for boundary radii and plasma current.

    Parameters
    ----------
    name
        Scenario identifier.
    ref_radii
        Callable with signature ``ref_radii(angles, t) -> np.ndarray`` that
        returns desired boundary radii at the given angles and time.
    Ip_ref
        Callable with signature ``Ip_ref(t) -> float`` that returns desired
        plasma current at time ``t``.
    """

    name: str
    ref_radii: Callable[[np.ndarray, float], np.ndarray]
    Ip_ref: Callable[[float], float]


def _resample_periodic_profile(profile: np.ndarray, size: int) -> np.ndarray:
    """Периодически пересэмплировать профиль на другое число углов."""
    profile = np.asarray(profile, dtype=float).reshape(-1)
    if profile.size == int(size):
        return profile.copy()
    if profile.size < 2:
        raise ValueError("Periodic profile must contain at least 2 samples")

    src = np.arange(profile.size + 1, dtype=float)
    ext = np.concatenate([profile, profile[:1]])
    dst = np.linspace(0.0, float(profile.size), int(size), endpoint=False, dtype=float)
    return np.interp(dst, src, ext)


def _ip_follow_fixed_shape_radii(base_radii: np.ndarray, high_radii_32: np.ndarray, ip_now: float, ip_peak: float, size: int) -> np.ndarray:
    """Построить reference boundary старым fixed-shape способом."""
    high_radii = _resample_periodic_profile(high_radii_32, int(size))
    base = np.asarray(base_radii, dtype=float)
    if base.shape != high_radii.shape:
        raise ValueError(f"Scenario 'ip_follow' requires base_radii with shape {high_radii.shape}, got {base.shape}")
    alpha = float(np.clip(abs(ip_now) / ip_peak, 0.0, 1.0))
    return (1.0 - alpha) * base + alpha * high_radii


def _ip_follow_t15_linear_radii(ip_now: float, size: int) -> np.ndarray:
    """Построить boundary reference по линейной аппроксимации радиусов от |Ip|."""
    intercept = _resample_periodic_profile(_IP_FOLLOW_T15_LINEAR_INTERCEPT_32, int(size))
    slope = _resample_periodic_profile(_IP_FOLLOW_T15_LINEAR_SLOPE_32, int(size))
    radii = intercept + slope * abs(float(ip_now))
    return np.clip(radii, 1e-6, None)


def make_nominal(base_radii: np.ndarray, Ip0: float) -> Scenario:
    """Return a scenario with fixed boundary radii and constant plasma current."""

    base_radii = np.asarray(base_radii, dtype=float)

    def _r(_angles: np.ndarray, _t: float) -> np.ndarray:
        return base_radii.copy()

    def _ip(_t: float) -> float:
        return float(Ip0)

    return Scenario("nominal", _r, _ip)


def make_boundary_step(
    base_radii: np.ndarray,
    Ip0: float,
    *,
    step_time: float,
    delta: float,
) -> Scenario:
    """Return a scenario with a step increase in all target radii at ``t >= step_time``."""

    base_radii = np.asarray(base_radii, dtype=float)

    def _r(_angles: np.ndarray, t: float) -> np.ndarray:
        return base_radii + (delta if t >= step_time else 0.0)

    def _ip(_t: float) -> float:
        return float(Ip0)

    return Scenario("boundary_step", _r, _ip)


def make_ip_ramp(
    base_radii: np.ndarray,
    Ip0: float,
    *,
    ramp_rate: float,
) -> Scenario:
    """Return a scenario with fixed boundary radii and linearly ramped plasma current."""

    base_radii = np.asarray(base_radii, dtype=float)

    def _r(_angles: np.ndarray, _t: float) -> np.ndarray:
        return base_radii.copy()

    def _ip(t: float) -> float:
        return float(Ip0 + ramp_rate * t)

    return Scenario("ip_ramp", _r, _ip)


def make_ip_flat_top(
    base_radii: np.ndarray,
    Ip0: float,
    *,
    ip_start: float,
    ip_flat: float,
    ip_end: float,
    t_ramp_up: float,
    t_flat: float,
    t_ramp_down: float,
) -> Scenario:
    """Return a fixed-boundary scenario with ramp-up, flat-top, and ramp-down plasma current."""

    del Ip0

    base_radii = np.asarray(base_radii, dtype=float)

    ip_start_value = float(ip_start)
    ip_flat_value = float(ip_flat)
    ip_end_value = float(ip_end)
    t_up = float(t_ramp_up)
    t_hold = float(t_flat)
    t_down = float(t_ramp_down)

    for name, value in (
        ("ip_start", ip_start_value),
        ("ip_flat", ip_flat_value),
        ("ip_end", ip_end_value),
    ):
        if not np.isfinite(value):
            raise ValueError(f"Scenario parameter '{name}' must be finite")

    if t_up <= 0.0:
        raise ValueError("Scenario parameter 't_ramp_up' must be > 0")
    if t_hold <= 0.0:
        raise ValueError("Scenario parameter 't_flat' must be > 0")
    if t_down <= 0.0:
        raise ValueError("Scenario parameter 't_ramp_down' must be > 0")

    t1 = t_up
    t2 = t_up + t_hold
    t3 = t_up + t_hold + t_down

    def _r(_angles: np.ndarray, _t: float) -> np.ndarray:
        return base_radii.copy()

    def _lerp(a: float, b: float, alpha: float) -> float:
        return float((1.0 - alpha) * a + alpha * b)

    def _ip(t: float) -> float:
        if t <= 0.0:
            return ip_start_value
        if t < t1:
            return _lerp(ip_start_value, ip_flat_value, t / t1)
        if t < t2:
            return ip_flat_value
        if t < t3:
            return _lerp(ip_flat_value, ip_end_value, (t - t2) / t_down)
        return ip_end_value

    return Scenario("ip_flat_top", _r, _ip)


def make_ip_jet_like(
    base_radii: np.ndarray,
    Ip0: float,
    *,
    ip_start: float = -425885.84375,
    ip_flat: float = -1989130.25,
    ip_end: float = -184966.0625,
    t_ramp_up: float = 2.0,
    t_flat: float = 3.0,
    t_ramp_down: float = 2.0,
) -> Scenario:
    """Return a fixed-boundary negative Ip ramp-up, flat-top, and ramp-down scenario with JET-like defaults."""

    del Ip0

    return make_ip_flat_top(
        base_radii,
        0.0,
        ip_start=ip_start,
        ip_flat=ip_flat,
        ip_end=ip_end,
        t_ramp_up=t_ramp_up,
        t_flat=t_flat,
        t_ramp_down=t_ramp_down,
    )


def make_boundary_pulse(
    base_radii: np.ndarray,
    Ip0: float,
    *,
    t0: float,
    t1: float,
    delta: float,
) -> Scenario:
    """Return a scenario with a temporary boundary pulse for ``t0 <= t < t1``."""

    base_radii = np.asarray(base_radii, dtype=float)

    def _r(_angles: np.ndarray, t: float) -> np.ndarray:
        active = (t >= t0) and (t < t1)
        return base_radii + (delta if active else 0.0)

    def _ip(_t: float) -> float:
        return float(Ip0)

    return Scenario("boundary_pulse", _r, _ip)


def make_joint_disturbance(
    base_radii: np.ndarray,
    Ip0: float,
    *,
    ramp_rate: float,
    t0: float,
    t1: float,
    delta: float,
) -> Scenario:
    """Return a scenario combining a boundary pulse and a linear plasma current ramp."""

    base_radii = np.asarray(base_radii, dtype=float)

    def _r(_angles: np.ndarray, t: float) -> np.ndarray:
        active = (t >= t0) and (t < t1)
        return base_radii + (delta if active else 0.0)

    def _ip(t: float) -> float:
        return float(Ip0 + ramp_rate * t)

    return Scenario("joint_disturbance", _r, _ip)


def _load_semicolon_csv(path: str | Path) -> np.ndarray:
    rows: list[list[float]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = [p for p in line.split(";") if p != ""]
            if not parts:
                continue
            rows.append([float(p) for p in parts])
    if not rows:
        raise ValueError(f"CSV file is empty: {path}")
    n_cols = len(rows[0])
    if any(len(r) != n_cols for r in rows):
        raise ValueError(f"CSV file has inconsistent row widths: {path}")
    return np.asarray(rows, dtype=float)


def _interp_clamped(query_t: float, t: np.ndarray, y: np.ndarray) -> float:
    tq = float(np.clip(query_t, float(t[0]), float(t[-1])))
    return float(np.interp(tq, t, y))


def _interp_polyline_clamped(
    query_t: float,
    t_samples: np.ndarray,
    r_samples: np.ndarray,
    z_samples: np.ndarray,
) -> np.ndarray:
    tq = float(np.clip(query_t, float(t_samples[0]), float(t_samples[-1])))
    hi = int(np.searchsorted(t_samples, tq, side="right"))

    if hi <= 0:
        return np.column_stack([r_samples[0], z_samples[0]])
    if hi >= len(t_samples):
        return np.column_stack([r_samples[-1], z_samples[-1]])

    lo = hi - 1
    t_lo = float(t_samples[lo])
    t_hi = float(t_samples[hi])
    if t_hi <= t_lo:
        return np.column_stack([r_samples[lo], z_samples[lo]])

    a = (tq - t_lo) / (t_hi - t_lo)
    r = (1.0 - a) * r_samples[lo] + a * r_samples[hi]
    z = (1.0 - a) * z_samples[lo] + a * z_samples[hi]
    return np.column_stack([r, z])


def make_shot_follow(
    base_radii: np.ndarray,
    Ip0: float,
    *,
    center: tuple[float, float] | None,
    ip_csv: str | Path,
    rboundary_csv: str | Path,
    zboundary_csv: str | Path,
) -> Scenario:
    """Return a scenario that follows Ip and boundary references from shot tables."""

    del base_radii, Ip0

    if center is None:
        raise ValueError("Scenario 'shot_follow' requires a non-None center")
    if not np.isfinite(float(center[0])) or not np.isfinite(float(center[1])):
        raise ValueError("Scenario 'shot_follow' requires a finite center")

    ip_mat = _load_semicolon_csv(ip_csv)
    rb_mat = _load_semicolon_csv(rboundary_csv)
    zb_mat = _load_semicolon_csv(zboundary_csv)

    if ip_mat.ndim != 2 or ip_mat.shape[1] != 2:
        raise ValueError("Scenario 'shot_follow' expects ip_csv with exactly 2 columns")
    if rb_mat.ndim != 2 or zb_mat.ndim != 2:
        raise ValueError("Scenario 'shot_follow' expects boundary CSVs to be 2-D")
    if rb_mat.shape != zb_mat.shape:
        raise ValueError("Scenario 'shot_follow' requires rboundary and zboundary to have identical shape")
    if rb_mat.shape[1] < 2:
        raise ValueError("Scenario 'shot_follow' boundary CSVs must contain time plus at least one boundary point")

    t_ip = np.asarray(ip_mat[:, 0], dtype=float)
    ip_vals = np.asarray(ip_mat[:, 1], dtype=float)
    t_b = np.asarray(rb_mat[:, 0], dtype=float)
    if not np.allclose(t_b, np.asarray(zb_mat[:, 0], dtype=float)):
        raise ValueError("Scenario 'shot_follow' requires matching boundary timestamps")

    if np.any(np.diff(t_ip) < 0.0):
        raise ValueError("Scenario 'shot_follow' requires nondecreasing ip timestamps")
    if np.any(np.diff(t_b) < 0.0):
        raise ValueError("Scenario 'shot_follow' requires nondecreasing boundary timestamps")

    r_samples = np.asarray(rb_mat[:, 1:], dtype=float)
    z_samples = np.asarray(zb_mat[:, 1:], dtype=float)
    def _r(angles: np.ndarray, t: float) -> np.ndarray:
        poly = _interp_polyline_clamped(float(t), t_b, r_samples, z_samples)
        return radii_from_polyline_ray_intersections(poly, center, np.asarray(angles, dtype=float))

    def _ip(t: float) -> float:
        return _interp_clamped(float(t), t_ip, ip_vals)

    return Scenario("shot_follow", _r, _ip)


def make_ip_table(
    base_radii: np.ndarray,
    Ip0: float,
    *,
    ip_csv: str | Path,
    time_offset: float | None = None,
) -> Scenario:
    """Return a fixed-boundary scenario whose plasma-current reference follows a time table."""

    del Ip0

    base_radii = np.asarray(base_radii, dtype=float)

    ip_mat = _load_semicolon_csv(ip_csv)
    if ip_mat.ndim != 2 or ip_mat.shape[1] != 2:
        raise ValueError("Scenario 'ip_table' expects ip_csv with exactly 2 columns")

    source_t = np.asarray(ip_mat[:, 0], dtype=float)
    ip_vals = np.asarray(ip_mat[:, 1], dtype=float)

    if not np.all(np.isfinite(source_t)):
        raise ValueError("Scenario 'ip_table' requires finite ip timestamps")
    if not np.all(np.isfinite(ip_vals)):
        raise ValueError("Scenario 'ip_table' requires finite Ip values")
    if np.any(np.diff(source_t) < 0.0):
        raise ValueError("Scenario 'ip_table' requires nondecreasing ip timestamps")

    if time_offset is None:
        origin = float(source_t[0])
    else:
        origin = float(time_offset)
        if not np.isfinite(origin):
            raise ValueError("Scenario parameter 'time_offset' must be finite when provided")

    t_ip = source_t - origin

    def _r(_angles: np.ndarray, _t: float) -> np.ndarray:
        return base_radii.copy()

    def _ip(t: float) -> float:
        return _interp_clamped(float(t), t_ip, ip_vals)

    return Scenario("ip_table", _r, _ip)


def make_ip_follow(
    base_radii: np.ndarray,
    Ip0: float,
    *,
    ip_csv: str | Path,
    boundary_mode: str = "fixed_shape",
) -> Scenario:
    """
    Return a scenario that follows a preprocessed Ip table and generates a fixed-shape
    boundary reference that interpolates between the startup boundary and a
    fixed high-current profile.

    Expected input
    --------------
    ip_csv must already be preprocessed:
    - column 1: local runtime time in seconds, starting from 0
    - column 2: plasma current in the simulator's runtime units
    """

    del Ip0

    base_radii = np.asarray(base_radii, dtype=float)
    boundary_mode_name = str(boundary_mode).strip().lower()

    ip_mat = _load_semicolon_csv(ip_csv)
    if ip_mat.ndim != 2 or ip_mat.shape[1] != 2:
        raise ValueError("Scenario 'ip_follow' expects ip_csv with exactly 2 columns")

    t_ip = np.asarray(ip_mat[:, 0], dtype=float)
    ip_vals = np.asarray(ip_mat[:, 1], dtype=float)

    if np.any(np.diff(t_ip) < 0.0):
        raise ValueError("Scenario 'ip_follow' requires nondecreasing ip timestamps")

    if boundary_mode_name not in {"fixed_shape", "t15_linear"}:
        raise ValueError(
            "Scenario 'ip_follow' received unsupported boundary_mode. "
            "Expected one of: fixed_shape, t15_linear"
        )

    ip_peak = float(np.max(np.abs(ip_vals)))
    if not np.isfinite(ip_peak) or ip_peak <= 0.0:
        raise ValueError("Scenario 'ip_follow' requires ip_csv with at least one nonzero finite Ip value")

    def _ip(t: float) -> float:
        return _interp_clamped(float(t), t_ip, ip_vals)

    def _r(angles: np.ndarray, t: float) -> np.ndarray:
        angles = np.asarray(angles, dtype=float)
        ip_now = _ip(t)
        if boundary_mode_name == "t15_linear":
            return _ip_follow_t15_linear_radii(ip_now, int(angles.size))
        return _ip_follow_fixed_shape_radii(base_radii, _IP_FOLLOW_HIGH_RADII_32, ip_now, ip_peak, int(angles.size))

    return Scenario("ip_follow", _r, _ip)


def _require_params(
    scenario_name: str,
    params: Mapping[str, object],
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> None:
    missing = [name for name in required if name not in params]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Scenario '{scenario_name}' requires parameter(s): {missing_str}"
        )

    allowed = set(required) | set(optional)
    unexpected = sorted(k for k in params if k not in allowed)
    if unexpected:
        unexpected_str = ", ".join(unexpected)
        raise ValueError(
            f"Scenario '{scenario_name}' received unexpected parameter(s): {unexpected_str}"
        )


def _coerce_float_param(params: Mapping[str, object], name: str) -> float:
    try:
        value = float(params[name])
    except KeyError as e:
        raise ValueError(f"Missing scenario parameter '{name}'") from e
    except (TypeError, ValueError) as e:
        raise ValueError(f"Scenario parameter '{name}' must be numeric") from e
    if not np.isfinite(value):
        raise ValueError(f"Scenario parameter '{name}' must be finite")
    return value


def _validate_nonnegative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"Scenario parameter '{name}' must be >= 0")


def _validate_time_window(t0: float, t1: float) -> None:
    if t1 <= t0:
        raise ValueError("Scenario parameters must satisfy t1 > t0")


def make_scenario(
    name: ScenarioName,
    base_radii: np.ndarray,
    Ip0: float,
    params: Mapping[str, object] | None = None,
    *,
    center: tuple[float, float] | None = None,
) -> Scenario:
    """
    Build a scenario by name using explicit runtime parameters.

    Notes
    -----
    This is the canonical scenario construction path for the standard
    simulation runner.
    """
    params = {} if params is None else dict(params)

    if name == "nominal":
        _require_params(name, params, required=())
        return make_nominal(base_radii, Ip0)

    if name == "boundary_step":
        _require_params(name, params, required=("step_time", "delta"))
        step_time = _coerce_float_param(params, "step_time")
        delta = _coerce_float_param(params, "delta")
        _validate_nonnegative("step_time", step_time)
        return make_boundary_step(base_radii, Ip0, step_time=step_time, delta=delta)

    if name == "ip_ramp":
        _require_params(name, params, required=("ramp_rate",))
        ramp_rate = _coerce_float_param(params, "ramp_rate")
        return make_ip_ramp(base_radii, Ip0, ramp_rate=ramp_rate)

    if name == "ip_flat_top":
        _require_params(
            name,
            params,
            required=("ip_start", "ip_flat", "ip_end", "t_ramp_up", "t_flat", "t_ramp_down"),
        )
        ip_start = _coerce_float_param(params, "ip_start")
        ip_flat = _coerce_float_param(params, "ip_flat")
        ip_end = _coerce_float_param(params, "ip_end")
        t_ramp_up = _coerce_float_param(params, "t_ramp_up")
        t_flat = _coerce_float_param(params, "t_flat")
        t_ramp_down = _coerce_float_param(params, "t_ramp_down")
        return make_ip_flat_top(
            base_radii,
            Ip0,
            ip_start=ip_start,
            ip_flat=ip_flat,
            ip_end=ip_end,
            t_ramp_up=t_ramp_up,
            t_flat=t_flat,
            t_ramp_down=t_ramp_down,
        )

    if name == "ip_jet_like":
        _require_params(
            name,
            params,
            required=(),
            optional=("ip_start", "ip_flat", "ip_end", "t_ramp_up", "t_flat", "t_ramp_down"),
        )
        ip_start = (
            -425885.84375 if "ip_start" not in params
            else _coerce_float_param(params, "ip_start")
        )
        ip_flat = (
            -1989130.25 if "ip_flat" not in params
            else _coerce_float_param(params, "ip_flat")
        )
        ip_end = (
            -184966.0625 if "ip_end" not in params
            else _coerce_float_param(params, "ip_end")
        )
        t_ramp_up = (
            2.0 if "t_ramp_up" not in params
            else _coerce_float_param(params, "t_ramp_up")
        )
        t_flat = (
            3.0 if "t_flat" not in params
            else _coerce_float_param(params, "t_flat")
        )
        t_ramp_down = (
            2.0 if "t_ramp_down" not in params
            else _coerce_float_param(params, "t_ramp_down")
        )
        return make_ip_jet_like(
            base_radii,
            Ip0,
            ip_start=ip_start,
            ip_flat=ip_flat,
            ip_end=ip_end,
            t_ramp_up=t_ramp_up,
            t_flat=t_flat,
            t_ramp_down=t_ramp_down,
        )

    if name == "boundary_pulse":
        _require_params(name, params, required=("t0", "t1", "delta"))
        t0 = _coerce_float_param(params, "t0")
        t1 = _coerce_float_param(params, "t1")
        delta = _coerce_float_param(params, "delta")
        _validate_nonnegative("t0", t0)
        _validate_nonnegative("t1", t1)
        _validate_time_window(t0, t1)
        return make_boundary_pulse(base_radii, Ip0, t0=t0, t1=t1, delta=delta)

    if name == "joint_disturbance":
        _require_params(name, params, required=("ramp_rate", "t0", "t1", "delta"))
        ramp_rate = _coerce_float_param(params, "ramp_rate")
        t0 = _coerce_float_param(params, "t0")
        t1 = _coerce_float_param(params, "t1")
        delta = _coerce_float_param(params, "delta")
        _validate_nonnegative("t0", t0)
        _validate_nonnegative("t1", t1)
        _validate_time_window(t0, t1)
        return make_joint_disturbance(
            base_radii,
            Ip0,
            ramp_rate=ramp_rate,
            t0=t0,
            t1=t1,
            delta=delta,
        )

    if name == "shot_follow":
        _require_params(
            name,
            params,
            required=("ip_csv", "rboundary_csv", "zboundary_csv"),
        )
        ip_csv = str(params["ip_csv"])
        rboundary_csv = str(params["rboundary_csv"])
        zboundary_csv = str(params["zboundary_csv"])
        return make_shot_follow(
            base_radii,
            Ip0,
            center=center,
            ip_csv=ip_csv,
            rboundary_csv=rboundary_csv,
            zboundary_csv=zboundary_csv,
        )

    if name == "ip_table":
        _require_params(
            name,
            params,
            required=("ip_csv",),
            optional=("time_offset",),
        )
        ip_csv = str(params["ip_csv"])
        time_offset = (
            None if "time_offset" not in params
            else _coerce_float_param(params, "time_offset")
        )
        return make_ip_table(
            base_radii,
            Ip0,
            ip_csv=ip_csv,
            time_offset=time_offset,
        )

    if name == "ip_follow":
        _require_params(
            name,
            params,
            required=("ip_csv",),
            optional=("boundary_mode",),
        )
        ip_csv = str(params["ip_csv"])
        boundary_mode = "fixed_shape" if "boundary_mode" not in params else str(params["boundary_mode"])
        return make_ip_follow(
            base_radii,
            Ip0,
            ip_csv=ip_csv,
            boundary_mode=boundary_mode,
        )

    raise ValueError(f"Unknown scenario name: {name}")
