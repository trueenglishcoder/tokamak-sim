"""Подбор параметров простой аналитической формы границы плазмы."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections


@dataclass(frozen=True, slots=True, repr=True)
class BoundaryParameters:
    """Параметры аналитической формы внешней границы плазмы."""

    R0: float
    Z0: float
    A0: float
    kappa: float
    delta: float

    def as_array(self) -> np.ndarray:
        """Вернуть параметры в стабильном порядке для численных проверок."""
        return np.array([self.R0, self.Z0, self.A0, self.kappa, self.delta], dtype=float)


@dataclass(frozen=True, slots=True, repr=True)
class BoundaryFitResult:
    """Результат подбора параметров для одного записанного контура."""

    parameters: BoundaryParameters | None
    rmse: float
    max_error: float
    n_boundary_points: int
    fit_status: str


def evaluate_parametric_boundary(theta: np.ndarray, parameters: BoundaryParameters) -> np.ndarray:
    """Вычислить точки границы по аналитической форме для заданных углов."""
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    sin_t = np.sin(theta_arr)
    R = float(parameters.R0) + float(parameters.A0) * np.cos(theta_arr) - float(parameters.delta) * float(parameters.A0) * sin_t * sin_t
    Z = float(parameters.Z0) + float(parameters.A0) * float(parameters.kappa) * sin_t
    return np.column_stack([R, Z])


def fit_parametric_boundary(
    polyline: np.ndarray,
    *,
    sample_count: int = 256,
    theta_count: int = 720,
    iterations: int = 8,
    min_points: int = 16,
    initial_parameters: BoundaryParameters | None = None,
) -> BoundaryFitResult:
    """Подобрать параметры аналитической формы к записанному контуру границы."""
    try:
        clean_polyline = _clean_polyline(polyline)
        if clean_polyline.shape[0] < int(min_points):
            return BoundaryFitResult(
                parameters=None,
                rmse=float("nan"),
                max_error=float("nan"),
                n_boundary_points=int(clean_polyline.shape[0]),
                fit_status="too_few_points",
            )
        points = _resample_closed_polyline(clean_polyline, int(sample_count))
        parameters = initial_parameters if _is_valid_initial(initial_parameters) else _initial_parameters_from_extrema(points)
        previous = parameters.as_array()
        for _ in range(max(int(iterations), 1)):
            theta = _nearest_model_angles(points, parameters, int(theta_count))
            parameters = _solve_parameters_for_angles(points, theta)
            current = parameters.as_array()
            scale = max(float(np.linalg.norm(previous)), 1.0)
            if float(np.linalg.norm(current - previous)) / scale < 1.0e-8:
                break
            previous = current
        rmse, max_error = _nearest_model_errors(points, parameters, int(theta_count))
        return BoundaryFitResult(
            parameters=parameters,
            rmse=rmse,
            max_error=max_error,
            n_boundary_points=int(clean_polyline.shape[0]),
            fit_status="ok",
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return BoundaryFitResult(
            parameters=None,
            rmse=float("nan"),
            max_error=float("nan"),
            n_boundary_points=0,
            fit_status=f"fit_failed:{exc}",
        )


def _is_valid_initial(parameters: BoundaryParameters | None) -> bool:
    """Проверить, можно ли использовать предыдущую успешную оценку как начальную."""
    if parameters is None:
        return False
    values = parameters.as_array()
    return bool(np.all(np.isfinite(values)) and parameters.A0 > 0.0 and parameters.kappa > 0.0)


def _clean_polyline(polyline: np.ndarray, *, duplicate_tol: float = 1.0e-10) -> np.ndarray:
    """Удалить NaN-заполнение и соседние повторяющиеся точки контура."""
    arr = np.asarray(polyline, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"polyline must have shape (N, 2), got {arr.shape}")
    finite = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite]
    if arr.shape[0] < 3:
        raise ValueError("polyline must contain at least three finite points")
    distances = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    keep = np.concatenate([np.array([True]), distances > float(duplicate_tol)])
    arr = arr[keep]
    if arr.shape[0] >= 2 and np.linalg.norm(arr[0] - arr[-1]) <= float(duplicate_tol):
        arr = arr[:-1]
    if arr.shape[0] < 3:
        raise ValueError("polyline must contain at least three distinct finite points")
    return arr


def _as_closed_polyline(polyline: np.ndarray) -> np.ndarray:
    """Вернуть замкнутую копию контура."""
    arr = np.asarray(polyline, dtype=float)
    if arr.shape[0] == 0:
        raise ValueError("polyline must be non-empty")
    if np.allclose(arr[0], arr[-1]):
        return arr.copy()
    return np.vstack([arr, arr[0]])


def _resample_closed_polyline(polyline: np.ndarray, sample_count: int) -> np.ndarray:
    """Пересэмплировать замкнутый контур равномерно по длине дуги."""
    if int(sample_count) < 8:
        raise ValueError("sample_count must be at least 8")
    closed = _as_closed_polyline(polyline)
    segments = np.diff(closed, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 1.0e-12
    if not np.any(valid):
        raise ValueError("polyline perimeter is zero")
    starts = closed[:-1][valid]
    vectors = segments[valid]
    lengths = lengths[valid]
    cumulative = np.concatenate([np.array([0.0]), np.cumsum(lengths)])
    perimeter = float(cumulative[-1])
    if perimeter <= 0.0:
        raise ValueError("polyline perimeter is zero")
    targets = np.linspace(0.0, perimeter, int(sample_count), endpoint=False)
    indices = np.searchsorted(cumulative, targets, side="right") - 1
    indices = np.clip(indices, 0, lengths.shape[0] - 1)
    local = (targets - cumulative[indices]) / lengths[indices]
    return starts[indices] + local[:, None] * vectors[indices]


def _initial_parameters_from_extrema(points: np.ndarray) -> BoundaryParameters:
    """Построить начальную оценку параметров по экстремумам контура."""
    R = points[:, 0]
    Z = points[:, 1]
    R_min = float(np.min(R))
    R_max = float(np.max(R))
    Z_min = float(np.min(Z))
    Z_max = float(np.max(Z))
    R0 = 0.5 * (R_min + R_max)
    Z0 = 0.5 * (Z_min + Z_max)
    A0 = max(0.5 * (R_max - R_min), 1.0e-9)
    kappa = max((Z_max - Z_min) / (2.0 * A0), 1.0e-9)
    top_index = int(np.argmax(Z))
    bottom_index = int(np.argmin(Z))
    vertical_mid_R = 0.5 * (float(R[top_index]) + float(R[bottom_index]))
    delta = (R0 - vertical_mid_R) / A0
    return BoundaryParameters(R0=R0, Z0=Z0, A0=A0, kappa=kappa, delta=delta)


def _nearest_model_angles(points: np.ndarray, parameters: BoundaryParameters, theta_count: int) -> np.ndarray:
    """Назначить каждой точке ближайший угол аналитической модели."""
    theta_grid = np.linspace(-np.pi, np.pi, max(int(theta_count), 32), endpoint=False)
    model = evaluate_parametric_boundary(theta_grid, parameters)
    deltas = points[:, None, :] - model[None, :, :]
    distances_sq = np.sum(deltas * deltas, axis=2)
    indices = np.argmin(distances_sq, axis=1)
    return theta_grid[indices]


def _solve_parameters_for_angles(points: np.ndarray, theta: np.ndarray) -> BoundaryParameters:
    """Решить линейную задачу МНК при фиксированном соответствии точек и углов."""
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    if theta_arr.shape[0] != points.shape[0]:
        raise ValueError("theta count must match point count")
    sin_t = np.sin(theta_arr)
    cos_t = np.cos(theta_arr)
    ones = np.ones_like(theta_arr)
    r_design = np.column_stack([ones, cos_t, sin_t * sin_t])
    z_design = np.column_stack([ones, sin_t])
    r_coeff, *_ = np.linalg.lstsq(r_design, points[:, 0], rcond=None)
    z_coeff, *_ = np.linalg.lstsq(z_design, points[:, 1], rcond=None)
    R0 = float(r_coeff[0])
    A0 = float(r_coeff[1])
    distortion_coeff = float(r_coeff[2])
    Z0 = float(z_coeff[0])
    vertical_coeff = float(z_coeff[1])
    if not np.all(np.isfinite(np.array([R0, A0, distortion_coeff, Z0, vertical_coeff], dtype=float))):
        raise ValueError("least-squares solution is not finite")
    if A0 <= 1.0e-12:
        raise ValueError("fitted minor radius is not positive")
    kappa = vertical_coeff / A0
    if kappa <= 1.0e-12:
        raise ValueError("fitted elongation is not positive")
    delta = -distortion_coeff / A0
    return BoundaryParameters(R0=R0, Z0=Z0, A0=A0, kappa=float(kappa), delta=float(delta))


def _nearest_model_errors(points: np.ndarray, parameters: BoundaryParameters, theta_count: int) -> tuple[float, float]:
    """Оценить ошибки как расстояния до ближайших точек плотной модели."""
    theta_grid = np.linspace(-np.pi, np.pi, max(int(theta_count), 32), endpoint=False)
    model = evaluate_parametric_boundary(theta_grid, parameters)
    deltas = points[:, None, :] - model[None, :, :]
    distances = np.sqrt(np.min(np.sum(deltas * deltas, axis=2), axis=1))
    return float(np.sqrt(np.mean(distances * distances))), float(np.max(distances))

@dataclass(frozen=True, slots=True, repr=True)
class BoundaryParameterBounds:
    """Границы допустимых значений параметров аналитической формы."""

    R0: tuple[float, float]
    Z0: tuple[float, float]
    A0: tuple[float, float]
    kappa: tuple[float, float]
    delta: tuple[float, float]

    def __post_init__(self) -> None:
        """Проверить все интервалы при создании объекта."""
        for name, interval in self.as_mapping().items():
            if len(tuple(interval)) != 2:
                raise ValueError(f"bounds interval {name} must contain two values")
            lo, hi = (float(interval[0]), float(interval[1]))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                raise ValueError(f"bounds interval {name} must satisfy finite hi > lo")

    def as_mapping(self) -> dict[str, tuple[float, float]]:
        """Вернуть интервалы в стабильном порядке имен параметров."""
        return {
            "R0": self.R0,
            "Z0": self.Z0,
            "A0": self.A0,
            "kappa": self.kappa,
            "delta": self.delta,
        }

    def contains(self, parameters: BoundaryParameters) -> bool:
        """Проверить, лежат ли параметры внутри интервалов."""
        values = _parameter_mapping(parameters)
        return all(float(lo) <= float(values[name]) <= float(hi) for name, (lo, hi) in self.as_mapping().items())


@dataclass(frozen=True, slots=True, repr=True)
class BoundaryParameterRateLimits:
    """Симметричные ограничения скоростей параметров аналитической формы."""

    R0: float
    Z0: float
    A0: float
    kappa: float
    delta: float

    def __post_init__(self) -> None:
        """Проверить, что все ограничения скоростей положительны."""
        for name, value in self.as_mapping().items():
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"rate limit {name} must be finite and > 0")

    def as_mapping(self) -> dict[str, float]:
        """Вернуть ограничения скоростей в стабильном порядке имен параметров."""
        return {
            "R0": self.R0,
            "Z0": self.Z0,
            "A0": self.A0,
            "kappa": self.kappa,
            "delta": self.delta,
        }


@dataclass(frozen=True, slots=True, repr=True)
class BoundaryParameterTrajectory:
    """Дискретная траектория параметров границы с равномерным шагом времени."""

    t: np.ndarray
    parameters: tuple[BoundaryParameters, ...]
    bounds: BoundaryParameterBounds
    rate_limits: BoundaryParameterRateLimits

    def __post_init__(self) -> None:
        """Проверить согласованность времени, параметров, границ и скоростей."""
        t_arr = np.asarray(self.t, dtype=float).reshape(-1)
        if t_arr.shape != (len(self.parameters),):
            raise ValueError("time and parameter trajectory lengths must match")
        if t_arr.size == 0:
            raise ValueError("trajectory must contain at least one timestep")
        if not np.all(np.isfinite(t_arr)):
            raise ValueError("trajectory time values must be finite")
        if t_arr.size > 1 and np.any(np.diff(t_arr) <= 0.0):
            raise ValueError("trajectory time values must be strictly increasing")
        validate_parameter_trajectory(self.parameters, t_arr, bounds=self.bounds, rate_limits=self.rate_limits)

    def parameter_matrix(self) -> np.ndarray:
        """Вернуть матрицу параметров формы `(n_steps, 5)`."""
        return np.stack([p.as_array() for p in self.parameters], axis=0)

    def at_time(self, time_s: float) -> BoundaryParameters:
        """Вернуть ближайшие параметры с удержанием на краях траектории."""
        t_arr = np.asarray(self.t, dtype=float).reshape(-1)
        idx = int(np.searchsorted(t_arr, float(time_s), side="right") - 1)
        idx = int(np.clip(idx, 0, len(self.parameters) - 1))
        return self.parameters[idx]


T15_REPLAY_ROBUST_BOUNDS = BoundaryParameterBounds(
    R0=(1.3266, 1.4756),
    Z0=(-0.0496, 0.0137),
    A0=(0.5200, 0.6641),
    kappa=(1.1212, 1.4966),
    delta=(0.1044, 0.3656),
)
"""Рекомендуемые первые границы параметров по 9 replay-прогонам T15."""

T15_REPLAY_SMOOTH_RATE_LIMITS = BoundaryParameterRateLimits(
    R0=0.30,
    Z0=0.70,
    A0=0.45,
    kappa=1.20,
    delta=0.80,
)
"""Рекомендуемые симметричные ограничения скоростей по сглаженным replay-оценкам."""


def validate_boundary_parameters(parameters: BoundaryParameters, *, bounds: BoundaryParameterBounds | None = None) -> None:
    """Проверить физическую и численную допустимость параметров границы."""
    values = _parameter_mapping(parameters)
    if not np.all(np.isfinite(np.array(list(values.values()), dtype=float))):
        raise ValueError("boundary parameters must be finite")
    if float(parameters.A0) <= 0.0:
        raise ValueError("A0 must be > 0")
    if float(parameters.kappa) <= 0.0:
        raise ValueError("kappa must be > 0")
    if bounds is not None and not bounds.contains(parameters):
        raise ValueError("boundary parameters are outside configured bounds")


def boundary_polyline_from_parameters(parameters: BoundaryParameters, *, theta_count: int = 256, close: bool = True) -> np.ndarray:
    """Построить полилинию границы по параметрам аналитической формы."""
    validate_boundary_parameters(parameters)
    if int(theta_count) < 8:
        raise ValueError("theta_count must be at least 8")
    theta = np.linspace(-np.pi, np.pi, int(theta_count), endpoint=False, dtype=float)
    poly = evaluate_parametric_boundary(theta, parameters)
    if close:
        poly = _as_closed_polyline(poly)
    return poly


def reference_radii_from_parameters(
    parameters: BoundaryParameters,
    *,
    center: tuple[float, float],
    angles: np.ndarray,
    theta_count: int = 512,
) -> np.ndarray:
    """Преобразовать аналитическую границу в reference radii для заданных лучей."""
    poly = boundary_polyline_from_parameters(parameters, theta_count=int(theta_count), close=True)
    return radii_from_polyline_ray_intersections(poly, center, np.asarray(angles, dtype=float))


def boundary_polyline_self_intersects(polyline: np.ndarray, *, atol: float = 1.0e-10) -> bool:
    """Проверить, имеет ли замкнутая полилиния самопересечения."""
    poly = _as_closed_polyline(_clean_polyline(polyline))
    starts = poly[:-1]
    ends = poly[1:]
    n_segments = starts.shape[0]
    for i in range(n_segments):
        for j in range(i + 1, n_segments):
            if _segments_are_adjacent(i, j, n_segments):
                continue
            if _segments_intersect(starts[i], ends[i], starts[j], ends[j], atol=float(atol)):
                return True
    return False


def boundary_polyline_inside_limiter(polyline: np.ndarray, limiter_shape: np.ndarray, *, atol: float = 1.0e-9) -> bool:
    """Проверить, что граница целиком находится внутри контура лимитера."""
    poly = _as_closed_polyline(_clean_polyline(polyline))
    limiter = _as_closed_polyline(_clean_polyline(limiter_shape))
    if not all(_point_in_closed_polygon(point, limiter, atol=float(atol)) for point in poly[:-1]):
        return False
    poly_starts = poly[:-1]
    poly_ends = poly[1:]
    limiter_starts = limiter[:-1]
    limiter_ends = limiter[1:]
    for a, b in zip(poly_starts, poly_ends, strict=True):
        for c, d in zip(limiter_starts, limiter_ends, strict=True):
            if _segments_intersect(a, b, c, d, atol=float(atol)):
                return False
    return True


def validate_reference_boundary(
    parameters: BoundaryParameters,
    *,
    bounds: BoundaryParameterBounds | None = None,
    limiter_shape: np.ndarray | None = None,
    theta_count: int = 256,
) -> np.ndarray:
    """Проверить аналитическую reference-границу и вернуть ее полилинию."""
    validate_boundary_parameters(parameters, bounds=bounds)
    poly = boundary_polyline_from_parameters(parameters, theta_count=int(theta_count), close=True)
    if boundary_polyline_self_intersects(poly):
        raise ValueError("reference boundary self-intersects")
    if limiter_shape is not None and not boundary_polyline_inside_limiter(poly, limiter_shape):
        raise ValueError("reference boundary is outside limiter")
    return poly


def generate_boundary_parameter_trajectory(
    *,
    step_count: int,
    t_step: float,
    seed: int,
    bounds: BoundaryParameterBounds = T15_REPLAY_ROBUST_BOUNDS,
    rate_limits: BoundaryParameterRateLimits = T15_REPLAY_SMOOTH_RATE_LIMITS,
    target_update_s: float = 0.20,
    initial_parameters: BoundaryParameters | None = None,
) -> BoundaryParameterTrajectory:
    """Сгенерировать детерминированную rate-limited траекторию параметров границы."""
    if int(step_count) <= 0:
        raise ValueError("step_count must be > 0")
    dt = float(t_step)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("t_step must be finite and > 0")
    if not np.isfinite(float(target_update_s)) or float(target_update_s) <= 0.0:
        raise ValueError("target_update_s must be finite and > 0")
    rng = np.random.default_rng(int(seed))
    current = _sample_parameters(rng, bounds) if initial_parameters is None else initial_parameters
    validate_boundary_parameters(current, bounds=bounds)
    target = _sample_parameters(rng, bounds)
    update_steps = max(1, int(round(float(target_update_s) / dt)))
    out: list[BoundaryParameters] = []
    for step_index in range(int(step_count)):
        if step_index % update_steps == 0:
            target = _sample_parameters(rng, bounds)
        out.append(current)
        current = _step_toward_target(current, target, dt=dt, bounds=bounds, rate_limits=rate_limits)
    t = np.arange(int(step_count), dtype=float) * dt
    return BoundaryParameterTrajectory(t=t, parameters=tuple(out), bounds=bounds, rate_limits=rate_limits)


def validate_parameter_trajectory(
    parameters: tuple[BoundaryParameters, ...] | list[BoundaryParameters],
    t: np.ndarray,
    *,
    bounds: BoundaryParameterBounds,
    rate_limits: BoundaryParameterRateLimits,
    rate_atol: float = 1.0e-12,
) -> None:
    """Проверить траекторию параметров по границам значений и скоростей."""
    if not parameters:
        raise ValueError("parameter trajectory must be non-empty")
    t_arr = np.asarray(t, dtype=float).reshape(-1)
    if t_arr.shape != (len(parameters),):
        raise ValueError("time and parameter trajectory lengths must match")
    for params in parameters:
        validate_boundary_parameters(params, bounds=bounds)
    if len(parameters) <= 1:
        return
    matrix = np.stack([p.as_array() for p in parameters], axis=0)
    dt = np.diff(t_arr)
    if np.any(dt <= 0.0) or not np.all(np.isfinite(dt)):
        raise ValueError("time values must be strictly increasing")
    rates = np.diff(matrix, axis=0) / dt[:, None]
    limits = np.array(list(rate_limits.as_mapping().values()), dtype=float)
    if np.any(np.abs(rates) > limits[None, :] + float(rate_atol)):
        raise ValueError("parameter trajectory violates rate limits")


def _parameter_mapping(parameters: BoundaryParameters) -> dict[str, float]:
    """Вернуть параметры в словаре со стабильными именами."""
    return {
        "R0": float(parameters.R0),
        "Z0": float(parameters.Z0),
        "A0": float(parameters.A0),
        "kappa": float(parameters.kappa),
        "delta": float(parameters.delta),
    }


def _validate_interval(name: str, interval: tuple[float, float]) -> None:
    """Проверить один численный интервал."""
    if len(tuple(interval)) != 2:
        raise ValueError(f"bounds interval {name} must contain two values")
    lo, hi = (float(interval[0]), float(interval[1]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"bounds interval {name} must satisfy finite hi > lo")


def _sample_parameters(rng: np.random.Generator, bounds: BoundaryParameterBounds) -> BoundaryParameters:
    """Случайно выбрать параметры внутри заданных интервалов."""
    values = {name: float(rng.uniform(lo, hi)) for name, (lo, hi) in bounds.as_mapping().items()}
    return BoundaryParameters(**values)


def _step_toward_target(
    current: BoundaryParameters,
    target: BoundaryParameters,
    *,
    dt: float,
    bounds: BoundaryParameterBounds,
    rate_limits: BoundaryParameterRateLimits,
) -> BoundaryParameters:
    """Сделать один ограниченный по скорости шаг к целевым параметрам."""
    current_values = _parameter_mapping(current)
    target_values = _parameter_mapping(target)
    next_values: dict[str, float] = {}
    for name, limit in rate_limits.as_mapping().items():
        lo, hi = bounds.as_mapping()[name]
        max_delta = float(limit) * float(dt)
        delta = float(np.clip(target_values[name] - current_values[name], -max_delta, max_delta))
        next_values[name] = float(np.clip(current_values[name] + delta, lo, hi))
    return BoundaryParameters(**next_values)


def _segments_are_adjacent(i: int, j: int, n_segments: int) -> bool:
    """Проверить, соседние ли сегменты замкнутого контура."""
    return bool(abs(int(i) - int(j)) <= 1 or (int(i) == 0 and int(j) == int(n_segments) - 1))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, *, atol: float) -> bool:
    """Проверить пересечение двух отрезков в плоскости."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    c_arr = np.asarray(c, dtype=float)
    d_arr = np.asarray(d, dtype=float)
    o1 = _orientation(a_arr, b_arr, c_arr)
    o2 = _orientation(a_arr, b_arr, d_arr)
    o3 = _orientation(c_arr, d_arr, a_arr)
    o4 = _orientation(c_arr, d_arr, b_arr)
    if (o1 * o2 < -float(atol)) and (o3 * o4 < -float(atol)):
        return True
    return bool(
        (abs(o1) <= float(atol) and _point_on_segment(c_arr, a_arr, b_arr, atol=float(atol)))
        or (abs(o2) <= float(atol) and _point_on_segment(d_arr, a_arr, b_arr, atol=float(atol)))
        or (abs(o3) <= float(atol) and _point_on_segment(a_arr, c_arr, d_arr, atol=float(atol)))
        or (abs(o4) <= float(atol) and _point_on_segment(b_arr, c_arr, d_arr, atol=float(atol)))
    )


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Вернуть ориентированную площадь треугольника ABC с точностью до множителя."""
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _point_on_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray, *, atol: float) -> bool:
    """Проверить, лежит ли точка на отрезке с учетом допуска."""
    return bool(
        min(float(a[0]), float(b[0])) - float(atol) <= float(point[0]) <= max(float(a[0]), float(b[0])) + float(atol)
        and min(float(a[1]), float(b[1])) - float(atol) <= float(point[1]) <= max(float(a[1]), float(b[1])) + float(atol)
    )


def _point_in_closed_polygon(point: np.ndarray, polygon: np.ndarray, *, atol: float) -> bool:
    """Проверить принадлежность точки замкнутому полигону с учетом границы."""
    p = np.asarray(point, dtype=float)
    poly = _as_closed_polyline(polygon)
    for a, b in zip(poly[:-1], poly[1:], strict=True):
        if abs(_orientation(a, b, p)) <= float(atol) and _point_on_segment(p, a, b, atol=float(atol)):
            return True
    inside = False
    x = float(p[0])
    y = float(p[1])
    for a, b in zip(poly[:-1], poly[1:], strict=True):
        yi = float(a[1])
        yj = float(b[1])
        xi = float(a[0])
        xj = float(b[0])
        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_cross + float(atol):
                inside = not inside
    return inside

