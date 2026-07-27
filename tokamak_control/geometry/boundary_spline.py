"""Периодический кубический сплайн для восстановления границы плазмы."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, repr=True)
class PeriodicSplineCoefficients:
    """Коэффициенты периодического кубического сплайна."""

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    n_segments: int

    def as_array(self) -> np.ndarray:
        """Вернуть коэффициенты в виде массива (4, n_segments)."""
        return np.stack([self.a, self.b, self.c, self.d], axis=0)


def fit_periodic_cubic_spline(angles_rad: np.ndarray, radii: np.ndarray) -> PeriodicSplineCoefficients:
    """Построить периодический кубический сплайн по углам и радиусам.

    Решает трёхдиагональную систему для вторых производных с периодическими
    граничными условиями через алгоритм Томаса с модификацией Шермана-Моррисона.

    :param angles_rad: массив углов (N,), равномерный от -π до π
    :param radii: массив радиусов (N,)
    :return: коэффициенты сплайна (a, b, c, d для каждого сегмента)
    """
    angles = np.asarray(angles_rad, dtype=float).reshape(-1)
    radii_arr = np.asarray(radii, dtype=float).reshape(-1)
    if angles.shape != radii_arr.shape:
        raise ValueError(f"angles and radi must have same shape, got {angles.shape} and {radii_arr.shape}")
    n = int(angles.shape[0])
    if n < 3:
        raise ValueError(f"need at least 3 points for periodic spline, got {n}")

    h = float(angles[1] - angles[0])
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError(f"angles must be uniformly spaced with positive step, got h={h}")

    for i in range(1, n):
        step = float(angles[i] - angles[i - 1])
        if abs(step - h) > 1.0e-9:
            raise ValueError(f"angles must be uniformly spaced, expected step {h}, got {step} at index {i}")

    a = radii_arr.copy()
    c = _solve_cyclic_tridiagonal_for_spline(radii_arr, h)
    b = np.zeros(n, dtype=float)
    d = np.zeros(n, dtype=float)

    for i in range(n):
        i_next = (i + 1) % n
        b[i] = (radii_arr[i_next] - radii_arr[i]) / h - h * (2.0 * c[i] + c[i_next]) / 3.0
        d[i] = (c[i_next] - c[i]) / (3.0 * h)

    return PeriodicSplineCoefficients(a=a, b=b, c=c, d=d, n_segments=n)


def _solve_cyclic_tridiagonal_for_spline(radii: np.ndarray, h: float) -> np.ndarray:
    """Решить циклическую трёхдиагональную систему для вторых производных сплайна.

    Система: h*c_{i-1} + 4*h*c_i + h*c_{i+1} = 3*(y_{i+1} - y_i)/h - 3*(y_i - y_{i-1})/h
    С периодическими условиями: c_0 = c_N, c_{-1} = c_{N-1}

    Используется модификация Шермана-Моррисона для циклических систем.
    """
    n = int(radii.shape[0])
    rhs = np.zeros(n, dtype=float)
    for i in range(n):
        i_prev = (i - 1) % n
        i_next = (i + 1) % n
        rhs[i] = 3.0 * (radii[i_next] - radii[i]) / h - 3.0 * (radii[i] - radii[i_prev]) / h

    lower = np.full(n, h, dtype=float)
    diag = np.full(n, 4.0 * h, dtype=float)
    upper = np.full(n, h, dtype=float)

    c = _solve_cyclic_tridiagonal(lower, diag, upper, rhs)
    return c


def _solve_cyclic_tridiagonal(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Решить циклическую трёхдиагональную систему через Шермана-Моррисона.

    Модифицирует систему, чтобы устранить цикличность, решает обычную
    трёхдиагональную систему, затем применяет поправку.
    """
    n = int(diag.shape[0])
    if n < 3:
        raise ValueError(f"cyclic tridiagonal requires n >= 3, got {n}")

    gamma = -diag[0]
    lower_mod = lower.copy()
    diag_mod = diag.copy()
    upper_mod = upper.copy()
    rhs_mod = rhs.copy()

    diag_mod[0] = diag[0] - gamma
    diag_mod[n - 1] = diag[n - 1] - upper[n - 1] * lower[0] / gamma

    lower_mod[0] = 0.0
    upper_mod[n - 1] = 0.0

    u = np.zeros(n, dtype=float)
    u[0] = gamma
    u[n - 1] = upper[n - 1]

    x = _solve_tridiagonal(lower_mod, diag_mod, upper_mod, rhs_mod)
    z = _solve_tridiagonal(lower_mod, diag_mod, upper_mod, u)

    v_factor = x[0] + x[n - 1] * lower[0] / gamma
    z_factor = z[0] + z[n - 1] * lower[0] / gamma

    if abs(z_factor) < 1.0e-15:
        return x

    correction = v_factor / z_factor
    return x - correction * z


def _solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Решить трёхдиагональную систему через алгоритм Томаса."""
    n = int(diag.shape[0])
    c_prime = np.zeros(n, dtype=float)
    d_prime = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)

    if abs(diag[0]) < 1.0e-15:
        raise ValueError("zero diagonal element in tridiagonal system")
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, n):
        denom = diag[i] - lower[i] * c_prime[i - 1]
        if abs(denom) < 1.0e-15:
            raise ValueError("zero pivot in tridiagonal system")
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom

    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


def evaluate_spline(coefficients: PeriodicSplineCoefficients, angles_rad: np.ndarray, query_angles: np.ndarray) -> np.ndarray:
    """Вычислить значения сплайна в заданных точках.

    :param coefficients: коэффициенты сплайна
    :param angles_rad: исходные узловые углы (N,)
    :param query_angles: углы для вычисления (M,)
    :return: значения сплайна (M,)
    """
    angles = np.asarray(angles_rad, dtype=float).reshape(-1)
    queries = np.asarray(query_angles, dtype=float).reshape(-1)
    n = int(coefficients.n_segments)
    h = float(angles[1] - angles[0])

    result = np.zeros(queries.shape[0], dtype=float)
    for k, q in enumerate(queries):
        q_wrapped = _wrap_angle_to_range(q, float(angles[0]), float(angles[-1]) + h)
        segment = int((q_wrapped - float(angles[0])) / h)
        segment = min(max(segment, 0), n - 1)
        dx = q_wrapped - float(angles[segment])
        result[k] = (
            coefficients.a[segment]
            + coefficients.b[segment] * dx
            + coefficients.c[segment] * dx * dx
            + coefficients.d[segment] * dx * dx * dx
        )
    return result


def _wrap_angle_to_range(angle: float, start: float, end: float) -> float:
    """Привести угол к диапазону [start, end) с учётом периодичности 2π."""
    span = end - start
    if span <= 0.0:
        return angle
    shifted = angle - start
    wrapped = shifted - span * np.floor(shifted / span)
    return wrapped + start


def spline_max_width(coefficients: PeriodicSplineCoefficients, angles_rad: np.ndarray, r_axis: float) -> float:
    """Вычислить ширину границы на экваториальной плоскости.

    Ширина = r(θ=0) + r(θ=π), где r — радиус от магнитной оси.

    :param coefficients: коэффициенты сплайна
    :param angles_rad: исходные узловые углы (N,)
    :param r_axis: радиус магнитной оси (не используется, радиусы уже от оси)
    :return: ширина в метрах
    """
    r_0 = float(evaluate_spline(coefficients, angles_rad, np.array([0.0]))[0])
    r_pi = float(evaluate_spline(coefficients, angles_rad, np.array([np.pi]))[0])
    return r_0 + r_pi


def fit_periodic_cubic_spline_torch(angles_rad, radii):
    """Построить периодический кубический сплайн для torch тензоров (batched).

    :param angles_rad: тензор углов (N,) или (B, N)
    :param radii: тензор радиусов (N,) или (B, N)
    :return: тензор коэффициентов (4, N) или (B, 4, N)
    """
    import torch

    if not torch.is_tensor(angles_rad):
        angles_rad = torch.as_tensor(angles_rad)
    if not torch.is_tensor(radii):
        radii = torch.as_tensor(radii)

    if angles_rad.ndim == 1:
        return _fit_periodic_spline_torch_single(angles_rad, radii)

    batch_size = int(angles_rad.shape[0])
    n = int(angles_rad.shape[1])
    coeffs = torch.zeros(batch_size, 4, n, dtype=radii.dtype, device=radii.device)
    for b in range(batch_size):
        coeffs[b] = _fit_periodic_spline_torch_single(angles_rad[b], radii[b])
    return coeffs


def _fit_periodic_spline_torch_single(angles_rad, radii):
    """Построить периодический кубический сплайн для одного набора данных (torch)."""
    import torch

    n = int(angles_rad.shape[0])
    h = float(angles_rad[1] - angles_rad[0])

    a = radii.clone()
    c = _solve_cyclic_tridiagonal_torch(radii, h)
    b = torch.zeros(n, dtype=radii.dtype, device=radii.device)
    d = torch.zeros(n, dtype=radii.dtype, device=radii.device)

    for i in range(n):
        i_next = (i + 1) % n
        b[i] = (radii[i_next] - radii[i]) / h - h * (2.0 * c[i] + c[i_next]) / 3.0
        d[i] = (c[i_next] - c[i]) / (3.0 * h)

    return torch.stack([a, b, c, d], dim=0)


def _solve_cyclic_tridiagonal_torch(radii, h: float):
    """Решить циклическую трёхдиагональную систему для torch тензора."""
    import torch

    n = int(radii.shape[0])
    rhs = torch.zeros(n, dtype=radii.dtype, device=radii.device)
    for i in range(n):
        i_prev = (i - 1) % n
        i_next = (i + 1) % n
        rhs[i] = 3.0 * (radii[i_next] - radii[i]) / h - 3.0 * (radii[i] - radii[i_prev]) / h

    lower = torch.full((n,), h, dtype=radii.dtype, device=radii.device)
    diag = torch.full((n,), 4.0 * h, dtype=radii.dtype, device=radii.device)
    upper = torch.full((n,), h, dtype=radii.dtype, device=radii.device)

    return _solve_cyclic_tridiagonal_system_torch(lower, diag, upper, rhs)


def _solve_cyclic_tridiagonal_system_torch(lower, diag, upper, rhs):
    """Решить циклическую трёхдиагональную систему через Шермана-Моррисона (torch)."""
    import torch

    n = int(diag.shape[0])
    gamma = -diag[0]

    lower_mod = lower.clone()
    diag_mod = diag.clone()
    upper_mod = upper.clone()
    rhs_mod = rhs.clone()

    diag_mod[0] = diag[0] - gamma
    diag_mod[n - 1] = diag[n - 1] - upper[n - 1] * lower[0] / gamma

    lower_mod[0] = 0.0
    upper_mod[n - 1] = 0.0

    u = torch.zeros(n, dtype=diag.dtype, device=diag.device)
    u[0] = gamma
    u[n - 1] = upper[n - 1]

    x = _solve_tridiagonal_torch(lower_mod, diag_mod, upper_mod, rhs_mod)
    z = _solve_tridiagonal_torch(lower_mod, diag_mod, upper_mod, u)

    v_factor = x[0] + x[n - 1] * lower[0] / gamma
    z_factor = z[0] + z[n - 1] * lower[0] / gamma

    if abs(float(z_factor)) < 1.0e-15:
        return x

    correction = v_factor / z_factor
    return x - correction * z


def _solve_tridiagonal_torch(lower, diag, upper, rhs):
    """Решить трёхдиагональную систему через алгоритм Томаса (torch)."""
    import torch

    n = int(diag.shape[0])
    c_prime = torch.zeros(n, dtype=diag.dtype, device=diag.device)
    d_prime = torch.zeros(n, dtype=diag.dtype, device=diag.device)
    x = torch.zeros(n, dtype=diag.dtype, device=diag.device)

    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, n):
        denom = diag[i] - lower[i] * c_prime[i - 1]
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom

    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


def evaluate_spline_torch(coefficients, angles_rad, query_angles):
    """Вычислить значения сплайна в заданных точках (torch).

    :param coefficients: тензор коэффициентов (4, N) или (B, 4, N)
    :param angles_rad: тензор узловых углов (N,) или (B, N)
    :param query_angles: тензор запросов (M,) или (B, M)
    :return: тензор значений (M,) или (B, M)
    """
    import torch

    if not torch.is_tensor(coefficients):
        coefficients = torch.as_tensor(coefficients)
    if not torch.is_tensor(angles_rad):
        angles_rad = torch.as_tensor(angles_rad)
    if not torch.is_tensor(query_angles):
        query_angles = torch.as_tensor(query_angles)

    if coefficients.ndim == 2:
        return _evaluate_spline_torch_single(coefficients, angles_rad, query_angles)

    batch_size = int(coefficients.shape[0])
    m = int(query_angles.shape[1])
    result = torch.zeros(batch_size, m, dtype=coefficients.dtype, device=coefficients.device)
    for b in range(batch_size):
        result[b] = _evaluate_spline_torch_single(coefficients[b], angles_rad[b], query_angles[b])
    return result


def _evaluate_spline_torch_single(coefficients, angles_rad, query_angles):
    """Вычислить значения сплайна для одного набора данных (torch)."""
    import torch

    n = int(coefficients.shape[1])
    h = float(angles_rad[1] - angles_rad[0])
    m = int(query_angles.shape[0])

    result = torch.zeros(m, dtype=coefficients.dtype, device=coefficients.device)
    for k in range(m):
        q = float(query_angles[k])
        q_wrapped = _wrap_angle_to_range(q, float(angles_rad[0]), float(angles_rad[-1]) + h)
        segment = int((q_wrapped - float(angles_rad[0])) / h)
        segment = min(max(segment, 0), n - 1)
        dx = q_wrapped - float(angles_rad[segment])
        result[k] = (
            coefficients[0, segment]
            + coefficients[1, segment] * dx
            + coefficients[2, segment] * dx * dx
            + coefficients[3, segment] * dx * dx * dx
        )
    return result


def spline_max_width_torch(coefficients, angles_rad, r_axis: float = 0.0) -> float:
    """Вычислить ширину границы на экваториальной плоскости (torch).

    :param coefficients: тензор коэффициентов (4, N)
    :param angles_rad: тензор узловых углов (N,)
    :param r_axis: радиус магнитной оси (не используется)
    :return: ширина в метрах
    """
    import torch

    r_0 = float(evaluate_spline_torch(coefficients, angles_rad, torch.tensor([0.0], device=coefficients.device))[0])
    r_pi = float(evaluate_spline_torch(coefficients, angles_rad, torch.tensor([np.pi], device=coefficients.device))[0])
    return r_0 + r_pi
