"""Сплайновая параметризация замкнутой границы по Е. П. Сучкову."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SUCHKOV_CONTROL_COUNT = 32
SUCHKOV_VALIDATION_COUNT = 256
SUCHKOV_SEARCH_ITERATIONS = 18


@dataclass(frozen=True, slots=True, repr=True)
class SuchkovSplinePlan:
    """Линейный план интерполяции замкнутой кривой кубическим сплайном."""

    control_angles: np.ndarray
    output_angles: np.ndarray
    interpolation_matrix: np.ndarray


def uniform_periodic_angles(count: int) -> np.ndarray:
    """Вернуть равномерные углы на периоде ``[-pi, pi)``."""
    n = int(count)
    if n < 4:
        raise ValueError("count must be >= 4")
    return np.linspace(-np.pi, np.pi, n, endpoint=False, dtype=np.float64)


def build_suchkov_spline_plan(
    output_angles: np.ndarray,
    *,
    control_count: int = 16,
) -> SuchkovSplinePlan:
    """Построить матрицу периодической кубической интерполяции ``R(xi), Z(xi)``.

    Метод использует равномерные узлы параметра ``xi`` и периодические условия
    гладкости. Интерполяция линейна по значениям координат в контрольных узлах,
    поэтому одна и та же матрица применяется к ``R`` и ``Z``.
    """
    output = _normalize_angles(np.asarray(output_angles, dtype=np.float64).reshape(-1))
    if output.size < 4:
        raise ValueError("output_angles must contain at least four values")
    controls = uniform_periodic_angles(int(control_count))
    matrix = _periodic_cubic_interpolation_matrix(controls, output)
    return SuchkovSplinePlan(
        control_angles=controls,
        output_angles=output,
        interpolation_matrix=matrix,
    )


def interpolate_closed_curve_numpy(
    control_points: np.ndarray,
    plan: SuchkovSplinePlan,
) -> np.ndarray:
    """Интерполировать замкнутую кривую по контрольным точкам ``(R, Z)``."""
    points = np.asarray(control_points, dtype=np.float64)
    expected = (plan.control_angles.size, 2)
    if points.shape != expected:
        raise ValueError(f"control_points must have shape {expected}, got {points.shape}")
    return np.asarray(plan.interpolation_matrix @ points, dtype=np.float64)


def interpolate_closed_curve_torch(control_points: object, interpolation_matrix: object) -> object:
    """Интерполировать batch замкнутых кривых на устройстве PyTorch."""
    torch = __import__("torch")
    points = torch.as_tensor(control_points)
    matrix = torch.as_tensor(interpolation_matrix, dtype=points.dtype, device=points.device)
    if points.ndim != 3 or int(points.shape[2]) != 2:
        raise ValueError(f"control_points must have shape (B, K, 2), got {tuple(points.shape)}")
    if matrix.ndim != 2 or int(matrix.shape[1]) != int(points.shape[1]):
        raise ValueError(
            "interpolation_matrix must have shape (A, K) compatible with control_points"
        )
    return torch.einsum("ak,bkd->bad", matrix, points)


def _normalize_angles(angles: np.ndarray) -> np.ndarray:
    """Нормализовать углы к полуинтервалу ``[-pi, pi)``."""
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def _periodic_cubic_interpolation_matrix(
    control_angles: np.ndarray,
    output_angles: np.ndarray,
) -> np.ndarray:
    """Построить матрицу интерполяции равномерного периодического кубического сплайна."""
    controls = np.asarray(control_angles, dtype=np.float64).reshape(-1)
    outputs = _normalize_angles(np.asarray(output_angles, dtype=np.float64).reshape(-1))
    n = int(controls.size)
    if n < 4:
        raise ValueError("control_angles must contain at least four values")
    if not np.all(np.isfinite(controls)) or not np.all(np.isfinite(outputs)):
        raise ValueError("angles must contain only finite values")

    expected = uniform_periodic_angles(n)
    if not np.allclose(controls, expected, rtol=0.0, atol=1.0e-12):
        raise ValueError("control_angles must be uniform on [-pi, pi)")

    h = 2.0 * np.pi / float(n)
    cyclic = np.zeros((n, n), dtype=np.float64)
    rhs_operator = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        cyclic[i, (i - 1) % n] = 1.0
        cyclic[i, i] = 4.0
        cyclic[i, (i + 1) % n] = 1.0
        rhs_operator[i, (i - 1) % n] = 6.0 / (h * h)
        rhs_operator[i, i] = -12.0 / (h * h)
        rhs_operator[i, (i + 1) % n] = 6.0 / (h * h)
    second_derivative_operator = np.linalg.solve(cyclic, rhs_operator)

    matrix = np.zeros((outputs.size, n), dtype=np.float64)
    phase = (outputs + np.pi) / h
    left_indices = np.floor(phase).astype(np.int64) % n
    local = phase - np.floor(phase)
    for row, (left, beta) in enumerate(zip(left_indices, local, strict=True)):
        right = (int(left) + 1) % n
        alpha = 1.0 - float(beta)
        matrix[row, int(left)] += alpha
        matrix[row, right] += float(beta)
        matrix[row] += ((alpha**3 - alpha) * second_derivative_operator[int(left)] + (float(beta) ** 3 - float(beta)) * second_derivative_operator[right]) * (h * h / 6.0)
    return matrix


@dataclass(frozen=True, slots=True, repr=True)
class SuchkovSplineTorchPlan:
    """Тензорный план интерполяции замкнутой кривой на GPU."""

    control_angles: object
    output_angles: object
    interpolation_matrix: object


def build_suchkov_spline_torch_plan(
    output_angles: object,
    *,
    control_count: int = 16,
) -> SuchkovSplineTorchPlan:
    """Построить тензорный план на устройстве входных углов."""
    torch = __import__("torch")
    output = torch.as_tensor(output_angles)
    if output.ndim != 1:
        output = output.reshape(-1)
    numpy_plan = build_suchkov_spline_plan(
        output.detach().cpu().numpy(),
        control_count=int(control_count),
    )
    return SuchkovSplineTorchPlan(
        control_angles=torch.as_tensor(
            numpy_plan.control_angles,
            dtype=output.dtype,
            device=output.device,
        ),
        output_angles=output,
        interpolation_matrix=torch.as_tensor(
            numpy_plan.interpolation_matrix,
            dtype=output.dtype,
            device=output.device,
        ),
    )
