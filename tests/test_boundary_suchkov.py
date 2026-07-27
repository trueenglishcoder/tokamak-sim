"""Проверки сплайнового восстановления границы по методу Сучкова."""

from __future__ import annotations

from pathlib import Path
import tomllib

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.geometry.boundary import find_plasma_boundary_with_status
from tokamak_control.geometry.boundary_gpu import (
    _sample_points,
    _suchkov_fixed_angle_search,
)
from tokamak_control.geometry.boundary_suchkov import (
    build_suchkov_spline_plan,
    build_suchkov_spline_torch_plan,
    uniform_periodic_angles,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _test_grid() -> Grid2D:
    """Создать квадратную тестовую сетку."""
    return Grid2D(
        r=Grid1D(start=0.0, step=1.0, size=40, center=19.5),
        z=Grid1D(start=0.0, step=1.0, size=40, center=19.5),
    )


def _quadratic_psi(grid: Grid2D) -> np.ndarray:
    """Создать поле с круговыми замкнутыми линиями уровня."""
    R, Z = grid.mesh()
    return (R - 19.5) ** 2 + (Z - 19.5) ** 2


def _square_limiter() -> np.ndarray:
    """Создать квадратный лимитер вокруг тестовой плазмы."""
    return np.asarray(
        [
            [5.0, 5.0],
            [34.0, 5.0],
            [34.0, 34.0],
            [5.0, 34.0],
            [5.0, 5.0],
        ],
        dtype=np.float64,
    )


def test_periodic_interpolation_matrix_matches_scipy() -> None:
    """Матрица интерполяции должна совпадать с эталонным CubicSpline."""
    control_angles = uniform_periodic_angles(16)
    output_angles = uniform_periodic_angles(64)
    values = 1.0 + 0.2 * np.cos(control_angles) + 0.1 * np.sin(2.0 * control_angles)
    plan = build_suchkov_spline_plan(output_angles, control_count=16)
    actual = plan.interpolation_matrix @ values

    scipy_angles = np.concatenate([control_angles, np.asarray([np.pi])])
    scipy_values = np.concatenate([values, values[:1]])
    expected = CubicSpline(scipy_angles, scipy_values, bc_type="periodic")(output_angles)

    assert np.allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_cpu_suchkov_boundary_returns_closed_spline() -> None:
    """CPU-режим должен вернуть замкнутую гладкую границу внутри лимитера."""
    grid = _test_grid()
    poly, level, status = find_plasma_boundary_with_status(
        _quadratic_psi(grid),
        grid,
        (19.5, 19.5),
        limiter_shape=_square_limiter(),
        boundary_mode="suchkov_spline_contour",
    )

    assert status == "suchkov_spline_contour_success"
    assert level == pytest.approx(210.31, abs=1.0)
    assert poly.shape == (129, 2)
    assert np.allclose(poly[0], poly[-1], rtol=0.0, atol=1.0e-12)
    assert float(np.min(poly[:, 0])) >= 5.0 - 1.0e-9
    assert float(np.max(poly[:, 0])) <= 34.0 + 1.0e-9
    assert float(np.min(poly[:, 1])) >= 5.0 - 1.0e-9
    assert float(np.max(poly[:, 1])) <= 34.0 + 1.0e-9


def test_torch_suchkov_search_is_batched_and_finite() -> None:
    """Тензорный поиск должен работать без Python-цикла по элементам batch."""
    torch = pytest.importorskip("torch")
    grid = _test_grid()
    psi_numpy = _quadratic_psi(grid)
    psi = torch.as_tensor(
        np.stack([psi_numpy, psi_numpy], axis=0),
        dtype=torch.float64,
    )
    centers = torch.tensor([[19.5, 19.5], [19.5, 19.5]], dtype=torch.float64)
    center_level = _sample_points(psi, grid, centers[:, None, :]).reshape(2)
    measurement_angles = torch.as_tensor(
        uniform_periodic_angles(32),
        dtype=torch.float64,
    )
    limiter = torch.as_tensor(_square_limiter(), dtype=torch.float64)
    dense_angles = torch.as_tensor(uniform_periodic_angles(64), dtype=torch.float64)
    plan = build_suchkov_spline_torch_plan(dense_angles, control_count=16)

    points, radii, found, levels = _suchkov_fixed_angle_search(
        psi=psi,
        grid=grid,
        center_points=centers,
        center_level=center_level,
        measurement_angles=measurement_angles,
        limiter=limiter,
        ray_samples=256,
        plan=plan,
    )

    assert tuple(points.shape) == (2, 32, 2)
    assert tuple(radii.shape) == (2, 32)
    assert bool(torch.all(found))
    assert bool(torch.all(torch.isfinite(radii)))
    assert bool(torch.all(torch.isfinite(levels)))
    assert float(torch.max(torch.abs(radii - 14.5))) < 1.0e-2


def test_t15_config_uses_new_suchkov_mode() -> None:
    """Основная конфигурация T15 должна включать только новую реализацию."""
    with (REPO_ROOT / "configs/T15MD.toml").open("rb") as handle:
        config = tomllib.load(handle)

    assert config["boundary"]["mode"] == "suchkov_spline_contour"
    assert config["boundary"]["legacy_precision_index2"] == pytest.approx(1.0e-6)
