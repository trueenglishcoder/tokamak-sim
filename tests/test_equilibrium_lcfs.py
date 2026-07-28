"""Acceptance tests for physical LCFS extraction from a known equilibrium."""

from __future__ import annotations

from pathlib import Path
import tomllib

import numpy as np
import pytest

from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.geometry.boundary_common import normalize_boundary_mode
from tokamak_control.geometry.boundary_projection import project_boundary_to_fixed_angles
from tokamak_control.geometry.lcfs import find_equilibrium_lcfs


REPO_ROOT = Path(__file__).resolve().parents[1]


def _grid(*, n: int = 241, low: float = -2.5, high: float = 2.5, center: tuple[float, float] = (0.0, 0.0)) -> Grid2D:
    step = (high - low) / float(n - 1)
    return Grid2D(
        r=Grid1D(start=low, step=step, size=n, center=center[0]),
        z=Grid1D(start=low, step=step, size=n, center=center[1]),
    )


def _square_limiter(half_width: float = 2.2) -> np.ndarray:
    a = float(half_width)
    return np.asarray([[-a, -a], [a, -a], [a, a], [-a, a], [-a, -a]], dtype=float)


def _angles(count: int = 32) -> np.ndarray:
    return np.linspace(-np.pi, np.pi, count, endpoint=False, dtype=float)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_limited_circle_for_both_flux_signs(sign: float) -> None:
    grid = _grid()
    R, Z = grid.mesh()
    psi = sign * (R * R + Z * Z)
    result = find_equilibrium_lcfs(
        psi,
        grid,
        center_hint=(0.0, 0.0),
        limiter_shape=_square_limiter(),
        fixed_angles=_angles(),
    )

    assert result.topology == "limited"
    assert len(result.limiter_contacts) == 4
    assert len(result.x_points) == 0
    assert result.fixed_angle_projection.valid
    assert np.allclose(result.fixed_angle_projection.radii, 2.2, atol=3.0e-3)
    assert result.quality.normalized_flux_residual < 2.0e-4
    assert result.quality.limiter_violation_count == 0


def test_single_null_preserves_explicit_saddle() -> None:
    grid = _grid(center=(0.0, 1.0))
    R, Z = grid.mesh()
    psi = R * R + (Z * Z - 1.0) ** 2
    result = find_equilibrium_lcfs(
        psi,
        grid,
        center_hint=(0.0, 1.0),
        limiter_shape=_square_limiter(),
        fixed_angles=_angles(),
    )

    assert result.topology == "single_null"
    assert len(result.x_points) == 1
    assert np.allclose(result.x_points[0].point, (0.0, 0.0), atol=2.0e-4)
    assert result.x_points[0].determinant < 0.0
    assert len(result.separatrix_branches) >= 1
    assert np.min(np.linalg.norm(result.core_boundary - np.asarray(result.x_points[0].point), axis=1)) < 1.0e-8
    assert result.quality.normalized_flux_residual < 2.0e-4


def test_double_null_preserves_both_saddles() -> None:
    grid = _grid()
    R, Z = grid.mesh()
    psi = R * R + Z * Z - 0.5 * Z**4
    result = find_equilibrium_lcfs(
        psi,
        grid,
        center_hint=(0.0, 0.0),
        limiter_shape=_square_limiter(),
        fixed_angles=_angles(),
    )

    assert result.topology == "double_null"
    assert len(result.x_points) == 2
    points = np.asarray(sorted((point.point for point in result.x_points), key=lambda point: point[1]))
    assert np.allclose(points, np.asarray([[0.0, -1.0], [0.0, 1.0]]), atol=2.0e-4)
    assert all(point.determinant < 0.0 for point in result.x_points)
    assert len(result.separatrix_branches) >= 2
    assert result.quality.normalized_flux_residual < 2.0e-4


def test_fixed_angle_projection_reports_non_star_shaped_boundary() -> None:
    boundary = np.asarray(
        [
            [-2.0, -2.0],
            [2.0, -2.0],
            [2.0, 2.0],
            [0.5, 2.0],
            [0.5, 0.2],
            [-0.5, 0.2],
            [-0.5, 2.0],
            [-2.0, 2.0],
            [-2.0, -2.0],
        ],
        dtype=float,
    )
    projection = project_boundary_to_fixed_angles(boundary, (0.0, 0.0), _angles(64))

    assert not projection.valid
    assert np.any(projection.intersection_counts != 1)
    assert projection.reason is not None
    assert "multi_hit_rays=" in projection.reason or "missing_rays=" in projection.reason


def test_old_suchkov_mode_is_rejected_with_migration_message() -> None:
    with pytest.raises(ValueError, match="equilibrium_lcfs"):
        normalize_boundary_mode("suchkov_spline_contour")


def test_t15_config_uses_equilibrium_lcfs() -> None:
    with (REPO_ROOT / "configs" / "T15MD.toml").open("rb") as stream:
        config = tomllib.load(stream)
    assert config["boundary"]["mode"] == "equilibrium_lcfs"


def test_shifted_magnetic_axis_is_refined_subgrid() -> None:
    center = (0.3, -0.2)
    grid = _grid(center=center)
    R, Z = grid.mesh()
    psi = (R - center[0]) ** 2 + (Z - center[1]) ** 2
    limiter = np.asarray([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0]], dtype=float)

    result = find_equilibrium_lcfs(
        psi,
        grid,
        center_hint=center,
        limiter_shape=limiter,
        fixed_angles=_angles(64),
    )

    assert result.topology == "limited"
    assert np.allclose(result.magnetic_axis.point, center, atol=1.0e-8)
    assert result.fixed_angle_projection.valid
    assert np.allclose(result.fixed_angle_projection.radii, 1.7, atol=1.0e-3)


def test_concave_limiter_contact_is_selected_continuously() -> None:
    grid = _grid()
    R, Z = grid.mesh()
    psi = R * R + Z * Z
    limiter = np.asarray(
        [
            [-2.0, -2.0],
            [2.0, -2.0],
            [2.0, -0.5],
            [1.0, -0.5],
            [1.0, 0.5],
            [2.0, 0.5],
            [2.0, 2.0],
            [-2.0, 2.0],
            [-2.0, -2.0],
        ],
        dtype=float,
    )

    result = find_equilibrium_lcfs(
        psi,
        grid,
        center_hint=(0.0, 0.0),
        limiter_shape=limiter,
        fixed_angles=_angles(64),
    )

    assert result.topology == "limited"
    assert result.psi_boundary == pytest.approx(1.0, abs=2.0e-6)
    assert len(result.limiter_contacts) == 1
    assert np.allclose(result.limiter_contacts[0].point, (1.0, 0.0), atol=2.0e-4)
    assert result.quality.limiter_violation_count == 0


def test_single_null_keeps_non_smooth_xpoint_geometry() -> None:
    grid = _grid(center=(0.0, 1.0))
    R, Z = grid.mesh()
    psi = R * R + (Z * Z - 1.0) ** 2
    result = find_equilibrium_lcfs(
        psi,
        grid,
        center_hint=(0.0, 1.0),
        limiter_shape=_square_limiter(),
        fixed_angles=_angles(64),
    )

    x_point = np.asarray(result.x_points[0].point, dtype=float)
    boundary = np.asarray(result.core_boundary[:-1], dtype=float)
    index = int(np.argmin(np.linalg.norm(boundary - x_point[None, :], axis=1)))
    assert np.linalg.norm(boundary[index] - x_point) < 1.0e-8

    before = boundary[(index - 1) % boundary.shape[0]] - x_point
    after = boundary[(index + 1) % boundary.shape[0]] - x_point
    before /= np.linalg.norm(before)
    after /= np.linalg.norm(after)
    # A globally smooth closed spline would force antiparallel one-sided rays.
    # The separatrix keeps the two distinct Hessian-null directions instead.
    assert abs(float(np.dot(before, after))) < 0.9


def test_single_null_solution_converges_when_grid_is_refined() -> None:
    limiter = _square_limiter()
    levels: list[float] = []
    residuals: list[float] = []
    x_points: list[np.ndarray] = []
    for n in (121, 181, 241):
        grid = _grid(n=n, center=(0.0, 1.0))
        R, Z = grid.mesh()
        psi = R * R + (Z * Z - 1.0) ** 2
        result = find_equilibrium_lcfs(
            psi,
            grid,
            center_hint=(0.0, 1.0),
            limiter_shape=limiter,
            fixed_angles=_angles(),
        )
        levels.append(result.psi_boundary)
        residuals.append(result.quality.normalized_flux_residual)
        x_points.append(np.asarray(result.x_points[0].point, dtype=float))

    assert max(abs(level - 1.0) for level in levels) < 2.0e-6
    assert residuals[2] < residuals[1] < residuals[0]
    assert all(np.linalg.norm(point) < 3.0e-4 for point in x_points)
