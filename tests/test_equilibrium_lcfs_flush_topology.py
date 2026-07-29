from __future__ import annotations

import numpy as np
import pytest

from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.geometry.critical_points import find_critical_points
from tokamak_control.geometry.equilibrium_field import EquilibriumField
from tokamak_control.geometry.lcfs import find_equilibrium_lcfs
from tokamak_control.geometry.level_set_graph import _build_topology_grid, _surface_segments


def _grid() -> Grid2D:
    size = 61
    step = 0.06
    start = -(size // 2) * step + 0.5 * step
    return Grid2D(
        r=Grid1D(start=start, step=step, size=size, center=0.0),
        z=Grid1D(start=start, step=step, size=size, center=0.0),
    )


def _angles() -> np.ndarray:
    return np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)


def test_limited_surface_uses_wall_contact_and_has_valid_projection() -> None:
    grid = _grid()
    R, Z = grid.mesh()
    limiter = np.asarray(
        [
            [-1.25, -1.25],
            [1.25, -1.25],
            [1.25, 1.25],
            [-1.25, 1.25],
            [-1.25, -1.25],
        ],
        dtype=float,
    )

    result = find_equilibrium_lcfs(
        R * R + Z * Z,
        grid,
        center_hint=(0.0, 0.0),
        limiter_shape=limiter,
        fixed_angles=_angles(),
    )

    assert result.topology == "limited"
    assert result.psi_boundary == pytest.approx(1.25**2, abs=2.0e-8)
    assert len(result.limiter_contacts) >= 4
    assert result.fixed_angle_projection.valid
    assert np.all(result.fixed_angle_projection.intersection_counts == 1)
    assert result.quality.normalized_flux_residual < 2.0e-4
    assert result.quality.limiter_violation_count == 0


@pytest.mark.parametrize("perturbation", [0.0, 1.0e-4, -1.0e-4, 5.0e-4])
def test_single_null_topology_is_stable_across_small_field_perturbations(
    perturbation: float,
) -> None:
    grid = _grid()
    R, Z = grid.mesh()
    limiter = np.asarray(
        [
            [-1.50, -1.40],
            [1.50, -1.40],
            [1.50, 1.60],
            [-1.50, 1.60],
            [-1.50, -1.40],
        ],
        dtype=float,
    )
    coupling = 0.30
    psi = (
        R * R
        + Z * Z
        - (2.0 / 3.0) * Z**3
        + coupling * R * Z
        + float(perturbation) * Z
    )

    result = find_equilibrium_lcfs(
        psi,
        grid,
        center_hint=(0.0, 0.0),
        limiter_shape=limiter,
        fixed_angles=_angles(),
    )

    assert result.topology == "single_null"
    assert len(result.x_points) == 1
    assert result.core_boundary.shape[0] >= 100
    assert np.linalg.norm(result.core_boundary[0] - result.core_boundary[-1]) < 1.0e-10
    assert result.fixed_angle_projection.valid
    assert np.all(result.fixed_angle_projection.intersection_counts == 1)
    assert result.quality.normalized_flux_residual < 2.0e-4
    assert result.quality.limiter_violation_count == 0


def test_xpoint_element_has_four_explicit_graph_branches() -> None:
    grid = _grid()
    R, Z = grid.mesh()
    limiter = np.asarray(
        [
            [-1.50, -1.40],
            [1.50, -1.40],
            [1.50, 1.60],
            [-1.50, 1.60],
            [-1.50, -1.40],
        ],
        dtype=float,
    )
    psi = R * R + Z * Z - (2.0 / 3.0) * Z**3 + 0.30 * R * Z
    field = EquilibriumField(grid=grid, psi=psi)
    critical = find_critical_points(
        field,
        center_hint=(0.0, 0.0),
        limiter_poly=limiter,
    )
    x_point = critical.x_points[0]
    topology_grid = _build_topology_grid(
        field,
        axis=critical.primary_axis.point,
        x_points=(x_point,),
        refinement=2,
    )
    segments, _points = _surface_segments(
        field,
        topology_grid,
        x_point.level,
        x_points=(x_point,),
    )

    x_key = ("x", 0)
    degree = sum(1 for segment in segments if x_key in segment)
    assert degree == 4
