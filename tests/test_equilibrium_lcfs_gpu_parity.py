"""CPU/GPU-kernel parity tests for the equilibrium LCFS fixed-angle signal."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.geometry.boundary_gpu import (
    _axis_search,
    _equilibrium_lcfs_fixed_angle_search,
    _sample_closed_polyline_numpy,
)
from tokamak_control.geometry.lcfs import find_equilibrium_lcfs


def _limiter() -> np.ndarray:
    return np.asarray([[-2.2, -2.2], [2.2, -2.2], [2.2, 2.2], [-2.2, 2.2], [-2.2, -2.2]], dtype=float)


def _case(name: str) -> tuple[Grid2D, np.ndarray, tuple[float, float], int]:
    n = 161
    low = -2.5
    high = 2.5
    step = (high - low) / float(n - 1)
    center = (0.0, 1.0) if name == "single_null" else (0.0, 0.0)
    grid = Grid2D(
        r=Grid1D(start=low, step=step, size=n, center=center[0]),
        z=Grid1D(start=low, step=step, size=n, center=center[1]),
    )
    R, Z = grid.mesh()
    if name == "limited":
        psi = R * R + Z * Z
        code = 1
    elif name == "single_null":
        psi = R * R + (Z * Z - 1.0) ** 2
        code = 2
    elif name == "double_null":
        psi = R * R + Z * Z - 0.5 * Z**4
        code = 3
    else:
        raise AssertionError(name)
    return grid, psi, center, code


@pytest.mark.parametrize("name", ["limited", "single_null", "double_null"])
def test_batched_kernel_matches_canonical_cpu(name: str) -> None:
    grid, psi, center, expected_code = _case(name)
    limiter = _limiter()
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    cpu = find_equilibrium_lcfs(
        psi,
        grid,
        center_hint=center,
        limiter_shape=limiter,
        fixed_angles=angles,
    )

    field = torch.as_tensor(psi[None], dtype=torch.float64)
    limiter_t = torch.as_tensor(limiter, dtype=torch.float64)
    axis_points, axis_level, axis_kind, axis_valid = _axis_search(field, grid, center, limiter_t)
    gpu_like, topology_code, selected_x = _equilibrium_lcfs_fixed_angle_search(
        psi=field,
        grid=grid,
        axis_points=axis_points,
        projection_center=torch.as_tensor([center], dtype=torch.float64),
        axis_level=axis_level,
        axis_kind=axis_kind,
        axis_valid=axis_valid,
        measurement_angles=torch.as_tensor(angles, dtype=torch.float64),
        limiter=limiter_t,
        limiter_samples=torch.as_tensor(_sample_closed_polyline_numpy(limiter, 512), dtype=torch.float64),
        ray_samples=512,
    )
    _points, radii, found, level = gpu_like

    assert bool(found[0])
    assert int(topology_code[0]) == expected_code
    assert float(level[0]) == pytest.approx(cpu.psi_boundary, abs=5.0e-4)
    assert np.allclose(np.asarray(radii[0]), cpu.fixed_angle_projection.radii, atol=2.0e-2, rtol=0.0)
    if expected_code == 1:
        assert not bool(torch.isfinite(selected_x[0]).any())
    else:
        assert int(torch.count_nonzero(torch.all(torch.isfinite(selected_x[0]), dim=1))) == expected_code - 1
