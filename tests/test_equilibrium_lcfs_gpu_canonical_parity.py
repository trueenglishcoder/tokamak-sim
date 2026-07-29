from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.geometry.boundary_gpu_canonical import fixed_angle_boundary_gpu_canonical
from tokamak_control.geometry.lcfs import find_equilibrium_lcfs


def _grid() -> Grid2D:
    size = 61
    step = 0.06
    start = -(size // 2) * step + 0.5 * step
    return Grid2D(
        r=Grid1D(start=start, step=step, size=size, center=0.0),
        z=Grid1D(start=start, step=step, size=size, center=0.0),
    )


def _limiter() -> np.ndarray:
    return np.asarray(
        [
            [-1.50, -1.40],
            [1.50, -1.40],
            [1.50, 1.60],
            [-1.50, 1.60],
            [-1.50, -1.40],
        ],
        dtype=np.float64,
    )


def _angles() -> np.ndarray:
    return np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=np.float64)


def test_canonical_gpu_bridge_matches_cpu_limited_and_single_null() -> None:
    grid = _grid()
    R, Z = grid.mesh()
    fields = np.stack(
        (
            R * R + Z * Z,
            R * R + Z * Z - (2.0 / 3.0) * Z**3 + 0.30 * R * Z,
        ),
        axis=0,
    )
    angles = _angles()
    limiter = _limiter()

    result = fixed_angle_boundary_gpu_canonical(
        field=torch.as_tensor(fields, dtype=torch.float64),
        grid=grid,
        center=(0.0, 0.0),
        angles=torch.as_tensor(angles, dtype=torch.float64),
        limiter=torch.as_tensor(limiter, dtype=torch.float64),
        return_dense_boundary=True,
    )

    assert result.found.tolist() == [True, True]
    assert result.topology_code.tolist() == [1, 2]
    assert result.status_code.tolist() == [8, 9]
    assert result.core_boundary_count[0].item() > 0
    assert result.core_boundary_count[1].item() > 0

    for index, psi in enumerate(fields):
        cpu = find_equilibrium_lcfs(
            psi,
            grid,
            center_hint=(0.0, 0.0),
            limiter_shape=limiter,
            fixed_angles=angles,
        )
        np.testing.assert_allclose(
            result.level[index].detach().cpu().numpy(),
            cpu.psi_boundary,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            result.radii[index].detach().cpu().numpy(),
            cpu.fixed_angle_projection.radii,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_array_equal(
            result.intersection_counts[index].detach().cpu().numpy(),
            cpu.fixed_angle_projection.intersection_counts,
        )
