"""Regression coverage for the shot 3863 LCFS topology transition.

The fixture stores reconstructed psi maps from run 1785266189979059307.  The
selected frames cover the early GPU failures, the large CPU/GPU disagreement
region, and the limited-to-single-null transition at steps 895/896.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.geometry.lcfs import find_equilibrium_lcfs


FIXTURE = Path(__file__).with_name("data") / "equilibrium_lcfs_3863_regression.npz"


def _load_fixture() -> tuple[np.lib.npyio.NpzFile, Grid2D]:
    data = np.load(FIXTURE, allow_pickle=False)
    grid = Grid2D(
        r=Grid1D(
            start=float(data["r_start"]),
            step=float(data["r_step"]),
            size=int(data["r_size"]),
            center=float(data["r_center"]),
        ),
        z=Grid1D(
            start=float(data["z_start"]),
            step=float(data["z_step"]),
            size=int(data["z_size"]),
            center=float(data["z_center"]),
        ),
    )
    return data, grid


def test_shot_3863_preserves_the_observed_topology_transition() -> None:
    data, grid = _load_fixture()
    observed: list[str] = []
    levels: list[float] = []

    for index, step in enumerate(data["steps"]):
        result = find_equilibrium_lcfs(
            data["psi"][index],
            grid,
            center_hint=tuple(float(value) for value in data["center"]),
            limiter_shape=data["limiter"],
            fixed_angles=data["angles"],
        )
        expected_topology = str(data["expected_topology"][index])
        observed.append(result.topology)
        levels.append(float(result.psi_boundary))

        assert result.topology == expected_topology, f"step {int(step)}"
        assert result.fixed_angle_projection.valid, f"step {int(step)}"
        assert len(result.x_points) == int(data["expected_x_count"][index]), f"step {int(step)}"
        np.testing.assert_allclose(
            result.magnetic_axis.point,
            data["expected_axis"][index],
            rtol=0.0,
            atol=2.0e-3,
        )
        np.testing.assert_allclose(
            result.psi_boundary,
            data["expected_psi_boundary"][index],
            rtol=0.0,
            atol=5.0e-5,
        )
        assert result.quality.limiter_violation_count == 0

    steps = data["steps"].tolist()
    assert observed[steps.index(895)] == "limited"
    assert observed[steps.index(896)] == "single_null"
    assert abs(levels[steps.index(896)] - levels[steps.index(895)]) < 1.0e-4


def test_canonical_gpu_dispatch_contract_matches_3863_cpu_frames() -> None:
    torch = pytest.importorskip("torch")
    from tokamak_control.geometry.boundary_gpu_canonical import fixed_angle_boundary_gpu_canonical

    data, grid = _load_fixture()
    selected_steps = (200, 896, 1192)
    indices = [data["steps"].tolist().index(step) for step in selected_steps]
    fields = torch.as_tensor(data["psi"][indices], dtype=torch.float64)
    angles = torch.as_tensor(data["angles"], dtype=torch.float64)
    limiter = torch.as_tensor(data["limiter"], dtype=torch.float64)

    gpu_contract = fixed_angle_boundary_gpu_canonical(
        field=fields,
        grid=grid,
        center=tuple(float(value) for value in data["center"]),
        angles=angles,
        limiter=limiter,
        return_dense_boundary=True,
    )

    assert gpu_contract.found.tolist() == [True, True, True]
    assert gpu_contract.topology_code.tolist() == [1, 2, 2]
    assert gpu_contract.status_code.tolist() == [8, 9, 9]

    for output_index, fixture_index in enumerate(indices):
        cpu = find_equilibrium_lcfs(
            data["psi"][fixture_index],
            grid,
            center_hint=tuple(float(value) for value in data["center"]),
            limiter_shape=data["limiter"],
            fixed_angles=data["angles"],
        )
        np.testing.assert_allclose(
            gpu_contract.level[output_index].detach().cpu().numpy(),
            cpu.psi_boundary,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            gpu_contract.radii[output_index].detach().cpu().numpy(),
            cpu.fixed_angle_projection.radii,
            rtol=0.0,
            atol=1.0e-12,
        )
