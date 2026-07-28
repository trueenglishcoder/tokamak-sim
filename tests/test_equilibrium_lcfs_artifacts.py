"""Artifact schema tests for dense LCFS topology channels."""

from __future__ import annotations

import numpy as np

from tokamak_control.io.data_io import RunWriter, load_run


def test_boundary_topology_round_trip(tmp_path) -> None:
    writer = RunWriter(
        output_dir=tmp_path,
        grid_shape=(2, 3),
        metadata={"boundary": {"mode": "equilibrium_lcfs"}},
    )
    boundary = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.0, 0.0]], dtype=float)
    branches = (
        np.asarray([[0.5, 1.0], [0.7, 1.4]], dtype=float),
        np.asarray([[0.5, 1.0], [0.3, 1.4], [0.2, 1.8]], dtype=float),
    )
    writer.append(
        t=0.0,
        Ip=1.0,
        pfc_currents=np.asarray([1.0, 2.0]),
        pfc_derivs=np.asarray([0.1, 0.2]),
        sol_currents=np.asarray([3.0]),
        sol_derivs=np.asarray([0.3]),
        psi=np.arange(6, dtype=float).reshape(2, 3),
        radii_true=np.asarray([0.9, 1.0, 1.1, 1.0]),
        radii_meas=np.asarray([0.9, 1.0, 1.1, 1.0]),
        boundary_poly_true=boundary,
        boundary_topology_code=2,
        boundary_axis=np.asarray([0.5, 0.5]),
        boundary_x_points=np.asarray([[0.5, 1.0]]),
        boundary_limiter_contacts=np.asarray([[0.7, 1.4], [0.3, 1.4]]),
        boundary_separatrix_branches=branches,
        boundary_fixed_angle_valid=True,
        boundary_fixed_angle_counts=np.ones((4,), dtype=int),
        boundary_quality=np.asarray([1e-9, 2e-8, 0.0, 0.1, 0.0, 12.0]),
    )
    path = writer.finalize()
    run = load_run(path)

    assert run["version"] == 4
    assert np.array_equal(run["boundary_topology_code"], np.asarray([2], dtype=np.int8))
    assert np.allclose(run["boundary_axis"], np.asarray([[0.5, 0.5]]))
    assert np.allclose(run["boundary_x_points"][0, 0], np.asarray([0.5, 1.0]))
    assert run["boundary_separatrix_branches"].shape == (1, 2, 3, 2)
    assert np.allclose(run["boundary_separatrix_branches"][0, 0, :2], branches[0])
    assert np.allclose(run["boundary_separatrix_branches"][0, 1, :3], branches[1])
    assert bool(run["boundary_fixed_angle_valid"][0])
    assert np.array_equal(run["boundary_fixed_angle_counts"], np.ones((1, 4)))
    assert np.allclose(run["boundary_quality"][0], np.asarray([1e-9, 2e-8, 0.0, 0.1, 0.0, 12.0]))
