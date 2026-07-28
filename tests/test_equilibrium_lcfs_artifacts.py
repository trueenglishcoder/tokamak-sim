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


def test_gpu_artifact_payload_uses_single_gpu_boundary_result() -> None:
    """Не смешивать плотный контур и радиусы из разных реализаций."""
    import sys
    from types import ModuleType, SimpleNamespace

    import torch

    try:
        import tomli_w  # noqa: F401
    except ModuleNotFoundError:
        stub = ModuleType("tomli_w")
        stub.dumps = lambda _value: ""  # type: ignore[attr-defined]
        sys.modules["tomli_w"] = stub

    from tokamak_control.cli.run_simulation import (
        _gpu_boundary_detail_payload,
        _gpu_dense_boundary_from_result,
    )

    dense = torch.as_tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.0, 0.0]]],
        dtype=torch.float64,
    )
    result = SimpleNamespace(
        boundary=SimpleNamespace(
            found=torch.as_tensor([True]),
            topology_code=torch.as_tensor([2]),
            axis_points=torch.as_tensor([[0.5, 0.5]], dtype=torch.float64),
            x_points=torch.as_tensor(
                [[[0.5, 1.0], [float("nan"), float("nan")]]],
                dtype=torch.float64,
            ),
            core_boundary=dense,
            core_boundary_count=torch.as_tensor([4]),
            limiter_contacts=torch.empty((1, 0, 2), dtype=torch.float64),
            limiter_contact_count=torch.as_tensor([0]),
            intersection_counts=torch.ones((1, 4), dtype=torch.int64),
            quality=torch.as_tensor(
                [[1.0e-9, 2.0e-8, 0.0, 0.1, 0.0, 12.0]],
                dtype=torch.float64,
            ),
        )
    )

    poly = _gpu_dense_boundary_from_result(result)
    payload = _gpu_boundary_detail_payload(result)

    assert poly is not None
    assert np.allclose(poly, np.asarray(dense[0]))
    assert payload["boundary_topology_code"] == 2
    assert np.array_equal(
        payload["boundary_fixed_angle_counts"],
        np.ones((4,), dtype=float),
    )
    assert bool(payload["boundary_fixed_angle_valid"])
    assert np.allclose(
        payload["boundary_quality"],
        np.asarray([1.0e-9, 2.0e-8, 0.0, 0.1, 0.0, 12.0]),
    )


def test_batched_gpu_artifact_path_does_not_call_cpu_lcfs() -> None:
    """GPU artifact path не должен повторно запускать CPU LCFS на каждом шаге."""
    import inspect
    import sys
    from types import ModuleType

    try:
        import tomli_w  # noqa: F401
    except ModuleNotFoundError:
        stub = ModuleType("tomli_w")
        stub.dumps = lambda _value: ""  # type: ignore[attr-defined]
        sys.modules["tomli_w"] = stub

    from tokamak_control.cli.run_simulation import _run_batched_gpu_fixed_angle_artifacts

    source = inspect.getsource(_run_batched_gpu_fixed_angle_artifacts)

    assert "find_equilibrium_boundary(" not in source
    assert "_gpu_dense_boundary_from_result(result)" in source
