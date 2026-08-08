from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.geometry.boundary_gpu import (
    fixed_angle_boundary_gpu,
    prepare_fixed_angle_boundary_gpu_geometry,
)
from tokamak_control.geometry.lcfs import find_equilibrium_lcfs
from tests.test_equilibrium_lcfs_gpu_parity import (
    _assert_unordered_points_close,
    _symmetric_polyline_hausdorff,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)
FIXTURE = (
    Path(__file__).resolve().parent
    / "data"
    / "equilibrium_lcfs_3863_regression.npz"
)
MAX_SINGLE_FRAME_SECONDS = float(
    os.environ.get("TOKAMAK_LCFS_MAX_SINGLE_FRAME_SECONDS", "0.35")
)
MAX_BATCH_SECONDS = float(
    os.environ.get("TOKAMAK_LCFS_MAX_BATCH_SECONDS", "1.00")
)
MAX_BATCH32_SECONDS = float(
    os.environ.get("TOKAMAK_LCFS_MAX_BATCH32_SECONDS", "2.00")
)
MAX_BATCH32_PER_LANE_SECONDS = float(
    os.environ.get("TOKAMAK_LCFS_MAX_BATCH32_PER_LANE_SECONDS", "0.08")
)
MAX_SINGLE_FRAME_MEMORY_MIB = float(
    os.environ.get("TOKAMAK_LCFS_MAX_SINGLE_FRAME_MEMORY_MIB", "1024")
)


def _fixture_grid(data: np.lib.npyio.NpzFile) -> Grid2D:
    return Grid2D(
        r=Grid1D(
            float(data["r_start"]),
            float(data["r_step"]),
            int(data["r_size"]),
            float(data["r_center"]),
        ),
        z=Grid1D(
            float(data["z_start"]),
            float(data["z_step"]),
            int(data["z_size"]),
            float(data["z_center"]),
        ),
    )


def _prepare_public_case(
    *,
    psi: np.ndarray,
    grid: Grid2D,
    center: tuple[float, float],
    limiter: np.ndarray,
    angles: np.ndarray,
) -> dict[str, object]:
    basis = np.asarray(psi, dtype=np.float64)
    if basis.ndim == 2:
        basis = basis[None]
    lane_count = int(basis.shape[0])
    geometry = prepare_fixed_angle_boundary_gpu_geometry(
        grid=grid,
        center=center,
        angles_rad=angles,
        limiter_shape=limiter,
        boundary_mode="equilibrium_lcfs",
        gpu_device="cuda:0",
        dtype=torch.float64,
        basis_fields=basis,
    )
    return {
        "psi": torch.as_tensor(basis, dtype=torch.float64, device="cuda:0"),
        "grid": grid,
        "center": center,
        "angles": angles,
        "limiter": limiter,
        "geometry": geometry,
        "amplitudes": torch.eye(
            lane_count, dtype=torch.float64, device="cuda:0"
        ),
    }


def _run_public_case(prepared: dict[str, object]):
    return fixed_angle_boundary_gpu(
        psi=prepared["psi"],
        grid=prepared["grid"],
        center=prepared["center"],
        angles_rad=prepared["angles"],
        limiter_shape=prepared["limiter"],
        boundary_mode="equilibrium_lcfs",
        gpu_device="cuda:0",
        prepared_geometry=prepared["geometry"],
        amplitudes=prepared["amplitudes"],
        return_dense_boundary=True,
    )


def _timed_runtime(prepared: dict[str, object], *, repeats: int = 3):
    # Static SciPy coefficient construction is excluded.  Warmup and timing both
    # exercise the complete public per-step path, including critical points,
    # candidate levels, half-edge topology, dense LCFS and 32-ray projection.
    for _ in range(2):
        result = _run_public_case(prepared)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(int(repeats)):
        result = _run_public_case(prepared)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) / float(repeats)
    peak_memory_mib = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return result, elapsed, peak_memory_mib


@pytest.mark.parametrize(
    "step",
    [200, 895, 896, 1192],
    ids=["limited-200", "limited-895", "single-null-896", "single-null-1192"],
)
def test_cuda_public_path_matches_cpu_on_selected_3863_frames(step: int) -> None:
    data = np.load(FIXTURE, allow_pickle=False)
    index = data["steps"].tolist().index(step)
    grid = _fixture_grid(data)
    psi = np.asarray(data["psi"][index], dtype=np.float64)
    center = tuple(float(value) for value in data["center"])
    limiter = np.asarray(data["limiter"], dtype=np.float64)
    angles = np.asarray(data["angles"], dtype=np.float64)
    cpu = find_equilibrium_lcfs(
        psi,
        grid,
        center_hint=center,
        limiter_shape=limiter,
        fixed_angles=angles,
    )
    prepared = _prepare_public_case(
        psi=psi,
        grid=grid,
        center=center,
        limiter=limiter,
        angles=angles,
    )
    result, elapsed, peak_memory_mib = _timed_runtime(prepared)
    print(
        f"step={step} runtime_s={elapsed:.6f} "
        f"peak_memory_mib={peak_memory_mib:.1f}"
    )
    expected_code = {
        "limited": 1,
        "single_null": 2,
        "double_null": 3,
        "multi_null": 4,
    }[cpu.topology]
    assert result.found.cpu().tolist() == [True]
    assert result.projection_valid.cpu().tolist() == [True]
    assert result.projection_error_code.cpu().tolist() == [0]
    assert result.topology_code.cpu().tolist() == [expected_code]
    np.testing.assert_allclose(
        result.axis_points.detach().cpu().numpy()[0],
        np.asarray(cpu.magnetic_axis.point, dtype=np.float64),
        atol=2.0e-6,
        rtol=0.0,
    )
    assert float(result.psi_axis.detach().cpu().numpy()[0]) == pytest.approx(
        cpu.psi_axis, abs=5.0e-7
    )
    selected_x = result.x_points.detach().cpu().numpy()[0]
    selected_x = selected_x[np.all(np.isfinite(selected_x), axis=1)]
    _assert_unordered_points_close(
        selected_x,
        np.asarray(
            [point.point for point in cpu.x_points], dtype=np.float64
        ).reshape(-1, 2),
        tolerance=2.0e-6,
    )
    np.testing.assert_array_equal(
        result.intersection_counts.detach().cpu().numpy()[0],
        cpu.fixed_angle_projection.intersection_counts,
    )
    np.testing.assert_allclose(
        result.level.detach().cpu().numpy(),
        [cpu.psi_boundary],
        atol=5.0e-7,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.radii.detach().cpu().numpy()[0],
        cpu.fixed_angle_projection.radii,
        atol=2.0e-6,
        rtol=0.0,
    )
    count = int(result.core_boundary_count.detach().cpu().numpy()[0])
    assert count >= 4
    gpu_boundary = result.core_boundary.detach().cpu().numpy()[0, :count]
    assert (
        _symmetric_polyline_hausdorff(cpu.core_boundary, gpu_boundary)
        <= 2.0e-6
    )
    contact_count = int(
        result.limiter_contact_count.detach().cpu().numpy()[0]
    )
    gpu_contacts = result.limiter_contacts.detach().cpu().numpy()[
        0, :contact_count
    ]
    _assert_unordered_points_close(
        gpu_contacts,
        np.asarray(
            [contact.point for contact in cpu.limiter_contacts],
            dtype=np.float64,
        ).reshape(-1, 2),
        tolerance=2.0e-6,
    )
    assert elapsed < MAX_SINGLE_FRAME_SECONDS
    assert peak_memory_mib < MAX_SINGLE_FRAME_MEMORY_MIB


def test_cuda_public_path_scales_on_heterogeneous_3863_batch() -> None:
    data = np.load(FIXTURE, allow_pickle=False)
    grid = _fixture_grid(data)
    selected_steps = [200, 895, 896, 1192]
    indices = [data["steps"].tolist().index(step) for step in selected_steps]
    psi = np.asarray(data["psi"][indices], dtype=np.float64)
    center = tuple(float(value) for value in data["center"])
    limiter = np.asarray(data["limiter"], dtype=np.float64)
    angles = np.asarray(data["angles"], dtype=np.float64)
    prepared = _prepare_public_case(
        psi=psi,
        grid=grid,
        center=center,
        limiter=limiter,
        angles=angles,
    )
    result, elapsed, peak_memory_mib = _timed_runtime(prepared, repeats=2)
    print(
        f"batch={len(indices)} runtime_s={elapsed:.6f} "
        f"per_lane_s={elapsed / len(indices):.6f} "
        f"peak_memory_mib={peak_memory_mib:.1f}"
    )
    assert result.found.cpu().tolist() == [True] * len(indices)
    assert result.projection_valid.cpu().tolist() == [True] * len(indices)
    assert result.projection_error_code.cpu().tolist() == [0] * len(indices)
    assert result.topology_code.cpu().tolist() == [1, 1, 2, 2]
    radii_np = result.radii.detach().cpu().numpy()
    levels_np = result.level.detach().cpu().numpy()
    counts_np = result.intersection_counts.detach().cpu().numpy()
    boundaries_np = result.core_boundary.detach().cpu().numpy()
    boundary_counts_np = result.core_boundary_count.detach().cpu().numpy()
    axes_np = result.axis_points.detach().cpu().numpy()
    x_points_np = result.x_points.detach().cpu().numpy()
    for lane, index in enumerate(indices):
        cpu = find_equilibrium_lcfs(
            np.asarray(data["psi"][index], dtype=np.float64),
            grid,
            center_hint=center,
            limiter_shape=limiter,
            fixed_angles=angles,
        )
        assert levels_np[lane] == pytest.approx(
            cpu.psi_boundary, abs=5.0e-7
        )
        np.testing.assert_allclose(
            axes_np[lane],
            np.asarray(cpu.magnetic_axis.point, dtype=np.float64),
            atol=2.0e-6,
            rtol=0.0,
        )
        selected_x = x_points_np[lane]
        selected_x = selected_x[np.all(np.isfinite(selected_x), axis=1)]
        _assert_unordered_points_close(
            selected_x,
            np.asarray(
                [point.point for point in cpu.x_points], dtype=np.float64
            ).reshape(-1, 2),
            tolerance=2.0e-6,
        )
        np.testing.assert_array_equal(
            counts_np[lane],
            cpu.fixed_angle_projection.intersection_counts,
        )
        np.testing.assert_allclose(
            radii_np[lane],
            cpu.fixed_angle_projection.radii,
            atol=2.0e-6,
            rtol=0.0,
        )
        count = int(boundary_counts_np[lane])
        assert count >= 4
        assert (
            _symmetric_polyline_hausdorff(
                cpu.core_boundary,
                boundaries_np[lane, :count],
            )
            <= 2.0e-6
        )
    assert elapsed < MAX_BATCH_SECONDS
    assert elapsed / len(indices) < MAX_SINGLE_FRAME_SECONDS
    assert peak_memory_mib < MAX_SINGLE_FRAME_MEMORY_MIB


def test_cuda_public_path_scales_to_training_sized_graph_chunk() -> None:
    """Exercise one full 32-lane graph chunk with heterogeneous topology.

    The four fixture equilibria are static spline basis fields.  Repeated one-hot
    amplitudes create 32 runtime lanes without inflating the basis count or
    changing any field.  This is the lane width used by the production bounded-
    memory topology path before larger training batches are chunked.
    """
    data = np.load(FIXTURE, allow_pickle=False)
    grid = _fixture_grid(data)
    selected_steps = [200, 895, 896, 1192]
    indices = [data["steps"].tolist().index(step) for step in selected_steps]
    basis = np.asarray(data["psi"][indices], dtype=np.float64)
    repeats = 8
    psi = np.tile(basis, (repeats, 1, 1))
    amplitudes = np.tile(np.eye(len(indices), dtype=np.float64), (repeats, 1))
    center = tuple(float(value) for value in data["center"])
    limiter = np.asarray(data["limiter"], dtype=np.float64)
    angles = np.asarray(data["angles"], dtype=np.float64)
    geometry = prepare_fixed_angle_boundary_gpu_geometry(
        grid=grid,
        center=center,
        angles_rad=angles,
        limiter_shape=limiter,
        boundary_mode="equilibrium_lcfs",
        gpu_device="cuda:0",
        dtype=torch.float64,
        basis_fields=basis,
    )
    prepared = {
        "psi": torch.as_tensor(psi, dtype=torch.float64, device="cuda:0"),
        "grid": grid,
        "center": center,
        "angles": angles,
        "limiter": limiter,
        "geometry": geometry,
        "amplitudes": torch.as_tensor(
            amplitudes, dtype=torch.float64, device="cuda:0"
        ),
    }
    result, elapsed, peak_memory_mib = _timed_runtime(prepared, repeats=2)
    expected_topology = [1, 1, 2, 2] * repeats
    print(
        f"batch=32 runtime_s={elapsed:.6f} "
        f"per_lane_s={elapsed / 32.0:.6f} "
        f"peak_memory_mib={peak_memory_mib:.1f}"
    )
    assert result.found.cpu().tolist() == [True] * 32
    assert result.projection_valid.cpu().tolist() == [True] * 32
    assert result.projection_error_code.cpu().tolist() == [0] * 32
    assert result.topology_code.cpu().tolist() == expected_topology
    assert elapsed < MAX_BATCH32_SECONDS
    assert elapsed / 32.0 < MAX_BATCH32_PER_LANE_SECONDS
    assert peak_memory_mib < MAX_SINGLE_FRAME_MEMORY_MIB
