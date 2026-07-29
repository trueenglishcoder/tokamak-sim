"""Проверки совпадения CPU- и GPU-реализаций equilibrium LCFS."""

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
    """Вернуть квадратный тестовый лимитер."""
    return np.asarray(
        [
            [-2.2, -2.2],
            [2.2, -2.2],
            [2.2, 2.2],
            [-2.2, 2.2],
            [-2.2, -2.2],
        ],
        dtype=float,
    )


def _case(name: str) -> tuple[Grid2D, np.ndarray, tuple[float, float], int]:
    """Построить аналитический limited или diverted equilibrium."""
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


def _gpu_result(
    *,
    psi: np.ndarray,
    grid: Grid2D,
    center: tuple[float, float],
    limiter: np.ndarray,
    angles: np.ndarray,
    return_dense_boundary: bool,
):
    """Запустить внутренний GPU-kernel на torch tensor без требования CUDA."""
    field = torch.as_tensor(psi[None], dtype=torch.float64)
    limiter_t = torch.as_tensor(limiter, dtype=torch.float64)
    axis_points, axis_level, axis_kind, axis_valid = _axis_search(
        field,
        grid,
        center,
        limiter_t,
    )
    validation_angles = torch.as_tensor(angles, dtype=torch.float64)
    dense_angles = torch.linspace(
        -float(np.pi),
        float(np.pi),
        257,
        dtype=torch.float64,
    )[:-1]
    return _equilibrium_lcfs_fixed_angle_search(
        psi=field,
        grid=grid,
        axis_points=axis_points,
        projection_center=torch.as_tensor([center], dtype=torch.float64),
        axis_level=axis_level,
        axis_kind=axis_kind,
        axis_valid=axis_valid,
        measurement_angles=torch.as_tensor(angles, dtype=torch.float64),
        validation_angles=validation_angles,
        dense_angles=dense_angles,
        limiter=limiter_t,
        limiter_samples=torch.as_tensor(
            _sample_closed_polyline_numpy(limiter, 512),
            dtype=torch.float64,
        ),
        ray_samples=512,
        return_dense_boundary=return_dense_boundary,
    )


def _bidirectional_boundary_error(first: np.ndarray, second: np.ndarray) -> float:
    """Вернуть симметричную максимальную ошибку двух дискретных контуров."""
    first_open = np.asarray(first, dtype=float)
    second_open = np.asarray(second, dtype=float)
    first_open = first_open[np.all(np.isfinite(first_open), axis=1)]
    second_open = second_open[np.all(np.isfinite(second_open), axis=1)]
    distances = np.linalg.norm(
        first_open[:, None, :] - second_open[None, :, :],
        axis=2,
    )
    return float(max(np.min(distances, axis=1).max(), np.min(distances, axis=0).max()))


@pytest.mark.parametrize("name", ["limited", "single_null", "double_null"])
def test_full_gpu_lcfs_matches_cpu_reference(name: str) -> None:
    """Проверить уровень, топологию, радиусы и плотный GPU-контур."""
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

    (
        gpu_signal,
        topology_code,
        selected_x,
        intersection_counts,
        core_boundary,
        core_boundary_count,
        contacts,
        contact_count,
        quality,
    ) = _gpu_result(
        psi=psi,
        grid=grid,
        center=center,
        limiter=limiter,
        angles=angles,
        return_dense_boundary=True,
    )
    _points, radii, found, level = gpu_signal

    assert bool(found[0])
    assert int(topology_code[0]) == expected_code
    assert float(level[0]) == pytest.approx(cpu.psi_boundary, abs=5.0e-3)
    assert np.allclose(
        np.asarray(radii[0]),
        cpu.fixed_angle_projection.radii,
        atol=2.0e-2,
        rtol=0.0,
    )
    assert np.array_equal(
        np.asarray(intersection_counts[0]),
        np.ones((angles.size,), dtype=np.int64),
    )
    count = int(core_boundary_count[0])
    assert count == 257
    gpu_boundary = np.asarray(core_boundary[0, :count])
    assert np.allclose(gpu_boundary[0], gpu_boundary[-1], atol=1.0e-12, rtol=0.0)
    assert _bidirectional_boundary_error(gpu_boundary, cpu.core_boundary) <= 1.1 * grid.r.step
    assert bool(torch.all(torch.isfinite(quality[0])))
    if expected_code == 1:
        assert not bool(torch.isfinite(selected_x[0]).any())
        assert int(contact_count[0]) == 1
        assert bool(torch.all(torch.isfinite(contacts[0, 0])))
    else:
        finite_x = torch.all(torch.isfinite(selected_x[0]), dim=1)
        assert int(torch.count_nonzero(finite_x)) == expected_code - 1
        assert int(contact_count[0]) == 0


def test_limited_wall_contact_accepts_machine_precision_endpoint() -> None:
    """Не терять wall-limited LCFS из-за округления уровня в конце луча."""
    n = 61
    low = -2.5
    high = 2.5
    step = (high - low) / float(n - 1)
    axis_r = -0.10684214033956077
    axis_z = -0.2972622725413387
    center = (0.10370255334983935, 0.23625515968942373)
    grid = Grid2D(
        r=Grid1D(start=low, step=step, size=n, center=center[0]),
        z=Grid1D(start=low, step=step, size=n, center=center[1]),
    )
    R, Z = grid.mesh()
    x = R - axis_r
    y = Z - axis_z
    psi = (
        0.6204192564138505 * x * x
        + 1.5296155176434978 * y * y
        - 0.16192904602321279 * x * y
        + 0.022997683349685452 * x**4
        + 0.02613183375724458 * y**4
    )
    limiter = np.asarray(
        [
            [-1.8915409839922626, -1.9499216874110183],
            [2.191629459226713, -1.9499216874110183],
            [2.191629459226713, 1.7139099353302147],
            [-1.8915409839922626, 1.7139099353302147],
            [-1.8915409839922626, -1.9499216874110183],
        ],
        dtype=float,
    )
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    cpu = find_equilibrium_lcfs(
        psi,
        grid,
        center_hint=center,
        limiter_shape=limiter,
        fixed_angles=angles,
    )

    gpu_signal, topology_code, *_rest = _gpu_result(
        psi=psi,
        grid=grid,
        center=center,
        limiter=limiter,
        angles=angles,
        return_dense_boundary=True,
    )
    _points, radii, found, level = gpu_signal

    assert bool(found[0])
    assert int(topology_code[0]) == 1
    assert float(level[0]) == pytest.approx(cpu.psi_boundary, abs=5.0e-3)
    assert np.allclose(
        np.asarray(radii[0]),
        cpu.fixed_angle_projection.radii,
        atol=3.0e-3,
        rtol=0.0,
    )


def test_batched_single_null_topology_sweep_matches_cpu_reference() -> None:
    """Keep the full GPU level-set graph valid across a diverted transition sweep."""
    n = 121
    low = -2.5
    high = 2.5
    step = (high - low) / float(n - 1)
    center = (0.0, 1.0)
    grid = Grid2D(
        r=Grid1D(start=low, step=step, size=n, center=center[0]),
        z=Grid1D(start=low, step=step, size=n, center=center[1]),
    )
    R, Z = grid.mesh()
    limiter = _limiter()
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    tilts = np.linspace(-0.08, 0.08, 9, dtype=float)
    psi_batch = np.stack(
        [R * R + (Z * Z - 1.0) ** 2 + float(tilt) * Z for tilt in tilts],
        axis=0,
    )

    field = torch.as_tensor(psi_batch, dtype=torch.float64)
    limiter_t = torch.as_tensor(limiter, dtype=torch.float64)
    axis_points, axis_level, axis_kind, axis_valid = _axis_search(
        field,
        grid,
        center,
        limiter_t,
    )
    gpu = _equilibrium_lcfs_fixed_angle_search(
        psi=field,
        grid=grid,
        axis_points=axis_points,
        projection_center=torch.as_tensor([center], dtype=torch.float64).expand(len(tilts), -1),
        axis_level=axis_level,
        axis_kind=axis_kind,
        axis_valid=axis_valid,
        measurement_angles=torch.as_tensor(angles, dtype=torch.float64),
        validation_angles=torch.as_tensor(angles, dtype=torch.float64),
        dense_angles=torch.linspace(-float(np.pi), float(np.pi), 257, dtype=torch.float64)[:-1],
        limiter=limiter_t,
        limiter_samples=torch.as_tensor(
            _sample_closed_polyline_numpy(limiter, 512),
            dtype=torch.float64,
        ),
        ray_samples=512,
        return_dense_boundary=True,
    )
    signal, topology, selected_x, counts, boundary, boundary_count, _contacts, contact_count, quality = gpu
    _points, radii, found, levels = signal

    assert bool(torch.all(found))
    assert bool(torch.all(topology == 2))
    assert bool(torch.all(boundary_count == 257))
    assert bool(torch.all(torch.isfinite(radii)))
    assert bool(torch.all(counts == 1))
    assert bool(torch.all(torch.isfinite(quality)))
    assert bool(torch.all(contact_count == 0))
    assert bool(torch.all(torch.sum(torch.all(torch.isfinite(selected_x), dim=2), dim=1) == 1))
    assert bool(torch.allclose(boundary[:, 0, :], boundary[:, -1, :], atol=1.0e-12, rtol=0.0))

    for index, psi in enumerate(psi_batch):
        cpu = find_equilibrium_lcfs(
            psi,
            grid,
            center_hint=center,
            limiter_shape=limiter,
            fixed_angles=angles,
        )
        assert cpu.topology == "single_null"
        assert float(levels[index]) == pytest.approx(cpu.psi_boundary, abs=5.0e-3)
        assert np.allclose(
            np.asarray(radii[index]),
            cpu.fixed_angle_projection.radii,
            atol=2.0e-2,
            rtol=0.0,
        )


def test_gpu_lcfs_uses_level_set_graph_not_radial_candidate_definition() -> None:
    """Guard the production GPU path against reverting to radial LCFS selection."""
    import inspect

    source = inspect.getsource(_equilibrium_lcfs_fixed_angle_search)

    assert "_marching_squares_segments_gpu" in source
    assert "_trace_core_cycle_gpu" in source
    assert "_target_level_contact_gpu" not in source
    assert "_ray_crossings_with_counts" not in source
