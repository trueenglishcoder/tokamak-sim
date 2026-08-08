from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

torch = pytest.importorskip("torch")

from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.geometry import level_set_graph
import tokamak_control.geometry.boundary_gpu as boundary_gpu_module
from tokamak_control.geometry.boundary_gpu import (
    _equilibrium_lcfs_fixed_angle_search,
    _sample_closed_polyline_numpy,
    _trace_level_set_faces_gpu,
    fixed_angle_boundary_gpu,
    prepare_fixed_angle_boundary_gpu_geometry,
)
from tokamak_control.geometry.equilibrium_lcfs_gpu import (
    combine_exact_spline_gpu_field,
    evaluate_exact_spline,
    evaluate_exact_spline_value,
    evaluate_exact_spline_value_gradient,
    find_critical_points_exact_gpu,
    _exact_edge_roots_gpu,
    prepare_exact_spline_gpu_geometry,
)
from tokamak_control.geometry.equilibrium_field import EquilibriumField
from tokamak_control.geometry.lcfs import find_equilibrium_lcfs


FIXTURE = Path(__file__).with_name("data") / "equilibrium_lcfs_3863_regression.npz"


def _limiter() -> np.ndarray:
    return np.asarray(
        [
            [-2.2, -2.2],
            [2.2, -2.2],
            [2.2, 2.2],
            [-2.2, 2.2],
            [-2.2, -2.2],
        ],
        dtype=np.float64,
    )


def _analytic_case(name: str) -> tuple[Grid2D, np.ndarray, tuple[float, float], int]:
    size = 61
    lower = -2.5
    upper = 2.5
    step = (upper - lower) / float(size - 1)
    center = (0.0, 1.0) if name == "single_null" else (0.0, 0.0)
    grid = Grid2D(
        r=Grid1D(lower, step, size, center[0]),
        z=Grid1D(lower, step, size, center[1]),
    )
    r, z = grid.mesh()
    if name == "limited":
        psi = r * r + z * z
        topology_code = 1
    elif name == "single_null":
        psi = r * r + (z * z - 1.0) ** 2
        topology_code = 2
    elif name == "double_null":
        psi = r * r + z * z - 0.5 * z**4
        topology_code = 3
    else:
        raise AssertionError(name)
    return grid, psi, center, topology_code


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


def _prepare_tensor_lcfs_case(
    *,
    psi: np.ndarray,
    grid: Grid2D,
    center: tuple[float, float],
    limiter: np.ndarray,
    angles: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    samples = _sample_closed_polyline_numpy(limiter, 512)
    geometry = prepare_exact_spline_gpu_geometry(
        grid=grid,
        basis_fields=np.asarray(psi, dtype=np.float64)[None],
        limiter_samples=samples,
        limiter_poly=limiter,
        device=device,
        dtype=torch.float64,
    )
    angle_tensor = torch.as_tensor(angles, dtype=torch.float64, device=device)
    return {
        "psi_tensor": torch.as_tensor(psi[None], dtype=torch.float64, device=device),
        "amplitudes": torch.ones((1, 1), dtype=torch.float64, device=device),
        "geometry": geometry,
        "grid": grid,
        "center": center,
        "projection_center": torch.as_tensor([center], dtype=torch.float64, device=device),
        "angles": angle_tensor,
        "limiter": torch.as_tensor(limiter, dtype=torch.float64, device=device),
        "limiter_samples": torch.as_tensor(samples, dtype=torch.float64, device=device),
    }


def _run_prepared_tensor_lcfs(
    prepared: dict[str, object],
    *,
    return_dense_boundary: bool = False,
):
    psi_tensor = prepared["psi_tensor"]
    amplitudes = prepared["amplitudes"]
    geometry = prepared["geometry"]
    grid = prepared["grid"]
    center = prepared["center"]
    exact_field = combine_exact_spline_gpu_field(
        amplitudes=amplitudes,
        geometry=geometry,
    )
    (
        axis_points,
        axis_level,
        axis_kind,
        axis_valid,
        x_points,
        x_levels,
        x_valid,
    ) = find_critical_points_exact_gpu(
        psi=psi_tensor,
        field=exact_field,
        center_hint=center,
        max_candidates=256,
        max_o_points=16,
        max_x_points=32,
    )
    return _equilibrium_lcfs_fixed_angle_search(
        psi=psi_tensor,
        grid=grid,
        axis_points=axis_points,
        projection_center=prepared["projection_center"],
        axis_level=axis_level,
        axis_kind=axis_kind,
        axis_valid=axis_valid,
        measurement_angles=prepared["angles"],
        limiter=prepared["limiter"],
        exact_field=exact_field,
        critical_x_points=x_points,
        critical_x_levels=x_levels,
        critical_x_valid=x_valid,
        return_dense_boundary=return_dense_boundary,
    )


def _tensor_lcfs(
    *,
    psi: np.ndarray,
    grid: Grid2D,
    center: tuple[float, float],
    limiter: np.ndarray,
    angles: np.ndarray,
    device: torch.device,
    return_dense_boundary: bool = False,
):
    prepared = _prepare_tensor_lcfs_case(
        psi=psi,
        grid=grid,
        center=center,
        limiter=limiter,
        angles=angles,
        device=device,
    )
    return _run_prepared_tensor_lcfs(
        prepared,
        return_dense_boundary=return_dense_boundary,
    )






def _run_public_tensor_lcfs_cpu(
    *,
    psi: np.ndarray,
    grid: Grid2D,
    center: tuple[float, float],
    limiter: np.ndarray,
    angles: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    return_dense_boundary: bool = True,
):
    """Exercise the complete public API while keeping tensors on the test CPU."""
    monkeypatch.setattr(boundary_gpu_module, "_torch", lambda _device: torch)
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
        gpu_device="cpu",
        dtype=torch.float64,
        basis_fields=basis,
    )
    return fixed_angle_boundary_gpu(
        psi=torch.as_tensor(basis, dtype=torch.float64),
        grid=grid,
        center=center,
        angles_rad=angles,
        limiter_shape=limiter,
        boundary_mode="equilibrium_lcfs",
        gpu_device="cpu",
        prepared_geometry=geometry,
        amplitudes=torch.eye(lane_count, dtype=torch.float64),
        return_dense_boundary=return_dense_boundary,
    )


def _directed_vertex_to_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> float:
    source = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    target = level_set_graph.close_poly(np.asarray(polyline, dtype=np.float64).reshape(-1, 2))
    if source.shape[0] == 0 or target.shape[0] < 2:
        return float("inf")
    starts = target[:-1]
    vectors = target[1:] - starts
    denominator = np.sum(vectors * vectors, axis=1)
    relative = source[:, None, :] - starts[None, :, :]
    fraction = np.divide(
        np.sum(relative * vectors[None, :, :], axis=2),
        denominator[None, :],
        out=np.zeros((source.shape[0], starts.shape[0]), dtype=np.float64),
        where=denominator[None, :] > 0.0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    nearest = starts[None, :, :] + fraction[:, :, None] * vectors[None, :, :]
    distance = np.linalg.norm(source[:, None, :] - nearest, axis=2)
    return float(np.max(np.min(distance, axis=1)))


def _symmetric_polyline_hausdorff(first: np.ndarray, second: np.ndarray) -> float:
    return max(
        _directed_vertex_to_polyline_distance(first, second),
        _directed_vertex_to_polyline_distance(second, first),
    )



def _assert_unordered_points_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    tolerance: float,
) -> None:
    actual_points = np.asarray(actual, dtype=np.float64).reshape(-1, 2)
    expected_points = np.asarray(expected, dtype=np.float64).reshape(-1, 2)
    assert actual_points.shape[0] == expected_points.shape[0]
    if actual_points.shape[0] == 0:
        return
    distance = np.linalg.norm(
        actual_points[:, None, :] - expected_points[None, :, :], axis=2
    )
    assert float(np.max(np.min(distance, axis=1))) <= float(tolerance)
    assert float(np.max(np.min(distance, axis=0))) <= float(tolerance)

def _legacy_face_oracle(
    edges: list[level_set_graph.LevelSetEdge],
) -> list[level_set_graph.LevelSetCycle]:
    """The previously accepted self-loop plus DFS cycle enumeration.

    The oracle intentionally lives in the test suite.  Production code contains
    only the half-edge face traversal.
    """
    cycles: list[level_set_graph.LevelSetCycle] = []
    for index, edge in enumerate(edges):
        if edge.start_node is not None and edge.start_node == edge.end_node:
            cycles.append(
                level_set_graph.LevelSetCycle(
                    level_set_graph.close_poly(edge.points),
                    (index,),
                    (int(edge.start_node),),
                )
            )

    adjacency: dict[int, list[int]] = {}
    for index, edge in enumerate(edges):
        if edge.start_node is None or edge.end_node is None or edge.start_node == edge.end_node:
            continue
        adjacency.setdefault(int(edge.start_node), []).append(index)
        adjacency.setdefault(int(edge.end_node), []).append(index)

    signatures: set[tuple[int, ...]] = set()

    def visit(
        *,
        start: int,
        current: int,
        path: list[tuple[int, bool]],
        nodes: list[int],
        used: set[int],
    ) -> None:
        if len(path) >= max(len(edges), 3):
            return
        for edge_index in adjacency.get(current, []):
            if edge_index in used:
                continue
            edge = edges[edge_index]
            if edge.start_node == current:
                nxt = int(edge.end_node)
                forward = True
            elif edge.end_node == current:
                nxt = int(edge.start_node)
                forward = False
            else:
                continue
            new_path = path + [(edge_index, forward)]
            if nxt == start and len(new_path) >= 2:
                signature = tuple(sorted(index for index, _forward in new_path))
                if signature in signatures:
                    continue
                signatures.add(signature)
                chunks: list[np.ndarray] = []
                for selected_index, selected_forward in new_path:
                    points = np.asarray(edges[selected_index].points, dtype=float)
                    if not selected_forward:
                        points = points[::-1]
                    chunks.append(points if not chunks else points[1:])
                cycles.append(
                    level_set_graph.LevelSetCycle(
                        points=level_set_graph.close_poly(np.vstack(chunks)),
                        edge_indices=tuple(index for index, _forward in new_path),
                        node_indices=tuple(sorted(set(nodes))),
                    )
                )
                continue
            if nxt in nodes:
                continue
            visit(
                start=start,
                current=nxt,
                path=new_path,
                nodes=nodes + [nxt],
                used=used | {edge_index},
            )

    for start in adjacency:
        visit(start=start, current=start, path=[], nodes=[start], used=set())

    canonical: list[level_set_graph.LevelSetCycle] = []
    for cycle in cycles:
        points = level_set_graph.close_poly(cycle.points)
        if level_set_graph.poly_area(points) < 0.0:
            points = level_set_graph.close_poly(points[-2::-1])
        canonical.append(
            level_set_graph.LevelSetCycle(
                points=points,
                edge_indices=cycle.edge_indices,
                node_indices=cycle.node_indices,
            )
        )
    return canonical


def test_combined_exact_spline_matches_scipy_value_gradient_and_hessian() -> None:
    grid, first, _center, _topology = _analytic_case("single_null")
    r, z = grid.mesh()
    second = 0.3 * r**3 - 0.7 * r * z + 0.2 * z**2
    amplitudes_np = np.asarray([[0.73, -1.17]], dtype=np.float64)
    combined = amplitudes_np[0, 0] * first + amplitudes_np[0, 1] * second
    limiter = _limiter()
    samples = _sample_closed_polyline_numpy(limiter, 256)
    geometry = prepare_exact_spline_gpu_geometry(
        grid=grid,
        basis_fields=np.stack((first, second)),
        limiter_samples=samples,
        limiter_poly=limiter,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    field = combine_exact_spline_gpu_field(
        amplitudes=torch.as_tensor(amplitudes_np, dtype=torch.float64),
        geometry=geometry,
    )
    rng = np.random.default_rng(20260804)
    points_np = rng.uniform(-2.0, 2.0, size=(1, 128, 2))
    points = torch.as_tensor(points_np, dtype=torch.float64)
    value_only, contains_value = evaluate_exact_spline_value(field=field, points=points)
    value_gradient, gradient_only, contains_gradient = evaluate_exact_spline_value_gradient(
        field=field,
        points=points,
    )
    value, gradient, hessian, contains = evaluate_exact_spline(field=field, points=points)

    spline = RectBivariateSpline(
        np.asarray(grid.r.coords(), dtype=np.float64),
        np.asarray(grid.z.coords(), dtype=np.float64),
        combined.T,
        kx=3,
        ky=3,
        s=0.0,
    )
    x = points_np[0, :, 0]
    y = points_np[0, :, 1]
    expected_value = spline.ev(x, y)
    expected_gradient = np.column_stack((spline.ev(x, y, dx=1), spline.ev(x, y, dy=1)))
    expected_hessian = np.stack(
        (
            np.column_stack((spline.ev(x, y, dx=2), spline.ev(x, y, dx=1, dy=1))),
            np.column_stack((spline.ev(x, y, dx=1, dy=1), spline.ev(x, y, dy=2))),
        ),
        axis=1,
    )
    assert bool(torch.all(contains_value & contains_gradient & contains))
    np.testing.assert_allclose(value_only[0].numpy(), expected_value, atol=2.0e-12, rtol=0.0)
    np.testing.assert_allclose(value_gradient[0].numpy(), expected_value, atol=2.0e-12, rtol=0.0)
    np.testing.assert_allclose(value[0].numpy(), expected_value, atol=2.0e-12, rtol=0.0)
    np.testing.assert_allclose(gradient_only[0].numpy(), expected_gradient, atol=2.0e-10, rtol=0.0)
    np.testing.assert_allclose(gradient[0].numpy(), expected_gradient, atol=2.0e-10, rtol=0.0)
    np.testing.assert_allclose(hessian[0].numpy(), expected_hessian, atol=2.0e-8, rtol=0.0)


def test_compacted_exact_edge_roots_match_cpu_exhaustive_search() -> None:
    size = 17
    step = 2.0 / float(size - 1)
    grid = Grid2D(
        r=Grid1D(-1.0, step, size, 0.0),
        z=Grid1D(-1.0, step, size, 0.0),
    )
    r, z = grid.mesh()
    psi = (r + 0.70) * (r + 0.10) * (r - 0.43) + 0.0 * z
    limiter = _limiter()
    samples = _sample_closed_polyline_numpy(limiter, 256)
    geometry = prepare_exact_spline_gpu_geometry(
        grid=grid,
        basis_fields=psi[None],
        limiter_samples=samples,
        limiter_poly=limiter,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    field = combine_exact_spline_gpu_field(
        amplitudes=torch.ones((1, 1), dtype=torch.float64),
        geometry=geometry,
    )
    start = np.asarray([-1.0, 0.0], dtype=np.float64)
    end = np.asarray([1.0, 0.0], dtype=np.float64)
    cpu_field = EquilibriumField(grid=grid, psi=psi)
    cpu_roots = level_set_graph._edge_roots(
        cpu_field,
        start,
        end,
        0.0,
        1.0e-12,
        exhaustive=True,
    )
    start_value = float(cpu_field.value(start)[0])
    end_value = float(cpu_field.value(end)[0])
    _points, roots_t, _nodes, root_count = _exact_edge_roots_gpu(
        field=field,
        starts=torch.as_tensor(np.asarray([[start]], dtype=np.float64)),
        ends=torch.as_tensor(np.asarray([[end]], dtype=np.float64)),
        start_values=torch.as_tensor([[start_value]], dtype=torch.float64),
        end_values=torch.as_tensor([[end_value]], dtype=torch.float64),
        edge_valid=torch.ones((1, 1), dtype=torch.bool),
        exhaustive=torch.ones((1, 1), dtype=torch.bool),
        level=torch.zeros((1,), dtype=torch.float64),
        tolerance=torch.full((1,), 1.0e-12, dtype=torch.float64),
        start_vertex=torch.as_tensor([0], dtype=torch.int64),
        end_vertex=torch.as_tensor([1], dtype=torch.int64),
        interior_node_base=2,
        root_iterations=20,
        max_roots=4,
    )
    count = int(root_count[0, 0])
    assert count == len(cpu_roots) == 3
    np.testing.assert_allclose(
        roots_t[0, 0, :count].numpy(),
        np.asarray([root[0] for root in cpu_roots], dtype=np.float64),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_compacted_exact_edge_roots_handle_ordinary_and_endpoint_roots() -> None:
    grid, base, _center, _topology = _analytic_case("limited")
    r, _z = grid.mesh()
    field_values = r - 0.25
    limiter = _limiter()
    samples = _sample_closed_polyline_numpy(limiter, 128)
    geometry = prepare_exact_spline_gpu_geometry(
        grid=grid,
        basis_fields=field_values[None],
        limiter_samples=samples,
        limiter_poly=limiter,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    field = combine_exact_spline_gpu_field(
        amplitudes=torch.ones((1, 1), dtype=torch.float64),
        geometry=geometry,
    )
    starts = torch.as_tensor([[[-1.0, 0.0], [0.25, -1.0]]], dtype=torch.float64)
    ends = torch.as_tensor([[[1.0, 0.0], [1.0, -1.0]]], dtype=torch.float64)
    start_values, _ = evaluate_exact_spline_value(field=field, points=starts)
    end_values, _ = evaluate_exact_spline_value(field=field, points=ends)
    _points, roots_t, _nodes, root_count = _exact_edge_roots_gpu(
        field=field,
        starts=starts,
        ends=ends,
        start_values=start_values,
        end_values=end_values,
        edge_valid=torch.ones((1, 2), dtype=torch.bool),
        exhaustive=torch.zeros((1, 2), dtype=torch.bool),
        level=torch.zeros((1,), dtype=torch.float64),
        tolerance=torch.full((1,), 1.0e-12, dtype=torch.float64),
        start_vertex=torch.as_tensor([0, 2], dtype=torch.int64),
        end_vertex=torch.as_tensor([1, 3], dtype=torch.int64),
        interior_node_base=4,
        root_iterations=20,
        max_roots=4,
    )
    assert root_count.tolist() == [[1, 1]]
    assert float(roots_t[0, 0, 0]) == pytest.approx(0.625, abs=1.0e-6)
    assert float(roots_t[0, 1, 0]) == pytest.approx(0.0, abs=1.0e-12)


@pytest.mark.parametrize("name", ["limited", "single_null", "double_null"])
def test_halfedge_cpu_path_matches_legacy_dfs_oracle_analytic(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid, psi, center, _topology = _analytic_case(name)
    limiter = _limiter()
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=np.float64)
    halfedge = find_equilibrium_lcfs(
        psi,
        grid,
        center_hint=center,
        limiter_shape=limiter,
        fixed_angles=angles,
    )
    monkeypatch.setattr(level_set_graph, "_enumerate_graph_faces", _legacy_face_oracle)
    legacy = find_equilibrium_lcfs(
        psi,
        grid,
        center_hint=center,
        limiter_shape=limiter,
        fixed_angles=angles,
    )
    assert halfedge.topology == legacy.topology
    assert halfedge.psi_boundary == pytest.approx(legacy.psi_boundary, abs=1.0e-13)
    np.testing.assert_allclose(
        halfedge.fixed_angle_projection.radii,
        legacy.fixed_angle_projection.radii,
        atol=1.0e-12,
        rtol=0.0,
    )


def test_halfedge_cpu_path_matches_legacy_dfs_oracle_shot_3863(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = np.load(FIXTURE, allow_pickle=False)
    grid = _fixture_grid(data)
    center = tuple(float(value) for value in data["center"])
    halfedge_results = [
        find_equilibrium_lcfs(
            psi,
            grid,
            center_hint=center,
            limiter_shape=data["limiter"],
            fixed_angles=data["angles"],
        )
        for psi in data["psi"]
    ]
    monkeypatch.setattr(level_set_graph, "_enumerate_graph_faces", _legacy_face_oracle)
    legacy_results = [
        find_equilibrium_lcfs(
            psi,
            grid,
            center_hint=center,
            limiter_shape=data["limiter"],
            fixed_angles=data["angles"],
        )
        for psi in data["psi"]
    ]
    for step, halfedge, legacy in zip(data["steps"], halfedge_results, legacy_results, strict=True):
        assert halfedge.topology == legacy.topology, int(step)
        assert halfedge.psi_boundary == pytest.approx(legacy.psi_boundary, abs=1.0e-13), int(step)
        np.testing.assert_allclose(
            halfedge.fixed_angle_projection.radii,
            legacy.fixed_angle_projection.radii,
            atol=1.0e-12,
            rtol=0.0,
            err_msg=f"step {int(step)}",
        )


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("name", ["limited", "single_null", "double_null"])
def test_tensor_topology_path_matches_cpu_analytic(name: str, sign: float) -> None:
    grid, psi, center, expected_topology = _analytic_case(name)
    psi = float(sign) * psi
    limiter = _limiter()
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=np.float64)
    cpu = find_equilibrium_lcfs(
        psi,
        grid,
        center_hint=center,
        limiter_shape=limiter,
        fixed_angles=angles,
    )
    signal, topology_code, *_rest = _tensor_lcfs(
        psi=psi,
        grid=grid,
        center=center,
        limiter=limiter,
        angles=angles,
        device=torch.device("cpu"),
    )
    _points, radii, found, level = signal
    assert found.tolist() == [True]
    assert topology_code.tolist() == [expected_topology]
    assert float(level[0]) == pytest.approx(cpu.psi_boundary, abs=2.0e-8)
    np.testing.assert_allclose(
        radii[0].numpy(),
        cpu.fixed_angle_projection.radii,
        atol=2.0e-6,
        rtol=0.0,
    )


@pytest.mark.parametrize("step", [0, 1, 200, 400, 440, 894, 895, 896, 897, 900, 1000, 1192])
def test_tensor_topology_path_matches_cpu_shot_3863(step: int) -> None:
    data = np.load(FIXTURE, allow_pickle=False)
    grid = _fixture_grid(data)
    index = data["steps"].tolist().index(step)
    psi = data["psi"][index]
    center = tuple(float(value) for value in data["center"])
    cpu = find_equilibrium_lcfs(
        psi,
        grid,
        center_hint=center,
        limiter_shape=data["limiter"],
        fixed_angles=data["angles"],
    )
    signal, topology_code, selected_x, counts, *_rest = _tensor_lcfs(
        psi=psi,
        grid=grid,
        center=center,
        limiter=data["limiter"],
        angles=data["angles"],
        device=torch.device("cpu"),
    )
    _points, radii, found, level = signal
    expected_code = {"limited": 1, "single_null": 2, "double_null": 3, "multi_null": 4}[
        cpu.topology
    ]
    assert found.tolist() == [True]
    assert topology_code.tolist() == [expected_code]
    assert float(level[0]) == pytest.approx(cpu.psi_boundary, abs=5.0e-7)
    assert int(torch.sum(torch.all(torch.isfinite(selected_x[0]), dim=1))) == len(cpu.x_points)
    np.testing.assert_array_equal(counts[0].numpy(), cpu.fixed_angle_projection.intersection_counts)
    np.testing.assert_allclose(
        radii[0].numpy(),
        cpu.fixed_angle_projection.radii,
        atol=2.0e-6,
        rtol=0.0,
    )




@pytest.mark.parametrize(
    "step", [0, 1, 200, 400, 440, 894, 895, 896, 897, 900, 1000, 1192]
)
def test_tensor_critical_points_and_contacts_match_cpu_shot_3863(step: int) -> None:
    data = np.load(FIXTURE, allow_pickle=False)
    grid = _fixture_grid(data)
    index = data["steps"].tolist().index(step)
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
    prepared = _prepare_tensor_lcfs_case(
        psi=psi,
        grid=grid,
        center=center,
        limiter=limiter,
        angles=angles,
        device=torch.device("cpu"),
    )
    exact_field = combine_exact_spline_gpu_field(
        amplitudes=prepared["amplitudes"],
        geometry=prepared["geometry"],
    )
    (
        axis_points,
        axis_level,
        _axis_kind,
        axis_valid,
        _x_points,
        _x_levels,
        _x_valid,
    ) = find_critical_points_exact_gpu(
        psi=prepared["psi_tensor"],
        field=exact_field,
        center_hint=center,
        max_candidates=256,
        max_o_points=16,
        max_x_points=32,
    )
    assert axis_valid.tolist() == [True]
    np.testing.assert_allclose(
        axis_points[0].numpy(),
        np.asarray(cpu.magnetic_axis.point, dtype=np.float64),
        atol=2.0e-6,
        rtol=0.0,
    )
    assert float(axis_level[0]) == pytest.approx(cpu.psi_axis, abs=5.0e-7)

    result = _run_prepared_tensor_lcfs(
        prepared, return_dense_boundary=True
    )
    selected_x = result[2][0]
    selected_x = selected_x[torch.all(torch.isfinite(selected_x), dim=1)].numpy()
    _assert_unordered_points_close(
        selected_x,
        np.asarray([point.point for point in cpu.x_points], dtype=np.float64).reshape(-1, 2),
        tolerance=2.0e-6,
    )
    limiter_contacts = result[6][0]
    limiter_contact_count = int(result[7][0])
    gpu_contacts = limiter_contacts[:limiter_contact_count].numpy()
    _assert_unordered_points_close(
        gpu_contacts,
        np.asarray([contact.point for contact in cpu.limiter_contacts], dtype=np.float64).reshape(-1, 2),
        tolerance=2.0e-6,
    )


@pytest.mark.parametrize("step", [200, 895, 896, 1192])
def test_tensor_dense_lcfs_matches_cpu_graph_geometry(step: int) -> None:
    data = np.load(FIXTURE, allow_pickle=False)
    grid = _fixture_grid(data)
    index = data["steps"].tolist().index(step)
    psi = data["psi"][index]
    center = tuple(float(value) for value in data["center"])
    cpu = find_equilibrium_lcfs(
        psi,
        grid,
        center_hint=center,
        limiter_shape=data["limiter"],
        fixed_angles=data["angles"],
    )
    prepared = _prepare_tensor_lcfs_case(
        psi=psi,
        grid=grid,
        center=center,
        limiter=data["limiter"],
        angles=data["angles"],
        device=torch.device("cpu"),
    )
    result = _run_prepared_tensor_lcfs(
        prepared,
        return_dense_boundary=True,
    )
    core_boundary = result[4]
    core_boundary_count = result[5]
    count = int(core_boundary_count[0])
    assert count >= 4
    gpu_boundary = core_boundary[0, :count].numpy()
    assert np.all(np.isfinite(gpu_boundary))
    assert np.linalg.norm(gpu_boundary[0] - gpu_boundary[-1]) <= 1.0e-12
    distance = _symmetric_polyline_hausdorff(cpu.core_boundary, gpu_boundary)
    assert distance <= 2.0e-6, f"step {step}: dense LCFS Hausdorff distance {distance:.9e} m"


def test_tensor_topology_path_handles_heterogeneous_batch() -> None:
    data = np.load(FIXTURE, allow_pickle=False)
    grid = _fixture_grid(data)
    selected_steps = [200, 896]
    indices = [data["steps"].tolist().index(step) for step in selected_steps]
    psi = np.asarray(data["psi"][indices], dtype=np.float64)
    center = tuple(float(value) for value in data["center"])
    limiter = np.asarray(data["limiter"], dtype=np.float64)
    angles = np.asarray(data["angles"], dtype=np.float64)
    samples = _sample_closed_polyline_numpy(limiter, 512)
    geometry = prepare_exact_spline_gpu_geometry(
        grid=grid,
        basis_fields=psi,
        limiter_samples=samples,
        limiter_poly=limiter,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    exact_field = combine_exact_spline_gpu_field(
        amplitudes=torch.eye(2, dtype=torch.float64),
        geometry=geometry,
    )
    psi_tensor = torch.as_tensor(psi, dtype=torch.float64)
    (
        axis_points,
        axis_level,
        axis_kind,
        axis_valid,
        x_points,
        x_levels,
        x_valid,
    ) = find_critical_points_exact_gpu(
        psi=psi_tensor,
        field=exact_field,
        center_hint=center,
        max_candidates=256,
        max_o_points=16,
        max_x_points=32,
    )
    result = _equilibrium_lcfs_fixed_angle_search(
        psi=psi_tensor,
        grid=grid,
        axis_points=axis_points,
        projection_center=torch.as_tensor([center, center], dtype=torch.float64),
        axis_level=axis_level,
        axis_kind=axis_kind,
        axis_valid=axis_valid,
        measurement_angles=torch.as_tensor(angles, dtype=torch.float64),
        limiter=torch.as_tensor(limiter, dtype=torch.float64),
        exact_field=exact_field,
        critical_x_points=x_points,
        critical_x_levels=x_levels,
        critical_x_valid=x_valid,
        return_dense_boundary=True,
    )
    signal, topology_code, _selected_x, counts, core_boundary, core_count, *_rest = result
    _points, radii, found, levels = signal
    assert found.tolist() == [True, True]
    assert topology_code.tolist() == [1, 2]
    for lane, index in enumerate(indices):
        cpu = find_equilibrium_lcfs(
            data["psi"][index],
            grid,
            center_hint=center,
            limiter_shape=limiter,
            fixed_angles=angles,
        )
        assert float(levels[lane]) == pytest.approx(cpu.psi_boundary, abs=5.0e-7)
        np.testing.assert_array_equal(
            counts[lane].numpy(),
            cpu.fixed_angle_projection.intersection_counts,
        )
        np.testing.assert_allclose(
            radii[lane].numpy(),
            cpu.fixed_angle_projection.radii,
            atol=2.0e-6,
            rtol=0.0,
        )
        count = int(core_count[lane])
        assert count >= 4
        distance = _symmetric_polyline_hausdorff(
            cpu.core_boundary,
            core_boundary[lane, :count].numpy(),
        )
        assert distance <= 2.0e-6

def _trace_handcrafted_graph(
    *,
    coordinates: list[tuple[float, float]],
    edges: list[tuple[int, int]],
    x_nodes: list[int],
    axis: tuple[float, float],
    require_x: bool,
):
    coordinate_tensor = torch.as_tensor(coordinates, dtype=torch.float64)
    edge_tensor = torch.as_tensor(edges, dtype=torch.int64)
    segment_points = coordinate_tensor[edge_tensor][None, :, :, :]
    segment_nodes = edge_tensor[None, :, :]
    segment_valid = torch.ones((1, len(edges)), dtype=torch.bool)
    node_valid = torch.ones((1, len(coordinates)), dtype=torch.bool)
    node_is_x = torch.zeros_like(node_valid)
    node_x_slot = torch.full((1, len(coordinates)), -1, dtype=torch.int64)
    for slot, node in enumerate(x_nodes):
        node_is_x[0, node] = True
        node_x_slot[0, node] = slot
    grid = Grid2D(
        r=Grid1D(-4.0, 0.1, 81, 0.0),
        z=Grid1D(-4.0, 0.1, 81, 0.0),
    )
    limiter = torch.as_tensor(
        [[-3.0, -3.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0], [-3.0, -3.0]],
        dtype=torch.float64,
    )
    return _trace_level_set_faces_gpu(
        segment_points=segment_points,
        segment_nodes=segment_nodes,
        segment_valid=segment_valid,
        node_valid=node_valid,
        node_is_x=node_is_x,
        node_x_slot=node_x_slot,
        axis_points=torch.as_tensor([axis], dtype=torch.float64),
        require_x=torch.as_tensor([require_x], dtype=torch.bool),
        limiter=limiter,
        grid=grid,
        x_count=len(x_nodes),
    )


@pytest.mark.parametrize(
    ("coordinates", "edges", "x_nodes", "axis", "require_x", "expected_x_count"),
    [
        (
            [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
            [(0, 1), (1, 2), (2, 3), (3, 0)],
            [],
            (0.0, 0.0),
            False,
            0,
        ),
        (
            [(0.0, 0.0), (-1.0, 1.0), (-2.0, 0.0), (-1.0, -1.0), (1.0, 1.0), (1.0, -1.0)],
            [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (0, 5)],
            [0],
            (-1.0, 0.0),
            True,
            1,
        ),
        (
            [(0.0, 1.0), (0.0, -1.0), (-1.0, 0.0), (1.0, 0.0), (-1.0, 2.0), (1.0, 2.0), (-1.0, -2.0), (1.0, -2.0)],
            [(0, 2), (2, 1), (1, 3), (3, 0), (0, 4), (0, 5), (1, 6), (1, 7)],
            [0, 1],
            (0.0, 0.0),
            True,
            2,
        ),
        (
            [
                (0.0, 1.0),
                (-1.0, -1.0),
                (1.0, -1.0),
                (-1.0, 2.0),
                (1.0, 2.0),
                (-2.0, -1.0),
                (-1.0, -2.0),
                (2.0, -1.0),
                (1.0, -2.0),
            ],
            [
                (0, 1),
                (1, 2),
                (2, 0),
                (0, 3),
                (0, 4),
                (1, 5),
                (1, 6),
                (2, 7),
                (2, 8),
            ],
            [0, 1, 2],
            (0.0, -0.25),
            True,
            3,
        ),
    ],
    ids=[
        "regular",
        "single-null-open-branches",
        "double-null-open-branches",
        "multi-null-open-branches",
    ],
)
def test_tensor_halfedge_traversal_selects_axis_face_on_handcrafted_graphs(
    coordinates: list[tuple[float, float]],
    edges: list[tuple[int, int]],
    x_nodes: list[int],
    axis: tuple[float, float],
    require_x: bool,
    expected_x_count: int,
) -> None:
    boundary, count, found, used_x = _trace_handcrafted_graph(
        coordinates=coordinates,
        edges=edges,
        x_nodes=x_nodes,
        axis=axis,
        require_x=require_x,
    )
    assert found.tolist() == [True]
    assert int(count[0]) >= 4
    assert int(torch.sum(used_x[0]).item()) == expected_x_count
    points = boundary[0, : int(count[0])].numpy()
    assert level_set_graph.poly_area(points) > 0.0
    assert bool(
        level_set_graph.points_in_or_on_polygon(
            np.asarray(axis, dtype=np.float64)[None, :],
            points,
            tol=1.0e-12,
        )[0]
    )


def test_tensor_halfedge_rejects_open_graph_without_bounded_axis_face() -> None:
    _boundary, count, found, used_x = _trace_handcrafted_graph(
        coordinates=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        edges=[(0, 1), (1, 2)],
        x_nodes=[],
        axis=(1.0, 0.1),
        require_x=False,
    )
    assert found.tolist() == [False]
    assert count.tolist() == [0]
    assert int(used_x.numel()) == 0


@pytest.mark.parametrize("name", ["limited", "single_null", "double_null"])
def test_public_tensor_api_matches_cpu_analytic(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid, psi, center, expected_topology = _analytic_case(name)
    limiter = _limiter()
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=np.float64)
    cpu = find_equilibrium_lcfs(
        psi,
        grid,
        center_hint=center,
        limiter_shape=limiter,
        fixed_angles=angles,
    )
    result = _run_public_tensor_lcfs_cpu(
        psi=psi,
        grid=grid,
        center=center,
        limiter=limiter,
        angles=angles,
        monkeypatch=monkeypatch,
    )
    assert result.found.tolist() == [True]
    assert result.projection_valid.tolist() == [True]
    assert result.topology_code.tolist() == [expected_topology]
    assert float(result.level[0]) == pytest.approx(cpu.psi_boundary, abs=5.0e-7)
    np.testing.assert_array_equal(
        result.intersection_counts[0].numpy(),
        cpu.fixed_angle_projection.intersection_counts,
    )
    np.testing.assert_allclose(
        result.radii[0].numpy(),
        cpu.fixed_angle_projection.radii,
        atol=2.0e-6,
        rtol=0.0,
    )
    count = int(result.core_boundary_count[0])
    assert count >= 4
    assert (
        _symmetric_polyline_hausdorff(
            cpu.core_boundary,
            result.core_boundary[0, :count].numpy(),
        )
        <= 2.0e-6
    )


def test_public_tensor_api_matches_cpu_heterogeneous_3863_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = np.load(FIXTURE, allow_pickle=False)
    grid = _fixture_grid(data)
    selected_steps = [200, 895, 896, 1192]
    indices = [data["steps"].tolist().index(step) for step in selected_steps]
    psi = np.asarray(data["psi"][indices], dtype=np.float64)
    center = tuple(float(value) for value in data["center"])
    limiter = np.asarray(data["limiter"], dtype=np.float64)
    angles = np.asarray(data["angles"], dtype=np.float64)
    result = _run_public_tensor_lcfs_cpu(
        psi=psi,
        grid=grid,
        center=center,
        limiter=limiter,
        angles=angles,
        monkeypatch=monkeypatch,
    )
    assert result.found.tolist() == [True, True, True, True]
    assert result.projection_valid.tolist() == [True, True, True, True]
    assert result.topology_code.tolist() == [1, 1, 2, 2]
    for lane, index in enumerate(indices):
        cpu = find_equilibrium_lcfs(
            np.asarray(data["psi"][index], dtype=np.float64),
            grid,
            center_hint=center,
            limiter_shape=limiter,
            fixed_angles=angles,
        )
        assert float(result.level[lane]) == pytest.approx(
            cpu.psi_boundary, abs=5.0e-7
        )
        np.testing.assert_array_equal(
            result.intersection_counts[lane].numpy(),
            cpu.fixed_angle_projection.intersection_counts,
        )
        np.testing.assert_allclose(
            result.radii[lane].numpy(),
            cpu.fixed_angle_projection.radii,
            atol=2.0e-6,
            rtol=0.0,
        )
        count = int(result.core_boundary_count[lane])
        assert count >= 4
        assert (
            _symmetric_polyline_hausdorff(
                cpu.core_boundary,
                result.core_boundary[lane, :count].numpy(),
            )
            <= 2.0e-6
        )


def test_runtime_source_has_one_bounded_halfedge_algorithm() -> None:
    import tokamak_control.geometry.boundary_gpu as boundary_gpu
    import tokamak_control.geometry.equilibrium_lcfs_gpu as equilibrium_gpu

    boundary_source = inspect.getsource(boundary_gpu)
    equilibrium_source = inspect.getsource(equilibrium_gpu)
    level_graph_source = inspect.getsource(level_set_graph)
    source = boundary_source + equilibrium_source
    forbidden = (
        "_cycle_subset_masks",
        "1 << edge_count",
        "enumerate_branch_cycles_dfs_gpu",
        "_trace_cpu_x_cycles_gpu",
        "_trace_cpu_regular_cycles_gpu",
        "_core_component_valid_cpu_exact_gpu",
        "boundary_gpu_canonical",
        "extract_native_lcfs_gpu",
        "cpp_extension",
        "max_steps_per_start",
        "for candidate_index in range(flat_count)",
        "for _ in range(nr + nz)",
        "for angle_index in range",
        "_enumerate_graph_cycles",
        "_dfs_cycles",
    )
    for marker in forbidden:
        assert marker not in source + level_graph_source

    candidate_search_source = inspect.getsource(
        boundary_gpu._equilibrium_lcfs_fixed_angle_search_impl
    )
    compact_graph_source = inspect.getsource(boundary_gpu._compact_level_set_segments_gpu)
    face_trace_source = inspect.getsource(boundary_gpu._trace_level_set_faces_gpu)
    limiter_source = inspect.getsource(equilibrium_gpu.limiter_flux_candidates_exact_gpu)
    root_source = inspect.getsource(equilibrium_gpu._exact_edge_roots_gpu)
    x_group_source = inspect.getsource(boundary_gpu._group_xpoint_candidates_exact_gpu)
    # Every host synchronization has one explicit capacity/control purpose.  A
    # total-count assertion ensures a future edit cannot add a hidden sync in an
    # unreviewed helper while preserving the per-function checks below.
    assert candidate_search_source.count(".item()") == 2
    assert compact_graph_source.count(".item()") == 1
    assert face_trace_source.count(".item()") == 1
    assert limiter_source.count(".item()") == 3
    assert root_source.count(".item()") == 2
    assert x_group_source.count(".item()") == 1
    assert source.count(".item()") == 10
    assert ".cpu()" not in source
    assert ".numpy()" not in source
    assert "repeat_exact_spline_gpu_field" in boundary_source
    assert "candidate_block_size" in boundary_source
    assert "clockwise predecessor" in boundary_source
    assert "torch.einsum" not in inspect.getsource(equilibrium_gpu.evaluate_exact_spline_value)
