"""FLUSH-style flux-surface graph on the same spline used for critical points.

The equilibrium is represented by one bicubic ``EquilibriumField``.  A tailored
rectangular topology grid is built for every requested surface.  The magnetic
axis is inserted on grid lines and every selected X-point is placed near the
centre of one element.  Surface intersections are solved on element edges from
the bicubic field itself.  Four-intersection X-point elements are split into
four branches meeting at the explicit saddle node before the global graph is
assembled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from tokamak_control.geometry.boundary_common import (
    close_poly,
    encloses_center,
    points_in_or_on_polygon,
    poly_area,
)
from tokamak_control.geometry.critical_points import CriticalPoint
from tokamak_control.geometry.equilibrium_field import EquilibriumField


@dataclass(frozen=True, slots=True, repr=True)
class LevelSetEdge:
    start_node: int | None
    end_node: int | None
    points: np.ndarray


@dataclass(frozen=True, slots=True, repr=True)
class LevelSetCycle:
    points: np.ndarray
    edge_indices: tuple[int, ...]
    node_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True, repr=True)
class LevelSetGraphResult:
    core_boundary: np.ndarray
    separatrix_branches: tuple[np.ndarray, ...]
    used_x_points: tuple[int, ...]
    all_edges: tuple[LevelSetEdge, ...]


@dataclass(frozen=True, slots=True, repr=True)
class TopologyGrid:
    r: np.ndarray
    z: np.ndarray
    psi: np.ndarray


def extract_core_level_set(
    field: EquilibriumField,
    *,
    level: float,
    axis: tuple[float, float],
    limiter_poly: np.ndarray,
    x_points: tuple[CriticalPoint, ...] = (),
    require_x_cycle: bool = False,
    refinement: int = 2,
) -> LevelSetGraphResult | None:
    """Extract the axis-enclosing cycle of ``psi = level``.

    ``refinement=2`` gives a topology grid with half the native R and Z spacing.
    Selected X-points are centred in local cells, which is the key robustness
    condition used by FLUSH near a separatrix.
    """
    topology_grid = _build_topology_grid(
        field,
        axis=axis,
        x_points=x_points,
        refinement=max(int(refinement), 1),
        limiter_poly=limiter_poly,
    )
    segments, node_points = _surface_segments(
        field,
        topology_grid,
        float(level),
        x_points=x_points,
    )
    if not segments:
        return None
    edges = _compress_segments(segments, node_points)
    if not edges:
        return None

    cycles: list[LevelSetCycle] = []
    for index, edge in enumerate(edges):
        closed_regular = edge.start_node is None and edge.end_node is None and _is_closed(edge.points)
        if closed_regular:
            cycles.append(LevelSetCycle(close_poly(edge.points), (index,), ()))
        elif edge.start_node is not None and edge.start_node == edge.end_node:
            cycles.append(
                LevelSetCycle(
                    close_poly(edge.points),
                    (index,),
                    (int(edge.start_node),),
                )
            )
    cycles.extend(_enumerate_graph_cycles(edges))

    valid: list[LevelSetCycle] = []
    limiter_tolerance = 0.35 * field.grid_scale
    for cycle in cycles:
        points = close_poly(np.asarray(cycle.points, dtype=float))
        if points.shape[0] < 4:
            continue
        if require_x_cycle and not cycle.node_indices:
            continue
        if not encloses_center(points, axis):
            continue
        if not bool(np.all(points_in_or_on_polygon(points[:-1], limiter_poly, tol=limiter_tolerance))):
            continue
        area = abs(poly_area(points))
        if np.isfinite(area) and area > 0.0:
            valid.append(LevelSetCycle(points, cycle.edge_indices, cycle.node_indices))
    if not valid:
        return None

    selected = max(valid, key=lambda cycle: abs(poly_area(cycle.points)))
    selected_boundary = close_poly(np.asarray(selected.points, dtype=float))
    used_edges = set(selected.edge_indices)
    branches = tuple(
        np.asarray(edge.points, dtype=float).copy()
        for index, edge in enumerate(edges)
        if index not in used_edges and edge.points.shape[0] >= 2
    )
    return LevelSetGraphResult(
        core_boundary=selected_boundary,
        separatrix_branches=branches,
        used_x_points=tuple(sorted(set(selected.node_indices))),
        all_edges=tuple(edges),
    )


def _build_topology_grid(
    field: EquilibriumField,
    *,
    axis: tuple[float, float],
    x_points: tuple[CriticalPoint, ...],
    refinement: int,
    limiter_poly: np.ndarray | None = None,
) -> TopologyGrid:
    native_r = np.asarray(field.grid.r.coords(), dtype=float)
    native_z = np.asarray(field.grid.z.coords(), dtype=float)
    r_step = abs(float(field.grid.r.step)) / float(refinement)
    z_step = abs(float(field.grid.z.step)) / float(refinement)
    r_lower = float(native_r[0])
    r_upper = float(native_r[-1])
    z_lower = float(native_z[0])
    z_upper = float(native_z[-1])
    if limiter_poly is not None:
        limiter = np.asarray(limiter_poly, dtype=float).reshape(-1, 2)
        padding = 1.25 * field.grid_scale
        r_lower = max(r_lower, float(np.min(limiter[:, 0])) - padding)
        r_upper = min(r_upper, float(np.max(limiter[:, 0])) + padding)
        z_lower = max(z_lower, float(np.min(limiter[:, 1])) - padding)
        z_upper = min(z_upper, float(np.max(limiter[:, 1])) + padding)
    r = _tailored_coordinates(
        r_lower,
        r_upper,
        r_step,
        nodes=(float(axis[0]),),
        centers=tuple(float(point.point[0]) for point in x_points),
    )
    z = _tailored_coordinates(
        z_lower,
        z_upper,
        z_step,
        nodes=(float(axis[1]),),
        centers=tuple(float(point.point[1]) for point in x_points),
    )
    R, Z = np.meshgrid(r, z, indexing="xy")
    points = np.column_stack((R.reshape(-1), Z.reshape(-1)))
    psi = field.value(points).reshape(z.size, r.size)
    return TopologyGrid(r=r, z=z, psi=psi)


def _tailored_coordinates(
    lower: float,
    upper: float,
    step: float,
    *,
    nodes: tuple[float, ...],
    centers: tuple[float, ...],
) -> np.ndarray:
    if upper <= lower or step <= 0.0:
        raise ValueError("invalid topology-grid bounds or step")
    # A fixed irrational-like phase prevents ordinary tangencies from landing
    # exactly on topology-grid vertices.  The physical domain endpoints remain
    # exact, while interior spacing never exceeds the requested step.
    phase = 0.3819660112501051
    values = [float(lower), float(upper)]
    coordinate = float(lower) + phase * float(step)
    while coordinate < float(upper):
        values.append(float(coordinate))
        coordinate += float(step)

    # Reserve one full local element around each X-point.  Processing points in
    # coordinate order keeps disjoint X-point cells stable for double-null cases.
    for center in sorted(centers):
        if not (lower < center < upper):
            continue
        half = 0.5 * step
        left = max(lower, center - half)
        right = min(upper, center + half)
        if right - left < 0.5 * step:
            continue
        values = [value for value in values if not (left < value < right)]
        values.extend((left, right))

    for node in nodes:
        if lower < node < upper:
            # Do not split a deliberately centred X-point cell.
            in_reserved = any(abs(node - center) < 0.49 * step for center in centers)
            if not in_reserved:
                values.append(float(node))

    ordered = sorted(values)
    unique: list[float] = []
    tolerance = 1.0e-12 * max(abs(lower), abs(upper), 1.0)
    for value in ordered:
        clipped = float(np.clip(value, lower, upper))
        if not unique or abs(clipped - unique[-1]) > tolerance:
            unique.append(clipped)
    return np.asarray(unique, dtype=float)


def _surface_segments(
    field: EquilibriumField,
    topology_grid: TopologyGrid,
    level: float,
    *,
    x_points: tuple[CriticalPoint, ...],
) -> tuple[list[tuple[tuple, tuple]], dict[tuple, np.ndarray]]:
    r = topology_grid.r
    z = topology_grid.z
    psi = topology_grid.psi
    flux_tolerance = max(
        2.0e-12 * field.flux_scale,
        128.0 * np.finfo(float).eps * max(abs(float(level)), 1.0),
    )

    x_by_cell: dict[tuple[int, int], tuple[int, CriticalPoint]] = {}
    for index, point in enumerate(x_points):
        i = int(np.clip(np.searchsorted(r, float(point.point[0]), side="right") - 1, 0, r.size - 2))
        j = int(np.clip(np.searchsorted(z, float(point.point[1]), side="right") - 1, 0, z.size - 2))
        if abs(float(point.level) - float(level)) <= 8.0 * flux_tolerance:
            x_by_cell[(j, i)] = (index, point)

    x_halo: set[tuple[int, int]] = set()
    for x_j, x_i in x_by_cell:
        for delta_j in (-1, 0, 1):
            for delta_i in (-1, 0, 1):
                candidate = (x_j + delta_j, x_i + delta_i)
                if 0 <= candidate[0] < z.size - 1 and 0 <= candidate[1] < r.size - 1:
                    x_halo.add(candidate)

    segments: list[tuple[tuple, tuple]] = []
    node_points: dict[tuple, np.ndarray] = {}
    edge_root_cache: dict[tuple[float, float, float, float], tuple[np.ndarray, ...]] = {}
    for j in range(z.size - 1):
        for i in range(r.size - 1):
            corners = np.array(
                [psi[j, i], psi[j, i + 1], psi[j + 1, i + 1], psi[j + 1, i]],
                dtype=float,
            )
            exact_x = x_by_cell.get((j, i))
            if exact_x is None and (j, i) not in x_halo:
                if float(level) < float(np.min(corners)) - flux_tolerance:
                    continue
                if float(level) > float(np.max(corners)) + flux_tolerance:
                    continue

            roots = _cell_perimeter_roots(
                field,
                r,
                z,
                i,
                j,
                float(level),
                flux_tolerance,
                exhaustive=((j, i) in x_halo),
                root_cache=edge_root_cache,
            )
            if exact_x is not None:
                x_index, x_point = exact_x
                x_key = ("x", int(x_index))
                x_coord = np.asarray(x_point.point, dtype=float)
                node_points[x_key] = x_coord
                # A well-resolved saddle has four perimeter exits.  More roots
                # indicate that this topology grid is too coarse for the cell.
                if len(roots) != 4:
                    continue
                for key, point in roots:
                    node_points[key] = point
                    if float(np.linalg.norm(point - x_coord)) > 1.0e-13:
                        segments.append((key, x_key))
                continue

            if len(roots) == 2:
                (key_a, point_a), (key_b, point_b) = roots
                node_points[key_a] = point_a
                node_points[key_b] = point_b
                if key_a != key_b:
                    segments.append((key_a, key_b))
                continue
            if len(roots) == 4:
                for key, point in roots:
                    node_points[key] = point
                for first, second in _pair_four_roots(field, roots, r, z, i, j, float(level)):
                    segments.append((first[0], second[0]))
                continue
            # Degenerate and unresolved cells are rejected rather than joined by
            # a distance heuristic.  Refinement can then be increased explicitly.
    return segments, node_points


def _cell_perimeter_roots(
    field: EquilibriumField,
    r: np.ndarray,
    z: np.ndarray,
    i: int,
    j: int,
    level: float,
    tolerance: float,
    *,
    exhaustive: bool,
    root_cache: dict[tuple[float, float, float, float], tuple[np.ndarray, ...]],
) -> list[tuple[tuple, np.ndarray]]:
    # Counter-clockwise perimeter: bottom, right, top reversed, left reversed.
    edges = (
        (np.array([r[i], z[j]]), np.array([r[i + 1], z[j]]), ("h", j, i)),
        (np.array([r[i + 1], z[j]]), np.array([r[i + 1], z[j + 1]]), ("v", j, i + 1)),
        (np.array([r[i + 1], z[j + 1]]), np.array([r[i], z[j + 1]]), ("h", j + 1, i)),
        (np.array([r[i], z[j + 1]]), np.array([r[i], z[j]]), ("v", j, i)),
    )
    roots: list[tuple[tuple, np.ndarray, float]] = []
    perimeter_position = 0.0
    for edge_index, (a, b, _base_key) in enumerate(edges):
        cache_key = _canonical_edge_key(a, b)
        cached = root_cache.get(cache_key)
        if cached is None:
            cached = tuple(
                point
                for _t_value, point in _edge_roots(
                    field,
                    a,
                    b,
                    level,
                    tolerance,
                    exhaustive=exhaustive,
                )
            )
            root_cache[cache_key] = cached
        vector = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        denominator = float(np.dot(vector, vector))
        for point in cached:
            t_value = (
                0.0
                if denominator <= 0.0
                else float(np.clip(np.dot(point - np.asarray(a, dtype=float), vector) / denominator, 0.0, 1.0))
            )
            # Coordinate quantisation makes the same physical root from adjacent
            # cells share a graph node even if Brent iterations differ by ulps.
            key = (
                "p",
                round(float(point[0]), 12),
                round(float(point[1]), 12),
            )
            roots.append((key, point, perimeter_position + t_value))
        perimeter_position += 1.0

    roots.sort(key=lambda item: item[2])
    deduplicated: list[tuple[tuple, np.ndarray, float]] = []
    spatial_tolerance = 2.0e-10 * max(field.grid_scale, 1.0)
    for item in roots:
        if deduplicated and float(np.linalg.norm(item[1] - deduplicated[-1][1])) <= spatial_tolerance:
            continue
        deduplicated.append(item)
    if len(deduplicated) > 1 and float(np.linalg.norm(deduplicated[0][1] - deduplicated[-1][1])) <= spatial_tolerance:
        deduplicated.pop()
    return [(item[0], item[1]) for item in deduplicated]


def _canonical_edge_key(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    first = (round(float(a[0]), 14), round(float(a[1]), 14))
    second = (round(float(b[0]), 14), round(float(b[1]), 14))
    if second < first:
        first, second = second, first
    return first[0], first[1], second[0], second[1]


def _edge_roots(
    field: EquilibriumField,
    a: np.ndarray,
    b: np.ndarray,
    level: float,
    tolerance: float,
    *,
    exhaustive: bool,
) -> list[tuple[float, np.ndarray]]:
    vector = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if exhaustive:
        samples = np.linspace(0.0, 1.0, 9, dtype=float)
    else:
        samples = np.asarray([0.0, 1.0], dtype=float)
    points = np.asarray(a, dtype=float)[None, :] + samples[:, None] * vector[None, :]
    values = field.value(points) - float(level)
    roots: list[float] = []
    for index, value in enumerate(values):
        if abs(float(value)) <= tolerance:
            roots.append(float(samples[index]))
    for index in range(samples.size - 1):
        left = float(values[index])
        right = float(values[index + 1])
        if not np.isfinite(left) or not np.isfinite(right) or left * right >= 0.0:
            continue

        def residual(t_value: float) -> float:
            point = np.asarray(a, dtype=float) + float(t_value) * vector
            return float(field.value(point)[0] - float(level))

        try:
            root = brentq(
                residual,
                float(samples[index]),
                float(samples[index + 1]),
                xtol=1.0e-13,
                rtol=4.0 * np.finfo(float).eps,
                maxiter=64,
            )
        except ValueError:
            continue
        roots.append(float(root))
    roots.sort()
    unique: list[float] = []
    for value in roots:
        if not unique or abs(value - unique[-1]) > 1.0e-9:
            unique.append(float(np.clip(value, 0.0, 1.0)))
    return [(value, np.asarray(a, dtype=float) + value * vector) for value in unique]


def _pair_four_roots(
    field: EquilibriumField,
    roots: list[tuple[tuple, np.ndarray]],
    r: np.ndarray,
    z: np.ndarray,
    i: int,
    j: int,
    level: float,
) -> tuple[tuple[tuple[tuple, np.ndarray], tuple[tuple, np.ndarray]], ...]:
    # The four roots are in perimeter order.  There are two non-crossing
    # pairings.  Test a point just inside the corner between roots 0 and 1.
    center = np.array([0.5 * (r[i] + r[i + 1]), 0.5 * (z[j] + z[j + 1])], dtype=float)
    center_positive = float(field.value(center)[0]) > float(level)
    corner = np.array([r[i], z[j]], dtype=float)
    corner_positive = float(field.value(corner)[0]) > float(level)
    if center_positive == corner_positive:
        return ((roots[0], roots[1]), (roots[2], roots[3]))
    return ((roots[1], roots[2]), (roots[3], roots[0]))


def _compress_segments(
    segments: list[tuple[tuple, tuple]],
    node_points: dict[tuple, np.ndarray],
) -> list[LevelSetEdge]:
    adjacency: dict[tuple, list[int]] = {}
    for index, (a, b) in enumerate(segments):
        adjacency.setdefault(a, []).append(index)
        adjacency.setdefault(b, []).append(index)
    visited: set[int] = set()
    out: list[LevelSetEdge] = []

    def is_junction(key: tuple) -> bool:
        return key[0] == "x" or len(adjacency.get(key, ())) != 2

    ordered = sorted(
        range(len(segments)),
        key=lambda idx: not (is_junction(segments[idx][0]) or is_junction(segments[idx][1])),
    )
    for first_edge in ordered:
        if first_edge in visited:
            continue
        a0, b0 = segments[first_edge]
        if is_junction(a0):
            start = a0
        elif is_junction(b0):
            start = b0
        else:
            start = a0
        points = [np.asarray(node_points[start], dtype=float)]
        current = start
        edge_index = first_edge
        end = start
        while True:
            if edge_index in visited:
                break
            visited.add(edge_index)
            a, b = segments[edge_index]
            nxt = b if a == current else a
            points.append(np.asarray(node_points[nxt], dtype=float))
            end = nxt
            if nxt == start:
                break
            if is_junction(nxt):
                break
            candidates = [candidate for candidate in adjacency[nxt] if candidate != edge_index and candidate not in visited]
            if not candidates:
                break
            current = nxt
            edge_index = candidates[0]
        array = _remove_consecutive_duplicates(np.asarray(points, dtype=float), tolerance=1.0e-13)
        if array.shape[0] < 2:
            continue
        start_node = int(start[1]) if start[0] == "x" else None
        end_node = int(end[1]) if end[0] == "x" else None
        if start == end and start_node is None:
            array = close_poly(array)
        out.append(LevelSetEdge(start_node=start_node, end_node=end_node, points=array))
    return out


def _enumerate_graph_cycles(edges: list[LevelSetEdge]) -> list[LevelSetCycle]:
    adjacency: dict[int, list[int]] = {}
    for index, edge in enumerate(edges):
        if edge.start_node is None or edge.end_node is None or edge.start_node == edge.end_node:
            continue
        adjacency.setdefault(int(edge.start_node), []).append(index)
        adjacency.setdefault(int(edge.end_node), []).append(index)
    cycles: list[LevelSetCycle] = []
    signatures: set[tuple[int, ...]] = set()
    for start in adjacency:
        _dfs_cycles(
            start=start,
            current=start,
            adjacency=adjacency,
            edges=edges,
            path=[],
            nodes=[start],
            used=set(),
            signatures=signatures,
            out=cycles,
            max_depth=max(len(edges), 3),
        )
    return cycles


def _dfs_cycles(
    *,
    start: int,
    current: int,
    adjacency: dict[int, list[int]],
    edges: list[LevelSetEdge],
    path: list[tuple[int, bool]],
    nodes: list[int],
    used: set[int],
    signatures: set[tuple[int, ...]],
    out: list[LevelSetCycle],
    max_depth: int,
) -> None:
    if len(path) >= max_depth:
        return
    for edge_index in adjacency.get(current, []):
        if edge_index in used:
            continue
        edge = edges[edge_index]
        if edge.start_node == current:
            nxt = int(edge.end_node)  # type: ignore[arg-type]
            forward = True
        elif edge.end_node == current:
            nxt = int(edge.start_node)  # type: ignore[arg-type]
            forward = False
        else:
            continue
        new_path = path + [(edge_index, forward)]
        if nxt == start and len(new_path) >= 2:
            signature = tuple(sorted(index for index, _ in new_path))
            if signature in signatures:
                continue
            signatures.add(signature)
            out.append(
                LevelSetCycle(
                    points=close_poly(_concatenate_edges(edges, new_path)),
                    edge_indices=tuple(index for index, _ in new_path),
                    node_indices=tuple(sorted(set(nodes))),
                )
            )
            continue
        if nxt in nodes:
            continue
        _dfs_cycles(
            start=start,
            current=nxt,
            adjacency=adjacency,
            edges=edges,
            path=new_path,
            nodes=nodes + [nxt],
            used=used | {edge_index},
            signatures=signatures,
            out=out,
            max_depth=max_depth,
        )


def _concatenate_edges(edges: list[LevelSetEdge], path: list[tuple[int, bool]]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for edge_index, forward in path:
        points = np.asarray(edges[edge_index].points, dtype=float)
        if not forward:
            points = points[::-1]
        chunks.append(points if not chunks else points[1:])
    return _remove_consecutive_duplicates(np.vstack(chunks), tolerance=1.0e-13)


def _is_closed(points: np.ndarray) -> bool:
    return points.shape[0] >= 4 and float(np.linalg.norm(points[0] - points[-1])) <= 1.0e-10


def _remove_consecutive_duplicates(points: np.ndarray, *, tolerance: float) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.shape[0] <= 1:
        return arr.copy()
    keep = np.ones((arr.shape[0],), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(arr, axis=0), axis=1) > float(tolerance)
    return arr[keep]
