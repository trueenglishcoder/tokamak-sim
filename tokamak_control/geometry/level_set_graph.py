"""Topology-preserving extraction of level sets from a known equilibrium field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from contourpy import contour_generator

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


def extract_core_level_set(
    field: EquilibriumField,
    *,
    level: float,
    axis: tuple[float, float],
    limiter_poly: np.ndarray,
    x_points: tuple[CriticalPoint, ...] = (),
    require_x_cycle: bool = False,
) -> LevelSetGraphResult | None:
    """Extract the cycle of ``psi=level`` bounding the primary core.

    X-points are explicit graph nodes. A contour passing through a saddle is
    split into branches at every occurrence of that node, and the graph cycle
    enclosing the primary magnetic axis is selected without imposing a smooth
    tangent through the saddle.
    """
    raw = _contours_at_level(field, float(level))
    if not raw:
        return None
    x_coords = np.asarray([point.point for point in x_points], dtype=float).reshape(-1, 2)
    snap_tolerance = 1.75 * field.grid_scale
    edges: list[LevelSetEdge] = []
    regular_cycles: list[LevelSetCycle] = []

    for polyline in raw:
        closed = _is_closed(polyline, tolerance=1.5 * field.grid_scale)
        refined = _refine_level_polyline(
            field,
            polyline,
            float(level),
            x_coords=x_coords,
            x_exclusion_radius=snap_tolerance,
        )
        if refined.shape[0] < 2:
            continue
        markers = _xpoint_markers_on_polyline(
            refined,
            x_coords,
            closed=closed,
            snap_tolerance=snap_tolerance,
        )
        if closed and not markers:
            loop = close_poly(refined)
            regular_cycles.append(LevelSetCycle(points=loop, edge_indices=(), node_indices=()))
            continue
        split_edges = _split_polyline_at_markers(refined, markers, closed=closed)
        edges.extend(split_edges)

    cycles = list(regular_cycles)
    cycles.extend(_enumerate_graph_cycles(edges))
    valid_cycles: list[LevelSetCycle] = []
    for cycle in cycles:
        if cycle.points.shape[0] < 4:
            continue
        if require_x_cycle and not cycle.node_indices:
            continue
        if not encloses_center(cycle.points, axis):
            continue
        inside = points_in_or_on_polygon(cycle.points[:-1], limiter_poly, tol=1.5 * field.grid_scale)
        if not bool(np.all(inside)):
            continue
        area = abs(poly_area(cycle.points))
        if not np.isfinite(area) or area <= 0.0:
            continue
        valid_cycles.append(cycle)

    if not valid_cycles:
        return None
    # A single equilibrium level can contain islands and secondary lobes. The
    # primary core cycle is the largest cycle that contains the selected axis.
    selected = max(valid_cycles, key=lambda cycle: abs(poly_area(cycle.points)))
    used = set(selected.edge_indices)
    branches = tuple(
        np.asarray(edge.points, dtype=float).copy()
        for index, edge in enumerate(edges)
        if index not in used and edge.points.shape[0] >= 2
    )
    return LevelSetGraphResult(
        core_boundary=close_poly(np.asarray(selected.points, dtype=float)),
        separatrix_branches=branches,
        used_x_points=tuple(sorted(set(selected.node_indices))),
        all_edges=tuple(edges),
    )


def _contours_at_level(field: EquilibriumField, level: float) -> list[np.ndarray]:
    generator = contour_generator(
        x=np.asarray(field.grid.r.coords(), dtype=float),
        y=np.asarray(field.grid.z.coords(), dtype=float),
        z=np.asarray(field.psi, dtype=float),
        name="serial",
        line_type="Separate",
    )
    out: list[np.ndarray] = []
    for line in generator.lines(float(level)):
        arr = np.asarray(line, dtype=float)
        if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] == 2:
            out.append(arr)
    return out


def _refine_level_polyline(
    field: EquilibriumField,
    points: np.ndarray,
    level: float,
    *,
    x_coords: np.ndarray,
    x_exclusion_radius: float,
) -> np.ndarray:
    arr = _remove_consecutive_duplicates(np.asarray(points, dtype=float), tolerance=1.0e-12)
    if arr.shape[0] < 2:
        return arr
    project_mask = np.ones((arr.shape[0],), dtype=bool)
    if x_coords.size:
        distances = np.min(np.linalg.norm(arr[:, None, :] - x_coords[None, :, :], axis=2), axis=1)
        project_mask = distances > float(x_exclusion_radius)
    if bool(np.any(project_mask)):
        projected, converged = field.project_to_level(arr[project_mask], float(level))
        successful_indices = np.flatnonzero(project_mask)[converged]
        arr[successful_indices] = projected[converged]
    return _remove_consecutive_duplicates(arr, tolerance=1.0e-10)


def _xpoint_markers_on_polyline(
    points: np.ndarray,
    x_coords: np.ndarray,
    *,
    closed: bool,
    snap_tolerance: float,
) -> list[tuple[float, int, np.ndarray]]:
    if x_coords.size == 0 or points.shape[0] < 2:
        return []
    work = close_poly(points) if closed else np.asarray(points, dtype=float)
    seg_start = work[:-1]
    seg_end = work[1:]
    seg_vec = seg_end - seg_start
    seg_len = np.linalg.norm(seg_vec, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = float(cumulative[-1])
    markers: list[tuple[float, int, np.ndarray]] = []

    for node_index, x_point in enumerate(x_coords):
        denom = np.sum(seg_vec * seg_vec, axis=1)
        u = np.zeros_like(denom)
        valid = denom > 0.0
        u[valid] = np.clip(np.sum((x_point[None, :] - seg_start[valid]) * seg_vec[valid], axis=1) / denom[valid], 0.0, 1.0)
        nearest = seg_start + u[:, None] * seg_vec
        distance = np.linalg.norm(nearest - x_point[None, :], axis=1)
        eligible = distance <= float(snap_tolerance)
        clusters = _cyclic_true_clusters(eligible) if closed else _linear_true_clusters(eligible)
        for cluster in clusters:
            best_index = min(cluster, key=lambda index: float(distance[index]))
            s = float(cumulative[best_index] + u[best_index] * seg_len[best_index])
            if closed and total > 0.0:
                s %= total
            markers.append((s, int(node_index), np.asarray(x_point, dtype=float).copy()))

    markers.sort(key=lambda item: item[0])
    deduped: list[tuple[float, int, np.ndarray]] = []
    arclength_tol = 0.25 * float(snap_tolerance)
    for marker in markers:
        if deduped and marker[1] == deduped[-1][1] and abs(marker[0] - deduped[-1][0]) <= arclength_tol:
            continue
        deduped.append(marker)
    if closed and len(deduped) >= 2 and total > 0.0:
        first = deduped[0]
        last = deduped[-1]
        if first[1] == last[1] and (first[0] + total - last[0]) <= arclength_tol:
            deduped = deduped[:-1]
    return deduped


def _split_polyline_at_markers(
    points: np.ndarray,
    markers: list[tuple[float, int, np.ndarray]],
    *,
    closed: bool,
) -> list[LevelSetEdge]:
    work = close_poly(points) if closed else np.asarray(points, dtype=float)
    lengths = np.linalg.norm(np.diff(work, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if not markers:
        return [LevelSetEdge(None, None, work.copy())]

    out: list[LevelSetEdge] = []
    if closed:
        for index, start in enumerate(markers):
            end = markers[(index + 1) % len(markers)]
            path = _polyline_interval(work, cumulative, start, end, total=total, wrap=end[0] <= start[0])
            if path.shape[0] >= 2:
                out.append(LevelSetEdge(start[1], end[1], path))
    else:
        first = markers[0]
        prefix = _polyline_interval(
            work,
            cumulative,
            (0.0, -1, work[0]),
            first,
            total=total,
            wrap=False,
        )
        if prefix.shape[0] >= 2:
            out.append(LevelSetEdge(None, first[1], prefix))
        for start, end in zip(markers[:-1], markers[1:], strict=True):
            path = _polyline_interval(work, cumulative, start, end, total=total, wrap=False)
            if path.shape[0] >= 2:
                out.append(LevelSetEdge(start[1], end[1], path))
        last = markers[-1]
        suffix = _polyline_interval(
            work,
            cumulative,
            last,
            (total, -1, work[-1]),
            total=total,
            wrap=False,
        )
        if suffix.shape[0] >= 2:
            out.append(LevelSetEdge(last[1], None, suffix))
    return out


def _polyline_interval(
    work: np.ndarray,
    cumulative: np.ndarray,
    start: tuple[float, int, np.ndarray],
    end: tuple[float, int, np.ndarray],
    *,
    total: float,
    wrap: bool,
) -> np.ndarray:
    s0 = float(start[0])
    s1 = float(end[0])
    if not wrap:
        mids = work[(cumulative > s0 + 1.0e-12) & (cumulative < s1 - 1.0e-12)]
        return _remove_consecutive_duplicates(
            np.vstack((np.asarray(start[2], dtype=float), mids, np.asarray(end[2], dtype=float))),
            tolerance=1.0e-12,
        )
    tail = work[cumulative > s0 + 1.0e-12]
    head = work[cumulative < s1 - 1.0e-12]
    return _remove_consecutive_duplicates(
        np.vstack((np.asarray(start[2], dtype=float), tail, head, np.asarray(end[2], dtype=float))),
        tolerance=1.0e-12,
    )


def _enumerate_graph_cycles(edges: list[LevelSetEdge]) -> list[LevelSetCycle]:
    cycles: list[LevelSetCycle] = []
    for index, edge in enumerate(edges):
        if edge.start_node is not None and edge.start_node == edge.end_node:
            cycles.append(
                LevelSetCycle(
                    points=close_poly(edge.points),
                    edge_indices=(index,),
                    node_indices=(int(edge.start_node),),
                )
            )

    adjacency: dict[int, list[int]] = {}
    for index, edge in enumerate(edges):
        if edge.start_node is None or edge.end_node is None or edge.start_node == edge.end_node:
            continue
        adjacency.setdefault(int(edge.start_node), []).append(index)
        adjacency.setdefault(int(edge.end_node), []).append(index)

    signatures: set[tuple[int, ...]] = set()
    for start_node in adjacency:
        _dfs_cycles(
            start_node=start_node,
            current_node=start_node,
            adjacency=adjacency,
            edges=edges,
            path_edges=[],
            path_nodes=[start_node],
            used_edges=set(),
            cycles=cycles,
            signatures=signatures,
            max_depth=max(len(adjacency) + 1, 3),
        )
    return cycles


def _dfs_cycles(
    *,
    start_node: int,
    current_node: int,
    adjacency: dict[int, list[int]],
    edges: list[LevelSetEdge],
    path_edges: list[tuple[int, bool]],
    path_nodes: list[int],
    used_edges: set[int],
    cycles: list[LevelSetCycle],
    signatures: set[tuple[int, ...]],
    max_depth: int,
) -> None:
    if len(path_edges) >= max_depth:
        return
    for edge_index in adjacency.get(current_node, []):
        if edge_index in used_edges:
            continue
        edge = edges[edge_index]
        if edge.start_node == current_node:
            next_node = int(edge.end_node)  # type: ignore[arg-type]
            forward = True
        elif edge.end_node == current_node:
            next_node = int(edge.start_node)  # type: ignore[arg-type]
            forward = False
        else:
            continue
        next_edges = path_edges + [(edge_index, forward)]
        if next_node == start_node and len(next_edges) >= 2:
            signature = tuple(sorted(index for index, _forward in next_edges))
            if signature in signatures:
                continue
            signatures.add(signature)
            points = _concatenate_oriented_edges(edges, next_edges)
            node_indices = tuple(sorted(set(path_nodes)))
            cycles.append(
                LevelSetCycle(
                    points=close_poly(points),
                    edge_indices=tuple(index for index, _forward in next_edges),
                    node_indices=node_indices,
                )
            )
            continue
        if next_node in path_nodes:
            continue
        _dfs_cycles(
            start_node=start_node,
            current_node=next_node,
            adjacency=adjacency,
            edges=edges,
            path_edges=next_edges,
            path_nodes=path_nodes + [next_node],
            used_edges=used_edges | {edge_index},
            cycles=cycles,
            signatures=signatures,
            max_depth=max_depth,
        )


def _concatenate_oriented_edges(edges: list[LevelSetEdge], path: list[tuple[int, bool]]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for edge_index, forward in path:
        pts = np.asarray(edges[edge_index].points, dtype=float)
        if not forward:
            pts = pts[::-1]
        chunks.append(pts if not chunks else pts[1:])
    return _remove_consecutive_duplicates(np.vstack(chunks), tolerance=1.0e-12)


def _is_closed(points: np.ndarray, *, tolerance: float) -> bool:
    return points.shape[0] >= 3 and float(np.linalg.norm(points[0] - points[-1])) <= float(tolerance)


def _remove_consecutive_duplicates(points: np.ndarray, *, tolerance: float) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.shape[0] <= 1:
        return arr.copy()
    keep = np.ones((arr.shape[0],), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(arr, axis=0), axis=1) > float(tolerance)
    return arr[keep]


def _linear_true_clusters(mask: np.ndarray) -> list[list[int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    clusters: list[list[int]] = [[int(indices[0])]]
    for index in indices[1:]:
        if int(index) == clusters[-1][-1] + 1:
            clusters[-1].append(int(index))
        else:
            clusters.append([int(index)])
    return clusters


def _cyclic_true_clusters(mask: np.ndarray) -> list[list[int]]:
    clusters = _linear_true_clusters(mask)
    if len(clusters) >= 2 and clusters[0][0] == 0 and clusters[-1][-1] == int(mask.size) - 1:
        clusters[0] = clusters[-1] + clusters[0]
        clusters.pop()
    return clusters
