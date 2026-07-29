"""Physical LCFS and separatrix extraction from a fully known equilibrium field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import label
from scipy.optimize import minimize_scalar

from tokamak_control.core.grid import Grid2D
from tokamak_control.geometry.boundary_common import (
    close_poly,
    point_to_polyline_distance,
    points_in_or_on_polygon,
    prepare_limiter_shape,
)
from tokamak_control.geometry.boundary_projection import (
    FixedAngleProjection,
    project_boundary_to_fixed_angles,
)
from tokamak_control.geometry.critical_points import (
    CriticalPoint,
    find_critical_points,
)
from tokamak_control.geometry.equilibrium_field import EquilibriumField
from tokamak_control.geometry.level_set_graph import LevelSetGraphResult, extract_core_level_set


BoundaryTopology = Literal["limited", "single_null", "double_null", "multi_null"]


@dataclass(frozen=True, slots=True, repr=True)
class LimiterContact:
    point: tuple[float, float]
    segment_index: int
    psi: float
    boundary_distance: float


@dataclass(frozen=True, slots=True, repr=True)
class BoundaryQuality:
    max_flux_residual: float
    normalized_flux_residual: float
    closure_error: float
    minimum_regular_gradient: float
    limiter_violation_count: int
    core_component_size: int


@dataclass(frozen=True, slots=True, repr=True)
class EquilibriumBoundary:
    found: bool
    topology: BoundaryTopology
    psi_axis: float
    psi_boundary: float
    orientation: float
    magnetic_axis: CriticalPoint
    x_points: tuple[CriticalPoint, ...]
    limiter_contacts: tuple[LimiterContact, ...]
    core_boundary: np.ndarray
    separatrix_branches: tuple[np.ndarray, ...]
    fixed_angle_projection: FixedAngleProjection
    quality: BoundaryQuality


@dataclass(frozen=True, slots=True, repr=True)
class EquilibriumLcfsSettings:
    critical_level_relative_tolerance: float = 2.0e-4
    flux_residual_relative_tolerance: float = 2.0e-5
    limiter_contact_grid_tolerance: float = 0.75
    xpoint_core_grid_tolerance: float = 1.5
    minimum_positive_flux_fraction: float = 1.0e-9


def find_equilibrium_lcfs(
    psi: np.ndarray,
    grid: Grid2D,
    *,
    center_hint: tuple[float, float],
    limiter_shape: np.ndarray,
    fixed_angles: np.ndarray | None = None,
    settings: EquilibriumLcfsSettings | None = None,
) -> EquilibriumBoundary:
    """Extract the physical LCFS of the primary core from known ``psi(R,Z)``.

    Limited and diverted topology are not selected by the caller. Wall-contact
    and saddle-point flux candidates are ordered from the magnetic axis outward;
    the first candidate that forms the boundary of the connected primary core is
    returned.
    """
    options = EquilibriumLcfsSettings() if settings is None else settings
    limiter = prepare_limiter_shape(limiter_shape)
    if limiter is None:
        raise ValueError("equilibrium_lcfs requires limiter geometry")
    field = EquilibriumField(grid=grid, psi=np.asarray(psi, dtype=float))
    critical = find_critical_points(field, center_hint=center_hint, limiter_poly=limiter)
    axis = critical.primary_axis
    orientation = _axis_orientation(axis)
    flux_floor = float(options.minimum_positive_flux_fraction) * field.flux_scale

    wall_candidates = _limiter_flux_candidates(
        field,
        limiter,
        psi_axis=float(axis.level),
        orientation=orientation,
        flux_floor=flux_floor,
    )
    x_groups = _group_xpoint_candidates(
        critical.x_points,
        psi_axis=float(axis.level),
        orientation=orientation,
        relative_tolerance=float(options.critical_level_relative_tolerance),
        flux_scale=field.flux_scale,
        flux_floor=flux_floor,
    )

    candidates: list[tuple[float, str, object]] = []
    candidates.extend((candidate[0], "wall", candidate) for candidate in wall_candidates)
    candidates.extend((group[0], "x", group) for group in x_groups)
    candidates.sort(key=lambda item: item[0])
    if not candidates:
        raise RuntimeError("No positive wall-contact or X-point LCFS candidate was found")

    failure_reasons: list[str] = []
    for chi_value, kind, payload in candidates:
        if kind == "wall":
            _chi, level, wall_points = payload  # type: ignore[misc]
            graph = extract_core_level_set(
                field,
                level=float(level),
                axis=axis.point,
                limiter_poly=limiter,
            )
            if graph is None:
                failure_reasons.append(f"wall candidate chi={chi_value:.6e}: no closed primary-core level set")
                continue
            contacts = _match_limiter_contacts(
                field,
                graph.core_boundary,
                wall_points,
                tolerance=float(options.limiter_contact_grid_tolerance) * field.grid_scale,
            )
            if not contacts:
                failure_reasons.append(f"wall candidate chi={chi_value:.6e}: level set does not contact limiter")
                continue
            if not _core_component_valid(
                field,
                axis=axis,
                level=float(level),
                orientation=orientation,
                x_points=(),
                limiter=limiter,
                boundary=graph.core_boundary,
                xpoint_tolerance=float(options.xpoint_core_grid_tolerance) * field.grid_scale,
            ):
                failure_reasons.append(f"wall candidate chi={chi_value:.6e}: invalid connected core component")
                continue
            return _build_result(
                field=field,
                topology="limited",
                axis=axis,
                level=float(level),
                orientation=orientation,
                graph=graph,
                selected_x_points=(),
                contacts=contacts,
                limiter=limiter,
                projection_center=center_hint,
                fixed_angles=fixed_angles,
            )

        _chi, x_points = payload  # type: ignore[misc]
        x_tuple = tuple(x_points)
        representative_level = float(np.mean([point.level for point in x_tuple]))
        graph = extract_core_level_set(
            field,
            level=representative_level,
            axis=axis.point,
            limiter_poly=limiter,
            x_points=x_tuple,
            require_x_cycle=True,
        )
        if graph is None:
            failure_reasons.append(f"X-point candidate chi={chi_value:.6e}: no separatrix cycle around primary core")
            continue
        selected_x = tuple(x_tuple[index] for index in graph.used_x_points)
        if not selected_x:
            failure_reasons.append(f"X-point candidate chi={chi_value:.6e}: cycle did not use an X-point")
            continue
        if not _core_component_valid(
            field,
            axis=axis,
            level=representative_level,
            orientation=orientation,
            x_points=selected_x,
            limiter=limiter,
            boundary=graph.core_boundary,
            xpoint_tolerance=float(options.xpoint_core_grid_tolerance) * field.grid_scale,
        ):
            failure_reasons.append(f"X-point candidate chi={chi_value:.6e}: invalid connected core component")
            continue
        topology: BoundaryTopology
        if len(selected_x) == 1:
            topology = "single_null"
        elif len(selected_x) == 2:
            topology = "double_null"
        else:
            topology = "multi_null"
        branch_contacts = _branch_limiter_contacts(field, graph.separatrix_branches, limiter)
        return _build_result(
            field=field,
            topology=topology,
            axis=axis,
            level=representative_level,
            orientation=orientation,
            graph=graph,
            selected_x_points=selected_x,
            contacts=branch_contacts,
            limiter=limiter,
            projection_center=center_hint,
            fixed_angles=fixed_angles,
        )

    detail = "; ".join(failure_reasons[:8])
    raise RuntimeError(f"No physical LCFS candidate bounded the primary core. {detail}")


def _axis_orientation(axis: CriticalPoint) -> float:
    eig = np.asarray(axis.eigenvalues, dtype=float)
    if bool(np.all(eig > 0.0)):
        return 1.0
    if bool(np.all(eig < 0.0)):
        return -1.0
    raise RuntimeError("Primary magnetic axis is not an O-point extremum")


def _oriented_flux(value: np.ndarray | float, *, psi_axis: float, orientation: float) -> np.ndarray:
    return float(orientation) * (np.asarray(value, dtype=float) - float(psi_axis))


def _limiter_flux_candidates(
    field: EquilibriumField,
    limiter: np.ndarray,
    *,
    psi_axis: float,
    orientation: float,
    flux_floor: float,
) -> list[tuple[float, float, tuple[tuple[np.ndarray, int], ...]]]:
    """Find local outward-flux minima along the physical limiter."""
    raw: list[tuple[float, float, np.ndarray, int]] = []
    for segment_index, (a, b) in enumerate(zip(limiter[:-1], limiter[1:], strict=True)):
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        vector = b_arr - a_arr
        length = float(np.linalg.norm(vector))
        sample_count = max(int(np.ceil(length / max(0.25 * field.grid_scale, 1.0e-12))), 12)
        t_grid = np.linspace(0.0, 1.0, sample_count + 1, dtype=float)
        points = a_arr[None, :] + t_grid[:, None] * vector[None, :]
        levels = field.value(points)
        chi = _oriented_flux(levels, psi_axis=psi_axis, orientation=orientation)

        candidate_indices: list[int] = []
        for index in range(1, sample_count):
            if chi[index] <= chi[index - 1] and chi[index] <= chi[index + 1]:
                candidate_indices.append(index)
        if chi[0] <= chi[1]:
            candidate_indices.append(0)
        if chi[-1] <= chi[-2]:
            candidate_indices.append(sample_count)

        def chi_at(t_value: float) -> float:
            point = a_arr + float(t_value) * vector
            value = float(field.value(point)[0])
            return float(_oriented_flux(value, psi_axis=psi_axis, orientation=orientation))

        for index in candidate_indices:
            if index == 0 or index == sample_count:
                t_value = float(t_grid[index])
            else:
                result = minimize_scalar(
                    chi_at,
                    bounds=(float(t_grid[index - 1]), float(t_grid[index + 1])),
                    method="bounded",
                    options={"xatol": 1.0e-13, "maxiter": 64},
                )
                if not result.success:
                    continue
                t_value = float(np.clip(result.x, 0.0, 1.0))
            point = a_arr + t_value * vector
            chi_value = chi_at(t_value)
            if not np.isfinite(chi_value) or chi_value <= float(flux_floor):
                continue
            level = float(field.value(point)[0])
            raw.append((chi_value, level, point, segment_index))

    raw.sort(key=lambda item: item[0])
    groups: list[list[tuple[float, float, np.ndarray, int]]] = []
    flux_tolerance = max(1.0e-10 * field.flux_scale, 1.0e-7 * max(raw[0][0] if raw else 1.0, 1.0e-12))
    for candidate in raw:
        if groups and abs(candidate[0] - groups[-1][0][0]) <= flux_tolerance:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    return [
        (
            float(np.mean([item[0] for item in group])),
            float(np.mean([item[1] for item in group])),
            tuple((np.asarray(item[2], dtype=float).copy(), int(item[3])) for item in group),
        )
        for group in groups
    ]


def _group_xpoint_candidates(
    x_points: tuple[CriticalPoint, ...],
    *,
    psi_axis: float,
    orientation: float,
    relative_tolerance: float,
    flux_scale: float,
    flux_floor: float,
) -> list[tuple[float, tuple[CriticalPoint, ...]]]:
    values: list[tuple[float, CriticalPoint]] = []
    for point in x_points:
        chi = float(_oriented_flux(point.level, psi_axis=psi_axis, orientation=orientation))
        if np.isfinite(chi) and chi > float(flux_floor):
            values.append((chi, point))
    values.sort(key=lambda item: item[0])
    groups: list[list[tuple[float, CriticalPoint]]] = []
    for value in values:
        tolerance = max(1.0e-10 * float(flux_scale), float(relative_tolerance) * max(value[0], 1.0e-12))
        if groups and abs(value[0] - float(np.mean([item[0] for item in groups[-1]]))) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [
        (float(np.mean([item[0] for item in group])), tuple(item[1] for item in group))
        for group in groups
    ]


def _match_limiter_contacts(
    field: EquilibriumField,
    boundary: np.ndarray,
    wall_points: tuple[tuple[np.ndarray, int], ...],
    *,
    tolerance: float,
) -> tuple[LimiterContact, ...]:
    contacts: list[LimiterContact] = []
    for point, segment_index in wall_points:
        distance = point_to_polyline_distance(np.asarray(point, dtype=float), boundary)
        if distance <= float(tolerance):
            contacts.append(
                LimiterContact(
                    point=(float(point[0]), float(point[1])),
                    segment_index=int(segment_index),
                    psi=float(field.value(point)[0]),
                    boundary_distance=float(distance),
                )
            )
    return tuple(contacts)


def _branch_limiter_contacts(
    field: EquilibriumField,
    branches: tuple[np.ndarray, ...],
    limiter: np.ndarray,
) -> tuple[LimiterContact, ...]:
    contacts: list[LimiterContact] = []
    tolerance = 0.75 * field.grid_scale
    for branch in branches:
        if branch.shape[0] == 0:
            continue
        for endpoint in (branch[0], branch[-1]):
            segment_index, closest, distance = _nearest_limiter_segment(endpoint, limiter)
            if distance <= tolerance:
                contacts.append(
                    LimiterContact(
                        point=(float(closest[0]), float(closest[1])),
                        segment_index=int(segment_index),
                        psi=float(field.value(closest)[0]),
                        boundary_distance=float(distance),
                    )
                )
    return tuple(contacts)


def _nearest_limiter_segment(point: np.ndarray, limiter: np.ndarray) -> tuple[int, np.ndarray, float]:
    p = np.asarray(point, dtype=float)
    best_index = -1
    best_point = np.zeros((2,), dtype=float)
    best_distance = float("inf")
    for index, (a, b) in enumerate(zip(limiter[:-1], limiter[1:], strict=True)):
        a_arr = np.asarray(a, dtype=float)
        vector = np.asarray(b, dtype=float) - a_arr
        denom = float(np.dot(vector, vector))
        t = 0.0 if denom <= 0.0 else float(np.clip(np.dot(p - a_arr, vector) / denom, 0.0, 1.0))
        closest = a_arr + t * vector
        distance = float(np.linalg.norm(p - closest))
        if distance < best_distance:
            best_index = index
            best_point = closest
            best_distance = distance
    return best_index, best_point, best_distance


def _core_component_valid(
    field: EquilibriumField,
    *,
    axis: CriticalPoint,
    level: float,
    orientation: float,
    x_points: tuple[CriticalPoint, ...],
    limiter: np.ndarray,
    boundary: np.ndarray,
    xpoint_tolerance: float,
) -> bool:
    chi = _oriented_flux(field.psi, psi_axis=float(axis.level), orientation=orientation)
    chi_boundary = float(_oriented_flux(level, psi_axis=float(axis.level), orientation=orientation))
    inside_domain = points_in_or_on_polygon(
        np.column_stack((field.grid.mesh()[0].reshape(-1), field.grid.mesh()[1].reshape(-1))),
        limiter,
        tol=0.0,
    ).reshape(field.grid.shape)
    mask = np.asarray((chi < chi_boundary) & inside_domain, dtype=bool)
    labels, _count = label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int))
    axis_i = field.grid.r.nearest_index(float(axis.point[0]))
    axis_j = field.grid.z.nearest_index(float(axis.point[1]))
    component_label = int(labels[axis_j, axis_i])
    if component_label == 0:
        return False
    component = labels == component_label
    if int(np.count_nonzero(component)) < 4:
        return False
    # The primary component must approach every selected X-point and the dense
    # boundary must remain in the physical limiter region.
    for x_point in x_points:
        grid_points = np.column_stack((field.grid.mesh()[0][component], field.grid.mesh()[1][component]))
        if grid_points.size == 0:
            return False
        distance = float(np.min(np.linalg.norm(grid_points - np.asarray(x_point.point)[None, :], axis=1)))
        if distance > float(xpoint_tolerance):
            return False
    boundary_inside = points_in_or_on_polygon(boundary[:-1], limiter, tol=0.35 * field.grid_scale)
    return bool(np.all(boundary_inside))


def _build_result(
    *,
    field: EquilibriumField,
    topology: BoundaryTopology,
    axis: CriticalPoint,
    level: float,
    orientation: float,
    graph: LevelSetGraphResult,
    selected_x_points: tuple[CriticalPoint, ...],
    contacts: tuple[LimiterContact, ...],
    limiter: np.ndarray,
    projection_center: tuple[float, float],
    fixed_angles: np.ndarray | None,
) -> EquilibriumBoundary:
    boundary = close_poly(np.asarray(graph.core_boundary, dtype=float))
    projection = (
        FixedAngleProjection(
            radii=np.zeros((0,), dtype=float),
            intersection_counts=np.zeros((0,), dtype=np.int64),
            valid=True,
            reason=None,
        )
        if fixed_angles is None
        else project_boundary_to_fixed_angles(boundary, projection_center, np.asarray(fixed_angles, dtype=float))
    )
    quality = _quality_metrics(
        field,
        boundary=boundary,
        level=float(level),
        x_points=selected_x_points,
        limiter=limiter,
        axis=axis,
        orientation=orientation,
    )
    return EquilibriumBoundary(
        found=True,
        topology=topology,
        psi_axis=float(axis.level),
        psi_boundary=float(level),
        orientation=float(orientation),
        magnetic_axis=axis,
        x_points=selected_x_points,
        limiter_contacts=contacts,
        core_boundary=boundary,
        separatrix_branches=(
            ()
            if topology == "limited"
            else tuple(np.asarray(branch, dtype=float).copy() for branch in graph.separatrix_branches)
        ),
        fixed_angle_projection=projection,
        quality=quality,
    )


def _quality_metrics(
    field: EquilibriumField,
    *,
    boundary: np.ndarray,
    level: float,
    x_points: tuple[CriticalPoint, ...],
    limiter: np.ndarray,
    axis: CriticalPoint,
    orientation: float,
) -> BoundaryQuality:
    vertices = np.asarray(boundary[:-1], dtype=float)
    residual = np.abs(field.value(vertices) - float(level))
    max_residual = float(np.max(residual)) if residual.size else float("inf")
    flux_span = abs(float(level) - float(axis.level))
    r_span = abs(float(field.grid.r.coords()[-1] - field.grid.r.coords()[0]))
    z_span = abs(float(field.grid.z.coords()[-1] - field.grid.z.coords()[0]))
    domain_diagonal = max(float(np.hypot(r_span, z_span)), field.grid_scale)
    interpolation_floor = (field.grid_scale / domain_diagonal) ** 4
    normalized = max(max_residual / max(flux_span, 1.0e-30), interpolation_floor)
    closure = float(np.linalg.norm(boundary[0] - boundary[-1]))
    regular = boundary[:-1]
    if x_points and regular.shape[0]:
        x_coords = np.asarray([point.point for point in x_points], dtype=float)
        distance = np.min(np.linalg.norm(regular[:, None, :] - x_coords[None, :, :], axis=2), axis=1)
        regular = regular[distance > 1.5 * field.grid_scale]
    gradients = np.linalg.norm(field.gradient(regular), axis=1) if regular.shape[0] else np.zeros((0,), dtype=float)
    min_gradient = float(np.min(gradients)) if gradients.size else 0.0
    inside = points_in_or_on_polygon(boundary[:-1], limiter, tol=0.35 * field.grid_scale)
    violation_count = int(np.count_nonzero(~inside))
    chi = _oriented_flux(field.psi, psi_axis=float(axis.level), orientation=orientation)
    chi_boundary = float(_oriented_flux(level, psi_axis=float(axis.level), orientation=orientation))
    component_size = int(np.count_nonzero(chi < chi_boundary))
    return BoundaryQuality(
        max_flux_residual=max_residual,
        normalized_flux_residual=float(normalized),
        closure_error=closure,
        minimum_regular_gradient=min_gradient,
        limiter_violation_count=violation_count,
        core_component_size=component_size,
    )
