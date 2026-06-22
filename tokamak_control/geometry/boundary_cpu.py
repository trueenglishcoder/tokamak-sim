# tokamak_control/geometry/boundary_cpu.py
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal
import numpy as np
from contourpy import contour_generator

from tokamak_control.core.grid import Grid2D
from tokamak_control.geometry.boundary_common import (
    BoundaryMode,
    BoundaryStatus,
    BoundaryNotFoundError as _SharedBoundaryNotFoundError,
    normalize_boundary_mode,
)
from tokamak_control.geometry.xpoints import find_x_points
from tokamak_control.io.logger import get_logger
from tokamak_control.io.profiling import Profiler


_PROFILER = Profiler(
    enabled=False,
    summary_every=0,
    logger=get_logger("geometry.boundary.profiling"),
)
_time_block = _PROFILER.time_block
_record_path = _PROFILER.record_path
_LEGACY_LIMITER_CONTAINMENT_TOL_M = 1.0e-9

class BoundaryNotFoundError(_SharedBoundaryNotFoundError):
    """Ошибка отсутствия физически определенной границы плазмы."""


@dataclass(frozen=True, slots=True, repr=True)
class _MagneticAxis:
    """Магнитная ось, найденная как ближайший экстремум psi."""

    point: tuple[float, float]
    level: float
    kind: Literal["maximum", "minimum"]


def configure_boundary_profiling(
    *,
    enabled: bool,
    summary_every: int = 0,
    reset: bool = True,
) -> None:
    """Настроить сбор профиля поиска границы плазмы."""
    _PROFILER.configure(
        enabled=enabled,
        summary_every=summary_every,
        logger=get_logger("geometry.boundary.profiling"),
        reset=reset,
    )


def boundary_profiling_snapshot() -> dict[str, object]:
    """Вернуть текущую сводку профиля поиска границы."""
    return _PROFILER.summary_dict(
        total_key="boundary_total",
        keys=(
            "axis_search",
            "contours_at_level",
            "legacy_contour_search",
        ),
        path_keys=(
            "legacy_contour_success",
            "legacy_contour_limited_success",
            "tracked_flux_contour_success",
            "tracked_flux_contour_reset",
        ),
        title="boundary",
    )


def log_boundary_profiling_summary() -> None:
    """Записать в лог сводку профиля поиска границы."""
    _PROFILER.log_summary(
        total_key="boundary_total",
        keys=(
            "axis_search",
            "contours_at_level",
            "legacy_contour_search",
        ),
        path_keys=(
            "legacy_contour_success",
            "legacy_contour_limited_success",
            "tracked_flux_contour_success",
            "tracked_flux_contour_reset",
        ),
        title="boundary",
    )


def find_plasma_boundary_cpu_with_status(
    psi: np.ndarray,
    grid: Grid2D,
    center: tuple[float, float],
    n_levels: int = 10,
    prev_level: float | None = None,
    prev_poly: np.ndarray | None = None,
    local_n_levels: int = 7,
    local_span_frac: float = 0.02,
    target_mean_radius: float | None = None,
    target_switch_ratio: float = 1.15,
    target_switch_abs_delta: float = 0.10,
    local_bbox_pad_r: float | None = None,
    local_bbox_pad_z: float | None = None,
    limiter_shape: np.ndarray | None = None,
    boundary_mode: BoundaryMode = "legacy_contour",
    boundary_base_mode: BoundaryMode = "legacy_contour_limited",
    legacy_precision_index2: float = 1.0e-3,
    track_level: bool = False,
    level_smoothing_alpha: float = 1.0,
    level_search_span_fraction: float = 0.02,
    continuity_weight_radii: float = 1.0,
    continuity_weight_mean_radius: float = 0.3,
    continuity_weight_center: float = 0.2,
    continuity_weight_area: float = 0.2,
    continuity_weight_level: float = 0.1,
) -> tuple[np.ndarray, float, BoundaryStatus]:
    """Найти старый MATLAB ``PlasmaBoundary``-контур по полю psi."""
    mode = normalize_boundary_mode(boundary_mode)
    base_mode = normalize_boundary_mode(boundary_base_mode)
    if mode == "tracked_flux_contour" and base_mode == "tracked_flux_contour":
        raise ValueError("tracked_flux_contour boundary mode requires a strict legacy base_mode")

    with _time_block("boundary_total"):
        if psi.shape != grid.shape:
            raise ValueError(f"psi shape {psi.shape} != grid shape {grid.shape}")

        del n_levels, local_n_levels, local_span_frac
        del target_mean_radius, target_switch_ratio, target_switch_abs_delta
        del local_bbox_pad_r, local_bbox_pad_z
        limiter_poly = None
        strict_mode = base_mode if mode == "tracked_flux_contour" else mode
        if strict_mode == "legacy_contour_limited":
            limiter_poly = _prepare_limiter_shape(limiter_shape)
            if limiter_poly is None:
                raise BoundaryNotFoundError(f"{strict_mode} boundary mode requires limiter geometry")
        should_track = mode == "tracked_flux_contour" or bool(track_level)
        if should_track and prev_poly is not None and prev_level is not None and np.isfinite(float(prev_level)):
            tracked_boundary = _legacy_tracked_contour_boundary(
                psi=psi,
                grid=grid,
                center=center,
                prev_level=float(prev_level),
                prev_poly=np.asarray(prev_poly, dtype=float),
                level_smoothing_alpha=float(level_smoothing_alpha),
                level_search_span_fraction=float(level_search_span_fraction),
                limiter_poly=limiter_poly,
                continuity_weight_radii=float(continuity_weight_radii),
                continuity_weight_mean_radius=float(continuity_weight_mean_radius),
                continuity_weight_center=float(continuity_weight_center),
                continuity_weight_area=float(continuity_weight_area),
                continuity_weight_level=float(continuity_weight_level),
            )
            if tracked_boundary is None and float(level_search_span_fraction) > 0.0:
                tracked_boundary = _legacy_tracked_contour_boundary(
                    psi=psi,
                    grid=grid,
                    center=center,
                    prev_level=float(prev_level),
                    prev_poly=np.asarray(prev_poly, dtype=float),
                    level_smoothing_alpha=float(level_smoothing_alpha),
                    level_search_span_fraction=3.0 * float(level_search_span_fraction),
                    limiter_poly=limiter_poly,
                    continuity_weight_radii=float(continuity_weight_radii),
                    continuity_weight_mean_radius=float(continuity_weight_mean_radius),
                    continuity_weight_center=float(continuity_weight_center),
                    continuity_weight_area=float(continuity_weight_area),
                    continuity_weight_level=float(continuity_weight_level),
                )
            if tracked_boundary is not None:
                poly, level = tracked_boundary
                status: BoundaryStatus = "tracked_flux_contour_success" if mode == "tracked_flux_contour" else ("legacy_contour_limited_success" if limiter_poly is not None else "legacy_contour_success")
                _record_path(status)
                return _close_poly(poly), float(level), status
            if mode != "tracked_flux_contour":
                raise BoundaryNotFoundError(f"No tracked {strict_mode} plasma boundary found")
        legacy_boundary = _legacy_contour_boundary(
            psi=psi,
            grid=grid,
            center=center,
            precision_index2=legacy_precision_index2,
            limiter_poly=limiter_poly,
        )
        if legacy_boundary is not None:
            poly, level = legacy_boundary
            if mode == "tracked_flux_contour":
                status = "tracked_flux_contour_reset"
            else:
                status = "legacy_contour_limited_success" if limiter_poly is not None else "legacy_contour_success"
            _record_path(status)
            return _close_poly(poly), float(level), status

        raise BoundaryNotFoundError(f"No {mode} plasma boundary found")


def boundary_status_is_real(status: BoundaryStatus) -> bool:
    """Вернуть True для статусов реально найденной границы."""
    return status in {
        "legacy_contour_success",
        "legacy_contour_limited_success",
        "tracked_flux_contour_success",
        "tracked_flux_contour_reset",
    }


def _legacy_contour_boundary(
    *,
    psi: np.ndarray,
    grid: Grid2D,
    center: tuple[float, float],
    precision_index2: float,
    limiter_poly: np.ndarray | None = None,
) -> tuple[np.ndarray, float] | None:
    """Reproduce tokamak-sim-0 MATLAB ``PlasmaBoundary`` in index space."""
    psi_arr = np.asarray(psi, dtype=float)
    if psi_arr.shape != grid.shape or not bool(np.any(np.isfinite(psi_arr))):
        return None
    precision = float(precision_index2)
    if not np.isfinite(precision) or precision <= 0.0:
        raise ValueError(f"legacy_precision_index2 must be finite and > 0, got {precision_index2!r}")

    o = _physical_center_to_legacy_index(grid, center)
    if not np.all(np.isfinite(o)):
        return None

    p = o.copy()
    new_step = -o / 2.0
    best_poly_index: np.ndarray | None = None
    best_level: float | None = None
    best_len = 0

    def contour_fits_limiter(contour_index: np.ndarray) -> bool:
        if limiter_poly is None:
            return True
        contour_physical = _legacy_index_poly_to_physical(grid, contour_index)
        return _poly_fits_limiter(
            _close_poly(contour_physical),
            limiter_poly,
            tol=_LEGACY_LIMITER_CONTAINMENT_TOL_M,
        )

    with _time_block("legacy_contour_search"):
        while float(np.dot(new_step, new_step)) >= precision and p[0] >= 1.0 and p[1] >= 1.0:
            p = p + new_step
            level = _legacy_sample_center_level(psi_arr, p)
            if level is None:
                break

            accepted = _legacy_first_accepted_contour(psi_arr, level, o, accept_predicate=contour_fits_limiter)
            if accepted is not None:
                length, contour = accepted
                if int(length) > best_len:
                    best_len = int(length)
                    best_poly_index = np.asarray(contour, dtype=float)
                    best_level = float(level)
                new_step = -np.abs(new_step) / 2.0
            else:
                new_step = np.abs(new_step) / 2.0

    if best_poly_index is None or best_level is None or best_poly_index.shape[0] < 3:
        return None
    return _legacy_index_poly_to_physical(grid, best_poly_index), float(best_level)


def _legacy_tracked_contour_boundary(
    *,
    psi: np.ndarray,
    grid: Grid2D,
    center: tuple[float, float],
    prev_level: float,
    prev_poly: np.ndarray,
    level_smoothing_alpha: float,
    level_search_span_fraction: float,
    limiter_poly: np.ndarray | None,
    continuity_weight_radii: float,
    continuity_weight_mean_radius: float,
    continuity_weight_center: float,
    continuity_weight_area: float,
    continuity_weight_level: float,
) -> tuple[np.ndarray, float] | None:
    """Continue the previous flux-surface identity with real current-frame contours."""
    psi_arr = np.asarray(psi, dtype=float)
    prev = _close_poly(np.asarray(prev_poly, dtype=float).reshape(-1, 2))
    if psi_arr.shape != grid.shape or prev.shape[0] < 4:
        return None
    alpha = float(np.clip(level_smoothing_alpha, 0.0, 1.0))
    span_fraction = max(float(level_search_span_fraction), 0.0)
    prev_center = np.asarray(center, dtype=float).reshape(1, 2)
    current_center = np.asarray(center, dtype=float).reshape(1, 2)
    predicted_prev = _close_poly(current_center + (prev - prev_center))
    sampled = _sample_psi_bilinear(psi_arr, grid, predicted_prev[:-1])
    sampled = sampled[np.isfinite(sampled)]
    if sampled.size == 0:
        return None

    continued_level = float(np.median(sampled))
    if np.isfinite(prev_level):
        level0 = alpha * float(prev_level) + (1.0 - alpha) * continued_level
    else:
        level0 = continued_level
    if not np.isfinite(level0):
        return None

    finite_values = psi_arr[np.isfinite(psi_arr)]
    if finite_values.size == 0:
        return None
    value_span = float(np.nanmax(finite_values) - np.nanmin(finite_values))
    level_span = span_fraction * max(value_span, abs(level0), 1.0e-12)
    offsets = np.asarray([0.0], dtype=float)
    if level_span > 0.0:
        offsets = np.asarray([0.0, -0.25, 0.25, -0.5, 0.5, -0.75, 0.75, -1.0, 1.0], dtype=float) * level_span

    o = _physical_center_to_legacy_index(grid, center)
    if not np.all(np.isfinite(o)):
        return None

    def contour_fits_limiter(contour_index: np.ndarray) -> bool:
        if limiter_poly is None:
            return True
        contour_physical = _legacy_index_poly_to_physical(grid, contour_index)
        return _poly_fits_limiter(
            _close_poly(contour_physical),
            limiter_poly,
            tol=_LEGACY_LIMITER_CONTAINMENT_TOL_M,
        )

    best: tuple[float, np.ndarray, float] | None = None
    for level in level0 + offsets:
        if not np.isfinite(float(level)):
            continue
        for contour in _legacy_accepted_contours(psi_arr, float(level), o, accept_predicate=contour_fits_limiter):
            contour_physical = _close_poly(_legacy_index_poly_to_physical(grid, contour))
            score = _contour_continuity_score(
                contour_physical,
                predicted_prev,
                center,
                level=float(level),
                level_guess=float(level0),
                level_span=float(max(level_span, 1.0e-12)),
                continuity_weight_radii=float(continuity_weight_radii),
                continuity_weight_mean_radius=float(continuity_weight_mean_radius),
                continuity_weight_center=float(continuity_weight_center),
                continuity_weight_area=float(continuity_weight_area),
                continuity_weight_level=float(continuity_weight_level),
            )
            if best is None or score < best[0]:
                best = (float(score), contour_physical, float(level))

    if best is None:
        return None
    _score, poly, level = best
    return poly, level


def _physical_center_to_legacy_index(grid: Grid2D, center: tuple[float, float]) -> np.ndarray:
    """Convert physical ``(R, Z)`` into MATLAB/contourc 1-based coordinates."""
    r0 = float(grid.r.coords()[0])
    z0 = float(grid.z.coords()[0])
    return np.asarray(
        [
            1.0 + (float(center[0]) - r0) / float(grid.r.step),
            1.0 + (float(center[1]) - z0) / float(grid.z.step),
        ],
        dtype=float,
    )


def _legacy_sample_center_level(psi: np.ndarray, p: np.ndarray) -> float | None:
    """Sample psi exactly like the old MATLAB ``PlasmaBoundary`` diagonal interpolation."""
    rows, cols = psi.shape
    p = np.asarray(p, dtype=float).reshape(2)
    r1 = np.floor(p).astype(int)
    r2 = np.ceil(p).astype(int)
    if r1[0] < 1 or r1[1] < 1:
        r1 = r2.copy()
    if r1[0] < 1 or r1[1] < 1 or r2[0] < 1 or r2[1] < 1:
        return None
    if r1[0] > cols or r2[0] > cols or r1[1] > rows or r2[1] > rows:
        return None

    value1 = float(psi[r1[1] - 1, r1[0] - 1])
    value2 = float(psi[r2[1] - 1, r2[0] - 1])
    if not np.isfinite(value1) or not np.isfinite(value2):
        return None

    rr1 = float(np.dot(r1.astype(float), r1.astype(float)))
    rr2 = float(np.dot(r2.astype(float), r2.astype(float)))
    rr = float(np.dot(p, p))
    if rr2 > rr1:
        return float(((rr2 - rr) * value1 + (rr - rr1) * value2) / (rr2 - rr1))
    return value2


def _legacy_first_accepted_contour(
    psi: np.ndarray,
    level: float,
    center_index: np.ndarray,
    accept_predicate: Callable[[np.ndarray], bool] | None = None,
) -> tuple[int, np.ndarray] | None:
    """Equivalent of old ``LineIsOk`` for contourpy-separated contours."""
    for contour in _legacy_accepted_contours(psi, float(level), center_index, accept_predicate=accept_predicate):
        return int(contour.shape[0]), contour
    return None


def _legacy_accepted_contours(
    psi: np.ndarray,
    level: float,
    center_index: np.ndarray,
    accept_predicate: Callable[[np.ndarray], bool] | None = None,
) -> list[np.ndarray]:
    contours = _contours_at_level_index_space(psi, float(level))
    o = np.asarray(center_index, dtype=float).reshape(2)
    accepted: list[np.ndarray] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        if not _legacy_contour_is_closed(contour):
            continue
        points = contour[:-1] if np.allclose(contour[0], contour[-1]) else contour
        min_values = np.min(points, axis=0)
        max_values = np.max(points, axis=0)
        if bool(np.all(min_values < o) and np.all(o < max_values)):
            if accept_predicate is not None and not bool(accept_predicate(contour)):
                continue
            accepted.append(np.asarray(contour, dtype=float))
    return accepted


def _legacy_contour_is_closed(contour: np.ndarray) -> bool:
    arr = np.asarray(contour, dtype=float)
    if arr.shape[0] < 3:
        return False
    return bool(np.allclose(arr[0], arr[-1], rtol=0.0, atol=1.0e-9))


def _contours_at_level_index_space(psi: np.ndarray, level: float) -> list[np.ndarray]:
    """Build contours in MATLAB ``contourc(Z)`` coordinates: x=1..n, y=1..m."""
    with _time_block("contours_at_level"):
        psi_view = np.asarray(psi, dtype=float)
        if psi_view.shape[0] < 2 or psi_view.shape[1] < 2:
            return []
        x = np.arange(1, psi_view.shape[1] + 1, dtype=float)
        y = np.arange(1, psi_view.shape[0] + 1, dtype=float)
        cg = contour_generator(x=x, y=y, z=psi_view, name="serial", line_type="Separate")
        out: list[np.ndarray] = []
        for contour in cg.lines(float(level)):
            arr = np.asarray(contour, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] == 2:
                out.append(arr)
        return out


def _legacy_index_poly_to_physical(grid: Grid2D, poly_index: np.ndarray) -> np.ndarray:
    arr = np.asarray(poly_index, dtype=float).reshape(-1, 2)
    r0 = float(grid.r.coords()[0])
    z0 = float(grid.z.coords()[0])
    out = np.empty_like(arr, dtype=float)
    out[:, 0] = r0 + (arr[:, 0] - 1.0) * float(grid.r.step)
    out[:, 1] = z0 + (arr[:, 1] - 1.0) * float(grid.z.step)
    return out


def _find_magnetic_axis(
    *,
    psi: np.ndarray,
    grid: Grid2D,
    center: tuple[float, float],
    limiter_poly: np.ndarray | None,
    limiter_tol: float,
) -> _MagneticAxis:
    """Найти магнитную ось как ближайший локальный экстремум psi."""
    psi_arr = np.asarray(psi, dtype=float)
    finite = np.isfinite(psi_arr)
    if psi_arr.shape != grid.shape or not bool(np.any(finite)):
        raise BoundaryNotFoundError("No finite psi values available for magnetic axis search")

    allowed = finite.copy()
    if limiter_poly is not None:
        R, Z = grid.mesh()
        points = np.column_stack([R.reshape(-1), Z.reshape(-1)])
        limiter_mask = _points_in_or_on_polygon(points, limiter_poly, tol=float(limiter_tol)).reshape(grid.shape)
        allowed &= limiter_mask
        if not bool(np.any(allowed)):
            raise BoundaryNotFoundError("No finite psi values inside limiter for magnetic axis search")

    candidates = _local_axis_candidates(psi_arr, grid, allowed)
    if candidates:
        return _select_axis_candidate(candidates, center)

    return _global_axis_candidate(psi_arr, grid, allowed, center)


def _local_axis_candidates(
    psi: np.ndarray,
    grid: Grid2D,
    allowed: np.ndarray,
) -> list[_MagneticAxis]:
    """Собрать локальные максимумы и минимумы psi на сетке."""
    r_coords = grid.r.coords()
    z_coords = grid.z.coords()
    candidates: list[_MagneticAxis] = []
    nz, nr = psi.shape
    for j in range(1, nz - 1):
        for i in range(1, nr - 1):
            if not bool(allowed[j, i]):
                continue
            value = float(psi[j, i])
            window = psi[j - 1 : j + 2, i - 1 : i + 2]
            valid_window = window[np.isfinite(window)]
            if valid_window.size == 0:
                continue
            if value >= float(np.max(valid_window)):
                candidates.append(_MagneticAxis(point=(float(r_coords[i]), float(z_coords[j])), level=value, kind="maximum"))
            if value <= float(np.min(valid_window)):
                candidates.append(_MagneticAxis(point=(float(r_coords[i]), float(z_coords[j])), level=value, kind="minimum"))
    return candidates


def _select_axis_candidate(
    candidates: Sequence[_MagneticAxis],
    center: tuple[float, float],
) -> _MagneticAxis:
    """Выбрать ближайший к расчетному центру кандидат магнитной оси."""
    c = np.asarray(center, dtype=float)
    return min(candidates, key=lambda item: float(np.linalg.norm(np.asarray(item.point, dtype=float) - c)))


def _global_axis_candidate(
    psi: np.ndarray,
    grid: Grid2D,
    allowed: np.ndarray,
    center: tuple[float, float],
) -> _MagneticAxis:
    """Выбрать ближайший глобальный экстремум, если локальных осей нет."""
    masked = np.where(allowed, psi, np.nan)
    if not bool(np.any(np.isfinite(masked))):
        raise BoundaryNotFoundError("No finite psi values available for magnetic axis search")

    max_idx = np.unravel_index(int(np.nanargmax(masked)), masked.shape)
    min_idx = np.unravel_index(int(np.nanargmin(masked)), masked.shape)
    r_coords = grid.r.coords()
    z_coords = grid.z.coords()
    candidates = [
        _MagneticAxis(point=(float(r_coords[max_idx[1]]), float(z_coords[max_idx[0]])), level=float(masked[max_idx]), kind="maximum"),
        _MagneticAxis(point=(float(r_coords[min_idx[1]]), float(z_coords[min_idx[0]])), level=float(masked[min_idx]), kind="minimum"),
    ]
    return _select_axis_candidate(candidates, center)


def _separatrix_boundary(
    *,
    psi: np.ndarray,
    grid: Grid2D,
    axis: tuple[float, float],
    limiter_poly: np.ndarray | None,
    limiter_tol: float,
) -> tuple[np.ndarray, float] | None:
    """Найти сепаратрису по X-point, если она представлена замкнутым контуром."""
    min_sep = 2.0 * float(max(abs(float(grid.r.step)), abs(float(grid.z.step))))
    with _time_block("xpoint_search"):
        x_points = find_x_points(psi, grid, max_points=8, min_separation=min_sep)
    if x_points.shape[0] == 0:
        return None

    candidates: list[tuple[np.ndarray, float, float]] = []
    for point in x_points:
        level = _psi_at_nearest_grid_point(psi, grid, (float(point[0]), float(point[1])))
        if not np.isfinite(level):
            continue
        for poly in _contours_at_level(psi, grid, float(level)):
            with _time_block("candidate_filtering"):
                if not _is_closed_poly(poly):
                    continue
                if not _encloses_center(poly, axis):
                    continue
                closed = _close_poly(poly)
                if _point_to_polyline_distance(np.asarray(point, dtype=float), closed) > min_sep:
                    continue
                if limiter_poly is not None and not _poly_fits_limiter(closed, limiter_poly, tol=limiter_tol):
                    continue
                area = abs(_poly_area(closed))
                if np.isfinite(area) and area > 0.0:
                    candidates.append((closed, float(level), float(area)))

    if not candidates:
        return None

    best_poly, best_level, _area = max(candidates, key=lambda item: item[2])
    return best_poly, best_level


def _limited_boundary(
    *,
    psi: np.ndarray,
    grid: Grid2D,
    axis: _MagneticAxis,
    limiter_poly: np.ndarray | None,
    limiter_tol: float,
    n_levels: int,
) -> tuple[np.ndarray, float] | None:
    """Найти limited LCFS как внешнюю замкнутую поверхность внутри лимитера."""
    if limiter_poly is None:
        raise RuntimeError("Limited boundary mode requires limiter geometry")

    contact_boundary = _limited_limiter_contact_boundary(
        psi=psi,
        grid=grid,
        axis=axis,
        limiter_poly=limiter_poly,
        limiter_tol=limiter_tol,
    )
    if contact_boundary is not None:
        return contact_boundary

    return _outermost_limited_boundary(
        psi=psi,
        grid=grid,
        axis=axis,
        limiter_poly=limiter_poly,
        limiter_tol=limiter_tol,
        n_levels=n_levels,
    )


def _limited_limiter_contact_boundary(
    *,
    psi: np.ndarray,
    grid: Grid2D,
    axis: _MagneticAxis,
    limiter_poly: np.ndarray,
    limiter_tol: float,
) -> tuple[np.ndarray, float] | None:
    """Найти limited LCFS как flux-поверхность первого контакта с лимитером."""
    with _time_block("limited_contact_levels"):
        limiter_points = _sample_limiter_points(limiter_poly, grid)
        limiter_psi = _sample_psi_bilinear(psi, grid, limiter_points)
        valid = np.isfinite(limiter_psi)
    if not bool(np.any(valid)):
        return None

    points = limiter_points[valid]
    values = limiter_psi[valid]
    rounded_values = np.round(values, 14)
    touch_tol = 2.5 * float(max(abs(float(grid.r.step)), abs(float(grid.z.step)), float(limiter_tol)))
    levels = _ordered_limiter_contact_levels(values, axis)

    for contact_level in levels:
        contact_points = points[rounded_values == float(contact_level)]
        if contact_points.shape[0] == 0:
            continue
        candidates: list[tuple[np.ndarray, float]] = []
        for poly in _contours_at_level(psi, grid, float(contact_level)):
            with _time_block("candidate_filtering"):
                if not _is_closed_poly(poly):
                    continue
                if not _encloses_center(poly, axis.point):
                    continue
                closed = _close_poly(poly)
                if not _poly_fits_limiter(closed, limiter_poly, tol=limiter_tol):
                    continue
                contact_dist = _polyline_to_points_distance(closed, contact_points)
                if contact_dist > touch_tol:
                    continue
                candidates.append((closed, float(contact_dist)))

        if candidates:
            best_poly, _dist = min(candidates, key=lambda item: item[1])
            return best_poly, float(contact_level)

    return None


def _outermost_limited_boundary(
    *,
    psi: np.ndarray,
    grid: Grid2D,
    axis: _MagneticAxis,
    limiter_poly: np.ndarray,
    limiter_tol: float,
    n_levels: int,
) -> tuple[np.ndarray, float] | None:
    """Найти наибольшую замкнутую flux-поверхность внутри лимитера."""
    levels = _limited_candidate_levels(
        psi=psi,
        grid=grid,
        axis=axis,
        limiter_poly=limiter_poly,
        limiter_tol=limiter_tol,
        n_levels=n_levels,
    )
    if levels.size == 0:
        return None

    candidates: list[tuple[np.ndarray, float, float]] = []
    for level in levels:
        for poly in _contours_at_level(psi, grid, float(level)):
            with _time_block("candidate_filtering"):
                if not _is_closed_poly(poly):
                    continue
                if not _encloses_center(poly, axis.point):
                    continue
                closed = _close_poly(poly)
                if not _poly_fits_limiter(closed, limiter_poly, tol=limiter_tol):
                    continue
                area = abs(_poly_area(closed))
                if np.isfinite(area) and area > 0.0:
                    candidates.append((closed, float(level), float(area)))

    if not candidates:
        return None

    best_poly, best_level, _area = max(candidates, key=lambda item: item[2])
    return best_poly, best_level


def _limited_candidate_levels(
    *,
    psi: np.ndarray,
    grid: Grid2D,
    axis: _MagneticAxis,
    limiter_poly: np.ndarray,
    limiter_tol: float,
    n_levels: int,
) -> np.ndarray:
    """Построить уровни psi для поиска внешнего контура внутри лимитера."""
    psi_arr = np.asarray(psi, dtype=float)
    R, Z = grid.mesh()
    points = np.column_stack([R.reshape(-1), Z.reshape(-1)])
    limiter_mask = _points_in_or_on_polygon(points, limiter_poly, tol=float(limiter_tol)).reshape(grid.shape)
    inside_values = psi_arr[limiter_mask & np.isfinite(psi_arr)]
    if inside_values.size == 0:
        return np.zeros((0,), dtype=float)

    base_level_count = max(int(n_levels), 1024)
    axis_level = float(axis.level)
    if axis.kind == "maximum":
        edge_level = float(np.min(inside_values))
        if edge_level >= axis_level:
            return np.zeros((0,), dtype=float)
        scan_levels = np.linspace(axis_level, edge_level, base_level_count + 2, dtype=float)[1:-1]
        return np.sort(np.unique(np.round(scan_levels, 14)))[::-1]

    edge_level = float(np.max(inside_values))
    if edge_level <= axis_level:
        return np.zeros((0,), dtype=float)
    scan_levels = np.linspace(axis_level, edge_level, base_level_count + 2, dtype=float)[1:-1]
    return np.sort(np.unique(np.round(scan_levels, 14)))


def _ordered_limiter_contact_levels(values: np.ndarray, axis: _MagneticAxis) -> np.ndarray:
    """Вернуть уровни psi на лимитере в порядке выхода от магнитной оси."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros((0,), dtype=float)
    unique = np.unique(np.round(vals, 14))
    if axis.kind == "maximum":
        return np.sort(unique)[::-1]
    return np.sort(unique)


def _polyline_to_points_distance(polyline: np.ndarray, points: np.ndarray) -> float:
    """Вычислить минимальное расстояние от набора точек до ломаной."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return float("inf")
    poly = _close_poly(np.asarray(polyline, dtype=float))
    if poly.shape[0] < 2:
        return float("inf")
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    denom = np.sum(ab * ab, axis=1)
    valid = denom > 0.0
    if not bool(np.any(valid)):
        diff = pts[:, None, :] - a[None, :, :]
        return float(np.sqrt(np.min(np.sum(diff * diff, axis=2))))
    a_valid = a[valid]
    ab_valid = ab[valid]
    denom_valid = denom[valid]
    ap = pts[:, None, :] - a_valid[None, :, :]
    u = np.clip(np.sum(ap * ab_valid[None, :, :], axis=2) / denom_valid[None, :], 0.0, 1.0)
    nearest = a_valid[None, :, :] + u[:, :, None] * ab_valid[None, :, :]
    diff = pts[:, None, :] - nearest
    return float(np.sqrt(np.min(np.sum(diff * diff, axis=2))))


def _nearest_grid_index(grid: Grid2D, center: tuple[float, float]) -> tuple[int, int]:
    """Вернуть индексы ближайшего узла сетки к заданной точке."""
    r_coords = grid.r.coords()
    z_coords = grid.z.coords()
    i = int(np.argmin(np.abs(r_coords - float(center[0]))))
    j = int(np.argmin(np.abs(z_coords - float(center[1]))))
    return i, j


def _sample_limiter_points(limiter_poly: np.ndarray, grid: Grid2D) -> np.ndarray:
    """Плотно дискретизировать контур лимитера точками."""
    poly = _close_poly(np.asarray(limiter_poly, dtype=float))
    step = 0.5 * float(min(abs(float(grid.r.step)), abs(float(grid.z.step))))
    if not np.isfinite(step) or step <= 0.0:
        step = 1e-3
    points: list[np.ndarray] = []
    for a, b in zip(poly[:-1], poly[1:], strict=True):
        length = float(np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)))
        n = max(int(np.ceil(length / step)), 1)
        for k in range(n):
            alpha = float(k) / float(n)
            points.append((1.0 - alpha) * np.asarray(a, dtype=float) + alpha * np.asarray(b, dtype=float))
    points.append(poly[-1])
    return np.asarray(points, dtype=float).reshape(-1, 2)


def _sample_psi_bilinear(psi: np.ndarray, grid: Grid2D, points: np.ndarray) -> np.ndarray:
    """Интерполировать psi в произвольных точках билинейно."""
    psi_arr = np.asarray(psi, dtype=float)
    pts = np.asarray(points, dtype=float)
    r_coords = np.asarray(grid.r.coords(), dtype=float)
    z_coords = np.asarray(grid.z.coords(), dtype=float)
    out = np.full((pts.shape[0],), np.nan, dtype=float)

    for idx, (r_value, z_value) in enumerate(pts):
        if r_value < r_coords[0] or r_value > r_coords[-1] or z_value < z_coords[0] or z_value > z_coords[-1]:
            continue
        i0 = int(np.searchsorted(r_coords, float(r_value), side="right") - 1)
        j0 = int(np.searchsorted(z_coords, float(z_value), side="right") - 1)
        i0 = min(max(i0, 0), grid.r.size - 2)
        j0 = min(max(j0, 0), grid.z.size - 2)
        r0 = float(r_coords[i0])
        r1 = float(r_coords[i0 + 1])
        z0 = float(z_coords[j0])
        z1 = float(z_coords[j0 + 1])
        if r1 <= r0 or z1 <= z0:
            continue
        ar = (float(r_value) - r0) / (r1 - r0)
        az = (float(z_value) - z0) / (z1 - z0)
        q00 = float(psi_arr[j0, i0])
        q10 = float(psi_arr[j0, i0 + 1])
        q01 = float(psi_arr[j0 + 1, i0])
        q11 = float(psi_arr[j0 + 1, i0 + 1])
        out[idx] = (1.0 - ar) * (1.0 - az) * q00 + ar * (1.0 - az) * q10 + (1.0 - ar) * az * q01 + ar * az * q11
    return out


def _contours_at_level(
    psi: np.ndarray,
    grid: Grid2D,
    level: float,
    *,
    roi: tuple[int, int, int, int] | None = None,
) -> list[np.ndarray]:
    """Построить контуры psi на заданном уровне."""
    with _time_block("contours_at_level"):
        z_coords = grid.z.coords()
        r_coords = grid.r.coords()

        if roi is None:
            psi_view = np.asarray(psi, dtype=float)
            x = np.asarray(r_coords, dtype=float)
            y = np.asarray(z_coords, dtype=float)
        else:
            i0, i1, j0, j1 = roi
            psi_view = np.asarray(psi[j0 : j1 + 1, i0 : i1 + 1], dtype=float)
            x = np.asarray(r_coords[i0 : i1 + 1], dtype=float)
            y = np.asarray(z_coords[j0 : j1 + 1], dtype=float)

        if psi_view.shape[0] < 2 or psi_view.shape[1] < 2:
            return []

        cg = contour_generator(x=x, y=y, z=psi_view, name="serial", line_type="Separate")
        contours_xy = cg.lines(float(level))

        polys: list[np.ndarray] = []
        for contour in contours_xy:
            arr = np.asarray(contour, dtype=float)
            if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
                continue
            polys.append(arr)
        return polys


def _close_poly(poly: np.ndarray) -> np.ndarray:
    """Вернуть замкнутую копию ломаной."""
    if np.allclose(poly[0], poly[-1]):
        return np.asarray(poly, dtype=float)
    return np.vstack([np.asarray(poly, dtype=float), np.asarray(poly[0], dtype=float)])


def _contour_continuity_score(
    candidate: np.ndarray,
    previous: np.ndarray,
    center: tuple[float, float],
    *,
    level: float,
    level_guess: float,
    level_span: float,
    continuity_weight_radii: float,
    continuity_weight_mean_radius: float,
    continuity_weight_center: float,
    continuity_weight_area: float,
    continuity_weight_level: float,
) -> float:
    """Score real contours by continuity to the previous extracted boundary."""
    cand = _close_poly(np.asarray(candidate, dtype=float).reshape(-1, 2))
    prev = _close_poly(np.asarray(previous, dtype=float).reshape(-1, 2))
    if cand.shape[0] < 4 or prev.shape[0] < 4:
        return float("inf")
    angles = np.linspace(-np.pi, np.pi, 96, endpoint=False, dtype=float)
    cand_r = _radii_at_angles_for_score(cand, center, angles)
    prev_r = _radii_at_angles_for_score(prev, center, angles)
    finite = np.isfinite(cand_r) & np.isfinite(prev_r)
    if not bool(np.any(finite)):
        return float("inf")
    rmse = float(np.sqrt(np.mean((cand_r[finite] - prev_r[finite]) ** 2)))
    mean_term = abs(float(np.nanmean(cand_r[finite])) - float(np.nanmean(prev_r[finite])))
    center_term = float(np.linalg.norm(np.nanmean(cand[:-1], axis=0) - np.nanmean(prev[:-1], axis=0)))
    cand_area = abs(_poly_area(cand))
    prev_area = abs(_poly_area(prev))
    if cand_area > 0.0 and prev_area > 0.0:
        area_term = abs(float(np.log(cand_area / prev_area)))
    else:
        area_term = float("inf")
    level_term = abs(float(level) - float(level_guess)) / max(float(level_span), 1.0e-12)
    return (
        max(float(continuity_weight_radii), 0.0) * rmse
        + max(float(continuity_weight_mean_radius), 0.0) * mean_term
        + max(float(continuity_weight_center), 0.0) * center_term
        + max(float(continuity_weight_area), 0.0) * area_term
        + max(float(continuity_weight_level), 0.0) * level_term
    )


def _radii_at_angles_for_score(poly: np.ndarray, center: tuple[float, float], angles: np.ndarray) -> np.ndarray:
    pts = np.asarray(poly, dtype=float)
    c = np.asarray(center, dtype=float).reshape(1, 2)
    rel = pts - c
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    radii = np.sqrt(np.sum(rel * rel, axis=1))
    order = np.argsort(theta)
    theta_sorted = theta[order]
    radii_sorted = radii[order]
    if theta_sorted.size == 0:
        return np.full_like(angles, np.nan, dtype=float)
    theta_ext = np.concatenate([theta_sorted - 2.0 * np.pi, theta_sorted, theta_sorted + 2.0 * np.pi])
    radii_ext = np.concatenate([radii_sorted, radii_sorted, radii_sorted])
    return np.interp(np.asarray(angles, dtype=float), theta_ext, radii_ext)


def _is_closed_poly(poly: np.ndarray, *, tol: float = 2.5e-2) -> bool:
    """Проверить, что контур уже замкнут с заданной точностью."""
    if poly.shape[0] < 3:
        return False
    return float(np.linalg.norm(np.asarray(poly[-1], dtype=float) - np.asarray(poly[0], dtype=float))) <= float(tol)


def _encloses_center(poly: np.ndarray, center: tuple[float, float]) -> bool:
    """Проверить, что полигон охватывает заданную точку."""
    pts = _close_poly(poly)
    x = float(center[0])
    y = float(center[1])
    inside = False
    for i in range(pts.shape[0] - 1):
        x0, y0 = float(pts[i, 0]), float(pts[i, 1])
        x1, y1 = float(pts[i + 1, 0]), float(pts[i + 1, 1])
        cond = ((y0 > y) != (y1 > y))
        if cond:
            x_cross = x0 + (y - y0) * (x1 - x0) / ((y1 - y0) + 1e-30)
            if x_cross > x:
                inside = not inside
    return bool(inside)


def _prepare_limiter_shape(limiter_shape: np.ndarray | None) -> np.ndarray | None:
    """Проверить и замкнуть полигон лимитера, если он задан."""
    if limiter_shape is None:
        return None
    arr = np.asarray(limiter_shape, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
        raise ValueError(f"limiter_shape must have shape (n, 2), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("limiter_shape must contain only finite values")
    return _close_poly(arr)


def _poly_fits_limiter(poly: np.ndarray, limiter_poly: np.ndarray, *, tol: float) -> bool:
    """Проверить, что все точки контура лежат внутри или на лимитере."""
    points = np.asarray(poly, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return False
    return bool(np.all(_points_in_or_on_polygon(points[:-1], limiter_poly, tol=float(tol))))


def _points_in_or_on_polygon(points: np.ndarray, polygon: np.ndarray, tol: float) -> np.ndarray:
    """Вернуть маску точек внутри полигона или достаточно близко к его ребрам."""
    pts = np.asarray(points, dtype=float)
    poly = _close_poly(np.asarray(polygon, dtype=float))
    inside = np.zeros((pts.shape[0],), dtype=bool)
    x = pts[:, 0]
    y = pts[:, 1]

    for i in range(poly.shape[0] - 1):
        x0 = float(poly[i, 0])
        y0 = float(poly[i, 1])
        x1 = float(poly[i + 1, 0])
        y1 = float(poly[i + 1, 1])
        crosses = (y0 > y) != (y1 > y)
        x_cross = x0 + (y - y0) * (x1 - x0) / ((y1 - y0) + 1e-30)
        inside ^= crosses & (x_cross > x)

    if float(tol) <= 0.0:
        return inside

    on_edge = np.zeros_like(inside)
    tol2 = float(tol) * float(tol)
    for i in range(poly.shape[0] - 1):
        a = poly[i]
        b = poly[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 0.0:
            d2 = np.sum((pts - a) ** 2, axis=1)
        else:
            u = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
            nearest = a + u[:, None] * ab
            d2 = np.sum((pts - nearest) ** 2, axis=1)
        on_edge |= d2 <= tol2
    return inside | on_edge


def _poly_area(poly: np.ndarray) -> float:
    """Вычислить ориентированную площадь замкнутого контура."""
    pts = _close_poly(poly)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _psi_at_nearest_grid_point(psi: np.ndarray, grid: Grid2D, point: tuple[float, float]) -> float:
    """Вернуть значение psi в ближайшем узле сетки."""
    i, j = _nearest_grid_index(grid, point)
    psi_arr = np.asarray(psi, dtype=float)
    if psi_arr.shape != grid.shape:
        return float("nan")
    return float(psi_arr[j, i])


def _point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    """Вычислить минимальное расстояние от точки до ломаной."""
    p = np.asarray(point, dtype=float)
    poly = _close_poly(np.asarray(polyline, dtype=float))
    best = float("inf")
    for a, b in zip(poly[:-1], poly[1:], strict=True):
        ab = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        denom = float(np.dot(ab, ab))
        if denom <= 0.0:
            q = np.asarray(a, dtype=float)
        else:
            u = float(np.clip(np.dot(p - np.asarray(a, dtype=float), ab) / denom, 0.0, 1.0))
            q = np.asarray(a, dtype=float) + u * ab
        dist = float(np.linalg.norm(p - q))
        if dist < best:
            best = dist
    return best
