# tokamak_control/geometry/boundary.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from contourpy import contour_generator

from tokamak_control.compute import ComputeBackend, normalize_compute_backend
from tokamak_control.core.grid import Grid2D
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

BoundaryMode = Literal[
    "limited",
    "diverted",
]

BoundaryStatus = Literal[
    "limited_success",
    "separatrix_success",
]


class BoundaryNotFoundError(RuntimeError):
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
) -> None:
    """Настроить сбор профиля поиска границы плазмы."""
    _PROFILER.configure(
        enabled=enabled,
        summary_every=summary_every,
        logger=get_logger("geometry.boundary.profiling"),
        reset=True,
    )


def boundary_profiling_snapshot() -> dict[str, object]:
    """Вернуть текущую сводку профиля поиска границы."""
    return _PROFILER.summary_dict(
        total_key="boundary_total",
        keys=(
            "axis_search",
            "limited_contact_levels",
            "contours_at_level",
            "candidate_filtering",
            "xpoint_search",
        ),
        path_keys=(
            "limited_success",
            "separatrix_success",
        ),
        title="boundary",
    )


def log_boundary_profiling_summary() -> None:
    """Записать в лог сводку профиля поиска границы."""
    _PROFILER.log_summary(
        total_key="boundary_total",
        keys=(
            "axis_search",
            "limited_contact_levels",
            "contours_at_level",
            "candidate_filtering",
            "xpoint_search",
        ),
        path_keys=(
            "limited_success",
            "separatrix_success",
        ),
        title="boundary",
    )


def find_plasma_boundary_with_status(
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
    boundary_mode: BoundaryMode = "limited",
    compute_backend: ComputeBackend | str = "cpu",
    gpu_device: str = "cuda:0",
) -> tuple[np.ndarray, float, BoundaryStatus]:
    """Найти контур границы плазмы по полю psi.

    В режиме ``limited`` граница задается внешней валидной замкнутой flux-
    поверхностью вокруг магнитной оси, которая целиком лежит внутри лимитера.
    В режиме ``diverted`` граница задается сепаратрисой на уровне X-point.
    Если выбранное физическое правило не дает контура, граница не считается
    найденной.
    """
    mode = _normalize_boundary_mode(boundary_mode)
    backend = normalize_compute_backend(compute_backend)
    if backend == "gpu":
        from tokamak_control.geometry.boundary_gpu import find_plasma_boundary_gpu_with_status

        return find_plasma_boundary_gpu_with_status(
            psi,
            grid,
            center,
            n_levels=n_levels,
            prev_level=prev_level,
            prev_poly=prev_poly,
            local_n_levels=local_n_levels,
            local_span_frac=local_span_frac,
            target_mean_radius=target_mean_radius,
            target_switch_ratio=target_switch_ratio,
            target_switch_abs_delta=target_switch_abs_delta,
            local_bbox_pad_r=local_bbox_pad_r,
            local_bbox_pad_z=local_bbox_pad_z,
            limiter_shape=limiter_shape,
            boundary_mode=mode,
            gpu_device=gpu_device,
        )

    with _time_block("boundary_total"):
        if psi.shape != grid.shape:
            raise ValueError(f"psi shape {psi.shape} != grid shape {grid.shape}")

        limiter_poly = _prepare_limiter_shape(limiter_shape)
        limiter_tol = 0.5 * min(abs(float(grid.r.step)), abs(float(grid.z.step)))
        with _time_block("axis_search"):
            axis = _find_magnetic_axis(
                psi=psi,
                grid=grid,
                center=center,
                limiter_poly=limiter_poly,
                limiter_tol=limiter_tol,
            )
        search_center = axis.point

        if mode == "limited":
            limited_boundary = _limited_boundary(
                psi=psi,
                grid=grid,
                axis=axis,
                limiter_poly=limiter_poly,
                limiter_tol=limiter_tol,
                n_levels=n_levels,
            )
            if limited_boundary is not None:
                poly, level = limited_boundary
                _record_path("limited_success")
                return _close_poly(poly), float(level), "limited_success"
            raise BoundaryNotFoundError("No limited plasma boundary found inside limiter")

        separatrix = _separatrix_boundary(
            psi=psi,
            grid=grid,
            axis=search_center,
            limiter_poly=limiter_poly,
            limiter_tol=limiter_tol,
        )
        if separatrix is not None:
            poly, level = separatrix
            _record_path("separatrix_success")
            return _close_poly(poly), float(level), "separatrix_success"

        raise BoundaryNotFoundError("No diverted plasma separatrix boundary found")


def boundary_status_is_real(status: BoundaryStatus) -> bool:
    """Вернуть True для статусов реально найденной границы."""
    return status in {"limited_success", "separatrix_success"}


def _normalize_boundary_mode(mode: BoundaryMode | str) -> BoundaryMode:
    """Нормализовать режим физического определения границы."""
    mode_text = str(mode).strip().lower()
    if mode_text not in {"limited", "diverted"}:
        raise ValueError(f"boundary_mode must be 'limited' or 'diverted', got {mode!r}")
    return cast(BoundaryMode, mode_text)


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
