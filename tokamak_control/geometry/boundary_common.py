from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from tokamak_control.core.grid import Grid2D


BoundaryMode = Literal["legacy_contour", "legacy_contour_limited", "tracked_flux_contour"]
BoundaryStatus = Literal[
    "legacy_contour_success",
    "legacy_contour_limited_success",
    "tracked_flux_contour_success",
    "tracked_flux_contour_reset",
]


class BoundaryNotFoundError(RuntimeError):
    """No physically defined plasma boundary was found."""


@dataclass(frozen=True, slots=True, repr=True)
class MagneticAxis:
    point: tuple[float, float]
    level: float
    kind: Literal["maximum", "minimum"]


def boundary_status_is_real(status: BoundaryStatus) -> bool:
    """Return True when a boundary status string represents a found boundary."""
    return status in {
        "legacy_contour_success",
        "legacy_contour_limited_success",
        "tracked_flux_contour_success",
        "tracked_flux_contour_reset",
    }


def normalize_boundary_mode(mode: BoundaryMode | str) -> BoundaryMode:
    """Normalize and validate the old-parity boundary extraction mode."""
    mode_text = str(mode).strip().lower()
    if mode_text not in {"legacy_contour", "legacy_contour_limited", "tracked_flux_contour"}:
        raise ValueError(
            "boundary_mode must be 'legacy_contour', 'legacy_contour_limited', "
            "or 'tracked_flux_contour', "
            f"got {mode!r}"
        )
    return cast(BoundaryMode, mode_text)


def close_poly(poly: np.ndarray) -> np.ndarray:
    arr = np.asarray(poly, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != 2:
        return arr.reshape(-1, 2)
    if np.allclose(arr[0], arr[-1]):
        return arr
    return np.vstack([arr, arr[0]])


def is_closed_poly(poly: np.ndarray, *, tol: float = 2.5e-2) -> bool:
    arr = np.asarray(poly, dtype=float)
    if arr.shape[0] < 3:
        return False
    return float(np.linalg.norm(arr[-1] - arr[0])) <= float(tol)


def encloses_center(poly: np.ndarray, center: tuple[float, float]) -> bool:
    pts = close_poly(poly)
    x = float(center[0])
    y = float(center[1])
    inside = False
    for i in range(pts.shape[0] - 1):
        x0, y0 = float(pts[i, 0]), float(pts[i, 1])
        x1, y1 = float(pts[i + 1, 0]), float(pts[i + 1, 1])
        cond = (y0 > y) != (y1 > y)
        if cond:
            x_cross = x0 + (y - y0) * (x1 - x0) / ((y1 - y0) + 1e-30)
            if x_cross > x:
                inside = not inside
    return bool(inside)


def prepare_limiter_shape(limiter_shape: np.ndarray | None) -> np.ndarray | None:
    if limiter_shape is None:
        return None
    arr = np.asarray(limiter_shape, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
        raise ValueError(f"limiter_shape must have shape (n, 2), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("limiter_shape must contain only finite values")
    return close_poly(arr)


def poly_fits_limiter(poly: np.ndarray, limiter_poly: np.ndarray, *, tol: float) -> bool:
    points = np.asarray(poly, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return False
    return bool(np.all(points_in_or_on_polygon(points[:-1], limiter_poly, tol=float(tol))))


def points_in_or_on_polygon(points: np.ndarray, polygon: np.ndarray, tol: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    poly = close_poly(np.asarray(polygon, dtype=float))
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


def poly_area(poly: np.ndarray) -> float:
    pts = close_poly(poly)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    p = np.asarray(point, dtype=float)
    poly = close_poly(np.asarray(polyline, dtype=float))
    best = float("inf")
    for a, b in zip(poly[:-1], poly[1:], strict=True):
        ab = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        denom = float(np.dot(ab, ab))
        if denom <= 0.0:
            q = np.asarray(a, dtype=float)
        else:
            u = float(np.clip(np.dot(p - np.asarray(a, dtype=float), ab) / denom, 0.0, 1.0))
            q = np.asarray(a, dtype=float) + u * ab
        best = min(best, float(np.linalg.norm(p - q)))
    return best


def polyline_to_points_distance(polyline: np.ndarray, points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return float("inf")
    poly = close_poly(np.asarray(polyline, dtype=float))
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


def sample_limiter_points(limiter_poly: np.ndarray, grid: Grid2D) -> np.ndarray:
    poly = close_poly(np.asarray(limiter_poly, dtype=float))
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


def ordered_limiter_contact_levels(values: np.ndarray, axis: MagneticAxis) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros((0,), dtype=float)
    unique = np.unique(np.round(vals, 14))
    if axis.kind == "maximum":
        return np.sort(unique)[::-1]
    return np.sort(unique)


def nearest_grid_index(grid: Grid2D, point: tuple[float, float]) -> tuple[int, int]:
    r_coords = grid.r.coords()
    z_coords = grid.z.coords()
    i = int(np.argmin(np.abs(r_coords - float(point[0]))))
    j = int(np.argmin(np.abs(z_coords - float(point[1]))))
    return i, j
