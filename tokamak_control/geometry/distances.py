from __future__ import annotations

from typing import Tuple

import numpy as np


def closest_point_on_segment(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Return the closest point to ``p`` on segment AB and the segment parameter."""
    ap = np.asarray(p, dtype=float) - np.asarray(a, dtype=float)
    ab = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    denom = float(np.dot(ab, ab))
    if denom <= 0.0:
        return np.asarray(a, dtype=float).copy(), 0.0
    t = float(np.clip(np.dot(ap, ab) / denom, 0.0, 1.0))
    return np.asarray(a, dtype=float) + t * ab, t


def closest_points_on_polyline(
    points: np.ndarray, polyline: np.ndarray
) -> np.ndarray:
    """For each point, return the closest point on the piecewise-linear polyline."""
    points = np.asarray(points, dtype=float)
    polyline = np.asarray(polyline, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (M, 2)")
    if polyline.ndim != 2 or polyline.shape[1] != 2 or polyline.shape[0] < 2:
        raise ValueError("polyline must have shape (N, 2) with N >= 2")

    seg_starts = polyline[:-1]
    seg_ends = polyline[1:]

    out = np.empty_like(points)
    for i, p in enumerate(points):
        best_d2 = np.inf
        best_q = None
        for a, b in zip(seg_starts, seg_ends):
            q, _ = closest_point_on_segment(p, a, b)
            d2 = float(np.sum((p - q) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best_q = q
        out[i] = best_q
    return out


def compute_signed_radial_errors(
    real_pts: np.ndarray,
    desired_pts: np.ndarray,
    center: Tuple[float, float],
) -> np.ndarray:
    """
    Compute signed geometric errors between corresponding real and desired points.

    Magnitude is the Euclidean point-to-point distance. The sign is negative
    when the real point is closer to ``center`` than the desired point, and
    positive otherwise.
    """
    real_pts = np.asarray(real_pts, dtype=float)
    desired_pts = np.asarray(desired_pts, dtype=float)
    if real_pts.shape != desired_pts.shape or real_pts.ndim != 2 or real_pts.shape[1] != 2:
        raise ValueError("real_pts and desired_pts must have shape (M,2) and match")

    center_arr = np.asarray(center, dtype=float).reshape(1, 2)
    real_r = np.linalg.norm(real_pts - center_arr, axis=1)
    desired_r = np.linalg.norm(desired_pts - center_arr, axis=1)
    sign = np.where(real_r < desired_r, -1.0, 1.0)
    mag = np.linalg.norm(real_pts - desired_pts, axis=1)
    return sign * mag
