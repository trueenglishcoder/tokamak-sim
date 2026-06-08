from __future__ import annotations

from typing import Tuple

import numpy as np


def cart_to_polar(points: np.ndarray, center: Tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian (R, Z) points to polar coordinates around ``center``."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")

    R0, Z0 = center
    dR = points[:, 0] - float(R0)
    dZ = points[:, 1] - float(Z0)
    angles = np.arctan2(dZ, dR)
    radii = np.hypot(dR, dZ)
    return angles, radii


def sort_by_angle(angles: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort angle and radius pairs by increasing angle."""
    angles = np.asarray(angles, dtype=float).reshape(-1)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    if angles.shape != radii.shape:
        raise ValueError(
            f"angles and radii must have the same shape, got {angles.shape} and {radii.shape}"
        )
    idx = np.argsort(angles)
    return angles[idx], radii[idx], idx


def _as_closed_polyline(polyline: np.ndarray) -> np.ndarray:
    poly = np.asarray(polyline, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("polyline must have shape (N, 2)")
    if poly.shape[0] < 2:
        raise ValueError("polyline must contain at least two vertices")
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly


def _ray_segment_intersection_radius(
    center: tuple[float, float],
    angle: float,
    a: np.ndarray,
    b: np.ndarray,
    *,
    atol: float = 1e-12,
) -> float | None:
    """Return the positive ray parameter where the ray at ``angle`` hits segment AB."""
    d = np.array([np.cos(float(angle)), np.sin(float(angle))], dtype=float)
    c = np.array(center, dtype=float)
    s = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rhs = np.asarray(a, dtype=float) - c

    M = np.array([[d[0], -s[0]], [d[1], -s[1]]], dtype=float)
    det = float(np.linalg.det(M))
    if abs(det) <= atol:
        return None

    t_ray, u_seg = np.linalg.solve(M, rhs)
    t_ray = float(t_ray)
    u_seg = float(u_seg)
    if t_ray < 0.0 or u_seg < -atol or u_seg > 1.0 + atol:
        return None
    return max(0.0, t_ray)


def interpolate_radii_at_angles(
    boundary_angles: np.ndarray,
    boundary_radii: np.ndarray,
    query_angles: np.ndarray,
) -> np.ndarray:
    """
    Interpolate boundary radii at given angles using periodic extension.

    This helper remains available for angle-sampled data, but the primary
    boundary-to-radii path for closed polylines should use
    ``radii_from_polyline_ray_intersections`` in ``control.linearization``.
    """
    ba = np.asarray(boundary_angles, dtype=float).reshape(-1)
    br = np.asarray(boundary_radii, dtype=float).reshape(-1)
    qa = np.asarray(query_angles, dtype=float).reshape(-1)

    if ba.shape != br.shape:
        raise ValueError(
            f"boundary_angles and boundary_radii must have the same shape, got {ba.shape} and {br.shape}"
        )
    if ba.size == 0:
        raise ValueError("boundary_angles and boundary_radii must be non-empty")

    two_pi = 2.0 * np.pi
    ba = (ba + two_pi) % two_pi
    qa = (qa + two_pi) % two_pi

    order = np.argsort(ba)
    ba = ba[order]
    br = br[order]

    ba_ext = np.concatenate([ba - two_pi, ba, ba + two_pi])
    br_ext = np.concatenate([br, br, br])

    return np.interp(qa, ba_ext, br_ext)


def radii_from_polyline_ray_intersections(
    polyline: np.ndarray,
    center: tuple[float, float],
    query_angles: np.ndarray,
) -> np.ndarray:
    """
    Evaluate radii by intersecting each query ray with the polyline exactly.

    For each angle, all segment intersections with the ray starting at
    ``center`` are collected and the farthest positive radius is used. This is
    more faithful to a measuring-ray construction than direct interpolation in
    angle space.
    """
    poly = _as_closed_polyline(polyline)
    qa = np.asarray(query_angles, dtype=float).reshape(-1)

    if qa.size == 0:
        return np.empty((0,), dtype=float)

    c = np.array(center, dtype=float).reshape(2)
    seg_starts = np.asarray(poly[:-1], dtype=float)
    seg_vecs = np.asarray(poly[1:] - poly[:-1], dtype=float)

    directions = np.stack([np.cos(qa), np.sin(qa)], axis=1)
    rhs = seg_starts[None, :, :] - c.reshape(1, 1, 2)
    d = directions[:, None, :]
    s = seg_vecs[None, :, :]

    den = d[..., 0] * s[..., 1] - d[..., 1] * s[..., 0]
    rhs_cross_s = rhs[..., 0] * s[..., 1] - rhs[..., 1] * s[..., 0]
    rhs_cross_d = rhs[..., 0] * d[..., 1] - rhs[..., 1] * d[..., 0]

    valid_den = np.abs(den) > 1.0e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ray = rhs_cross_s / den
        u_seg = rhs_cross_d / den

    valid = valid_den & (t_ray >= 0.0) & (u_seg >= -1.0e-12) & (u_seg <= 1.0 + 1.0e-12)
    hits = np.where(valid, np.maximum(t_ray, 0.0), -np.inf)
    out = np.max(hits, axis=1)
    if not np.all(np.isfinite(out)):
        raise ValueError("No ray/polyline intersection found for a measurement angle")
    return np.asarray(out, dtype=float)
