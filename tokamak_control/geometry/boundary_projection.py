"""Projection of a dense physical boundary to fixed-angle RL observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokamak_control.geometry.boundary_common import close_poly


@dataclass(frozen=True, slots=True, repr=True)
class FixedAngleProjection:
    radii: np.ndarray
    intersection_counts: np.ndarray
    valid: bool
    reason: str | None


def project_boundary_to_fixed_angles(
    boundary: np.ndarray,
    center: tuple[float, float],
    angles: np.ndarray,
    *,
    dedup_tolerance: float = 1.0e-9,
) -> FixedAngleProjection:
    """Intersect each center-origin ray with the dense core boundary.

    A valid radial representation requires exactly one forward intersection per
    ray. Multiple intersections are reported explicitly instead of silently
    choosing a near or far branch.
    """
    poly = close_poly(np.asarray(boundary, dtype=float))
    if poly.ndim != 2 or poly.shape[0] < 4 or poly.shape[1] != 2:
        raise ValueError(f"boundary must be a closed polyline with shape (N, 2), got {poly.shape}")
    query = np.asarray(angles, dtype=float).reshape(-1)
    radii = np.full((query.size,), np.nan, dtype=float)
    counts = np.zeros((query.size,), dtype=np.int64)
    c = np.asarray(center, dtype=float).reshape(2)
    starts = poly[:-1]
    vectors = poly[1:] - poly[:-1]

    for angle_index, angle in enumerate(query):
        d = np.array([np.cos(float(angle)), np.sin(float(angle))], dtype=float)
        rhs = starts - c[None, :]
        denominator = d[0] * vectors[:, 1] - d[1] * vectors[:, 0]
        valid_den = np.abs(denominator) > 1.0e-14
        t = np.full_like(denominator, np.nan, dtype=float)
        u = np.full_like(denominator, np.nan, dtype=float)
        t[valid_den] = (rhs[valid_den, 0] * vectors[valid_den, 1] - rhs[valid_den, 1] * vectors[valid_den, 0]) / denominator[valid_den]
        u[valid_den] = (rhs[valid_den, 0] * d[1] - rhs[valid_den, 1] * d[0]) / denominator[valid_den]
        hits = t[valid_den & (t >= 0.0) & (u >= -1.0e-12) & (u <= 1.0 + 1.0e-12)]
        if hits.size:
            hits = np.sort(hits[np.isfinite(hits)])
            unique: list[float] = []
            for value in hits:
                if not unique or abs(float(value) - unique[-1]) > float(dedup_tolerance):
                    unique.append(float(value))
            counts[angle_index] = len(unique)
            if unique:
                radii[angle_index] = unique[0]

    if bool(np.all(counts == 1)) and bool(np.all(np.isfinite(radii))):
        return FixedAngleProjection(radii=radii, intersection_counts=counts, valid=True, reason=None)
    missing = int(np.count_nonzero(counts == 0))
    multiple = int(np.count_nonzero(counts > 1))
    reason = f"fixed-angle boundary representation is not single-valued: missing_rays={missing}, multi_hit_rays={multiple}"
    return FixedAngleProjection(radii=radii, intersection_counts=counts, valid=False, reason=reason)
