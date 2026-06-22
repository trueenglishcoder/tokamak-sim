from __future__ import annotations

import numpy as np

from tokamak_control.geometry.coordinates import cart_to_polar, interpolate_radii_at_angles


def legacy_measurement_angles_from_actuators(
    boundary_poly: np.ndarray,
    center: tuple[float, float],
    actuator_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize old tokamak-sim-0 boundary measurements from actuator centroids.

    The MATLAB startup path selected boundary measuring points by nearest match
    to PFC actuator geometry, then tracked the boundary by angle/radius. This
    helper returns the corresponding center-relative angles and initial radii.
    """
    poly = _open_polyline(boundary_poly)
    actuators = np.asarray(actuator_positions, dtype=float).reshape(-1, 2)
    if poly.shape[0] < 3:
        raise ValueError("boundary_poly must contain at least three points")
    if actuators.shape[0] == 0:
        raise ValueError("actuator_positions must contain at least one actuator")

    chosen: list[np.ndarray] = []
    for pos in actuators:
        dist2 = np.sum((poly - pos.reshape(1, 2)) ** 2, axis=1)
        chosen.append(poly[int(np.argmin(dist2))])
    points = np.stack(chosen, axis=0)
    angles, radii = cart_to_polar(points, center)
    return np.asarray(angles, dtype=float), np.asarray(radii, dtype=float)


def legacy_radii_at_angles(
    boundary_poly: np.ndarray,
    center: tuple[float, float],
    query_angles: np.ndarray,
) -> np.ndarray:
    """Return old-style periodic angle/radius interpolated boundary radii."""
    poly = _open_polyline(boundary_poly)
    angles, radii = cart_to_polar(poly, center)
    return interpolate_radii_at_angles(angles, radii, np.asarray(query_angles, dtype=float))


def legacy_boundary_errors(
    boundary_poly: np.ndarray,
    center: tuple[float, float],
    measure_angles: np.ndarray,
    ref_radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute old-style measured radii and radius errors."""
    measured = legacy_radii_at_angles(boundary_poly, center, measure_angles)
    ref = np.asarray(ref_radii, dtype=float).reshape(-1)
    if measured.shape != ref.shape:
        raise ValueError(f"legacy measured radii shape {measured.shape} != ref_radii {ref.shape}")
    return measured, measured - ref


def _open_polyline(polyline: np.ndarray) -> np.ndarray:
    poly = np.asarray(polyline, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("polyline must have shape (N, 2)")
    if poly.shape[0] >= 2 and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    return poly
