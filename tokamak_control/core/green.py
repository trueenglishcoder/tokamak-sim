from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import ellipk, ellipe


def green_axisymmetric(
    R: np.ndarray,
    Z: np.ndarray,
    Rp: np.ndarray,
    Zp: np.ndarray,
) -> np.ndarray:
    """Axisymmetric Green's function G(R, Z; Rp, Zp) for poloidal flux."""
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    Rp = np.asarray(Rp, dtype=float)
    Zp = np.asarray(Zp, dtype=float)

    denom = (R + Rp) ** 2 + (Z - Zp) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (4.0 * R * Rp) / np.where(denom == 0.0, np.inf, denom)

    m = np.clip(m, 0.0, 1.0 - 1e-15)
    K = ellipk(m)
    E = ellipe(m)

    k = np.sqrt(np.maximum(m, 1e-30))
    pref = np.sqrt(np.maximum(R, 0.0) * np.maximum(Rp, 0.0)) / (np.pi * k)
    val = (1.0 - 0.5 * m) * K - E
    G = pref * val
    return np.where((R <= 0.0) | (Rp <= 0.0), 0.0, G)


def _normalize_grouped_positions(
    coil_element_positions: Sequence[np.ndarray] | np.ndarray,
) -> list[np.ndarray]:
    """Normalize grouped actuator element positions into ``(n_i, 2)`` arrays."""
    if isinstance(coil_element_positions, np.ndarray):
        arr = np.asarray(coil_element_positions, dtype=float)
        if arr.size == 0:
            return []
        if arr.ndim == 2 and arr.shape[1] == 2:
            return [arr[i:i + 1].copy() for i in range(arr.shape[0])]
        raise ValueError(
            "coil_element_positions ndarray must have shape (n_actuators, 2) when passed directly"
        )

    grouped: list[np.ndarray] = []
    for i, group in enumerate(coil_element_positions):
        arr = np.asarray(group, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"coil_element_positions[{i}] must have shape (n_elements, 2)")
        if arr.shape[0] == 0:
            raise ValueError(f"coil_element_positions[{i}] cannot be empty")
        grouped.append(arr)
    return grouped


def _normalize_grouped_weights(
    grouped_positions: Sequence[np.ndarray],
    coil_element_weights: Sequence[np.ndarray] | None,
) -> list[np.ndarray]:
    """Нормализовать веса физических элементов в список векторов по актуаторам."""
    if coil_element_weights is None:
        return [np.ones((group.shape[0],), dtype=float) for group in grouped_positions]
    if len(coil_element_weights) != len(grouped_positions):
        raise ValueError("coil_element_weights must match grouped actuator count")

    grouped_weights: list[np.ndarray] = []
    for i, (group, weights_raw) in enumerate(zip(grouped_positions, coil_element_weights, strict=True)):
        weights = np.asarray(weights_raw, dtype=float).reshape(-1)
        if weights.shape != (group.shape[0],):
            raise ValueError(f"coil_element_weights[{i}] must have shape ({group.shape[0]},)")
        if not np.all(np.isfinite(weights)):
            raise ValueError(f"coil_element_weights[{i}] must contain only finite values")
        grouped_weights.append(weights.copy())
    return grouped_weights


def build_green_for_coils(
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    coil_element_positions: Sequence[np.ndarray] | np.ndarray,
    coil_element_weights: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Precompute one Green-response field per runtime actuator."""
    grouped = _normalize_grouped_positions(coil_element_positions)
    grouped_weights = _normalize_grouped_weights(grouped, coil_element_weights)
    nz, nr = Z_grid.shape
    out = np.zeros((len(grouped), nz, nr), dtype=float)
    for i, (group, weights) in enumerate(zip(grouped, grouped_weights, strict=True)):
        acc = np.zeros((nz, nr), dtype=float)
        for (Rp, Zp), weight in zip(group, weights, strict=True):
            acc += float(weight) * green_axisymmetric(R_grid, Z_grid, Rp, Zp)
        out[i] = acc
    return out


def build_green_for_plasma_center(
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    R0: float,
    Z0: float,
) -> np.ndarray:
    """Green array for a unit plasma current concentrated at ``(R0, Z0)``."""
    return green_axisymmetric(R_grid, Z_grid, R0, Z0)


def build_green_for_eind(
    R0: float,
    Z0: float,
    coil_element_positions: Sequence[np.ndarray] | np.ndarray,
    coil_element_weights: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Compute the Ip-to-actuator coupling vector at ``(R0, Z0)``."""
    grouped = _normalize_grouped_positions(coil_element_positions)
    grouped_weights = _normalize_grouped_weights(grouped, coil_element_weights)
    if not grouped:
        return np.zeros((0,), dtype=float)

    out = np.zeros((len(grouped),), dtype=float)
    for i, (group, weights) in enumerate(zip(grouped, grouped_weights, strict=True)):
        Rp = group[:, 0]
        Zp = group[:, 1]
        out[i] = float(np.sum(np.asarray(weights, dtype=float) * green_axisymmetric(R0, Z0, Rp, Zp)))
    return out
