from __future__ import annotations

import numpy as np


def _margin(values: np.ndarray, limits: np.ndarray, value_name: str, limit_name: str) -> np.ndarray:
    """Вычислить относительный запас до симметричного предела по модулю."""
    val = np.asarray(values, dtype=float).reshape(-1)
    lim = np.asarray(limits, dtype=float).reshape(-1)
    if val.shape != lim.shape:
        raise ValueError(f"{value_name} shape {val.shape} must match {limit_name} shape {lim.shape}")
    if not np.all(np.isfinite(val)):
        raise ValueError(f"{value_name} must contain only finite values")
    if not np.all(np.isfinite(lim)) or np.any(lim <= 0.0):
        raise ValueError(f"{limit_name} must contain only finite positive values")
    return 1.0 - np.abs(val) / lim


def current_limit_margin(currents: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Вернуть относительный запас активных токов до пределов."""
    return _margin(currents, limits, "currents", "limits")


def derivative_limit_margin(derivatives: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Вернуть относительный запас производных токов до пределов."""
    return _margin(derivatives, limits, "derivatives", "limits")
