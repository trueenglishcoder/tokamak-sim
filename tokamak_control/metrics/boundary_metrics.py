from __future__ import annotations

import numpy as np


def _as_matching_1d(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> tuple[np.ndarray, np.ndarray]:
    """Привести два радиальных профиля к согласованным одномерным массивам."""
    arr_a = np.asarray(a, dtype=float).reshape(-1)
    arr_b = np.asarray(b, dtype=float).reshape(-1)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"{name_a} shape {arr_a.shape} must match {name_b} shape {arr_b.shape}")
    if arr_a.size == 0:
        raise ValueError(f"{name_a} and {name_b} must not be empty")
    return arr_a, arr_b


def radii_error(radii: np.ndarray, radii_ref: np.ndarray) -> np.ndarray:
    """Вернуть покомпонентную ошибку sampled radii относительно reference radii."""
    r, ref = _as_matching_1d(radii, radii_ref, "radii", "radii_ref")
    err = r - ref
    if not np.all(np.isfinite(err)):
        raise ValueError("radii error must contain only finite values")
    return err


def radii_rmse(radii: np.ndarray, radii_ref: np.ndarray) -> float:
    """Вернуть RMSE sampled radii относительно reference radii."""
    err = radii_error(radii, radii_ref)
    return float(np.sqrt(np.mean(err * err)))


def normalized_radii_rmse(radii: np.ndarray, radii_ref: np.ndarray, radius_scale: float) -> float:
    """Вернуть RMSE sampled radii, нормированный на радиальный масштаб."""
    scale = float(radius_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"radius_scale must be finite and > 0, got {radius_scale!r}")
    return float(radii_rmse(radii, radii_ref) / scale)
