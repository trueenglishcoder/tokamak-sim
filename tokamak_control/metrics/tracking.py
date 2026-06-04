from __future__ import annotations

import numpy as np


def ip_abs_error(ip: float, ip_ref: float) -> float:
    """Вернуть абсолютную ошибку тока плазмы."""
    value = abs(float(ip) - float(ip_ref))
    if not np.isfinite(value):
        raise ValueError("Ip error must be finite")
    return float(value)


def ip_normalized_abs_error(ip: float, ip_ref: float, ip_scale: float) -> float:
    """Вернуть абсолютную ошибку тока плазмы, нормированную на масштаб."""
    scale = float(ip_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"ip_scale must be finite and > 0, got {ip_scale!r}")
    return float(ip_abs_error(ip, ip_ref) / scale)
