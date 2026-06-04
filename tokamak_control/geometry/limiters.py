"""Именованные контуры лимитеров для расчета и визуализации."""

from __future__ import annotations

import numpy as np


_T15MD_LIMITER = np.asarray(
    [
        [0.80, 1.10],
        [1.10, 1.38],
        [1.75, 1.38],
        [1.75, 1.18],
        [1.92, 1.05],
        [2.08, 0.82],
        [2.24, 0.52],
        [2.25, 0.16],
        [2.25, -0.46],
        [2.12, -0.74],
        [1.88, -0.95],
        [1.63, -1.18],
        [1.43, -1.47],
        [1.43, -1.70],
        [1.08, -1.70],
        [0.82, -1.25],
        [0.80, -0.72],
        [0.80, -0.30],
        [0.80, 0.28],
        [0.80, 0.76],
        [0.80, 1.10],
    ],
    dtype=float,
)

_LIMITERS: dict[str, np.ndarray] = {
    "T15MD": _T15MD_LIMITER,
}


def limiter_names() -> tuple[str, ...]:
    """Вернуть имена поддерживаемых лимитеров."""
    return tuple(sorted(_LIMITERS))


def get_limiter_shape(name: str | None) -> np.ndarray | None:
    """Вернуть копию именованного контура лимитера или None."""
    if name is None:
        return None
    key = str(name).upper()
    if key not in _LIMITERS:
        choices = ", ".join(limiter_names())
        raise ValueError(f"Unknown limiter {name!r}. Available limiters: {choices}")
    return _LIMITERS[key].copy()
