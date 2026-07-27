"""Именованные контуры лимитеров для расчета и визуализации."""

from __future__ import annotations

import numpy as np


# Координаты заданы как пары (R, Z) в метрах.
# Контур замкнут повторением первой точки в конце массива.
_T15MD_LIMITER = np.asarray(
    [
        [0.82, 1.08],
        [1.00, 1.38],
        [1.62, 1.38],
        [1.65, 1.17],
        [2.02, 0.79],
        [2.25, 0.45],
        [2.25, -0.45],
        [2.14, -0.74],
        [1.87, -0.94],
        [1.63, -1.18],
        [1.45, -1.41],
        [1.44, -1.74],
        [1.00, -1.74],
        [0.82, -1.44],
        [0.82, -0.83],
        [0.77, -0.70],
        [0.77, -0.55],
        [0.77, 0.29],
        [0.77, 0.53],
        [0.82, 0.63],
        [0.82, 1.08],
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
