"""Геометрия стенки камеры для модели токов в стенке."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, repr=True)
class WallGeometry:
    """Геометрия стенки камеры, представленная как набор сегментов.

    Стенка разбивается на N сегментов, каждый сегмент — отрезок между
    соседними точками полигона вакуумной камеры. Центры сегментов
    используются как точки приложения токов в стенке.

    Attributes
    ----------
    wall_points : np.ndarray
        Центры сегментов стенки, shape (N, 2), где N — количество сегментов.
        Каждая строка содержит (R, Z) координаты центра сегмента.
    wall_lengths : np.ndarray
        Длины сегментов стенки, shape (N,). Используется для вычисления
        полного тока и нормализации.
    n_segments : int
        Количество сегментов стенки.
    """

    wall_points: np.ndarray
    wall_lengths: np.ndarray
    n_segments: int

    @classmethod
    def from_polygon(cls, polygon: np.ndarray) -> WallGeometry:
        """Создать геометрию стенки из произвольного полигона.

        Parameters
        ----------
        polygon : np.ndarray
            Полигон, shape (M, 2), где M — количество точек.
            Полигон должен быть замкнутым (первая и последняя точки совпадают)
            или будет автоматически замкнут.

        Returns
        -------
        WallGeometry
            Геометрия стенки с N = M-1 сегментами (если полигон замкнут)
            или N = M сегментами (если полигон не замкнут).
        """
        poly = np.asarray(polygon, dtype=float)
        if poly.ndim != 2 or poly.shape[1] != 2:
            raise ValueError(f"polygon must have shape (M, 2), got {poly.shape}")
        if poly.shape[0] < 3:
            raise ValueError(f"polygon must have at least 3 points, got {poly.shape[0]}")

        # Замыкаем полигон, если необходимо
        if not np.allclose(poly[0], poly[-1], rtol=0.0, atol=1.0e-9):
            poly = np.vstack([poly, poly[0:1]])

        # Создаём сегменты: каждый сегмент — отрезок между соседними точками
        n_segments = poly.shape[0] - 1
        wall_points = np.zeros((n_segments, 2), dtype=float)
        wall_lengths = np.zeros(n_segments, dtype=float)

        for i in range(n_segments):
            p0 = poly[i]
            p1 = poly[i + 1]
            # Центр сегмента
            wall_points[i] = 0.5 * (p0 + p1)
            # Длина сегмента
            wall_lengths[i] = float(np.linalg.norm(p1 - p0))

        return cls(wall_points=wall_points, wall_lengths=wall_lengths, n_segments=n_segments)

    @classmethod
    def from_limiter(cls, limiter_shape: np.ndarray) -> WallGeometry:
        """Создать геометрию стенки из полигона лимитера (устаревший метод).

        Рекомендуется использовать from_polygon или from_vacuum_chamber.
        """
        return cls.from_polygon(limiter_shape)

    @classmethod
    def from_vacuum_chamber(cls, vacuum_chamber_shape: np.ndarray) -> WallGeometry:
        """Создать геометрию стенки из полигона вакуумной камеры.

        Parameters
        ----------
        vacuum_chamber_shape : np.ndarray
            Полигон вакуумной камеры, shape (M, 2).

        Returns
        -------
        WallGeometry
            Геометрия стенки.
        """
        return cls.from_polygon(vacuum_chamber_shape)

    def total_perimeter(self) -> float:
        """Вернуть полный периметр стенки (сумма длин сегментов)."""
        return float(np.sum(self.wall_lengths))
