"""Модель токов в стенке камеры для tokamak-sim.

Стенка камеры моделируется как резистивный проводник. При изменении магнитного
потока через стенку в ней индуцируются вихревые токи, которые создают своё
магнитное поле и влияют на psi-поле плазмы.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tokamak_control.core.green import build_green_for_wall, build_green_wall_flux
from tokamak_control.geometry.wall import WallGeometry


@dataclass
class WallModel:
    """Модель токов в стенке камеры (резистивный проводник).

    Стенка разбита на N сегментов. В каждом сегменте течёт ток J_wall[i].
    Уравнение ОДУ для токов:
        L · dJ/dt + R · J = -dΦ/dt
    где L = G_wall_flux (матрица индуктивностей), R = R_wall * I (сопротивление).

    При R_wall = 0 работает как идеальный проводник (старое поведение).

    Attributes
    ----------
    wall_geometry : WallGeometry
        Геометрия стенки (точки и длины сегментов).
    G_wall : np.ndarray
        Функции Грина от сегментов стенки ко всем точкам сетки, shape (N, nz, nr).
    G_wall_flux : np.ndarray
        Взаимная индуктивность между сегментами стенки, shape (N, N).
    G_wall_inv : np.ndarray
        Предвычисленная обратная матрица G_wall_flux для быстрого решения, shape (N, N).
    J_wall : np.ndarray
        Токи в сегментах стенки, shape (N,). Инициализируются нулями.
    R_wall : float
        Сопротивление стенки в Омах. По умолчанию 0.0 (идеальный проводник).
    dt : float
        Временной шаг для интегрирования ОДУ. По умолчанию 1e-5.
    psi_wall_prev : np.ndarray
        Предыдущее значение psi в точках стенки для вычисления dΦ/dt.
    LHS_inv : np.ndarray
        Предвычисленная обратная матрица (L/dt + R/2) для метода трапеций.
    """

    wall_geometry: WallGeometry
    G_wall: np.ndarray
    G_wall_flux: np.ndarray
    G_wall_inv: np.ndarray
    J_wall: np.ndarray = field(default_factory=lambda: np.array([]))
    R_wall: float = 0.0
    dt: float = 1.0e-5
    psi_wall_prev: np.ndarray = field(default_factory=lambda: np.array([]))
    LHS_inv: np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def from_polygon(
        cls,
        polygon: np.ndarray,
        R_grid: np.ndarray,
        Z_grid: np.ndarray,
        regularization: float = 1.0e-3,
        R_wall: float = 0.0,
        dt: float = 1.0e-5,
    ) -> WallModel:
        """Создать модель стенки из произвольного полигона.

        Parameters
        ----------
        polygon : np.ndarray
            Полигон, shape (M, 2).
        R_grid : np.ndarray
            Сетка радиальных координат, shape (nz, nr).
        Z_grid : np.ndarray
            Сетка вертикальных координат, shape (nz, nr).
        regularization : float
            Параметр регуляризации для самовлияния (по умолчанию 1e-3).
        R_wall : float
            Сопротивление стенки в Омах. По умолчанию 0.0 (идеальный проводник).
        dt : float
            Временной шаг для интегрирования ОДУ. По умолчанию 1e-5.

        Returns
        -------
        WallModel
            Инициализированная модель стенки.
        """
        wall_geometry = WallGeometry.from_polygon(polygon)
        G_wall = build_green_for_wall(R_grid, Z_grid, wall_geometry.wall_points)
        G_wall_flux = build_green_wall_flux(wall_geometry.wall_points, regularization=regularization)

        # Предвычисляем обратную матрицу для быстрого решения
        # Используем псевдообратную для устойчивости
        G_wall_inv = np.linalg.pinv(G_wall_flux, rcond=1.0e-10)

        n_segments = wall_geometry.n_segments
        J_wall = np.zeros(n_segments, dtype=float)
        psi_wall_prev = np.zeros(n_segments, dtype=float)

        # Предвычисляем LHS_inv для метода трапеций
        # (L/dt + R/2) · J_new = ...
        L = G_wall_flux
        R = R_wall * np.eye(n_segments)
        LHS = L / dt + R / 2.0
        LHS_inv = np.linalg.pinv(LHS, rcond=1.0e-10)

        return cls(
            wall_geometry=wall_geometry,
            G_wall=G_wall,
            G_wall_flux=G_wall_flux,
            G_wall_inv=G_wall_inv,
            J_wall=J_wall,
            R_wall=R_wall,
            dt=dt,
            psi_wall_prev=psi_wall_prev,
            LHS_inv=LHS_inv,
        )

    @classmethod
    def from_limiter(
        cls,
        limiter_shape: np.ndarray,
        R_grid: np.ndarray,
        Z_grid: np.ndarray,
        regularization: float = 1.0e-3,
        R_wall: float = 0.0,
        dt: float = 1.0e-5,
    ) -> WallModel:
        """Создать модель стенки из полигона лимитера (устаревший метод).

        Рекомендуется использовать from_polygon или from_vacuum_chamber.
        """
        return cls.from_polygon(
            limiter_shape, R_grid, Z_grid,
            regularization=regularization, R_wall=R_wall, dt=dt,
        )

    @classmethod
    def from_vacuum_chamber(
        cls,
        vacuum_chamber_shape: np.ndarray,
        R_grid: np.ndarray,
        Z_grid: np.ndarray,
        regularization: float = 1.0e-3,
        R_wall: float = 0.0,
        dt: float = 1.0e-5,
    ) -> WallModel:
        """Создать модель стенки из полигона вакуумной камеры.

        Parameters
        ----------
        vacuum_chamber_shape : np.ndarray
            Полигон вакуумной камеры, shape (M, 2).
        R_grid : np.ndarray
            Сетка радиальных координат, shape (nz, nr).
        Z_grid : np.ndarray
            Сетка вертикальных координат, shape (nz, nr).
        regularization : float
            Параметр регуляризации для самовлияния (по умолчанию 1e-3).
        R_wall : float
            Сопротивление стенки в Омах. По умолчанию 0.0 (идеальный проводник).
        dt : float
            Временной шаг для интегрирования ОДУ. По умолчанию 1e-5.

        Returns
        -------
        WallModel
            Инициализированная модель стенки.
        """
        return cls.from_polygon(
            vacuum_chamber_shape, R_grid, Z_grid,
            regularization=regularization, R_wall=R_wall, dt=dt,
        )

    def step(
        self,
        psi_wall_prev: np.ndarray,
        psi_wall_curr: np.ndarray,
    ) -> np.ndarray:
        """Вычислить новые токи в стенке методом трапеций.

        Уравнение ОДУ:
            L · dJ/dt + R · J = -dΦ/dt

        Метод трапеций:
            (L/dt + R/2) · J_new = (L/dt - R/2) · J_old - (psi_new - psi_old)/dt

        Parameters
        ----------
        psi_wall_prev : np.ndarray
            Psi в точках стенки на предыдущем шаге, shape (N,).
        psi_wall_curr : np.ndarray
            Psi в точках стенки на текущем шаге, shape (N,).

        Returns
        -------
        np.ndarray
            Обновлённые токи в стенке, shape (N,).
        """
        psi_prev = np.asarray(psi_wall_prev, dtype=float)
        psi_curr = np.asarray(psi_wall_curr, dtype=float)
        n = self.wall_geometry.n_segments

        if psi_prev.shape != (n,):
            raise ValueError(f"psi_wall_prev must have shape ({n},), got {psi_prev.shape}")
        if psi_curr.shape != (n,):
            raise ValueError(f"psi_wall_curr must have shape ({n},), got {psi_curr.shape}")

        # Проверка на NaN в psi (fallback на нулевые токи)
        if not (np.all(np.isfinite(psi_prev)) and np.all(np.isfinite(psi_curr))):
            self.J_wall = np.zeros(n, dtype=float)
            self.psi_wall_prev = psi_curr.copy()
            return self.J_wall

        # dΦ/dt = (psi_curr - psi_prev) / dt
        dpsi_dt = (psi_curr - psi_prev) / self.dt

        # RHS = (L/dt - R/2) · J_old - dΦ/dt
        L = self.G_wall_flux
        R = self.R_wall * np.eye(n)
        RHS = (L / self.dt - R / 2.0) @ self.J_wall - dpsi_dt

        # Проверка на NaN в RHS (fallback на нулевые токи)
        if not np.all(np.isfinite(RHS)):
            self.J_wall = np.zeros(n, dtype=float)
            self.psi_wall_prev = psi_curr.copy()
            return self.J_wall

        # J_new = inv(L/dt + R/2) · RHS
        self.J_wall = self.LHS_inv @ RHS

        # Финальная проверка на NaN в результате
        if not np.all(np.isfinite(self.J_wall)):
            self.J_wall = np.zeros(n, dtype=float)

        # Сохраняем текущее psi для следующего шага
        self.psi_wall_prev = psi_curr.copy()

        return self.J_wall

    def compute_psi_wall_contribution(self) -> np.ndarray:
        """Вычислить вклад токов стенки в psi на сетке.

        Returns
        -------
        np.ndarray
            Вклад стенки в psi, shape (nz, nr).
        """
        if self.J_wall.size == 0:
            return np.zeros(self.G_wall.shape[1:], dtype=float)
        return np.tensordot(self.J_wall, self.G_wall, axes=(0, 0))

    def sample_psi_wall(self, points: np.ndarray) -> np.ndarray:
        """Вычислить psi от токов стенки в произвольных точках.

        Parameters
        ----------
        points : np.ndarray
            Точки для вычисления, shape (M, 2).

        Returns
        -------
        np.ndarray
            Psi от стенки в точках, shape (M,).
        """
        from tokamak_control.core.green import green_axisymmetric

        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points must have shape (M, 2), got {pts.shape}")

        n_pts = pts.shape[0]
        psi = np.zeros(n_pts, dtype=float)

        for i in range(self.wall_geometry.n_segments):
            Rw = float(self.wall_geometry.wall_points[i, 0])
            Zw = float(self.wall_geometry.wall_points[i, 1])
            G_i = green_axisymmetric(pts[:, 0], pts[:, 1], Rw, Zw)
            psi += float(self.J_wall[i]) * G_i

        return psi

    def total_wall_current(self) -> float:
        """Вернуть полный ток в стенке (сумма токов по всем сегментам)."""
        return float(np.sum(self.J_wall))

    def reset(self) -> None:
        """Сбросить токи в стенке к нулю."""
        self.J_wall = np.zeros(self.wall_geometry.n_segments, dtype=float)
