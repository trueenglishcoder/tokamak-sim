"""Утилиты визуализации psi-полей, границы плазмы, рядов, кадров и видео."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, cast

import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec

from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.geometry.boundary import (
    BoundaryMode,
    boundary_status_is_real,
    find_plasma_boundary_with_status,
)
from tokamak_control.geometry.xpoints import find_x_points
from tokamak_control.geometry.limiters import get_limiter_shape
from tokamak_control.io.data_io import load_run


def _grid_from_meta(meta: dict) -> Grid2D:
    grid_meta = meta.get("grid")
    if not isinstance(grid_meta, dict):
        raise ValueError("Run metadata is missing grid information")
    r = Grid1D(
        start=float(grid_meta["r_start"]),
        step=float(grid_meta["r_step"]),
        size=int(grid_meta["r_size"]),
        center=float(grid_meta.get("r_center", 0.0)),
    )
    z = Grid1D(
        start=float(grid_meta["z_start"]),
        step=float(grid_meta["z_step"]),
        size=int(grid_meta["z_size"]),
        center=float(grid_meta.get("z_center", 0.0)),
    )
    return Grid2D(r=r, z=z)


def _center_from_meta(meta: dict) -> tuple[float, float]:
    center_meta = meta.get("center")
    if not isinstance(center_meta, dict):
        raise ValueError("Run metadata is missing center information")
    return float(center_meta["R0"]), float(center_meta["Z0"])


def _coil_positions_by_bank_from_meta(meta: dict) -> dict[str, np.ndarray] | None:
    coils_meta = meta.get("coil_positions")
    if not isinstance(coils_meta, dict):
        return None

    out: dict[str, np.ndarray] = {}
    for bank in ("pfc", "sol"):
        bank_positions = coils_meta.get(bank)
        if bank_positions is None:
            continue
        arr = np.asarray(bank_positions, dtype=float)
        if arr.size == 0:
            continue
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Invalid {bank} coil positions shape in run metadata: {arr.shape}")
        out[bank] = arr

    return out or None


def _limiter_shape_from_meta(meta: dict) -> np.ndarray | None:
    """Прочитать контур лимитера из metadata запуска."""
    limiter_meta = meta.get("limiter")
    if not isinstance(limiter_meta, dict):
        return None
    raw_shape = limiter_meta.get("shape")
    if raw_shape is not None:
        arr = np.asarray(raw_shape, dtype=float)
        if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] == 2:
            return arr
        raise ValueError(f"Invalid limiter shape in run metadata: {arr.shape}")
    raw_name = limiter_meta.get("name")
    if isinstance(raw_name, str) and raw_name.strip():
        return get_limiter_shape(raw_name)
    return None


def _polyline_from_padded_row(row: np.ndarray) -> np.ndarray:
    """Прочитать один NaN-заполненный контур из массива артефакта."""
    arr = np.asarray(row, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Boundary polyline row must have shape (N, 2), got {arr.shape}")
    valid = np.all(np.isfinite(arr), axis=1)
    poly = arr[valid]
    if poly.shape[0] < 3:
        raise RuntimeError("Stored boundary polyline is missing or too short")
    return poly


def _boundary_mode_from_meta(meta: dict) -> BoundaryMode:
    """Прочитать режим физического определения границы из metadata запуска."""
    boundary_meta = meta.get("boundary")
    if not isinstance(boundary_meta, dict):
        return "legacy_contour"
    mode = str(boundary_meta.get("mode", "legacy_contour")).strip().lower()
    if mode not in {"legacy_contour", "legacy_contour_limited"}:
        raise ValueError(f"Invalid boundary mode in run metadata: {mode!r}")
    return cast(BoundaryMode, mode)


def _all_coil_positions_from_meta(meta: dict) -> np.ndarray | None:
    by_bank = _coil_positions_by_bank_from_meta(meta)
    if not by_bank:
        return None
    return np.vstack([by_bank[bank] for bank in ("pfc", "sol") if bank in by_bank])


def _plot_coil_markers(
    ax: plt.Axes,
    coil_positions: Optional[np.ndarray | dict[str, np.ndarray]],
) -> None:
    if coil_positions is None:
        return

    if isinstance(coil_positions, dict):
        pfc = coil_positions.get("pfc")
        sol = coil_positions.get("sol")
        if pfc is not None and len(pfc):
            pos = np.atleast_2d(np.asarray(pfc, dtype=float))
            ax.plot(pos[:, 0], pos[:, 1], "x", ms=6, color="tab:blue", zorder=5)
        if sol is not None and len(sol):
            pos = np.atleast_2d(np.asarray(sol, dtype=float))
            ax.plot(pos[:, 0], pos[:, 1], "+", ms=6, color="tab:orange", zorder=5)
        return

    pos = np.atleast_2d(np.asarray(coil_positions, dtype=float))
    if len(pos):
        ax.plot(pos[:, 0], pos[:, 1], "x", ms=6, zorder=5)


def _plot_limiter_shape(ax: plt.Axes, limiter_shape: Optional[np.ndarray]) -> None:
    """Нарисовать контур лимитера, если он задан."""
    if limiter_shape is None:
        return
    limiter = np.asarray(limiter_shape, dtype=float)
    if limiter.ndim != 2 or limiter.shape[1] != 2:
        raise ValueError(f"limiter_shape must have shape (n, 2), got {limiter.shape}")
    if limiter.shape[0] < 2:
        return
    ax.plot(limiter[:, 0], limiter[:, 1], "-", lw=2.0, color="magenta", zorder=6)


def _plot_x_points(ax: plt.Axes, psi: np.ndarray, grid: Grid2D) -> None:
    """Найти и отметить X-точки на графике поля."""
    try:
        sep = 4.0 * max(abs(float(grid.r.step)), abs(float(grid.z.step)))
        points = find_x_points(psi, grid, max_points=8, min_separation=sep)
    except Exception:
        return
    if points.size == 0:
        return
    ax.plot(
        points[:, 0],
        points[:, 1],
        linestyle="none",
        marker="x",
        ms=7,
        mew=1.8,
        color="black",
        zorder=7,
    )


def fig_psi_contours(
    psi: np.ndarray,
    grid: Grid2D,
    *,
    levels: int = 512,
    title: Optional[str] = None,
    center: Optional[Tuple[float, float]] = None,
    coil_positions: Optional[np.ndarray | dict[str, np.ndarray]] = None,
    limiter_shape: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Contour plot of ψ with optional center and coil markers."""
    if psi.shape != grid.shape:
        raise ValueError(f"psi shape {psi.shape} != grid shape {grid.shape}")

    R, Z = grid.mesh()
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.contour(R, Z, psi, levels=levels, linestyles=":", linewidths=0.5)

    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    if title:
        ax.set_title(title)
    if center is not None:
        ax.plot([center[0]], [center[1]], "x", ms=6, color="red", zorder=4)
    _plot_limiter_shape(ax, limiter_shape)
    _plot_x_points(ax, psi, grid)
    _plot_coil_markers(ax, coil_positions)
    ax.set_aspect("equal", adjustable="box")
    return fig


def _find_boundary_tracked(
    psi: np.ndarray,
    grid: Grid2D,
    center: Tuple[float, float],
    *,
    n_levels_search: int,
    prev_level: Optional[float] = None,
    prev_poly: Optional[np.ndarray] = None,
    target_mean_radius: Optional[float] = None,
    limiter_shape: Optional[np.ndarray] = None,
    boundary_mode: BoundaryMode = "legacy_contour",
) -> tuple[np.ndarray | None, float | None, bool]:
    poly, level, status = find_plasma_boundary_with_status(
        psi=np.asarray(psi, dtype=float),
        grid=grid,
        center=center,
        n_levels=max(int(n_levels_search), 3),
        prev_level=prev_level,
        prev_poly=prev_poly,
        local_n_levels=3,
        local_span_frac=0.02,
        target_mean_radius=target_mean_radius,
        limiter_shape=limiter_shape,
        boundary_mode=boundary_mode,
    )
    is_real = boundary_status_is_real(status)
    if not is_real:
        return None, None, False
    return poly, level, True


def _fig_boundary_from_poly(
    psi: np.ndarray,
    grid: Grid2D,
    center: Tuple[float, float],
    poly: Optional[np.ndarray] = None,
    *,
    n_contours: int = 512,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    coil_positions: Optional[np.ndarray | dict[str, np.ndarray]] = None,
    limiter_shape: Optional[np.ndarray] = None,
    boundary_mode: BoundaryMode = "legacy_contour",
) -> plt.Figure:
    if psi.shape != grid.shape:
        raise ValueError(f"psi shape {psi.shape} != grid shape {grid.shape}")

    R, Z = grid.mesh()
    if vmin is None:
        vmin = float(np.nanmin(psi))
    if vmax is None:
        vmax = float(np.nanmax(psi))

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    pcm = ax.pcolormesh(R, Z, psi, shading="auto", vmin=vmin, vmax=vmax)
    ax.contour(R, Z, psi, levels=n_contours, colors="k", linestyles=":", linewidths=0.5, zorder=2)
    if poly is not None:
        poly_arr = np.asarray(poly, dtype=float)
        if poly_arr.ndim != 2 or poly_arr.shape[1] != 2:
            raise ValueError(f"boundary polyline must have shape (n, 2), got {poly_arr.shape}")
        ax.plot(poly_arr[:, 0], poly_arr[:, 1], "-", lw=2.0, color="red", zorder=4)
    ax.plot([center[0]], [center[1]], "x", ms=6, color="red", zorder=5)

    _plot_limiter_shape(ax, limiter_shape)
    _plot_x_points(ax, psi, grid)
    _plot_coil_markers(ax, coil_positions)

    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    if title:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(pcm, ax=ax, label="ψ")
    return fig


def fig_boundary_from_poly(
    psi: np.ndarray,
    grid: Grid2D,
    center: Tuple[float, float],
    poly: np.ndarray,
    *,
    n_contours: int = 160,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    coil_positions: Optional[np.ndarray | dict[str, np.ndarray]] = None,
    limiter_shape: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Построить psi и уже найденный в расчете физический контур плазмы."""
    return _fig_boundary_from_poly(
        psi=psi,
        grid=grid,
        center=center,
        poly=poly,
        n_contours=n_contours,
        vmin=vmin,
        vmax=vmax,
        title=title,
        coil_positions=coil_positions,
        limiter_shape=limiter_shape,
    )


def _draw_boundary_panel(
    ax: plt.Axes,
    psi: np.ndarray,
    grid: Grid2D,
    center: Tuple[float, float],
    poly: Optional[np.ndarray],
    *,
    n_contours: int,
    vmin: float,
    vmax: float,
    title: str,
    coil_positions: Optional[np.ndarray | dict[str, np.ndarray]],
    limiter_shape: Optional[np.ndarray],
) -> object:
    """Нарисовать панель поля ψ и границы на существующей оси."""
    R, Z = grid.mesh()
    pcm = ax.pcolormesh(R, Z, psi, shading="auto", vmin=vmin, vmax=vmax)
    ax.contour(R, Z, psi, levels=n_contours, colors="k", linestyles=":", linewidths=0.45, zorder=2)
    if poly is not None:
        poly_arr = np.asarray(poly, dtype=float)
        ax.plot(poly_arr[:, 0], poly_arr[:, 1], "-", lw=2.4, color="red", zorder=4)
    ax.plot([center[0]], [center[1]], "x", ms=7, color="red", zorder=5)
    _plot_limiter_shape(ax, limiter_shape)
    _plot_x_points(ax, psi, grid)
    _plot_coil_markers(ax, coil_positions)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    return pcm


def _as_matrix(series: np.ndarray) -> np.ndarray:
    """Вернуть временной ряд в форме (T, N)."""
    arr = np.asarray(series, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D time series, got {arr.shape}")
    return arr


def _fixed_series_ylim(y: np.ndarray, ref: np.ndarray | None = None) -> tuple[float, float] | None:
    """Return padded y limits from a complete time series."""
    y_mat = _as_matrix(y)
    finite_parts = [y_mat[np.isfinite(y_mat)]]
    if ref is not None:
        ref_arr = np.asarray(ref, dtype=float)
        finite_parts.append(ref_arr[np.isfinite(ref_arr)])
    finite = (
        np.concatenate([part for part in finite_parts if part.size], axis=0)
        if any(part.size for part in finite_parts)
        else np.zeros((0,), dtype=float)
    )
    if not finite.size:
        return None
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    pad = 0.08 * max(abs(hi - lo), abs(hi), abs(lo), 1.0)
    return lo - pad, hi + pad


def _plot_live_series(
    ax: plt.Axes,
    t: np.ndarray,
    y: np.ndarray,
    current_index: int,
    *,
    ylabel: str,
    labels: list[str],
    ref: np.ndarray | None = None,
    channel_plot_limit: int = 12,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Нарисовать временной ряд до текущего кадра и вертикальный маркер времени."""
    if t.size == 0:
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        return

    k = min(max(int(current_index), 0), int(t.size) - 1)
    t_now = float(t[k])
    y_mat = _as_matrix(y)
    end = k + 1

    if ref is not None:
        ref_arr = np.asarray(ref, dtype=float)
        if ref_arr.shape == (t.size,):
            ax.plot(t[:end], ref_arr[:end], color="black", linestyle="--", linewidth=1.3, label="ref")
        else:
            ref_arr = None
    else:
        ref_arr = None

    if y_mat.shape[1] > max(int(channel_plot_limit), 0):
        y_slice = y_mat[:end]
        y_min = np.nanmin(y_slice, axis=1)
        y_max = np.nanmax(y_slice, axis=1)
        y_mean = np.nanmean(y_slice, axis=1)
        ax.fill_between(t[:end], y_min, y_max, alpha=0.18, label="channel range")
        ax.plot(t[:end], y_mean, linewidth=1.5, label="channel mean")
    else:
        for j in range(y_mat.shape[1]):
            label = labels[j] if j < len(labels) else f"ch{j + 1}"
            ax.plot(t[:end], y_mat[:end, j], linewidth=1.25, label=label)

    ax.axvline(t_now, color="red", linewidth=1.0, alpha=0.8)
    ax.set_xlim(float(t[0]), float(t[-1]) if t.size > 1 else float(t[0]) + 1.0e-6)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        dynamic_ylim = _fixed_series_ylim(y_mat[:end], ref_arr[:end] if ref_arr is not None else None)
        if dynamic_ylim is not None:
            ax.set_ylim(*dynamic_ylim)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if y_mat.shape[1] <= 6:
        ax.legend(loc="upper right", fontsize=7, ncol=min(3, y_mat.shape[1] + (1 if ref is not None else 0)))
    elif y_mat.shape[1] > max(int(channel_plot_limit), 0):
        ax.legend(loc="upper right", fontsize=7)


def _fig_boundary_with_live_timeseries(
    *,
    psi: np.ndarray,
    grid: Grid2D,
    center: Tuple[float, float],
    poly: Optional[np.ndarray],
    n_contours: int,
    vmin: float,
    vmax: float,
    title: str,
    coil_positions: Optional[np.ndarray | dict[str, np.ndarray]],
    limiter_shape: Optional[np.ndarray],
    t: np.ndarray,
    frame_index: int,
    Ip: np.ndarray,
    Ip_ref: np.ndarray | None,
    pfc_currents: np.ndarray,
    sol_currents: np.ndarray,
    Ip_ylim: tuple[float, float] | None = None,
    pfc_ylim: tuple[float, float] | None = None,
    sol_ylim: tuple[float, float] | None = None,
) -> plt.Figure:
    """Собрать широкий кадр: сечение токамака слева, live-графики справа."""
    fig = plt.figure(figsize=(14.5, 7.2), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0, 1.0])

    ax_boundary = fig.add_subplot(gs[:, 0])
    pcm = _draw_boundary_panel(
        ax_boundary,
        psi,
        grid,
        center,
        poly,
        n_contours=n_contours,
        vmin=vmin,
        vmax=vmax,
        title=title,
        coil_positions=coil_positions,
        limiter_shape=limiter_shape,
    )
    fig.colorbar(pcm, ax=ax_boundary, label="ψ", shrink=0.88)

    ax_ip = fig.add_subplot(gs[0, 1])
    _plot_live_series(
        ax_ip,
        t,
        Ip,
        frame_index,
        ylabel="Ip [A]",
        labels=["Ip"],
        ref=Ip_ref,
        ylim=Ip_ylim,
    )

    ax_pfc = fig.add_subplot(gs[1, 1], sharex=ax_ip)
    pfc = _as_matrix(pfc_currents)
    _plot_live_series(
        ax_pfc,
        t,
        pfc,
        frame_index,
        ylabel="PFC [A]",
        labels=[f"PFC{i + 1}" for i in range(pfc.shape[1])],
        ylim=pfc_ylim,
    )

    ax_sol = fig.add_subplot(gs[2, 1], sharex=ax_ip)
    sol = _as_matrix(sol_currents)
    _plot_live_series(
        ax_sol,
        t,
        sol,
        frame_index,
        ylabel="SOL [A]",
        labels=[f"SOL{i + 1}" for i in range(sol.shape[1])],
        ylim=sol_ylim,
    )
    ax_sol.set_xlabel("t [s]")
    return fig


def fig_boundary(
    psi: np.ndarray,
    grid: Grid2D,
    center: Tuple[float, float],
    *,
    n_levels_search: int = 40,
    n_contours: int = 512,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    coil_positions: Optional[np.ndarray | dict[str, np.ndarray]] = None,
    limiter_shape: Optional[np.ndarray] = None,
    boundary_mode: BoundaryMode = "legacy_contour",
    prev_level: Optional[float] = None,
    prev_poly: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Построить psi, контуры и физически найденную границу плазмы."""
    poly, _, is_real = _find_boundary_tracked(
        psi,
        grid,
        center,
        n_levels_search=n_levels_search,
        prev_level=prev_level,
        prev_poly=prev_poly,
        limiter_shape=limiter_shape,
        boundary_mode=boundary_mode,
    )
    if not is_real or poly is None:
        raise RuntimeError("Boundary finder returned a non-physical boundary status")

    return _fig_boundary_from_poly(
        psi=psi,
        grid=grid,
        center=center,
        poly=poly,
        n_contours=n_contours,
        vmin=vmin,
        vmax=vmax,
        title=title,
        coil_positions=coil_positions,
        limiter_shape=limiter_shape,
    )


def fig_boundary_vs_reference(
    psi: np.ndarray,
    grid: Grid2D,
    center: Tuple[float, float],
    angles: np.ndarray,
    ref_radii: np.ndarray,
    *,
    n_levels_search: int = 40,
    title: Optional[str] = None,
    limiter_shape: Optional[np.ndarray] = None,
    boundary_mode: BoundaryMode = "legacy_contour",
) -> plt.Figure:
    """Построить границу плазмы и опорную полярную кривую."""
    poly, _, is_real = _find_boundary_tracked(
        psi,
        grid,
        center,
        n_levels_search=n_levels_search,
        limiter_shape=limiter_shape,
        boundary_mode=boundary_mode,
    )
    if not is_real or poly is None:
        raise RuntimeError("Boundary finder returned a non-physical boundary status")

    R0, Z0 = center
    R_ref = R0 + ref_radii * np.cos(angles)
    Z_ref = Z0 + ref_radii * np.sin(angles)

    R, Z = grid.mesh()
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.contour(R, Z, psi, levels=120, linestyles=":", linewidths=0.5)
    ax.plot(poly[:, 0], poly[:, 1], "-", lw=2, color="red", label="current boundary")
    ax.plot(np.r_[R_ref, R_ref[0]], np.r_[Z_ref, Z_ref[0]], "--", lw=2, label="reference")
    ax.plot([R0], [Z0], "x", ms=6, color="red", label="center")
    _plot_limiter_shape(ax, limiter_shape)
    _plot_x_points(ax, psi, grid)

    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    return fig


def fig_time_series_from_npz(npz_path: str | Path) -> plt.Figure:
    """Build a compact time-series figure for tracking, coil currents, and actuation."""
    run = load_run(npz_path)

    t = np.asarray(run["t"], dtype=float)
    Ip = np.asarray(run["Ip"], dtype=float)
    Ip_ref = np.asarray(run["Ip_ref"], dtype=float) if "Ip_ref" in run else None

    radii_true = np.asarray(run["radii_true"], dtype=float) if "radii_true" in run else None
    radii_ref = np.asarray(run["radii_ref"], dtype=float) if "radii_ref" in run else None

    pfc_curr = np.asarray(run["pfc_currents"], dtype=float)
    sol_curr = np.asarray(run["sol_currents"], dtype=float)

    pfc_cmd = np.asarray(run["pfc_derivs_cmd"], dtype=float) if "pfc_derivs_cmd" in run else np.asarray(run["pfc_derivs"], dtype=float)
    sol_cmd = np.asarray(run["sol_derivs_cmd"], dtype=float) if "sol_derivs_cmd" in run else np.asarray(run["sol_derivs"], dtype=float)

    if pfc_curr.ndim == 1:
        pfc_curr = pfc_curr[:, None]
    if sol_curr.ndim == 1:
        sol_curr = sol_curr[:, None]
    if pfc_cmd.ndim == 1:
        pfc_cmd = pfc_cmd[:, None]
    if sol_cmd.ndim == 1:
        sol_cmd = sol_cmd[:, None]

    fig, axes = plt.subplots(6, 1, figsize=(8, 14), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(t, Ip, label="Ip")
    if Ip_ref is not None and Ip_ref.shape == Ip.shape:
        ax.plot(t, Ip_ref, label="Ip_ref")
    ax.set_ylabel("Ip [A]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    if radii_true is not None and radii_ref is not None:
        rt = np.asarray(radii_true, dtype=float)
        rr = np.asarray(radii_ref, dtype=float)
        if rt.ndim == 1:
            rt = rt[:, None]
        if rr.ndim == 1:
            rr = rr[:, None]
        mean_true = np.nanmean(rt, axis=1)
        mean_ref = np.nanmean(rr, axis=1)
        rmse = np.sqrt(np.nanmean((rt - rr) ** 2, axis=1))

        ax.plot(t, mean_true, label="boundary_mean")
        ax.plot(t, mean_ref, label="boundary_ref_mean")
        ax.set_ylabel("mean radius [m]")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(t, rmse, linestyle="--", label="boundary_rmse")
        ax2.set_ylabel("RMSE [m]")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "boundary reference not stored in run", transform=ax.transAxes, ha="center", va="center")
        ax.set_ylabel("boundary")
        ax.grid(True, alpha=0.3)

    ax = axes[2]
    for i in range(pfc_curr.shape[1]):
        ax.plot(t, pfc_curr[:, i], label=f"PFC{i}")
    ax.set_ylabel("I_pfc [A]")
    ax.grid(True, alpha=0.3)
    if pfc_curr.shape[1] <= 12:
        ax.legend(loc="best", fontsize=8)

    ax = axes[3]
    for i in range(sol_curr.shape[1]):
        ax.plot(t, sol_curr[:, i], label=f"SOL{i}")
    ax.set_ylabel("I_sol [A]")
    ax.grid(True, alpha=0.3)
    if sol_curr.shape[1] <= 12:
        ax.legend(loc="best", fontsize=8)

    ax = axes[4]
    for i in range(pfc_cmd.shape[1]):
        ax.plot(t, pfc_cmd[:, i], label=f"PFC{i}")
    ax.set_ylabel("dI_pfc [A/s]")
    ax.grid(True, alpha=0.3)
    if pfc_cmd.shape[1] <= 12:
        ax.legend(loc="best", fontsize=8)

    ax = axes[5]
    for i in range(sol_cmd.shape[1]):
        ax.plot(t, sol_cmd[:, i], label=f"SOL{i}")
    ax.set_ylabel("dI_sol [A/s]")
    ax.grid(True, alpha=0.3)
    if sol_cmd.shape[1] <= 12:
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("t [s]")
    return fig


def save_run_frames(
    npz_path: str | Path,
    frames_dir: str | Path,
    *,
    n_levels_search: int = 40,
    n_contours: int = 512,
    coil_positions: Optional[np.ndarray | dict[str, np.ndarray]] = None,
    limiter_shape: Optional[np.ndarray] = None,
    frame_stride: int = 1,
    dpi: int = 240,
) -> List[Path]:
    """Сохранить покадровые графики psi и границы с фиксированной шкалой цвета."""
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    run = load_run(npz_path)
    meta = run["meta"]
    grid = _grid_from_meta(meta)
    center = _center_from_meta(meta)
    if coil_positions is None:
        coil_positions = _coil_positions_by_bank_from_meta(meta)
    if limiter_shape is None:
        limiter_shape = _limiter_shape_from_meta(meta)

    psi_snaps = np.asarray(run["psi_snaps"], dtype=float)
    if psi_snaps.size == 0:
        raise RuntimeError("No psi snapshots found in run NPZ; ensure snapshot_every is enabled")
    if psi_snaps.ndim != 3:
        raise ValueError(f"psi_snaps must have shape (S, nz, nr), got {psi_snaps.shape}")
    if psi_snaps.shape[1:] != grid.shape:
        raise ValueError(f"psi snapshot grid shape {psi_snaps.shape[1:]} != {grid.shape}")

    psi_min = float(np.nanmin(psi_snaps))
    psi_max = float(np.nanmax(psi_snaps))
    snap_steps = np.asarray(run.get("psi_snap_steps", np.arange(psi_snaps.shape[0])), dtype=int)
    t = np.asarray(run["t"], dtype=float)
    Ip = np.asarray(run["Ip"], dtype=float)
    Ip_ref = np.asarray(run["Ip_ref"], dtype=float) if "Ip_ref" in run else None
    radii_ref = np.asarray(run["radii_ref"], dtype=float) if "radii_ref" in run else None
    pfc_currents = _as_matrix(np.asarray(run["pfc_currents"], dtype=float))
    sol_currents = _as_matrix(np.asarray(run["sol_currents"], dtype=float))
    Ip_ylim = _fixed_series_ylim(Ip, Ip_ref)
    pfc_ylim = _fixed_series_ylim(pfc_currents)
    sol_ylim = _fixed_series_ylim(sol_currents)
    boundary_polys = np.asarray(run.get("boundary_poly_true"), dtype=float) if "boundary_poly_true" in run else None
    if boundary_polys is not None and boundary_polys.ndim != 3:
        raise RuntimeError(f"boundary_poly_true must have shape (T, N, 2), got {boundary_polys.shape}")
    steps = np.asarray(run["step"], dtype=int)
    step_to_index = {int(step): step_idx for step_idx, step in enumerate(steps)}

    frame_paths: List[Path] = []
    stride = max(int(frame_stride), 1)
    render_indices = range(0, int(psi_snaps.shape[0]), stride)

    for frame_idx, idx in enumerate(render_indices):
        psi = np.asarray(psi_snaps[idx], dtype=float)
        step_label = int(snap_steps[idx]) if idx < snap_steps.shape[0] else idx
        if step_label not in step_to_index:
            raise RuntimeError(f"No stored time-series row for snapshot step {step_label}")
        if boundary_polys is None:
            poly = None
        else:
            try:
                poly = _polyline_from_padded_row(boundary_polys[step_to_index[step_label]])
            except RuntimeError:
                poly = None
        frame_title = f"ψ and boundary (step {step_label})"
        fig = _fig_boundary_with_live_timeseries(
            psi=psi,
            grid=grid,
            center=center,
            poly=poly,
            n_contours=n_contours,
            vmin=psi_min,
            vmax=psi_max,
            title=frame_title,
            coil_positions=coil_positions,
            limiter_shape=limiter_shape,
            t=t,
            frame_index=step_label,
            Ip=Ip,
            Ip_ref=Ip_ref,
            pfc_currents=pfc_currents,
            sol_currents=sol_currents,
            Ip_ylim=Ip_ylim,
            pfc_ylim=pfc_ylim,
            sol_ylim=sol_ylim,
        )
        frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
        fig.savefig(frame_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

    return frame_paths


def frames_to_video(
    frames_dir: str | Path,
    video_path: str | Path,
    fps: int = 10,
) -> None:
    """Закодировать сохраненные кадры в MP4 через Matplotlib FFMpegWriter."""
    frames_dir = Path(frames_dir)
    video_path = Path(video_path)

    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    first_img = plt.imread(frame_paths[0])
    height_px, width_px = first_img.shape[:2]
    dpi = 160
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), constrained_layout=True)
    ax.axis("off")

    writer = animation.FFMpegWriter(fps=fps)
    with writer.saving(fig, str(video_path), dpi=dpi):
        for frame_path in frame_paths:
            img = plt.imread(frame_path)
            ax.imshow(img, aspect="equal")
            ax.axis("off")
            writer.grab_frame()
            ax.cla()

    plt.close(fig)
