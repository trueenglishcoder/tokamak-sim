# scripts/fit_sigma_L_gradient.py
"""
Stochastic gradient-style fitting of sigma and inductance_L using the project PlasmaModel.

This is a drop-in alternative to the grid-search fitter. It uses SPSA (Simultaneous
Perturbation Stochastic Approximation), which is "gradient descent with random bumps":
each iteration estimates a gradient from two objective evaluations using random ±1
perturbations, then takes a descent step. Multiple random restarts reduce local-minimum
risk.

Inputs and data semantics match the grid-search fitter:
- TOML config defines tokamak geometry, dt, center, mu0, and any sign conventions.
- Shot tables provide applied coil currents and measured Ip.
- The fitter resamples both tables to the model dt and replays coil currents through
  PlasmaModel.step_currents().
- For fitting, actuator lag and clipping are disabled so the replay matches the tables.

Outputs:
- Writes a CSV with the top-k best parameter pairs found (same columns as grid search).
- Prints the best parameters and objective on stdout.
- Writes artifacts into one timestamped run directory per invocation.
- If --plot is set, writes plots into <run_dir>/plots/ with subfolders:
    ip_overlay/     true Ip highlighted + best Ip highlighted + tried candidates faint
    ip_best/        true Ip vs best Ip only
    coil_currents/  table coil currents vs simulation-used coil currents (best params)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    class tqdm:
        """Минимальная заглушка прогресс-бара, когда пакет tqdm не установлен."""

        def __init__(self, *args, **kwargs) -> None:
            """Принять аргументы tqdm без вывода прогресса."""
            self.iterable = args[0] if args else None

        def __iter__(self):
            """Вернуть итератор исходной последовательности."""
            return iter(()) if self.iterable is None else iter(self.iterable)

        def __enter__(self):
            """Вернуть объект контекстного менеджера."""
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            """Закрыть контекст без подавления исключений."""
            return False

        def update(self, _n: int = 1) -> None:
            """Игнорировать обновление прогресса."""
            return None

        def set_postfix_str(self, _text: str) -> None:
            """Игнорировать текстовый статус прогресса."""
            return None

        def close(self) -> None:
            """Закрыть заглушку прогресс-бара."""
            return None

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def _ensure_repo_root_on_path() -> None:
    """Добавить корень репозитория в путь импорта при прямом запуске скрипта."""
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from tokamak_control.core.plasma_model import PlasmaModel
from tokamak_control.core.plasma_state import PlasmaState
from tokamak_control.io.config_io import LoadedConfig, load_config
from tokamak_control.io.run_dirs import allocate_artifact_run_dir, slugify, timestamped_stem


IP_CSV_PATTERN = re.compile(r"t15md_(\d+)_ip\.csv$", re.IGNORECASE)
COIL_CSV_PATTERN = re.compile(r"t15md_(\d+)_coils\.csv$", re.IGNORECASE)

IP_TXT_PATTERN = re.compile(r"T15md_(\d+)_Pcs\.Ip\.S\.txt$", re.IGNORECASE)
COIL_TXT_PATTERN = re.compile(r"T15md_(\d+)_Pcs\.Pf5\.S\.TXT$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Shot:
    shot_id: str
    t: np.ndarray  # seconds, starts at 0
    ip: np.ndarray  # A
    pfc_curr: np.ndarray  # (T, n_pfc) A
    sol_curr: np.ndarray  # (T, n_sol) A
    cfg: LoadedConfig | None = None


@dataclass(frozen=True, slots=True)
class PairScore:
    sigma: float
    inductance_L: float
    tau: float
    mean_rmse: float
    mean_nrmse: float
    n_shots: int


def _read_numeric_rows(path: Path, expected_cols: int) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith(";"):
                continue

            if ";" in line:
                parts = [p.strip() for p in line.split(";") if p.strip() != ""]
            elif "," in line:
                parts = [p.strip() for p in line.split(",") if p.strip() != ""]
            else:
                parts = line.split()

            if len(parts) != expected_cols:
                continue

            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue

            rows.append(vals)

    if not rows:
        raise ValueError(f"No numeric {expected_cols}-column rows found in {path}")
    return np.asarray(rows, dtype=float)


def _normalize_time_seconds(raw_t: np.ndarray, *, time_divisor: float) -> np.ndarray:
    raw_t = np.asarray(raw_t, dtype=float)
    if raw_t.size == 0:
        return raw_t.copy()
    div = float(time_divisor)
    if not np.isfinite(div) or div <= 0.0:
        raise ValueError(f"time_divisor must be finite and > 0, got {time_divisor}")
    return raw_t / div


def _discover_pairs(ip_dir: Path, coils_dir: Path) -> list[tuple[str, Path, Path]]:
    ip_files: dict[str, Path] = {}
    coil_files: dict[str, Path] = {}

    for p in ip_dir.iterdir():
        if not p.is_file():
            continue
        m = IP_CSV_PATTERN.search(p.name)
        if m:
            ip_files[m.group(1)] = p
            continue
        m = IP_TXT_PATTERN.search(p.name)
        if m and m.group(1) not in ip_files:
            ip_files[m.group(1)] = p

    for p in coils_dir.iterdir():
        if not p.is_file():
            continue
        m = COIL_CSV_PATTERN.search(p.name)
        if m:
            coil_files[m.group(1)] = p
            continue
        m = COIL_TXT_PATTERN.search(p.name)
        if m and m.group(1) not in coil_files:
            coil_files[m.group(1)] = p

    shot_ids = sorted(set(ip_files) & set(coil_files), key=lambda s: int(s))
    if not shot_ids:
        raise FileNotFoundError(
            "No matching shot pairs found.\n"
            f"ip_dir={ip_dir}\n"
            f"coils_dir={coils_dir}\n"
            f"Ip shots: {sorted(ip_files)}\n"
            f"Coil shots: {sorted(coil_files)}"
        )

    return [(sid, ip_files[sid], coil_files[sid]) for sid in shot_ids]


def _initial_currents_for_shot(config_path: Path, shot_id: str) -> Path | None:
    """Найти файл начального состояния для shot_id рядом с конфигами."""
    candidate = config_path.parent / "initial_currents" / f"{config_path.stem}_{shot_id}.toml"
    return candidate if candidate.exists() else None


def _resample_overlap(
    t_ip: np.ndarray,
    ip_raw: np.ndarray,
    t_coil: np.ndarray,
    sol_raw: np.ndarray,
    pfc_raw: np.ndarray,
    *,
    dt: float,
    t_min: float | None,
    t_max: float | None,
) -> Shot:
    t0 = max(float(t_ip[0]), float(t_coil[0]))
    t1 = min(float(t_ip[-1]), float(t_coil[-1]))
    if t_min is not None:
        t0 = max(t0, float(t_min))
    if t_max is not None:
        t1 = min(t1, float(t_max))
    if t1 <= t0:
        raise ValueError("empty overlap window after applying t_min/t_max")

    n = int(math.floor((t1 - t0) / float(dt))) + 1
    if n < 3:
        raise ValueError(f"fit window too short for dt={dt}")

    t = t0 + np.arange(n, dtype=float) * float(dt)
    ip = np.interp(t, t_ip, ip_raw)
    sol = np.column_stack([np.interp(t, t_coil, sol_raw[:, j]) for j in range(sol_raw.shape[1])])
    pfc = np.column_stack([np.interp(t, t_coil, pfc_raw[:, j]) for j in range(pfc_raw.shape[1])])

    t = t - float(t[0])
    return Shot(shot_id="?", t=t, ip=ip, pfc_curr=pfc, sol_curr=sol)


def _split_sol_group_sizes(cfg: LoadedConfig) -> list[int]:
    """Return the physical split-element count for each runtime SOL actuator."""
    return [int(np.asarray(group, dtype=float).reshape(-1, 2).shape[0]) for group in cfg.sol.element_positions]


def _expand_split_sol_rows(sol_raw: np.ndarray, group_sizes: list[int]) -> np.ndarray:
    """
    Expand SOL actuator currents into per-element currents for legacy fits.

    Each split point receives ``group_current / n_elements`` so the total
    current of the volumetric SOL actuator is conserved.
    """
    source = np.asarray(sol_raw, dtype=float)
    if source.ndim != 2 or source.shape[1] != len(group_sizes):
        raise ValueError("source SOL matrix shape does not match split group count")
    parts = [np.repeat(source[:, i : i + 1] / float(count), int(count), axis=1) for i, count in enumerate(group_sizes)]
    return np.concatenate(parts, axis=1)


def _load_shots(
    *,
    config_path: Path,
    ip_dir: Path,
    coils_dir: Path,
    time_divisor: float,
    t_min: float | None,
    t_max: float | None,
    only_shot: str | None,
) -> tuple[list[Shot], dict[str, object]]:
    cfg = load_config(config_path)

    dt = float(cfg.physics.t_step)

    pairs = _discover_pairs(ip_dir, coils_dir)
    if only_shot is not None:
        pairs = [p for p in pairs if p[0] == str(only_shot)]
        if not pairs:
            raise ValueError(f"Requested shot {only_shot} not found in provided directories")

    shots: list[Shot] = []
    skipped: list[tuple[str, str]] = []

    for shot_id, ip_path, coil_path in pairs:
        try:
            initial_path = _initial_currents_for_shot(config_path, str(shot_id))
            shot_cfg = load_config(config_path, initial_currents_path=initial_path)
            n_pfc = int(shot_cfg.pfc.n_coils)
            n_sol = int(shot_cfg.sol.n_coils)
            ip_rows = _read_numeric_rows(ip_path, expected_cols=2)
            try:
                coil_rows = _read_numeric_rows(coil_path, expected_cols=1 + n_sol + n_pfc)
                sol_source_cols = n_sol
            except Exception:
                group_sizes = _split_sol_group_sizes(shot_cfg)
                sol_source_cols = len(group_sizes)
                coil_rows = _read_numeric_rows(coil_path, expected_cols=1 + sol_source_cols + n_pfc)

            t_ip = _normalize_time_seconds(ip_rows[:, 0], time_divisor=time_divisor)
            ip_raw = ip_rows[:, 1]

            t_coil = _normalize_time_seconds(coil_rows[:, 0], time_divisor=time_divisor)
            sol_raw_source = coil_rows[:, 1 : 1 + sol_source_cols]
            if sol_source_cols == n_sol:
                sol_raw = sol_raw_source
            else:
                sol_raw = _expand_split_sol_rows(sol_raw_source, _split_sol_group_sizes(shot_cfg))
            pfc_raw = coil_rows[:, 1 + sol_source_cols : 1 + sol_source_cols + n_pfc]

            sh = _resample_overlap(
                t_ip,
                ip_raw,
                t_coil,
                sol_raw,
                pfc_raw,
                dt=dt,
                t_min=t_min,
                t_max=t_max,
            )
            shots.append(Shot(shot_id=str(shot_id), t=sh.t, ip=sh.ip, pfc_curr=sh.pfc_curr, sol_curr=sh.sol_curr, cfg=shot_cfg))
        except Exception as e:
            skipped.append((str(shot_id), str(e)))

    if not shots:
        msg = "No usable shots were loaded.\n"
        if skipped:
            msg += "First few skip reasons:\n" + "\n".join([f"  {sid}: {reason}" for sid, reason in skipped[:10]])
        raise ValueError(msg)

    meta = {"dt": dt, "cfg": cfg, "skipped": skipped}
    return shots, meta


def _init_model_for_shot(
    model: PlasmaModel,
    *,
    ip0: float,
    ip_at_t0: float,
    pfc0: np.ndarray,
    sol0: np.ndarray,
) -> None:
    model.actuator_tau = 0.0
    model.pfc_deriv_limit = None
    model.sol_deriv_limit = None
    model.pfc_current_limit = None
    model.sol_current_limit = None

    psi0 = model._compose_psi(float(ip_at_t0), np.asarray(pfc0, dtype=float), np.asarray(sol0, dtype=float))
    model.Ip0 = float(ip0)
    model.state = PlasmaState(
        t=0.0,
        step=0,
        Ip=float(ip_at_t0),
        Ip0=float(ip0),
        psi=psi0,
        pfc_currents=np.asarray(pfc0, dtype=float).copy(),
        pfc_current_derivs=np.zeros((model.pfc.n_coils,), dtype=float),
        sol_currents=np.asarray(sol0, dtype=float).copy(),
        sol_current_derivs=np.zeros((model.sol.n_coils,), dtype=float),
    )


def _simulate_shot_record(
    model: PlasmaModel,
    shot: Shot,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = float(model.t_step)
    T = int(shot.t.size)

    ip_pred = np.empty((T,), dtype=float)
    pfc_used = np.empty((T, model.pfc.n_coils), dtype=float)
    sol_used = np.empty((T, model.sol.n_coils), dtype=float)

    s0 = model.state
    if s0 is None:
        raise RuntimeError("Model state not initialized")
    ip_pred[0] = float(s0.Ip)
    pfc_used[0] = np.asarray(s0.pfc_currents, dtype=float)
    sol_used[0] = np.asarray(s0.sol_currents, dtype=float)

    for k in range(T - 1):
        state = model.step_currents(
            pfc_currents_next=shot.pfc_curr[k + 1],
            sol_currents_next=shot.sol_curr[k + 1],
        )
        ip_pred[k + 1] = float(state.Ip)
        pfc_used[k + 1] = np.asarray(state.pfc_currents, dtype=float)
        sol_used[k + 1] = np.asarray(state.sol_currents, dtype=float)

    return ip_pred, pfc_used, sol_used


def _model_for_scored_shot(shot: Shot, *, sigma: float, inductance_L: float) -> PlasmaModel:
    """Создать модель с активными катушками конкретного shot."""
    if shot.cfg is None:
        raise RuntimeError(f"Shot {shot.shot_id} has no loaded config")
    model = PlasmaModel.from_settings(grid=shot.cfg.grid, pfc=shot.cfg.pfc, sol=shot.cfg.sol, settings=shot.cfg.physics)
    model.sigma = float(sigma)
    model.inductance_L = float(inductance_L)
    model.actuator_tau = 0.0
    model.pfc_deriv_limit = None
    model.sol_deriv_limit = None
    model.pfc_current_limit = None
    model.sol_current_limit = None
    return model


def _score_params(
    base_model: PlasmaModel,
    shots: list[Shot],
    *,
    sigma: float,
    inductance_L: float,
    store_curves: dict[str, list[np.ndarray]] | None,
    store_max_curves: int,
    store_seen: dict[str, int] | None,
) -> PairScore:
    _ = base_model
    tau = float(sigma) * float(inductance_L)

    rmses: list[float] = []
    nrmse: list[float] = []

    for sh in shots:
        model = _model_for_scored_shot(sh, sigma=float(sigma), inductance_L=float(inductance_L))
        _init_model_for_shot(
            model,
            ip0=float(sh.ip[0]),
            ip_at_t0=float(sh.ip[0]),
            pfc0=sh.pfc_curr[0],
            sol0=sh.sol_curr[0],
        )
        pred, _, _ = _simulate_shot_record(model, sh)

        # Optional reservoir sampling of candidate curves for plotting.
        if store_curves is not None and store_seen is not None:
            sid = str(sh.shot_id)
            seen = store_seen.get(sid, 0) + 1
            store_seen[sid] = seen
            if sid in store_curves:
                curves = store_curves[sid]
                if len(curves) < store_max_curves:
                    curves.append(pred.astype(np.float32))
                else:
                    j = random.randrange(seen)
                    if j < store_max_curves:
                        curves[j] = pred.astype(np.float32)

        err = pred - sh.ip
        rmse = float(np.sqrt(np.mean(err * err)))
        peak = max(float(np.max(np.abs(sh.ip))), 1e-12)
        rmses.append(rmse)
        nrmse.append(float(rmse / peak))

    return PairScore(
        sigma=float(sigma),
        inductance_L=float(inductance_L),
        tau=tau,
        mean_rmse=float(np.mean(rmses)),
        mean_nrmse=float(np.mean(nrmse)),
        n_shots=len(shots),
    )


def _project_log_param(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _spsa_optimize(
    base_model: PlasmaModel,
    shots: list[Shot],
    *,
    sigma_min: float,
    sigma_max: float,
    L_min: float,
    L_max: float,
    iters: int,
    restarts: int,
    seed: int,
    store_curves: dict[str, list[np.ndarray]] | None,
    store_max_curves: int,
) -> tuple[PairScore, list[PairScore]]:
    rng = random.Random(int(seed))

    total_iters = int(restarts) * int(iters)
    pbar = tqdm(total=total_iters, desc="SPSA", unit="iter")

    # Optimize objective = mean_nrmse to equalize shots, but still record mean_rmse.
    def eval_score(sigma: float, L: float, seen_map: dict[str, int] | None) -> PairScore:
        return _score_params(
            base_model,
            shots,
            sigma=sigma,
            inductance_L=L,
            store_curves=store_curves,
            store_max_curves=store_max_curves,
            store_seen=seen_map,
        )

    # Log-space bounds
    log_sigma_lo = math.log(float(sigma_min))
    log_sigma_hi = math.log(float(sigma_max))
    log_L_lo = math.log(float(L_min))
    log_L_hi = math.log(float(L_max))

    # SPSA schedules
    # c_k: perturbation size in log-space
    c0 = 0.1
    gamma = 0.101
    # a_k: step size in log-space
    a0 = 0.2
    alpha = 0.602
    A = float(iters) * 0.1

    best_global: PairScore | None = None
    history: list[PairScore] = []

    for r in range(int(restarts)):
        # Random init within bounds (log-uniform)
        th0 = rng.uniform(log_sigma_lo, log_sigma_hi)
        th1 = rng.uniform(log_L_lo, log_L_hi)
        theta = np.asarray([th0, th1], dtype=float)

        # Use a separate seen counter for reservoir sampling so early restarts don't dominate.
        seen_map = {str(sh.shot_id): 0 for sh in shots} if store_curves is not None else None

        for k in range(int(iters)):
            ck = c0 / ((k + 1) ** gamma)
            ak = a0 / ((k + 1 + A) ** alpha)

            # Random ±1 perturbation
            delta = np.asarray([1.0 if rng.random() < 0.5 else -1.0, 1.0 if rng.random() < 0.5 else -1.0], dtype=float)

            theta_plus = theta + ck * delta
            theta_minus = theta - ck * delta

            # Project
            theta_plus[0] = _project_log_param(theta_plus[0], log_sigma_lo, log_sigma_hi)
            theta_plus[1] = _project_log_param(theta_plus[1], log_L_lo, log_L_hi)
            theta_minus[0] = _project_log_param(theta_minus[0], log_sigma_lo, log_sigma_hi)
            theta_minus[1] = _project_log_param(theta_minus[1], log_L_lo, log_L_hi)

            sigma_plus = float(math.exp(theta_plus[0]))
            L_plus = float(math.exp(theta_plus[1]))
            sigma_minus = float(math.exp(theta_minus[0]))
            L_minus = float(math.exp(theta_minus[1]))

            sc_plus = eval_score(sigma_plus, L_plus, seen_map)
            sc_minus = eval_score(sigma_minus, L_minus, seen_map)

            f_plus = float(sc_plus.mean_nrmse)
            f_minus = float(sc_minus.mean_nrmse)

            # Gradient estimate in log-space
            ghat = ((f_plus - f_minus) / (2.0 * ck)) * delta

            # Update
            theta = theta - ak * ghat
            theta[0] = _project_log_param(float(theta[0]), log_sigma_lo, log_sigma_hi)
            theta[1] = _project_log_param(float(theta[1]), log_L_lo, log_L_hi)

            sigma = float(math.exp(float(theta[0])))
            L = float(math.exp(float(theta[1])))

            sc = eval_score(sigma, L, seen_map)
            history.append(sc)

            if best_global is None or sc.mean_nrmse < best_global.mean_nrmse:
                best_global = sc

            pbar.update(1)
            pbar.set_postfix_str(f"restart={r+1}/{int(restarts)} best_nrmse={best_global.mean_nrmse:.4f}")

            # Random bump if stagnating: every 25 steps, small jitter
            if (k + 1) % 25 == 0:
                theta[0] = _project_log_param(float(theta[0] + rng.uniform(-0.05, 0.05)), log_sigma_lo, log_sigma_hi)
                theta[1] = _project_log_param(float(theta[1] + rng.uniform(-0.05, 0.05)), log_L_lo, log_L_hi)

    pbar.close()
    assert best_global is not None
    return best_global, history


def _set_ylim_from_true_best(ax: plt.Axes, y_true: np.ndarray, y_best: np.ndarray) -> None:
    y_min = float(min(np.min(y_true), np.min(y_best)))
    y_max = float(max(np.max(y_true), np.max(y_best)))
    pad = 0.05 * max(1e-12, (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)


def _plot_ip_overlay(
    *,
    shot: Shot,
    candidate_curves: list[np.ndarray],
    best_curve: np.ndarray,
    best_sigma: float,
    best_L: float,
    best_mean_rmse: float,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    for curve in candidate_curves:
        ax.plot(shot.t, curve, alpha=0.03, linewidth=0.8)

    ax.plot(shot.t, shot.ip, linewidth=2.8, label="True Ip (table)")
    ax.plot(
        shot.t,
        best_curve,
        linewidth=2.4,
        label=f"Best fit (sigma={best_sigma:.3g}, L={best_L:.3g}, mean_rmse={best_mean_rmse:.2f})",
    )

    ax.set_title(f"Ip overlay for shot {shot.shot_id}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Ip (A)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _set_ylim_from_true_best(ax, shot.ip, best_curve)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_ip_best_only(
    *,
    shot: Shot,
    best_curve: np.ndarray,
    best_sigma: float,
    best_L: float,
    best_mean_rmse: float,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(shot.t, shot.ip, linewidth=2.8, label="True Ip (table)")
    ax.plot(
        shot.t,
        best_curve,
        linewidth=2.4,
        label=f"Best fit (sigma={best_sigma:.3g}, L={best_L:.3g}, mean_rmse={best_mean_rmse:.2f})",
    )

    ax.set_title(f"Ip best-only for shot {shot.shot_id}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Ip (A)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _set_ylim_from_true_best(ax, shot.ip, best_curve)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_coil_currents(
    *,
    shot: Shot,
    pfc_used: np.ndarray,
    sol_used: np.ndarray,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    for j in range(shot.sol_curr.shape[1]):
        ax1.plot(shot.t, shot.sol_curr[:, j], linewidth=1.8, label=f"SOL{j+1} table")
        ax1.plot(shot.t, sol_used[:, j], linewidth=1.2, linestyle="--", label=f"SOL{j+1} sim")
    ax1.set_title(f"SOL currents: table vs simulation (shot {shot.shot_id})")
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("I (A)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", ncol=2)

    for j in range(shot.pfc_curr.shape[1]):
        ax2.plot(shot.t, shot.pfc_curr[:, j], linewidth=1.8, label=f"PFC{j+1} table")
        ax2.plot(shot.t, pfc_used[:, j], linewidth=1.2, linestyle="--", label=f"PFC{j+1} sim")
    ax2.set_title(f"PFC currents: table vs simulation (shot {shot.shot_id})")
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("I (A)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", ncol=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _resolve_output_layout(
    *,
    out_arg: str | None,
    out_csv_arg: str | None,
    name_arg: str | None,
    config_path: Path,
    n_shots: int,
    iters: int,
    restarts: int,
) -> tuple[Path, str, str]:
    """Определить корень вывода, имя директории запуска и имя CSV-файла."""
    if out_csv_arg is not None:
        legacy_csv = Path(out_csv_arg)
        output_root = Path(out_arg) if out_arg is not None else legacy_csv.parent
        if str(output_root) in ("", "."):
            output_root = Path("./output")
        csv_name = legacy_csv.name
        explicit_name = name_arg if name_arg is not None else legacy_csv.stem
    else:
        output_root = Path("./output") if out_arg is None else Path(out_arg)
        csv_name = "top_k.csv"
        explicit_name = name_arg

    if explicit_name is not None:
        stem = timestamped_stem("fit-sigma-L-gradient", explicit_name)
    else:
        stem = timestamped_stem(
            "fit-sigma-L-gradient",
            config_path.stem,
            f"shots-{int(n_shots)}",
            f"iters-{int(iters)}",
            f"restarts-{int(restarts)}",
        )
    safe_csv_name = slugify(csv_name)
    if not safe_csv_name.endswith(".csv"):
        safe_csv_name += ".csv"
    return output_root, stem, safe_csv_name


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    """Записать manifest JSON для одного запуска подбора."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Fit sigma and inductance_L with SPSA (gradient descent with random bumps), "
            "using PlasmaModel.step_currents() replay across shots. Writes top-k CSV like the grid fitter."
        )
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--ip-dir", required=True)
    ap.add_argument("--coils-dir", required=True)
    ap.add_argument("--time-divisor", type=float, default=1.0)
    ap.add_argument("--t-min", type=float, default=None)
    ap.add_argument("--t-max", type=float, default=None)
    ap.add_argument("--shot", type=str, default=None)

    ap.add_argument("--sigma-min", type=float, default=1.0)
    ap.add_argument("--sigma-max", type=float, default=1e9)
    ap.add_argument("--L-min", type=float, default=1e-8)
    ap.add_argument("--L-max", type=float, default=1e-2)

    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out", default=None, help="Output root for timestamped fit run folders. Defaults to ./output.")
    ap.add_argument("--name", default=None, help="Optional descriptive run name used in the generated run folder.")
    ap.add_argument("--out-csv", default=None, help="CSV filename, or legacy output/foo.csv path used to derive output root and run name.")

    ap.add_argument("--plot", action="store_true")

    args = ap.parse_args()

    shots, meta = _load_shots(
        config_path=Path(args.config),
        ip_dir=Path(args.ip_dir),
        coils_dir=Path(args.coils_dir),
        time_divisor=float(args.time_divisor),
        t_min=args.t_min,
        t_max=args.t_max,
        only_shot=args.shot,
    )

    cfg = meta["cfg"]
    base_model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics)

    # Applied-current tables: override actuator lag and disable clipping to avoid replay distortion.
    if float(cfg.physics.actuator_tau) != 0.0:
        sys.stderr.write(
            f"Note: overriding actuator_tau from {float(cfg.physics.actuator_tau):.6g} to 0.0 for fitting\n"
        )
    base_model.actuator_tau = 0.0
    base_model.pfc_deriv_limit = None
    base_model.sol_deriv_limit = None
    base_model.pfc_current_limit = None
    base_model.sol_current_limit = None

    if meta["skipped"]:
        sys.stderr.write("Skipping shots:\n")
        for sid, reason in meta["skipped"][:20]:
            sys.stderr.write(f"  shot {sid}: {reason}\n")

    output_root, run_stem, csv_name = _resolve_output_layout(
        out_arg=args.out,
        out_csv_arg=args.out_csv,
        name_arg=args.name,
        config_path=Path(args.config),
        n_shots=len(shots),
        iters=int(args.iters),
        restarts=int(args.restarts),
    )
    paths = allocate_artifact_run_dir(output_root, run_stem)
    out_csv = paths.run_dir / csv_name
    sys.stdout.write(f"{paths.run_dir}\n")

    plot_enabled = bool(args.plot)
    shot_ids = [str(sh.shot_id) for sh in shots]

    # Store up to this many candidate curves per shot for overlay plots.
    store_max_curves = 220
    store_curves: dict[str, list[np.ndarray]] | None = None
    store_seen: dict[str, int] | None = None
    if plot_enabled:
        store_curves = {sid: [] for sid in shot_ids}
        store_seen = {sid: 0 for sid in shot_ids}

    best, history = _spsa_optimize(
        base_model,
        shots,
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        L_min=float(args.L_min),
        L_max=float(args.L_max),
        iters=int(args.iters),
        restarts=int(args.restarts),
        seed=int(args.seed),
        store_curves=store_curves,
        store_max_curves=int(store_max_curves),
    )

    # Top-k results from the whole history.
    history_sorted = sorted(history, key=lambda r: (r.mean_nrmse, r.mean_rmse))
    top = history_sorted[: int(args.top_k)]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "sigma", "inductance_L", "tau", "mean_rmse", "mean_nrmse", "n_shots"])
        for i, r in enumerate(top, start=1):
            w.writerow(
                [
                    i,
                    f"{r.sigma:.16g}",
                    f"{r.inductance_L:.16g}",
                    f"{r.tau:.16g}",
                    f"{r.mean_rmse:.16g}",
                    f"{r.mean_nrmse:.16g}",
                    r.n_shots,
                ]
            )

    # Plots for best params
    plot_root: Path | None = None
    ip_overlay_dir: Path | None = None
    ip_best_dir: Path | None = None
    coil_dir: Path | None = None
    if plot_enabled:
        plot_root = paths.run_dir / "plots"
        ip_overlay_dir = plot_root / "ip_overlay"
        ip_best_dir = plot_root / "ip_best"
        coil_dir = plot_root / "coil_currents"

        for sh in shots:
            # Simulate under best params to get Ip and used coil currents.
            model = _model_for_scored_shot(sh, sigma=float(best.sigma), inductance_L=float(best.inductance_L))
            _init_model_for_shot(
                model,
                ip0=float(sh.ip[0]),
                ip_at_t0=float(sh.ip[0]),
                pfc0=sh.pfc_curr[0],
                sol0=sh.sol_curr[0],
            )
            ip_best, pfc_used, sol_used = _simulate_shot_record(model, sh)
            sid = str(sh.shot_id)

            _plot_ip_overlay(
                shot=sh,
                candidate_curves=(store_curves.get(sid, []) if store_curves is not None else []),
                best_curve=ip_best,
                best_sigma=float(best.sigma),
                best_L=float(best.inductance_L),
                best_mean_rmse=float(best.mean_rmse),
                out_path=ip_overlay_dir / f"shot_{sid}.png",
            )

            _plot_ip_best_only(
                shot=sh,
                best_curve=ip_best,
                best_sigma=float(best.sigma),
                best_L=float(best.inductance_L),
                best_mean_rmse=float(best.mean_rmse),
                out_path=ip_best_dir / f"shot_{sid}.png",
            )

            _plot_coil_currents(
                shot=sh,
                pfc_used=pfc_used,
                sol_used=sol_used,
                out_path=coil_dir / f"shot_{sid}.png",
            )

        sys.stdout.write(f"plot_dir={plot_root}\n")
        sys.stdout.write(f"plot_subdir_ip_overlay={ip_overlay_dir}\n")
        sys.stdout.write(f"plot_subdir_ip_best={ip_best_dir}\n")
        sys.stdout.write(f"plot_subdir_coil_currents={coil_dir}\n")

    sys.stdout.write(f"best_sigma={best.sigma:.16g}\n")
    sys.stdout.write(f"best_inductance_L={best.inductance_L:.16g}\n")
    sys.stdout.write(f"best_tau={best.tau:.16g}\n")
    sys.stdout.write(f"best_mean_rmse={best.mean_rmse:.16g}\n")
    sys.stdout.write(f"best_mean_nrmse={best.mean_nrmse:.16g}\n")
    sys.stdout.write(f"n_shots={best.n_shots}\n")
    sys.stdout.write(f"manifest_path={paths.manifest_path}\n")
    sys.stdout.write(f"top_k_csv={out_csv}\n")
    _write_manifest(
        paths.manifest_path,
        {
            "run_id": paths.run_id,
            "script": "fit_sigma_L_gradient",
            "config": str(Path(args.config)),
            "ip_dir": str(Path(args.ip_dir)),
            "coils_dir": str(Path(args.coils_dir)),
            "shot_ids": shot_ids,
            "skipped": meta["skipped"],
            "sigma_min": float(args.sigma_min),
            "sigma_max": float(args.sigma_max),
            "L_min": float(args.L_min),
            "L_max": float(args.L_max),
            "iters": int(args.iters),
            "restarts": int(args.restarts),
            "seed": int(args.seed),
            "top_k": int(args.top_k),
            "objective": "mean_nrmse",
            "best": {
                "sigma": float(best.sigma),
                "inductance_L": float(best.inductance_L),
                "tau": float(best.tau),
                "mean_rmse": float(best.mean_rmse),
                "mean_nrmse": float(best.mean_nrmse),
                "n_shots": int(best.n_shots),
            },
            "artifacts": {
                "top_k_csv": str(out_csv),
                "plot_dir": str(plot_root) if plot_root is not None else None,
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
