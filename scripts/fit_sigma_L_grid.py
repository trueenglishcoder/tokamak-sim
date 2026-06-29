# scripts/fit_sigma_L_grid.py
"""
Fit effective sigma and inductance_L for the T15MD tokamak config from processed shot CSVs.

Expected inputs
- Ip CSV files named t15md_<shot>_ip.csv
    * two semicolon-separated columns without a header:
      time_s;Ip
- Coil CSV files named t15md_<shot>_coils.csv
    * semicolon-separated columns without a header:
      time_s;SOL...;PFC...
    * the current columns must already be in project runtime units
    * the current columns must already match the config bank order

The input CSV time column is interpreted as seconds. This fitter is for already
processed local shot windows and deliberately does not perform raw T15MD unit
conversion.

Core fitting loop
- Load matching Ip and coil CSVs by shot number.
- Merge near-duplicate timestamps caused by floating-point preprocessing noise.
- Validate that cleaned times are finite and strictly increasing.
- Resample Ip and coil currents onto the model dt over the overlap window.
- For each (sigma, inductance_L) candidate:
    * override model sigma and inductance_L
    * initialize the model from the first sample of each shot
    * replay measured coil currents through PlasmaModel.step_currents()
    * score mean RMSE and mean NRMSE across shots

Fitting semantics
- Coil tables contain applied currents, so actuator lag must not be applied.
  The fitter forces actuator_tau = 0.0.
- To avoid distorting replay, the fitter disables derivative and current clipping.
- The optimization objective is mean NRMSE, where each shot RMSE is normalized by
  the Ip dynamic range in that fitted shot window.

Plots
When --plot is set, the script writes three subfolders under <run_dir>/plots/:
- ip_overlay/: for each shot, overlays all tried candidate Ip curves faintly,
  and highlights true Ip plus the globally best Ip curve.
- ip_best/: for each shot, plots only true Ip vs globally best Ip curve.
- coil_currents/: for each shot, plots coil currents from the table vs the coil
  currents actually used by the simulation for the globally best parameters.

Output artifacts are written into one timestamped run directory per invocation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

_TIME_EPS = 1.0e-12


@dataclass(frozen=True, slots=True)
class Shot:
    shot_id: str
    t: np.ndarray
    ip: np.ndarray
    pfc_curr: np.ndarray
    sol_curr: np.ndarray
    cfg: LoadedConfig | None = None


@dataclass(frozen=True, slots=True)
class PairScore:
    sigma: float
    inductance_L: float
    tau: float
    mean_rmse: float
    mean_nrmse: float
    n_shots: int


def _read_numeric_csv_rows(path: Path, expected_cols: int) -> np.ndarray:
    rows: list[list[float]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(";")]

            if parts and parts[-1] == "":
                parts = parts[:-1]

            if len(parts) != expected_cols:
                raise ValueError(
                    f"{path} line {line_no}: expected {expected_cols} semicolon-separated "
                    f"columns, got {len(parts)}"
                )

            try:
                vals = [float(x) for x in parts]
            except ValueError as exc:
                raise ValueError(f"{path} line {line_no}: non-numeric value") from exc

            if not all(np.isfinite(vals)):
                raise ValueError(f"{path} line {line_no}: non-finite numeric value")

            rows.append(vals)

    if not rows:
        raise ValueError(f"No numeric {expected_cols}-column rows found in {path}")

    return np.asarray(rows, dtype=float)


def _coalesce_near_duplicate_time_rows(rows: np.ndarray, *, path: Path) -> np.ndarray:
    rows = np.asarray(rows, dtype=float)

    if rows.ndim != 2:
        raise ValueError(f"{path}: expected a two-dimensional numeric table")
    if rows.shape[0] < 3:
        raise ValueError(f"{path}: at least 3 time samples are required")
    if not np.all(np.isfinite(rows)):
        raise ValueError(f"{path}: table contains non-finite values")

    merged: list[np.ndarray] = []
    group: list[np.ndarray] = [rows[0]]
    last_t = float(rows[0, 0])

    for row in rows[1:]:
        t = float(row[0])
        dt = t - last_t
        if dt < -_TIME_EPS:
            raise ValueError(
                f"{path}: time column must be nondecreasing before duplicate merging. "
                f"Found {last_t} followed by {t}"
            )
        if dt <= _TIME_EPS:
            group.append(row)
        else:
            merged.append(np.mean(np.vstack(group), axis=0))
            group = [row]
        last_t = t

    merged.append(np.mean(np.vstack(group), axis=0))
    out = np.vstack(merged)
    if out.shape[0] < 3:
        raise ValueError(f"{path}: fewer than 3 unique time samples after duplicate merging")
    return out


def _validate_time_seconds(t: np.ndarray, *, path: Path) -> np.ndarray:
    t = np.asarray(t, dtype=float)

    if t.ndim != 1:
        raise ValueError(f"{path}: time column is not one-dimensional")
    if t.size < 3:
        raise ValueError(f"{path}: at least 3 time samples are required")
    if not np.all(np.isfinite(t)):
        raise ValueError(f"{path}: time column contains non-finite values")

    dt = np.diff(t)
    bad = np.where(dt <= _TIME_EPS)[0]
    if bad.size > 0:
        i = int(bad[0])
        raise ValueError(
            f"{path}: time column must be strictly increasing. "
            f"Found t[{i}]={t[i]} and t[{i + 1}]={t[i + 1]}"
        )

    return t


def _discover_pairs(ip_dir: Path, coils_dir: Path) -> list[tuple[str, Path, Path]]:
    if not ip_dir.exists() or not ip_dir.is_dir():
        raise NotADirectoryError(f"Ip directory not found or not a directory: {ip_dir}")
    if not coils_dir.exists() or not coils_dir.is_dir():
        raise NotADirectoryError(f"Coil directory not found or not a directory: {coils_dir}")

    ip_files: dict[str, Path] = {}
    coil_files: dict[str, Path] = {}

    for p in sorted(ip_dir.iterdir()):
        if not p.is_file():
            continue
        m = IP_CSV_PATTERN.fullmatch(p.name)
        if not m:
            continue
        sid = m.group(1)
        if sid in ip_files:
            raise ValueError(f"Duplicate Ip files for shot {sid}: {ip_files[sid]} and {p}")
        ip_files[sid] = p

    for p in sorted(coils_dir.iterdir()):
        if not p.is_file():
            continue
        m = COIL_CSV_PATTERN.fullmatch(p.name)
        if not m:
            continue
        sid = m.group(1)
        if sid in coil_files:
            raise ValueError(f"Duplicate coil files for shot {sid}: {coil_files[sid]} and {p}")
        coil_files[sid] = p

    shot_ids = sorted(set(ip_files) & set(coil_files), key=lambda s: int(s))
    if not shot_ids:
        raise FileNotFoundError(
            "No matching preprocessed CSV shot pairs found.\n"
            f"ip_dir={ip_dir}\n"
            f"coils_dir={coils_dir}\n"
            f"Ip shots: {sorted(ip_files, key=int)}\n"
            f"Coil shots: {sorted(coil_files, key=int)}"
        )

    return [(sid, ip_files[sid], coil_files[sid]) for sid in shot_ids]


def _resample_matrix(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    y_src = np.asarray(y_src, dtype=float)

    if y_src.ndim != 2:
        raise ValueError("Expected a two-dimensional current matrix")

    return np.column_stack(
        [np.interp(t_dst, t_src, y_src[:, j]) for j in range(y_src.shape[1])]
    )


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


def _resample_overlap(
    *,
    shot_id: str,
    t_ip: np.ndarray,
    ip_raw: np.ndarray,
    t_coil: np.ndarray,
    sol_raw: np.ndarray,
    pfc_raw: np.ndarray,
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

    if not np.isfinite(t0) or not np.isfinite(t1):
        raise ValueError("Non-finite overlap window")
    if t1 <= t0:
        raise ValueError("Empty overlap window after applying t_min/t_max")

    duration = t1 - t0
    if duration < 2.0 * float(dt):
        raise ValueError(
            f"Fit window too short for model dt={dt}. "
            f"Window duration is {duration}. "
            "Input CSV times must be seconds. Do not divide already processed CSV time by 1000."
        )

    n = int(math.floor(duration / float(dt))) + 1
    if n < 3:
        raise ValueError(f"Fit window too short for dt={dt}")

    t_abs = t0 + np.arange(n, dtype=float) * float(dt)

    ip = np.interp(t_abs, t_ip, ip_raw)
    sol = _resample_matrix(t_coil, sol_raw, t_abs)
    pfc = _resample_matrix(t_coil, pfc_raw, t_abs)

    t_local = t_abs - float(t_abs[0])

    return Shot(
        shot_id=str(shot_id),
        t=t_local,
        ip=np.asarray(ip, dtype=float),
        pfc_curr=np.asarray(pfc, dtype=float),
        sol_curr=np.asarray(sol, dtype=float),
    )


def _load_shots(
    *,
    config_path: Path,
    ip_dir: Path,
    coils_dir: Path,
    t_min: float | None,
    t_max: float | None,
    only_shot: str | None,
) -> tuple[list[Shot], dict[str, object]]:
    cfg = load_config(config_path)

    dt = float(cfg.physics.t_step)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Config t_step must be finite and positive, got {dt}")

    pairs = _discover_pairs(ip_dir, coils_dir)
    if only_shot is not None:
        pairs = [p for p in pairs if p[0] == str(only_shot)]
        if not pairs:
            raise ValueError(f"Requested shot {only_shot} not found in provided directories")

    shots: list[Shot] = []
    skipped: list[tuple[str, str]] = []

    for shot_id, ip_path, coil_path in pairs:
        try:
            shot_cfg = load_config(config_path)
            n_pfc = int(shot_cfg.pfc.n_coils)
            n_sol = int(shot_cfg.sol.n_coils)
            expected_coil_cols = 1 + n_sol + n_pfc
            ip_rows = _coalesce_near_duplicate_time_rows(
                _read_numeric_csv_rows(ip_path, expected_cols=2),
                path=ip_path,
            )
            try:
                coil_rows = _coalesce_near_duplicate_time_rows(
                    _read_numeric_csv_rows(coil_path, expected_cols=expected_coil_cols),
                    path=coil_path,
                )
                sol_source_cols = n_sol
            except Exception:
                group_sizes = _split_sol_group_sizes(shot_cfg)
                sol_source_cols = len(group_sizes)
                coil_rows = _coalesce_near_duplicate_time_rows(
                    _read_numeric_csv_rows(coil_path, expected_cols=1 + sol_source_cols + n_pfc),
                    path=coil_path,
                )

            t_ip = _validate_time_seconds(ip_rows[:, 0], path=ip_path)
            ip_raw = np.asarray(ip_rows[:, 1], dtype=float)

            t_coil = _validate_time_seconds(coil_rows[:, 0], path=coil_path)
            sol_raw_source = np.asarray(coil_rows[:, 1 : 1 + sol_source_cols], dtype=float)
            if sol_source_cols == n_sol:
                sol_raw = sol_raw_source
            else:
                sol_raw = _expand_split_sol_rows(sol_raw_source, _split_sol_group_sizes(shot_cfg))
            pfc_raw = np.asarray(coil_rows[:, 1 + sol_source_cols : 1 + sol_source_cols + n_pfc], dtype=float)

            sh = _resample_overlap(
                shot_id=str(shot_id),
                t_ip=t_ip,
                ip_raw=ip_raw,
                t_coil=t_coil,
                sol_raw=sol_raw,
                pfc_raw=pfc_raw,
                dt=dt,
                t_min=t_min,
                t_max=t_max,
            )
            sh = Shot(shot_id=sh.shot_id, t=sh.t, ip=sh.ip, pfc_curr=sh.pfc_curr, sol_curr=sh.sol_curr, cfg=shot_cfg)
            shots.append(sh)
        except Exception as exc:
            skipped.append((str(shot_id), str(exc)))

    if not shots:
        msg = "No usable shots were loaded.\n"
        if skipped:
            msg += "Skip reasons:\n" + "\n".join([f"  {sid}: {reason}" for sid, reason in skipped[:20]])
        raise ValueError(msg)

    meta = {
        "dt": dt,
        "cfg": cfg,
        "skipped": skipped,
    }
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

    pfc0 = np.asarray(pfc0, dtype=float)
    sol0 = np.asarray(sol0, dtype=float)

    if pfc0.shape != (model.pfc.n_coils,):
        raise ValueError(f"pfc0 shape {pfc0.shape} does not match model PFC count {model.pfc.n_coils}")
    if sol0.shape != (model.sol.n_coils,):
        raise ValueError(f"sol0 shape {sol0.shape} does not match model SOL count {model.sol.n_coils}")

    psi0 = model._compose_psi(float(ip_at_t0), pfc0, sol0)

    model.Ip0 = float(ip0)
    model.state = PlasmaState(
        t=0.0,
        step=0,
        Ip=float(ip_at_t0),
        Ip0=float(ip0),
        psi=psi0,
        pfc_currents=pfc0.copy(),
        pfc_current_derivs=np.zeros((model.pfc.n_coils,), dtype=float),
        sol_currents=sol0.copy(),
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
    model = PlasmaModel.from_settings(grid=shot.cfg.grid, pfc=shot.cfg.pfc, sol=shot.cfg.sol, settings=shot.cfg.physics, ip0=float(shot.ip[0]))
    model.sigma = float(sigma)
    model.inductance_L = float(inductance_L)
    model.actuator_tau = 0.0
    model.pfc_deriv_limit = None
    model.sol_deriv_limit = None
    model.pfc_current_limit = None
    model.sol_current_limit = None
    return model


def _nrmse_scale(ip: np.ndarray) -> float:
    ip = np.asarray(ip, dtype=float)
    span = float(np.max(ip) - np.min(ip))
    return max(span, 1.0e-12)


def _score_pair(
    base_model: PlasmaModel,
    shots: list[Shot],
    *,
    sigma: float,
    inductance_L: float,
    collect_preds: bool,
) -> tuple[PairScore, dict[str, np.ndarray]]:
    _ = base_model
    tau = float(sigma) * float(inductance_L)

    rmses: list[float] = []
    nrmses: list[float] = []

    preds: dict[str, np.ndarray] = {}

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

        if collect_preds:
            preds[str(sh.shot_id)] = pred.copy()

        err = pred - sh.ip
        rmse = float(np.sqrt(np.mean(err ** 2)))
        nrmse = float(rmse / _nrmse_scale(sh.ip))

        rmses.append(rmse)
        nrmses.append(nrmse)

    sc = PairScore(
        sigma=float(sigma),
        inductance_L=float(inductance_L),
        tau=tau,
        mean_rmse=float(np.mean(rmses)),
        mean_nrmse=float(np.mean(nrmses)),
        n_shots=len(shots),
    )
    return sc, preds


def _logspace(min_val: float, max_val: float, n: int) -> np.ndarray:
    if min_val <= 0.0 or max_val <= 0.0:
        raise ValueError("logspace bounds must be > 0")
    if max_val < min_val:
        raise ValueError("max must be >= min")
    if n <= 0:
        raise ValueError("points must be > 0")
    return np.logspace(math.log10(float(min_val)), math.log10(float(max_val)), int(n))


def _set_ylim_from_true_best(ax: plt.Axes, y_true: np.ndarray, y_best: np.ndarray) -> None:
    y_min = float(min(np.min(y_true), np.min(y_best)))
    y_max = float(max(np.max(y_true), np.max(y_best)))
    pad = 0.05 * max(1.0e-12, (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)


def _plot_ip_overlay(
    *,
    shot: Shot,
    candidate_curves: list[np.ndarray],
    best_curve: np.ndarray,
    best_sigma: float,
    best_L: float,
    best_mean_nrmse: float,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    for curve in candidate_curves:
        ax.plot(shot.t, curve, alpha=0.03, linewidth=0.8)

    ax.plot(shot.t, shot.ip, linewidth=2.8, label="True Ip")
    ax.plot(
        shot.t,
        best_curve,
        linewidth=2.4,
        label=f"Best global fit (sigma={best_sigma:.3g}, L={best_L:.3g}, mean_nrmse={best_mean_nrmse:.4g})",
    )

    ax.set_title(f"Ip overlay for shot {shot.shot_id}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Ip")
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
    best_mean_nrmse: float,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(shot.t, shot.ip, linewidth=2.8, label="True Ip")
    ax.plot(
        shot.t,
        best_curve,
        linewidth=2.4,
        label=f"Best global fit (sigma={best_sigma:.3g}, L={best_L:.3g}, mean_nrmse={best_mean_nrmse:.4g})",
    )

    ax.set_title(f"Ip best-only for shot {shot.shot_id}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Ip")
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
        ax1.plot(shot.t, shot.sol_curr[:, j], linewidth=1.8, label=f"SOL{j + 1} table")
        ax1.plot(shot.t, sol_used[:, j], linewidth=1.2, linestyle="--", label=f"SOL{j + 1} sim")

    ax1.set_title(f"SOL currents: table vs simulation (shot {shot.shot_id})")
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("current")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", ncol=2)

    for j in range(shot.pfc_curr.shape[1]):
        ax2.plot(shot.t, shot.pfc_curr[:, j], linewidth=1.8, label=f"PFC{j + 1} table")
        ax2.plot(shot.t, pfc_used[:, j], linewidth=1.2, linestyle="--", label=f"PFC{j + 1} sim")

    ax2.set_title(f"PFC currents: table vs simulation (shot {shot.shot_id})")
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("current")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", ncol=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _validate_args(args: argparse.Namespace) -> None:
    if float(args.time_divisor) != 1.0:
        raise ValueError(
            "This version expects preprocessed CSV files whose time column is already in seconds. "
            "Do not pass --time-divisor 1000. Omit --time-divisor or set --time-divisor 1."
        )

    if args.t_min is not None and not np.isfinite(float(args.t_min)):
        raise ValueError("--t-min must be finite when provided")
    if args.t_max is not None and not np.isfinite(float(args.t_max)):
        raise ValueError("--t-max must be finite when provided")
    if args.t_min is not None and args.t_max is not None and float(args.t_max) <= float(args.t_min):
        raise ValueError("--t-max must be greater than --t-min")

    if int(args.top_k) <= 0:
        raise ValueError("--top-k must be greater than zero")


def _resolve_output_layout(
    *,
    out_arg: str | None,
    out_csv_arg: str | None,
    name_arg: str | None,
    config_path: Path,
    n_shots: int,
    sigma_points: int,
    L_points: int,
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
        stem = timestamped_stem("fit-sigma-L-grid", explicit_name)
    else:
        stem = timestamped_stem(
            "fit-sigma-L-grid",
            config_path.stem,
            f"shots-{int(n_shots)}",
            f"sigma-{int(sigma_points)}",
            f"L-{int(L_points)}",
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
            "Grid-search sigma and inductance_L by replaying preprocessed measured T15MD coil "
            "currents through PlasmaModel.step_currents(). Input CSV time is interpreted as seconds."
        )
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--ip-dir", required=True)
    ap.add_argument("--coils-dir", required=True)

    # Kept only so older commands fail with a clear message instead of silently compressing time.
    ap.add_argument("--time-divisor", type=float, default=1.0)

    ap.add_argument("--t-min", type=float, default=None)
    ap.add_argument("--t-max", type=float, default=None)
    ap.add_argument("--shot", type=str, default=None)

    ap.add_argument("--sigma-min", type=float, default=1.0)
    ap.add_argument("--sigma-max", type=float, default=1e9)
    ap.add_argument("--sigma-points", type=int, default=25)

    ap.add_argument("--L-min", type=float, default=1e-8)
    ap.add_argument("--L-max", type=float, default=1e-2)
    ap.add_argument("--L-points", type=int, default=25)

    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out", default=None, help="Output root for timestamped fit run folders. Defaults to ./output.")
    ap.add_argument("--name", default=None, help="Optional descriptive run name used in the generated run folder.")
    ap.add_argument("--out-csv", default=None, help="CSV filename, or legacy output/foo.csv path used to derive output root and run name.")

    ap.add_argument("--plot", action="store_true")

    args = ap.parse_args()
    _validate_args(args)

    shots, meta = _load_shots(
        config_path=Path(args.config),
        ip_dir=Path(args.ip_dir),
        coils_dir=Path(args.coils_dir),
        t_min=args.t_min,
        t_max=args.t_max,
        only_shot=args.shot,
    )

    cfg = meta["cfg"]
    base_model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics, ip0=1.0)

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
        for sid, reason in meta["skipped"][:50]:
            sys.stderr.write(f"  shot {sid}: {reason}\n")
        if len(meta["skipped"]) > 50:
            sys.stderr.write(f"  ... {len(meta['skipped']) - 50} more skipped shots\n")

    output_root, run_stem, csv_name = _resolve_output_layout(
        out_arg=args.out,
        out_csv_arg=args.out_csv,
        name_arg=args.name,
        config_path=Path(args.config),
        n_shots=len(shots),
        sigma_points=int(args.sigma_points),
        L_points=int(args.L_points),
    )
    paths = allocate_artifact_run_dir(output_root, run_stem)
    out_csv = paths.run_dir / csv_name
    sys.stdout.write(f"{paths.run_dir}\n")

    sigma_grid = _logspace(float(args.sigma_min), float(args.sigma_max), int(args.sigma_points))
    L_grid = _logspace(float(args.L_min), float(args.L_max), int(args.L_points))

    plot_enabled = bool(args.plot)
    shot_ids = [str(sh.shot_id) for sh in shots]
    candidate_ip: dict[str, list[np.ndarray]] = {sid: [] for sid in shot_ids} if plot_enabled else {}

    results: list[PairScore] = []
    best_seen = float("inf")
    total = int(sigma_grid.size) * int(L_grid.size)

    with tqdm(total=total, desc="Grid search", unit="pair") as pbar:
        for sigma in sigma_grid:
            for L in L_grid:
                sc, preds = _score_pair(
                    base_model,
                    shots,
                    sigma=float(sigma),
                    inductance_L=float(L),
                    collect_preds=plot_enabled,
                )
                results.append(sc)

                if plot_enabled:
                    for sid, pred in preds.items():
                        candidate_ip[sid].append(pred)

                if sc.mean_nrmse < best_seen:
                    best_seen = float(sc.mean_nrmse)

                pbar.update(1)
                pbar.set_postfix_str(f"best_nrmse={best_seen:.6g}")

    results.sort(key=lambda r: (r.mean_nrmse, r.mean_rmse))
    top = results[: int(args.top_k)]
    best = top[0]

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
                candidate_curves=candidate_ip.get(sid, []),
                best_curve=ip_best,
                best_sigma=float(best.sigma),
                best_L=float(best.inductance_L),
                best_mean_nrmse=float(best.mean_nrmse),
                out_path=ip_overlay_dir / f"shot_{sid}.png",
            )

            _plot_ip_best_only(
                shot=sh,
                best_curve=ip_best,
                best_sigma=float(best.sigma),
                best_L=float(best.inductance_L),
                best_mean_nrmse=float(best.mean_nrmse),
                out_path=ip_best_dir / f"shot_{sid}.png",
            )

            _plot_coil_currents(
                shot=sh,
                pfc_used=pfc_used,
                sol_used=sol_used,
                out_path=coil_dir / f"shot_{sid}.png",
            )

    sys.stdout.write(f"best_sigma={best.sigma:.16g}\n")
    sys.stdout.write(f"best_inductance_L={best.inductance_L:.16g}\n")
    sys.stdout.write(f"best_tau={best.tau:.16g}\n")
    sys.stdout.write(f"best_mean_rmse={best.mean_rmse:.16g}\n")
    sys.stdout.write(f"best_mean_nrmse={best.mean_nrmse:.16g}\n")
    sys.stdout.write(f"objective=mean_nrmse\n")
    sys.stdout.write(f"n_shots={best.n_shots}\n")
    sys.stdout.write(f"manifest_path={paths.manifest_path}\n")
    sys.stdout.write(f"top_k_csv={out_csv}\n")

    if plot_root is not None:
        sys.stdout.write(f"plot_dir={plot_root}\n")
        sys.stdout.write(f"plot_subdir_ip_overlay={ip_overlay_dir}\n")
        sys.stdout.write(f"plot_subdir_ip_best={ip_best_dir}\n")
        sys.stdout.write(f"plot_subdir_coil_currents={coil_dir}\n")

    _write_manifest(
        paths.manifest_path,
        {
            "run_id": paths.run_id,
            "script": "fit_sigma_L_grid",
            "config": str(Path(args.config)),
            "ip_dir": str(Path(args.ip_dir)),
            "coils_dir": str(Path(args.coils_dir)),
            "shot_ids": shot_ids,
            "skipped": meta["skipped"],
            "sigma_min": float(args.sigma_min),
            "sigma_max": float(args.sigma_max),
            "sigma_points": int(args.sigma_points),
            "L_min": float(args.L_min),
            "L_max": float(args.L_max),
            "L_points": int(args.L_points),
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
