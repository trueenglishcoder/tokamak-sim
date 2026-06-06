"""Извлечь временные ряды параметров границы из replay-артефактов T15MD."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import numpy as np


def _ensure_repo_root_on_path() -> None:
    """Добавить корень репозитория в путь импорта при прямом запуске скрипта."""
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from tokamak_control.geometry.parametric_boundary import BoundaryFitResult, BoundaryParameters, fit_parametric_boundary
from tokamak_control.io.data_io import load_run


@dataclass(frozen=True, slots=True, repr=True)
class RunArtifact:
    """NPZ-артефакт одного replay-запуска и путь для таблицы параметров."""

    run_name: str
    shot: str
    npz_path: Path
    output_csv_path: Path


def main(argv: list[str] | None = None) -> int:
    """Разобрать аргументы и записать таблицы параметров для replay-запусков."""
    parser = argparse.ArgumentParser(
        description="Fit parametric T15MD boundary time series from simulated replay run artifacts."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root directory containing simulated_replay_* run folders.",
    )
    parser.add_argument(
        "--run-glob",
        default="simulated_replay_*",
        help="Glob used inside --runs-root to select replay run groups.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/t15_boundary_parameters"),
        help="Output directory for per-run boundary parameter CSV files.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=256,
        help="Number of arc-length samples used from each recorded boundary.",
    )
    parser.add_argument(
        "--theta-count",
        type=int,
        default=720,
        help="Dense angular grid size used for nearest-angle assignment.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=8,
        help="Maximum nearest-angle and least-squares refinement iterations per step.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=16,
        help="Minimum number of finite boundary vertices required for a fit.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    artifacts = _discover_run_artifacts(
        runs_root=args.runs_root,
        run_glob=str(args.run_glob),
        output_dir=args.out,
    )
    if not artifacts:
        raise SystemExit(f"No run*.npz artifacts found under {args.runs_root / str(args.run_glob)}")

    args.out.mkdir(parents=True, exist_ok=True)
    for artifact in artifacts:
        rows_written = _write_boundary_parameter_table(
            artifact=artifact,
            sample_count=int(args.sample_count),
            theta_count=int(args.theta_count),
            iterations=int(args.iterations),
            min_points=int(args.min_points),
        )
        print(f"{artifact.output_csv_path} rows={rows_written}")
    return 0


def _discover_run_artifacts(runs_root: Path, run_glob: str, output_dir: Path) -> list[RunArtifact]:
    """Найти последний NPZ-артефакт внутри каждой папки simulated_replay_* ."""
    artifacts: list[RunArtifact] = []
    for group_dir in sorted(Path(runs_root).glob(run_glob)):
        if not group_dir.is_dir():
            continue
        npz_files = sorted(group_dir.rglob("run*.npz"), key=lambda path: (path.stat().st_mtime, str(path)))
        if not npz_files:
            continue
        npz_path = npz_files[-1]
        run_name = group_dir.name
        shot = _extract_shot(run_name)
        output_csv = output_dir / f"{run_name}_boundary_params.csv"
        artifacts.append(RunArtifact(run_name=run_name, shot=shot, npz_path=npz_path, output_csv_path=output_csv))
    return artifacts


def _extract_shot(run_name: str) -> str:
    """Извлечь номер импульса из имени simulated_replay_<shot>."""
    match = re.search(r"(\d+)$", run_name)
    return match.group(1) if match is not None else ""


def _write_progress_line(label: str, current: int, total: int) -> None:
    """Обновить одну строку terminal progress bar для текущего запуска."""
    total_safe = max(int(total), 1)
    current_safe = min(max(int(current), 0), total_safe)
    fraction = float(current_safe) / float(total_safe)
    width = 32
    filled = int(round(width * fraction))
    bar = "#" * filled + "-" * (width - filled)
    percent = 100.0 * fraction
    sys.stderr.write(f"\r{label} [{bar}] {current_safe}/{total_safe} {percent:5.1f}%")
    sys.stderr.flush()


def _finish_progress_line() -> None:
    """Завершить строку progress bar переносом строки."""
    sys.stderr.write("\n")
    sys.stderr.flush()


def _write_boundary_parameter_table(
    *,
    artifact: RunArtifact,
    sample_count: int,
    theta_count: int,
    iterations: int,
    min_points: int,
) -> int:
    """Записать CSV-временной ряд параметров для одного NPZ-артефакта."""
    run_data = load_run(artifact.npz_path)
    if "boundary_poly_true" not in run_data:
        raise ValueError(f"Artifact has no boundary_poly_true channel: {artifact.npz_path}")

    boundaries = np.asarray(run_data["boundary_poly_true"], dtype=float)
    if boundaries.ndim != 3 or boundaries.shape[2] != 2:
        raise ValueError(f"boundary_poly_true must have shape (T, N, 2), got {boundaries.shape}")

    steps = np.asarray(run_data.get("step", np.arange(boundaries.shape[0], dtype=int)), dtype=int)
    times = np.asarray(run_data.get("t", np.full((boundaries.shape[0],), np.nan, dtype=float)), dtype=float)
    ip_values = np.asarray(run_data.get("Ip", np.full((boundaries.shape[0],), np.nan, dtype=float)), dtype=float)
    ip_ref_values = _optional_float_series(run_data, "Ip_ref", boundaries.shape[0])

    artifact.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    previous_parameters: BoundaryParameters | None = None
    rows_written = 0
    total_steps = int(boundaries.shape[0])
    _write_progress_line(artifact.run_name, 0, total_steps)
    with artifact.output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_name",
                "shot",
                "source_npz",
                "step",
                "t",
                "Ip",
                "Ip_ref",
                "R0",
                "Z0",
                "A0",
                "kappa",
                "delta",
                "rmse",
                "max_error",
                "n_boundary_points",
                "fit_status",
            ],
        )
        writer.writeheader()
        for index in range(boundaries.shape[0]):
            result = fit_parametric_boundary(
                boundaries[index],
                sample_count=sample_count,
                theta_count=theta_count,
                iterations=iterations,
                min_points=min_points,
                initial_parameters=previous_parameters,
            )
            if result.parameters is not None and result.fit_status == "ok":
                previous_parameters = result.parameters
            writer.writerow(
                _row_from_fit_result(
                    artifact=artifact,
                    index=index,
                    result=result,
                    steps=steps,
                    times=times,
                    ip_values=ip_values,
                    ip_ref_values=ip_ref_values,
                )
            )
            rows_written += 1
            _write_progress_line(artifact.run_name, rows_written, total_steps)
    _finish_progress_line()
    return rows_written


def _optional_float_series(run_data: dict[str, object], key: str, size: int) -> np.ndarray:
    """Вернуть числовой канал из артефакта или NaN-серию нужной длины."""
    if key not in run_data:
        return np.full((int(size),), np.nan, dtype=float)
    values = np.asarray(run_data[key], dtype=float).reshape(-1)
    if values.shape[0] < int(size):
        padded = np.full((int(size),), np.nan, dtype=float)
        padded[: values.shape[0]] = values
        return padded
    return values[: int(size)]


def _series_value(values: np.ndarray, index: int, default: float) -> float:
    """Безопасно прочитать значение временного ряда по индексу."""
    if index < 0 or index >= values.shape[0]:
        return float(default)
    return float(values[index])


def _step_value(values: np.ndarray, index: int) -> int:
    """Безопасно прочитать номер шага по индексу."""
    if index < 0 or index >= values.shape[0]:
        return int(index)
    return int(values[index])


def _row_from_fit_result(
    *,
    artifact: RunArtifact,
    index: int,
    result: BoundaryFitResult,
    steps: np.ndarray,
    times: np.ndarray,
    ip_values: np.ndarray,
    ip_ref_values: np.ndarray,
) -> dict[str, object]:
    """Собрать одну CSV-строку из результата подбора."""
    parameters = result.parameters
    return {
        "run_name": artifact.run_name,
        "shot": artifact.shot,
        "source_npz": str(artifact.npz_path),
        "step": _step_value(steps, index),
        "t": _series_value(times, index, float("nan")),
        "Ip": _series_value(ip_values, index, float("nan")),
        "Ip_ref": _series_value(ip_ref_values, index, float("nan")),
        "R0": float("nan") if parameters is None else float(parameters.R0),
        "Z0": float("nan") if parameters is None else float(parameters.Z0),
        "A0": float("nan") if parameters is None else float(parameters.A0),
        "kappa": float("nan") if parameters is None else float(parameters.kappa),
        "delta": float("nan") if parameters is None else float(parameters.delta),
        "rmse": float(result.rmse),
        "max_error": float(result.max_error),
        "n_boundary_points": int(result.n_boundary_points),
        "fit_status": str(result.fit_status),
    }


if __name__ == "__main__":
    raise SystemExit(main())
