"""Построить графики восстановленных границ T15MD по таблицам параметров."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_repo_root_on_path() -> None:
    """Добавить корень репозитория в путь импорта при прямом запуске скрипта."""
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from tokamak_control.geometry.parametric_boundary import BoundaryParameters, evaluate_parametric_boundary
from tokamak_control.io.data_io import load_run


@dataclass(frozen=True, slots=True, repr=True)
class ParameterRow:
    """Одна валидная строка таблицы параметров восстановленной границы."""

    csv_index: int
    run_name: str
    shot: str
    source_npz: Path | None
    step: int
    t: float
    parameters: BoundaryParameters
    rmse: float
    max_error: float


@dataclass(frozen=True, slots=True, repr=True)
class ParameterTable:
    """Таблица параметров одного replay-запуска."""

    csv_path: Path
    run_name: str
    rows: tuple[ParameterRow, ...]


def main(argv: Sequence[str] | None = None) -> int:
    """Разобрать аргументы и построить графики восстановленных границ."""
    parser = argparse.ArgumentParser(
        description="Plot reconstructed T15MD boundaries from fitted boundary parameter CSV files."
    )
    parser.add_argument(
        "--params-root",
        type=Path,
        default=Path("output/t15_boundary_parameters"),
        help="Directory containing *_boundary_params.csv files.",
    )
    parser.add_argument(
        "--param-glob",
        default="*_boundary_params.csv",
        help="Glob used inside --params-root to select parameter CSV files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/t15_boundary_reconstructions"),
        help="Output directory for reconstruction PNG files.",
    )
    parser.add_argument(
        "--theta-count",
        type=int,
        default=360,
        help="Number of angle samples used to draw each reconstructed boundary.",
    )
    parser.add_argument(
        "--max-curves",
        type=int,
        default=64,
        help="Maximum number of reconstructed curves drawn in each overview image.",
    )
    parser.add_argument(
        "--comparison-count",
        type=int,
        default=6,
        help="Number of sampled steps used for original-vs-reconstructed comparison panels.",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Disable comparison panels against original boundary_poly_true artifacts.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    csv_paths = sorted(Path(args.params_root).glob(str(args.param_glob)))
    if not csv_paths:
        raise SystemExit(f"No parameter CSV files found under {args.params_root / str(args.param_glob)}")

    args.out.mkdir(parents=True, exist_ok=True)
    for csv_path in csv_paths:
        table = _load_parameter_table(csv_path)
        if not table.rows:
            print(f"{csv_path} skipped: no valid fitted rows")
            continue
        overview_path = args.out / f"{table.run_name}_reconstructed_overview.png"
        _write_overview_plot(
            table=table,
            output_path=overview_path,
            theta_count=int(args.theta_count),
            max_curves=int(args.max_curves),
        )
        print(str(overview_path))
        if not bool(args.skip_comparison):
            comparison_path = args.out / f"{table.run_name}_reconstruction_check.png"
            wrote_comparison = _write_comparison_plot(
                table=table,
                output_path=comparison_path,
                theta_count=int(args.theta_count),
                comparison_count=int(args.comparison_count),
            )
            if wrote_comparison:
                print(str(comparison_path))
    return 0


def _load_parameter_table(csv_path: Path) -> ParameterTable:
    """Прочитать CSV-таблицу параметров и оставить только валидные подгоны."""
    rows: list[ParameterRow] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for csv_index, raw_row in enumerate(reader):
            row = _parse_parameter_row(csv_index, raw_row)
            if row is not None:
                rows.append(row)
    if rows:
        run_name = rows[0].run_name
    else:
        run_name = _run_name_from_csv_path(csv_path)
    return ParameterTable(csv_path=Path(csv_path), run_name=run_name, rows=tuple(rows))


def _parse_parameter_row(csv_index: int, row: Mapping[str, str]) -> ParameterRow | None:
    """Преобразовать одну строку CSV в параметры или пропустить невалидный подбор."""
    if str(row.get("fit_status", "")).strip() != "ok":
        return None
    parameters = BoundaryParameters(
        R0=_parse_float(row, "R0"),
        Z0=_parse_float(row, "Z0"),
        A0=_parse_float(row, "A0"),
        kappa=_parse_float(row, "kappa"),
        delta=_parse_float(row, "delta"),
    )
    values = parameters.as_array()
    if not bool(np.all(np.isfinite(values)) and parameters.A0 > 0.0 and parameters.kappa > 0.0):
        return None
    source_raw = str(row.get("source_npz", "")).strip()
    source_npz = Path(source_raw) if source_raw else None
    return ParameterRow(
        csv_index=int(csv_index),
        run_name=str(row.get("run_name", "")).strip() or "boundary_parameters",
        shot=str(row.get("shot", "")).strip(),
        source_npz=source_npz,
        step=_parse_int(row, "step", csv_index),
        t=_parse_float(row, "t"),
        parameters=parameters,
        rmse=_parse_float(row, "rmse"),
        max_error=_parse_float(row, "max_error"),
    )


def _run_name_from_csv_path(csv_path: Path) -> str:
    """Получить имя запуска из имени CSV при отсутствии валидных строк."""
    stem = Path(csv_path).stem
    suffix = "_boundary_params"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _parse_float(row: Mapping[str, str], key: str) -> float:
    """Прочитать float-значение из CSV-строки."""
    raw = str(row.get(key, "nan")).strip()
    if not raw:
        return float("nan")
    return float(raw)


def _parse_int(row: Mapping[str, str], key: str, default: int) -> int:
    """Прочитать int-значение из CSV-строки."""
    raw = str(row.get(key, "")).strip()
    if not raw:
        return int(default)
    return int(float(raw))


def _theta_grid(theta_count: int) -> np.ndarray:
    """Построить сетку углов для отрисовки замкнутой аналитической границы."""
    count = max(int(theta_count), 32)
    return np.linspace(-np.pi, np.pi, count + 1, endpoint=True)


def _select_even_indices(size: int, max_count: int) -> np.ndarray:
    """Выбрать равномерные индексы из временного ряда."""
    n = int(size)
    if n <= 0:
        return np.zeros((0,), dtype=int)
    limit = int(max_count)
    if limit <= 0 or n <= limit:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, limit, dtype=int))


def _write_overview_plot(
    *,
    table: ParameterTable,
    output_path: Path,
    theta_count: int,
    max_curves: int,
) -> None:
    """Записать обзорный график восстановленных границ одного запуска."""
    theta = _theta_grid(theta_count)
    indices = _select_even_indices(len(table.rows), max_curves)
    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    for index in indices:
        row = table.rows[int(index)]
        polyline = evaluate_parametric_boundary(theta, row.parameters)
        ax.plot(polyline[:, 0], polyline[:, 1], linewidth=0.8, alpha=0.55)
    ax.set_title(f"{table.run_name}: reconstructed boundaries")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_comparison_plot(
    *,
    table: ParameterTable,
    output_path: Path,
    theta_count: int,
    comparison_count: int,
) -> bool:
    """Записать панели сравнения исходных и восстановленных границ."""
    loaded = _load_original_boundaries(table)
    if loaded is None:
        return False
    boundaries, steps = loaded
    indices = _select_even_indices(len(table.rows), comparison_count)
    if indices.size == 0:
        return False
    theta = _theta_grid(theta_count)
    step_to_index = _step_index_map(steps)
    n_cols = min(3, max(int(indices.size), 1))
    n_rows = int(np.ceil(float(indices.size) / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows), squeeze=False)
    flat_axes = np.asarray(axes, dtype=object).reshape(-1)
    for panel_index, row_index in enumerate(indices):
        ax = flat_axes[int(panel_index)]
        row = table.rows[int(row_index)]
        reconstructed = evaluate_parametric_boundary(theta, row.parameters)
        original = _original_polyline_for_row(boundaries, step_to_index, row)
        if original is not None:
            ax.plot(original[:, 0], original[:, 1], linewidth=1.0, label="original")
        ax.plot(reconstructed[:, 0], reconstructed[:, 1], linewidth=1.0, linestyle="--", label="reconstructed")
        ax.set_title(f"step={row.step} t={row.t:.4g}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.set_xlabel("R")
        ax.set_ylabel("Z")
        if panel_index == 0:
            ax.legend(loc="best")
    for extra_index in range(int(indices.size), flat_axes.shape[0]):
        flat_axes[extra_index].axis("off")
    fig.suptitle(f"{table.run_name}: original vs reconstructed boundary")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def _load_original_boundaries(table: ParameterTable) -> tuple[np.ndarray, np.ndarray] | None:
    """Загрузить исходные boundary_poly_true для проверки восстановления."""
    source_npz = _first_existing_source_npz(table.rows)
    if source_npz is None:
        return None
    run_data = load_run(source_npz)
    if "boundary_poly_true" not in run_data:
        return None
    boundaries = np.asarray(run_data["boundary_poly_true"], dtype=float)
    if boundaries.ndim != 3 or boundaries.shape[2] != 2:
        return None
    steps = np.asarray(run_data.get("step", np.arange(boundaries.shape[0], dtype=int)), dtype=int).reshape(-1)
    return boundaries, steps


def _first_existing_source_npz(rows: Sequence[ParameterRow]) -> Path | None:
    """Найти первый существующий NPZ-артефакт из строк таблицы."""
    for row in rows:
        if row.source_npz is not None and row.source_npz.exists():
            return row.source_npz
    return None


def _step_index_map(steps: np.ndarray) -> dict[int, int]:
    """Построить отображение номер шага -> индекс строки артефакта."""
    mapping: dict[int, int] = {}
    for index, step in enumerate(np.asarray(steps, dtype=int).reshape(-1)):
        mapping[int(step)] = int(index)
    return mapping


def _original_polyline_for_row(
    boundaries: np.ndarray,
    step_to_index: Mapping[int, int],
    row: ParameterRow,
) -> np.ndarray | None:
    """Получить исходный контур для строки параметров."""
    artifact_index = step_to_index.get(int(row.step), int(row.csv_index))
    if artifact_index < 0 or artifact_index >= boundaries.shape[0]:
        return None
    padded = np.asarray(boundaries[artifact_index], dtype=float)
    finite = np.all(np.isfinite(padded), axis=1)
    polyline = padded[finite]
    if polyline.shape[0] < 3:
        return None
    return polyline


if __name__ == "__main__":
    raise SystemExit(main())
