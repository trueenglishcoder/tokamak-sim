"""Сгенерировать синтетические таблицы Ip по образцу существующих T15-подобных данных."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def _ensure_repo_root_on_path() -> None:
    """Добавить корень репозитория в путь импорта при прямом запуске скрипта."""
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()


@dataclass(frozen=True, slots=True, repr=True)
class SourceIpShot:
    """Исходная таблица Ip, используемая как шаблон формы сигнала."""

    shot_id: str
    path: Path
    time_s: np.ndarray
    ip_a: np.ndarray

    @property
    def rows(self) -> int:
        """Вернуть число строк в таблице."""
        return int(self.time_s.size)

    @property
    def duration_s(self) -> float:
        """Вернуть длительность таблицы в секундах."""
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def peak_abs_ip(self) -> float:
        """Вернуть максимальный модуль тока плазмы."""
        return float(np.max(np.abs(self.ip_a)))


@dataclass(frozen=True, slots=True, repr=True)
class SyntheticIpSummary:
    """Краткая metadata по одной сгенерированной таблице Ip."""

    shot_id: str
    template_shot_id: str
    rows: int
    duration_s: float
    peak_abs_ip: float
    amplitude_scale: float
    duration_scale: float
    ip_csv: str
    initial_currents_toml: str


def _finite_positive(name: str, value: float) -> float:
    """Проверить, что аргумент конечен и строго положителен."""
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return out


def _finite_nonnegative(name: str, value: float) -> float:
    """Проверить, что аргумент конечен и неотрицателен."""
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return out


def _extract_shot_id(path: Path) -> str:
    """Вытащить номер shot из имени файла формата t15md_<shot>_ip.csv."""
    stem = path.stem.lower()
    prefix = "t15md_"
    suffix = "_ip"
    if not stem.startswith(prefix) or not stem.endswith(suffix):
        raise ValueError(f"Unsupported Ip filename: {path.name}")
    shot_id = stem[len(prefix) : -len(suffix)]
    if not shot_id.isdigit():
        raise ValueError(f"Could not parse shot id from filename: {path.name}")
    return shot_id


def _load_semicolon_ip_table(path: Path) -> SourceIpShot:
    """Прочитать двухколоночную таблицу time;Ip в каноническом формате сценариев."""
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = [part for part in line.split(";") if part != ""]
            if len(parts) != 2:
                raise ValueError(f"Expected 2 semicolon-separated columns in {path}, got {len(parts)}")
            rows.append([float(parts[0]), float(parts[1])])

    if len(rows) < 2:
        raise ValueError(f"Ip table must contain at least 2 rows: {path}")

    arr = np.asarray(rows, dtype=float)
    time_s = np.asarray(arr[:, 0], dtype=float)
    ip_a = np.asarray(arr[:, 1], dtype=float)

    if not np.all(np.isfinite(time_s)):
        raise ValueError(f"Ip table contains non-finite timestamps: {path}")
    if not np.all(np.isfinite(ip_a)):
        raise ValueError(f"Ip table contains non-finite Ip values: {path}")
    if np.any(np.diff(time_s) < 0.0):
        raise ValueError(f"Ip table timestamps must be nondecreasing: {path}")

    time_s = time_s - float(time_s[0])
    if float(time_s[-1]) <= 0.0:
        raise ValueError(f"Ip table duration must be positive: {path}")
    if not np.any(np.abs(ip_a) > 0.0):
        raise ValueError(f"Ip table must contain at least one nonzero Ip sample: {path}")

    return SourceIpShot(
        shot_id=_extract_shot_id(path),
        path=path,
        time_s=time_s,
        ip_a=ip_a,
    )


def _discover_source_shots(source_ip_dir: Path) -> list[SourceIpShot]:
    """Собрать и проверить все шаблонные Ip-таблицы из директории."""
    if not source_ip_dir.exists():
        raise FileNotFoundError(f"Source Ip directory not found: {source_ip_dir}")
    if not source_ip_dir.is_dir():
        raise NotADirectoryError(f"Source Ip path is not a directory: {source_ip_dir}")

    shots: list[SourceIpShot] = []
    for path in sorted(source_ip_dir.glob("t15md_*_ip.csv")):
        shots.append(_load_semicolon_ip_table(path))

    if not shots:
        raise FileNotFoundError(f"No t15md_*_ip.csv files found in {source_ip_dir}")
    return shots


def _source_initial_currents_path(
    source_initial_currents_dir: Path,
    *,
    initial_currents_prefix: str,
    shot_id: str,
) -> Path:
    """Построить путь к исходному TOML начальных токов для выбранного shot."""
    path = source_initial_currents_dir / f"{initial_currents_prefix}_{shot_id}.toml"
    if not path.exists():
        raise FileNotFoundError(f"Initial-currents TOML not found for shot {shot_id}: {path}")
    return path


def _copy_initial_currents_toml(
    source_initial_currents_dir: Path,
    out_initial_currents_dir: Path,
    *,
    initial_currents_prefix: str,
    source_shot_id: str,
    target_shot_id: str,
) -> Path:
    """Скопировать TOML начальных токов под новое имя синтетического shot."""
    source_path = _source_initial_currents_path(
        source_initial_currents_dir,
        initial_currents_prefix=initial_currents_prefix,
        shot_id=source_shot_id,
    )
    out_path = out_initial_currents_dir / f"{initial_currents_prefix}_{target_shot_id}.toml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    return out_path


def _sample_scale(rng: np.random.Generator, *, jitter: float) -> float:
    """Сэмплировать положительный масштаб около 1.0."""
    if float(jitter) <= 0.0:
        return 1.0
    scale = 1.0 + float(rng.normal(0.0, float(jitter)))
    return max(scale, 0.05)


def _shape_envelope(
    n_rows: int,
    rng: np.random.Generator,
    *,
    shape_jitter: float,
    n_anchors: int = 5,
) -> np.ndarray:
    """Собрать гладкую мультипликативную огибающую формы сигнала."""
    if n_rows < 2:
        raise ValueError(f"n_rows must be >= 2, got {n_rows}")
    if float(shape_jitter) <= 0.0:
        return np.ones((n_rows,), dtype=float)

    anchors_x = np.linspace(0.0, 1.0, int(n_anchors), dtype=float)
    anchors_y = rng.normal(0.0, float(shape_jitter), size=int(n_anchors))
    anchors_y[0] = 0.0
    anchors_y[-1] = 0.0
    query_x = np.linspace(0.0, 1.0, n_rows, dtype=float)
    envelope = 1.0 + np.interp(query_x, anchors_x, anchors_y)
    return np.clip(envelope, 0.1, None)


def _synthesize_ip_from_template(
    template: SourceIpShot,
    rng: np.random.Generator,
    *,
    amplitude_jitter: float,
    duration_jitter: float,
    shape_jitter: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Построить новую таблицу Ip, сохранив базовую форму и единицы шаблона."""
    rows = template.rows
    duration_scale = _sample_scale(rng, jitter=duration_jitter)
    amplitude_scale = _sample_scale(rng, jitter=amplitude_jitter)

    source_duration = template.duration_s
    target_duration = float(source_duration * duration_scale)
    query_phase = np.linspace(0.0, 1.0, rows, dtype=float)
    source_phase = template.time_s / float(source_duration)
    ip_base = np.interp(query_phase, source_phase, template.ip_a)
    envelope = _shape_envelope(rows, rng, shape_jitter=shape_jitter)
    ip_out = amplitude_scale * ip_base * envelope
    t_out = np.linspace(0.0, target_duration, rows, dtype=float)
    return t_out, ip_out, amplitude_scale, duration_scale


def _write_semicolon_csv(path: Path, arr: np.ndarray) -> None:
    """Записать массив в headerless CSV с разделителем `;`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(arr, dtype=float), delimiter=";", fmt="%.16g")


def _validate_args(args: argparse.Namespace) -> None:
    """Проверить аргументы CLI генератора."""
    if int(args.n_shots) <= 0:
        raise ValueError("--n-shots must be > 0")
    if int(args.shot_start) < 0:
        raise ValueError("--shot-start must be >= 0")

    _finite_nonnegative("--amplitude-jitter", args.amplitude_jitter)
    _finite_nonnegative("--duration-jitter", args.duration_jitter)
    _finite_nonnegative("--shape-jitter", args.shape_jitter)
    if str(args.initial_currents_prefix).strip() == "":
        raise ValueError("--initial-currents-prefix must be non-empty")


def main() -> int:
    """Прочитать шаблонные Ip-таблицы и записать набор синтетических вариантов."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic T15-like Ip reference tables by perturbing existing "
            "two-column semicolon-separated Ip tables."
        )
    )
    parser.add_argument("--source-ip-dir", required=True, help="Directory containing source t15md_*_ip.csv tables.")
    parser.add_argument("--out-root", required=True, help="Output root. Generated Ip tables are written into <out-root>/ip/.")
    parser.add_argument(
        "--source-initial-currents-dir",
        default="configs/initial_currents",
        help="Directory containing source initial-current TOML files.",
    )
    parser.add_argument(
        "--out-initial-currents-dir",
        default="configs/initial_currents",
        help="Directory where generated initial-current TOML files are written.",
    )
    parser.add_argument(
        "--initial-currents-prefix",
        default="T15MD_new_data",
        help="Filename prefix used for <prefix>_<shot>.toml initial-current files.",
    )
    parser.add_argument("--n-shots", type=int, default=8, help="Number of synthetic Ip tables to generate.")
    parser.add_argument("--shot-start", type=int, default=950001, help="First synthetic shot number used in filenames.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for reproducible synthetic tables.")
    parser.add_argument("--amplitude-jitter", type=float, default=0.05, help="Relative random scaling of the Ip magnitude.")
    parser.add_argument("--duration-jitter", type=float, default=0.05, help="Relative random scaling of the shot duration.")
    parser.add_argument("--shape-jitter", type=float, default=0.02, help="Low-frequency random deformation of the normalized Ip shape.")

    args = parser.parse_args()
    _validate_args(args)

    source_ip_dir = Path(args.source_ip_dir)
    source_initial_currents_dir = Path(args.source_initial_currents_dir)
    out_root = Path(args.out_root)
    out_ip_dir = out_root / "ip"
    out_initial_currents_dir = Path(args.out_initial_currents_dir)
    out_ip_dir.mkdir(parents=True, exist_ok=True)
    out_initial_currents_dir.mkdir(parents=True, exist_ok=True)

    source_shots = _discover_source_shots(source_ip_dir)
    rng = np.random.default_rng(int(args.seed))

    summaries: list[SyntheticIpSummary] = []
    for index in range(int(args.n_shots)):
        shot_id = str(int(args.shot_start) + index)
        template = source_shots[int(rng.integers(0, len(source_shots)))]
        t_out, ip_out, amplitude_scale, duration_scale = _synthesize_ip_from_template(
            template,
            rng,
            amplitude_jitter=float(args.amplitude_jitter),
            duration_jitter=float(args.duration_jitter),
            shape_jitter=float(args.shape_jitter),
        )

        ip_table = np.column_stack([t_out, ip_out])
        ip_path = out_ip_dir / f"t15md_{shot_id}_ip.csv"
        _write_semicolon_csv(ip_path, ip_table)
        initial_currents_path = _copy_initial_currents_toml(
            source_initial_currents_dir,
            out_initial_currents_dir,
            initial_currents_prefix=str(args.initial_currents_prefix),
            source_shot_id=template.shot_id,
            target_shot_id=shot_id,
        )

        summary = SyntheticIpSummary(
            shot_id=shot_id,
            template_shot_id=template.shot_id,
            rows=int(ip_table.shape[0]),
            duration_s=float(t_out[-1] - t_out[0]),
            peak_abs_ip=float(np.max(np.abs(ip_out))),
            amplitude_scale=float(amplitude_scale),
            duration_scale=float(duration_scale),
            ip_csv=str(ip_path),
            initial_currents_toml=str(initial_currents_path),
        )
        summaries.append(summary)

        print(
            f"Saved synthetic Ip shot {shot_id}: "
            f"template={template.shot_id}, rows={summary.rows}, "
            f"duration={summary.duration_s:.6f}s, max_abs_Ip={summary.peak_abs_ip:.6g}, "
            f"initial_currents={initial_currents_path}"
        )

    metadata = {
        "source_ip_dir": str(source_ip_dir),
        "source_initial_currents_dir": str(source_initial_currents_dir),
        "out_initial_currents_dir": str(out_initial_currents_dir),
        "initial_currents_prefix": str(args.initial_currents_prefix),
        "seed": int(args.seed),
        "amplitude_jitter": float(args.amplitude_jitter),
        "duration_jitter": float(args.duration_jitter),
        "shape_jitter": float(args.shape_jitter),
        "source_shots": [
            {
                "shot_id": shot.shot_id,
                "path": str(shot.path),
                "rows": shot.rows,
                "duration_s": shot.duration_s,
                "peak_abs_ip": shot.peak_abs_ip,
            }
            for shot in source_shots
        ],
        "generated_shots": [asdict(summary) for summary in summaries],
    }
    metadata_path = out_root / "synthetic_ip_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    print(f"ip_dir={out_ip_dir}")
    print(f"metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
