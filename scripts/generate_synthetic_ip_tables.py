"""Сгенерировать синтетические таблицы Ip по образцу существующих T15-подобных данных."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 local fallback
    import tomli as tomllib

import numpy as np


def _ensure_repo_root_on_path() -> None:
    """Добавить корень репозитория в путь импорта при прямом запуске скрипта."""
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from tokamak_control.config.ip_trajectories import (  # noqa: E402
    SyntheticIpConfig,
    discover_ip_templates,
    generate_ip_reference_from_template,
    write_semicolon_ip_table,
)


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
    initial_state_toml: str


def _finite_nonnegative(name: str, value: float) -> float:
    """Проверить, что аргумент конечен и неотрицателен."""
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return out


def _source_initial_state_path(
    source_initial_state_dir: Path,
    *,
    initial_state_prefix: str,
    shot_id: str,
) -> Path:
    """Построить путь к исходному TOML начального состояния для выбранного shot."""
    path = source_initial_state_dir / f"{initial_state_prefix}_{shot_id}.toml"
    if not path.exists():
        raise FileNotFoundError(f"Initial-state TOML not found for shot {shot_id}: {path}")
    return path


def _write_synthetic_initial_state_toml(
    source_initial_state_dir: Path,
    out_initial_state_dir: Path,
    *,
    initial_state_prefix: str,
    source_shot_id: str,
    target_shot_id: str,
    ip0: float,
) -> Path:
    """Скопировать токи из шаблона и записать Ip0 новой синтетической траектории."""
    source_path = _source_initial_state_path(
        source_initial_state_dir,
        initial_state_prefix=initial_state_prefix,
        shot_id=source_shot_id,
    )
    raw = tomllib.loads(source_path.read_text(encoding="utf-8"))
    pfc = raw["coils"]["pfc"]["currents"]
    sol = raw["coils"]["sol"]["currents"]
    out_path = out_initial_state_dir / f"{initial_state_prefix}_{target_shot_id}.toml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        (
            "version = 1\n\n"
            "[plasma]\n"
            f"Ip0 = {float(ip0):.17g}\n\n"
            "[coils.pfc]\n"
            f"currents = {_array(pfc)}\n\n"
            "[coils.sol]\n"
            f"currents = {_array(sol)}\n"
        ),
        encoding="utf-8",
    )
    return out_path


def _array(values: object) -> str:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return "[\n" + "".join(f"    {float(v):.17g},\n" for v in arr) + "]"


def _validate_args(args: argparse.Namespace) -> None:
    """Проверить аргументы CLI генератора."""
    if int(args.n_shots) <= 0:
        raise ValueError("--n-shots must be > 0")
    if int(args.shot_start) < 0:
        raise ValueError("--shot-start must be >= 0")

    _finite_nonnegative("--amplitude-jitter", args.amplitude_jitter)
    _finite_nonnegative("--duration-jitter", args.duration_jitter)
    _finite_nonnegative("--shape-jitter", args.shape_jitter)
    if str(args.initial_state_prefix).strip() == "":
        raise ValueError("--initial-state-prefix must be non-empty")


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
        "--source-initial-state-dir",
        default="configs/initial_states",
        help="Directory containing source initial-state TOML files.",
    )
    parser.add_argument(
        "--out-initial-state-dir",
        default="configs/initial_states",
        help="Directory where generated initial-state TOML files are written.",
    )
    parser.add_argument(
        "--initial-state-prefix",
        default="T15MD_new_data",
        help="Filename prefix used for <prefix>_<shot>.toml initial-state files.",
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
    source_initial_state_dir = Path(args.source_initial_state_dir)
    out_root = Path(args.out_root)
    out_ip_dir = out_root / "ip"
    out_initial_state_dir = Path(args.out_initial_state_dir)
    out_ip_dir.mkdir(parents=True, exist_ok=True)
    out_initial_state_dir.mkdir(parents=True, exist_ok=True)

    templates = discover_ip_templates(source_ip_dir)
    rng = np.random.default_rng(int(args.seed))
    config = SyntheticIpConfig(
        amplitude_jitter=float(args.amplitude_jitter),
        duration_jitter=float(args.duration_jitter),
        shape_jitter=float(args.shape_jitter),
    )

    summaries: list[SyntheticIpSummary] = []
    for index in range(int(args.n_shots)):
        shot_id = str(int(args.shot_start) + index)
        template = templates[int(rng.integers(0, len(templates)))]
        trajectory = generate_ip_reference_from_template(template, rng=rng, config=config)

        ip_path = out_ip_dir / f"t15md_{shot_id}_ip.csv"
        write_semicolon_ip_table(ip_path, trajectory)
        initial_state_path = _write_synthetic_initial_state_toml(
            source_initial_state_dir,
            out_initial_state_dir,
            initial_state_prefix=str(args.initial_state_prefix),
            source_shot_id=template.shot_id,
            target_shot_id=shot_id,
            ip0=float(trajectory.ip_a[0]),
        )

        summary = SyntheticIpSummary(
            shot_id=shot_id,
            template_shot_id=template.shot_id,
            rows=trajectory.rows,
            duration_s=trajectory.duration_s,
            peak_abs_ip=trajectory.peak_abs_ip,
            amplitude_scale=trajectory.amplitude_scale,
            duration_scale=trajectory.duration_scale,
            ip_csv=str(ip_path),
            initial_state_toml=str(initial_state_path),
        )
        summaries.append(summary)

        print(
            f"Saved synthetic Ip shot {shot_id}: "
            f"template={template.shot_id}, rows={summary.rows}, "
            f"duration={summary.duration_s:.6f}s, max_abs_Ip={summary.peak_abs_ip:.6g}, "
            f"initial_state={initial_state_path}"
        )

    metadata = {
        "source_ip_dir": str(source_ip_dir),
        "source_initial_state_dir": str(source_initial_state_dir),
        "out_initial_state_dir": str(out_initial_state_dir),
        "initial_state_prefix": str(args.initial_state_prefix),
        "seed": int(args.seed),
        "amplitude_jitter": float(args.amplitude_jitter),
        "duration_jitter": float(args.duration_jitter),
        "shape_jitter": float(args.shape_jitter),
        "source_shots": [
            {
                "shot_id": template.shot_id,
                "path": "" if template.path is None else str(template.path),
                "rows": template.rows,
                "duration_s": template.duration_s,
                "peak_abs_ip": template.peak_abs_ip,
            }
            for template in templates
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
