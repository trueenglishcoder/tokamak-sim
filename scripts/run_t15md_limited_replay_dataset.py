#!/usr/bin/env python3
"""Run exact limited T15MD current replays and export LQR boundary references."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter
from typing import Iterable
import tomllib

import numpy as np
import tomli_w


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tokamak_control.io.config_io import load_config
from tokamak_control.io.data_io import load_run


DEFAULT_SHOTS = ("3854", "3855", "3856", "3857", "3858", "3859", "3862", "3863", "3864")


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "t", "yes"}


def _count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _first_match(run_dir: Path, pattern: str) -> Path | None:
    matches = sorted(run_dir.glob(pattern))
    return matches[0] if matches else None


def _latest_subdir(parent: Path, *, after: set[Path] | None = None) -> Path | None:
    after = set() if after is None else after
    if not parent.exists():
        return None
    candidates = [p for p in parent.iterdir() if p.is_dir() and p.resolve() not in after]
    if not candidates:
        candidates = [p for p in parent.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_events(path: Path | None) -> tuple[list[dict[str, str]], Counter[str]]:
    if path is None or not path.exists():
        return [], Counter()
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    status_counts = Counter(str(row.get("boundary_status", "")) for row in rows if row.get("type", "step") == "step")
    return rows, status_counts


def _write_strict_limited_config(*, base_config: Path, out_path: Path, legacy_precision_index2: float) -> Path:
    with base_config.open("rb") as handle:
        data = tomllib.load(handle)
    boundary = data.setdefault("boundary", {})
    if not isinstance(boundary, dict):
        raise ValueError(f"{base_config} boundary section must be a TOML table")
    boundary.update(
        {
            "mode": "legacy_contour_limited",
            "base_mode": "legacy_contour_limited",
            "legacy_precision_index2": float(legacy_precision_index2),
            "track_level": False,
            "level_smoothing_alpha": 1.0,
            "level_search_span_fraction": 0.02,
            "continuity_weight_radii": 1.0,
            "continuity_weight_mean_radius": 0.3,
            "continuity_weight_center": 0.2,
            "continuity_weight_area": 0.2,
            "continuity_weight_level": 0.1,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        tomli_w.dump(data, handle)
    loaded = load_config(out_path)
    if loaded.boundary_mode != "legacy_contour_limited":
        raise RuntimeError(f"Strict replay config did not persist legacy_contour_limited: {loaded.boundary_mode!r}")
    if loaded.limiter_name is None:
        raise RuntimeError("Strict replay config has no limiter; limited replay dataset requires limiter geometry")
    return out_path


def _discover_data_shots(data_root: Path) -> list[str]:
    """Найти разряды по наличию пар ip/coils CSV."""
    ip_dir = data_root / "ip"
    coils_dir = data_root / "coils"
    if not ip_dir.is_dir() or not coils_dir.is_dir():
        raise FileNotFoundError(f"data root must contain ip/ and coils/: {data_root}")
    ip_shots = {path.name.removeprefix("t15md_").removesuffix("_ip.csv") for path in ip_dir.glob("t15md_*_ip.csv")}
    coil_shots = {path.name.removeprefix("t15md_").removesuffix("_coils.csv") for path in coils_dir.glob("t15md_*_coils.csv")}
    shots = sorted(ip_shots & coil_shots, key=lambda value: int(value))
    if not shots:
        raise FileNotFoundError(f"no paired replay shots found in {data_root}")
    return shots


def _validate_shot_files(shot: str, *, data_root: Path, initial_prefix: str) -> tuple[Path, Path, Path]:
    coils = data_root / "coils" / f"t15md_{shot}_coils.csv"
    ip = data_root / "ip" / f"t15md_{shot}_ip.csv"
    init = REPO_ROOT / "configs" / "initial_currents" / f"{initial_prefix}_{shot}.toml"
    missing = [str(p) for p in (coils, ip, init) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Shot {shot} is missing required files: {', '.join(missing)}")
    if _count_rows(coils) != _count_rows(ip):
        raise ValueError(
            f"Shot {shot} has mismatched row counts: coils={_count_rows(coils)} ip={_count_rows(ip)}"
        )
    return coils, ip, init


def _run_one_shot(
    *,
    shot: str,
    config: Path,
    data_root: Path,
    initial_prefix: str,
    dataset_root: Path,
    angles: int,
    frame_stride: int,
    frame_dpi: int,
    fps: int,
    dry_run: bool,
) -> Path | None:
    coils, ip, init = _validate_shot_files(shot, data_root=data_root, initial_prefix=initial_prefix)
    steps = _count_rows(ip)
    shot_parent = dataset_root / f"t15md_limited_replay_{shot}"
    before = {p.resolve() for p in shot_parent.iterdir() if p.is_dir()} if shot_parent.exists() else set()

    cmd = [
        sys.executable,
        "scripts/run_simulation_artifacts.py",
        "--config",
        str(config),
        "--initial-currents",
        str(init.relative_to(REPO_ROOT)),
        "--steps",
        str(steps),
        "--controller",
        "t15md_replay",
        "--controller-arg",
        f"replay_path={coils.relative_to(REPO_ROOT)}",
        "--angles",
        str(angles),
        "--scenario",
        "ip_table",
        "--scenario-arg",
        f"ip_csv={ip.relative_to(REPO_ROOT)}",
        "--out",
        str(shot_parent),
        "--compute-backend",
        "cpu",
        "--video",
        "--frame-stride",
        str(frame_stride),
        "--frame-dpi",
        str(frame_dpi),
        "--fps",
        str(fps),
        "--verbose",
        "--no-progress",
    ]

    print(f"===== shot {shot}: steps={steps} =====", flush=True)
    print(" ".join(cmd), flush=True)
    if dry_run:
        return None

    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(dataset_root / ".matplotlib"))
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Replay run failed for shot {shot} with exit code {result.returncode}")

    run_dir = _latest_subdir(shot_parent, after=before)
    if run_dir is None:
        raise RuntimeError(f"Replay run for shot {shot} finished but no run directory appeared under {shot_parent}")
    return run_dir


def _finite_row_mask(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        return np.zeros((0,), dtype=bool)
    return np.all(np.isfinite(arr), axis=1)


def _export_lqr_reference(*, shot: str, run_dir: Path, dataset_root: Path) -> dict[str, object]:
    npz_path = _first_match(run_dir, "run*.npz")
    events_path = _first_match(run_dir, "events*.csv")
    video_path = _first_match(run_dir, "video*.mp4") or _first_match(run_dir, "*.mp4")
    timeseries_path = _first_match(run_dir, "run_timeseries*.csv")
    manifest_path = _first_match(run_dir, "manifest*.json")

    if npz_path is None:
        raise FileNotFoundError(f"{run_dir} does not contain run*.npz")
    run = load_run(npz_path)

    radii_true = np.asarray(run.get("radii_true"), dtype=float)
    if radii_true.ndim != 2 or radii_true.shape[1] == 0:
        raise RuntimeError(f"{npz_path} has no usable radii_true array")
    n_angles = int(radii_true.shape[1])
    angles_rad = np.linspace(-np.pi, np.pi, n_angles, endpoint=False, dtype=float)

    finite_radii = _finite_row_mask(radii_true)
    if np.any(finite_radii):
        initial_radii_true = radii_true[int(np.flatnonzero(finite_radii)[0])].copy()
    else:
        initial_radii_true = np.full((n_angles,), np.nan, dtype=float)

    events, status_counts = _read_events(events_path)
    step_events = [row for row in events if row.get("type", "step") == "step"]
    boundary_found = np.asarray([_truthy(row.get("boundary_found", "")) for row in step_events], dtype=bool)
    if boundary_found.size == 0:
        boundary_found = finite_radii.astype(bool)
    boundary_found_fraction = float(np.mean(boundary_found)) if boundary_found.size else float("nan")
    missing_boundary_count = int(np.sum(~boundary_found)) if boundary_found.size else 0

    ref_npz = dataset_root / f"lqr_boundary_reference_{shot}.npz"
    ref_json = dataset_root / f"lqr_boundary_reference_{shot}.json"

    arrays: dict[str, object] = {
        "shot": np.asarray([int(shot)], dtype=int),
        "step": np.asarray(run["step"], dtype=int),
        "t": np.asarray(run["t"], dtype=float),
        "angles_rad": angles_rad,
        "radii_true": radii_true,
        "initial_radii_true": initial_radii_true,
        "boundary_found": boundary_found,
        "Ip": np.asarray(run["Ip"], dtype=float),
        "dIp_dt": np.asarray(run.get("dIp_dt"), dtype=float),
        "pfc_currents": np.asarray(run["pfc_currents"], dtype=float),
        "sol_currents": np.asarray(run["sol_currents"], dtype=float),
        "pfc_derivs": np.asarray(run["pfc_derivs"], dtype=float),
        "sol_derivs": np.asarray(run["sol_derivs"], dtype=float),
    }
    for key in (
        "radii_ref",
        "radii_meas",
        "boundary_poly_true",
        "boundary_poly_meas",
        "Ip_ref",
        "Ip_meas",
        "pfc_currents_cmd",
        "sol_currents_cmd",
        "pfc_currents_eff",
        "sol_currents_eff",
    ):
        if key in run:
            arrays[key] = np.asarray(run[key], dtype=float)
    np.savez_compressed(ref_npz, **arrays)

    meta = dict(run.get("meta", {}))
    schema = {
        "schema": "t15md_lqr_boundary_reference_v1",
        "shot": str(shot),
        "source_run_dir": str(run_dir),
        "source_run_npz": str(npz_path),
        "source_events_csv": "" if events_path is None else str(events_path),
        "source_timeseries_csv": "" if timeseries_path is None else str(timeseries_path),
        "source_manifest_json": "" if manifest_path is None else str(manifest_path),
        "video_path": "" if video_path is None else str(video_path),
        "reference_npz": str(ref_npz),
        "boundary_mode": meta.get("boundary", {}).get("mode"),
        "boundary_base_mode": meta.get("boundary", {}).get("base_mode"),
        "legacy_precision_index2": meta.get("boundary", {}).get("legacy_precision_index2"),
        "limiter": meta.get("limiter", {}).get("name"),
        "angles_count": n_angles,
        "boundary_found_fraction": boundary_found_fraction,
        "missing_boundary_count": missing_boundary_count,
        "boundary_status_counts": dict(status_counts),
        "usage": {
            "follow_recorded_boundary": "Use radii_true[t, angle_index] with angles_rad.",
            "hold_initial_boundary": "Use initial_radii_true with angles_rad.",
            "optional_polyline_reference": "Use boundary_poly_true[t, :, :] when a full contour is needed.",
        },
    }
    ref_json.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")

    ip = np.asarray(run["Ip"], dtype=float)
    ip_ref = np.asarray(run["Ip_ref"], dtype=float) if "Ip_ref" in run else np.full_like(ip, np.nan)
    ip_err = np.abs(ip - ip_ref)
    ip_err = ip_err[np.isfinite(ip_err)]

    return {
        "shot": str(shot),
        "status": "ok",
        "run_dir": str(run_dir),
        "row_count": int(np.asarray(run["t"]).shape[0]),
        "boundary_found_fraction": boundary_found_fraction,
        "missing_boundary_count": missing_boundary_count,
        "boundary_status_counts": json.dumps(dict(status_counts), sort_keys=True),
        "ip_error_mean_a": "" if ip_err.size == 0 else float(np.mean(ip_err)),
        "ip_error_max_a": "" if ip_err.size == 0 else float(np.max(ip_err)),
        "video_path": "" if video_path is None else str(video_path),
        "run_npz": str(npz_path),
        "lqr_reference_npz": str(ref_npz),
        "lqr_reference_json": str(ref_json),
    }


def _write_summary(dataset_root: Path, rows: Iterable[dict[str, object]]) -> Path:
    rows = list(rows)
    out = dataset_root / "batch_summary.csv"
    fieldnames = [
        "shot",
        "status",
        "run_dir",
        "row_count",
        "boundary_found_fraction",
        "missing_boundary_count",
        "boundary_status_counts",
        "ip_error_mean_a",
        "ip_error_max_a",
        "video_path",
        "run_npz",
        "lqr_reference_npz",
        "lqr_reference_json",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run exact T15MD limited current replays and export LQR boundary reference files."
    )
    parser.add_argument("--shots", nargs="+", default=list(DEFAULT_SHOTS), help="Shot ids to run, or 'auto' for every paired shot.")
    parser.add_argument("--config", default="configs/T15MD_new_data.toml", help="Base config to copy from.")
    parser.add_argument("--data-root", default="data/t15_data_new", help="Replay data root with ip/ and coils/ subdirectories.")
    parser.add_argument("--initial-prefix", default="T15MD_new_data", help="Prefix for configs/initial_currents/<prefix>_<shot>.toml")
    parser.add_argument("--strict-config-name", default=None, help="Filename for the generated strict replay config.")
    parser.add_argument("--out", default="runs/t15md_limited_replay_dataset", help="Dataset output root.")
    parser.add_argument("--angles", type=int, default=32, help="Boundary radii sample count to store.")
    parser.add_argument("--legacy-precision-index2", type=float, default=1.0e-3)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--frame-dpi", type=int, default=100)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Print commands and validate inputs without running.")
    args = parser.parse_args(argv)

    data_root = (REPO_ROOT / args.data_root).resolve() if not Path(args.data_root).is_absolute() else Path(args.data_root)
    shots = _discover_data_shots(data_root) if len(args.shots) == 1 and str(args.shots[0]).lower() == "auto" else [str(shot) for shot in args.shots]
    dataset_root = (REPO_ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    dataset_root.mkdir(parents=True, exist_ok=True)
    strict_config_name = args.strict_config_name or f"{Path(args.config).stem}_legacy_contour_limited_replay.toml"
    base_config = (REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = _write_strict_limited_config(
        base_config=base_config,
        out_path=dataset_root / strict_config_name,
        legacy_precision_index2=float(args.legacy_precision_index2),
    )

    for shot in shots:
        _validate_shot_files(str(shot), data_root=data_root, initial_prefix=str(args.initial_prefix))

    rows: list[dict[str, object]] = []
    for shot in shots:
        run_dir = _run_one_shot(
            shot=str(shot),
            config=config,
            data_root=data_root,
            initial_prefix=str(args.initial_prefix),
            dataset_root=dataset_root,
            angles=int(args.angles),
            frame_stride=int(args.frame_stride),
            frame_dpi=int(args.frame_dpi),
            fps=int(args.fps),
            dry_run=bool(args.dry_run),
        )
        if run_dir is None:
            continue
        rows.append(_export_lqr_reference(shot=str(shot), run_dir=run_dir, dataset_root=dataset_root))
        summary_path = _write_summary(dataset_root, rows)
        print(f"Wrote {summary_path}", flush=True)

    if args.dry_run:
        print(f"Dry run complete. Strict config would be: {config}")
        return 0

    summary_path = _write_summary(dataset_root, rows)
    print(json.dumps({"dataset_root": str(dataset_root), "summary": str(summary_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
