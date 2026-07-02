#!/usr/bin/env python3
"""Audit that a T15 replay run really used the intended trim50 replay table.

This script intentionally uses only the Python standard library so it can run on
the login node.  It checks three things:

1. The idealized table is aligned with the trim50 source table.
2. The idealized table does not have a dead/flat prefix when trim50 moves.
3. The generated run_timeseries currents match the replay table row-for-row.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


COIL_TABLE_COLUMNS = (
    "sol_current_0",
    "sol_current_1",
    "sol_current_2",
    "pfc_current_0",
    "pfc_current_1",
    "pfc_current_2",
    "pfc_current_3",
    "pfc_current_4",
    "pfc_current_5",
)


def _read_numeric_table(path: Path, *, delimiter: str) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if not row:
                continue
            rows.append([float(value) for value in row])
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _read_timeseries(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{key: float(value) for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_path(value: object, *, repo_root: Path) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return repo_root / path


def _read_manifest(run_dir: Path) -> tuple[Path, dict[str, object]]:
    path = _find_one(run_dir, "manifest*.json")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return path, data


def _nested(mapping: dict[str, object], *keys: str) -> object | None:
    current: object = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _manifest_file_hash(manifest: dict[str, object], key: str) -> str | None:
    node = _nested(manifest, "input_files", key)
    if not isinstance(node, dict):
        return None
    value = node.get("sha256")
    return None if value is None else str(value)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _max_abs_delta(rows: list[list[float]], *, first: int) -> list[float]:
    if len(rows) < 2:
        raise ValueError("need at least two rows to compute deltas")
    n = min(int(first), len(rows) - 1)
    out = [0.0 for _ in rows[0][1:]]
    for i in range(n):
        prev = rows[i][1:]
        curr = rows[i + 1][1:]
        for j, (a, b) in enumerate(zip(prev, curr, strict=True)):
            out[j] = max(out[j], abs(b - a))
    return out


def _max_abs_table_diff(a: list[list[float]], b: list[list[float]]) -> float:
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError(f"table shape mismatch: {len(a)}x{len(a[0])} vs {len(b)}x{len(b[0])}")
    max_diff = 0.0
    for ra, rb in zip(a, b, strict=True):
        for va, vb in zip(ra, rb, strict=True):
            max_diff = max(max_diff, abs(va - vb))
    return max_diff


def _max_abs_current_diff(a: list[list[float]], b: list[list[float]]) -> float:
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError(f"table shape mismatch: {len(a)}x{len(a[0])} vs {len(b)}x{len(b[0])}")
    max_diff = 0.0
    for ra, rb in zip(a, b, strict=True):
        for va, vb in zip(ra[1:], rb[1:], strict=True):
            max_diff = max(max_diff, abs(va - vb))
    return max_diff


def _find_latest_run_dir(run_root: Path, shot: str) -> Path:
    parent = run_root / f"t15md_limited_replay_{shot}"
    if not parent.is_dir():
        raise FileNotFoundError(f"missing replay shot directory: {parent}")
    candidates = [path for path in parent.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no run subdirectories under {parent}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _find_one(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no {pattern!r} under {path}")
    if len(matches) > 1:
        return max(matches, key=lambda p: p.stat().st_mtime)
    return matches[0]


def _format(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{float(value):.6g}" for value in values) + "]"


def _format_row(row: list[float], *, n: int = 10) -> str:
    return "[" + ", ".join(f"{float(value):.12g}" for value in row[:n]) + ("]" if len(row) <= n else ", ...]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True)
    parser.add_argument("--data-root", default="data/t15_data_new_trim50_idealized_matched")
    parser.add_argument("--reference-root", default="data/t15_data_new_trim50")
    parser.add_argument("--run-root", default="runs/t15md_limited_replay_dataset_trim50_idealized_matched_gpu_plain_1e6")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--first-steps", type=int, default=50)
    parser.add_argument("--max-current-deviation-a", type=float, default=250.0)
    parser.add_argument("--flat-ratio-threshold", type=float, default=0.2)
    parser.add_argument("--run-table-tolerance-a", type=float, default=1.0e-3)
    args = parser.parse_args(argv)

    shot = str(args.shot)
    repo_root = _repo_root()
    data_root = Path(args.data_root)
    reference_root = Path(args.reference_root)
    run_root = Path(args.run_root)
    if not data_root.is_absolute():
        data_root = repo_root / data_root
    if not reference_root.is_absolute():
        reference_root = repo_root / reference_root
    if not run_root.is_absolute():
        run_root = repo_root / run_root
    run_dir = Path(args.run_dir) if args.run_dir else _find_latest_run_dir(run_root, shot)
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir

    ideal_coils_path = data_root / "coils" / f"t15md_{shot}_coils.csv"
    ref_coils_path = reference_root / "coils" / f"t15md_{shot}_coils.csv"
    ideal_ip_path = data_root / "ip" / f"t15md_{shot}_ip.csv"
    ref_ip_path = reference_root / "ip" / f"t15md_{shot}_ip.csv"
    ideal_coils = _read_numeric_table(ideal_coils_path, delimiter=";")
    ref_coils = _read_numeric_table(ref_coils_path, delimiter=";")
    ideal_ip = _read_numeric_table(ideal_ip_path, delimiter=";")
    ref_ip = _read_numeric_table(ref_ip_path, delimiter=";")

    if len(ideal_coils) != len(ref_coils) or len(ideal_coils[0]) != len(ref_coils[0]):
        raise SystemExit(f"coil table shape mismatch: ideal={len(ideal_coils)} ref={len(ref_coils)}")
    if len(ideal_ip) != len(ref_ip) or len(ideal_ip[0]) != len(ref_ip[0]):
        raise SystemExit(f"Ip table shape mismatch: ideal={len(ideal_ip)} ref={len(ref_ip)}")

    ip_diff = _max_abs_table_diff(ideal_ip, ref_ip)
    current_diff = _max_abs_current_diff(ideal_coils, ref_coils)
    if ip_diff > 1.0e-9:
        raise SystemExit(f"idealized Ip is not an exact trim50 copy: max diff {ip_diff:.6g}")
    if current_diff > float(args.max_current_deviation_a) + 1.0e-6:
        raise SystemExit(
            f"idealized current deviation {current_diff:.6g} A exceeds "
            f"{float(args.max_current_deviation_a):.6g} A"
        )

    ref_first_delta = _max_abs_delta(ref_coils, first=int(args.first_steps))
    ideal_first_delta = _max_abs_delta(ideal_coils, first=int(args.first_steps))
    flat_cols = [
        idx
        for idx, (ref, ideal) in enumerate(zip(ref_first_delta, ideal_first_delta, strict=True))
        if ref > 1.0 and ideal < float(args.flat_ratio_threshold) * ref
    ]
    if flat_cols:
        raise SystemExit(
            f"idealized replay has a dead prefix in columns {flat_cols}; "
            f"ref_first_delta={_format(ref_first_delta)} ideal_first_delta={_format(ideal_first_delta)}"
        )

    manifest_path, manifest = _read_manifest(run_dir)
    replay_path = _resolve_manifest_path(_nested(manifest, "controller", "params", "replay_path"), repo_root=repo_root)
    ip_csv_path = _resolve_manifest_path(_nested(manifest, "scenario", "params", "ip_csv"), repo_root=repo_root)
    initial_state_path = _resolve_manifest_path(manifest.get("initial_state_source"), repo_root=repo_root)
    config_path = _resolve_manifest_path(manifest.get("config_source"), repo_root=repo_root)

    if replay_path is None:
        raise SystemExit(f"{manifest_path}: missing controller.params.replay_path")
    if ip_csv_path is None:
        raise SystemExit(f"{manifest_path}: missing scenario.params.ip_csv")
    if replay_path.resolve() != ideal_coils_path.resolve():
        raise SystemExit(
            "manifest replay_path does not point at the expected idealized coil table:\n"
            f"  manifest: {replay_path}\n"
            f"  expected: {ideal_coils_path}"
        )
    if ip_csv_path.resolve() != ideal_ip_path.resolve():
        raise SystemExit(
            "manifest ip_csv does not point at the expected idealized Ip table:\n"
            f"  manifest: {ip_csv_path}\n"
            f"  expected: {ideal_ip_path}"
        )

    if _sha256(replay_path) != _sha256(ideal_coils_path):
        raise SystemExit(f"manifest replay_path hash differs from expected path hash: {replay_path}")
    if _sha256(ip_csv_path) != _sha256(ideal_ip_path):
        raise SystemExit(f"manifest ip_csv hash differs from expected path hash: {ip_csv_path}")
    manifest_replay_hash = _manifest_file_hash(manifest, "controller.replay_path")
    manifest_ip_hash = _manifest_file_hash(manifest, "scenario.ip_csv")
    current_replay_hash = _sha256(ideal_coils_path)
    current_ip_hash = _sha256(ideal_ip_path)
    if manifest_replay_hash is not None and manifest_replay_hash != current_replay_hash:
        raise SystemExit(
            "run manifest replay_path hash does not match the current idealized table. "
            "This run was made with stale/different CSV contents:\n"
            f"  manifest sha256: {manifest_replay_hash}\n"
            f"  current  sha256: {current_replay_hash}\n"
            f"  path: {ideal_coils_path}"
        )
    if manifest_ip_hash is not None and manifest_ip_hash != current_ip_hash:
        raise SystemExit(
            "run manifest ip_csv hash does not match the current idealized Ip table. "
            "This run was made with stale/different CSV contents:\n"
            f"  manifest sha256: {manifest_ip_hash}\n"
            f"  current  sha256: {current_ip_hash}\n"
            f"  path: {ideal_ip_path}"
        )

    timeseries_path = _find_one(run_dir, "run_timeseries*.csv")
    ts = _read_timeseries(timeseries_path)
    n = min(int(args.first_steps), len(ts), len(ideal_coils) - 1)
    max_run_table_diff = 0.0
    worst: tuple[int, str, float, float] | None = None
    for i in range(n):
        row = ts[i]
        table_row = ideal_coils[i + 1]
        for col_idx, name in enumerate(COIL_TABLE_COLUMNS, start=1):
            got = float(row[name])
            expected = float(table_row[col_idx])
            diff = abs(got - expected)
            if diff > max_run_table_diff:
                max_run_table_diff = diff
                worst = (i + 1, name, got, expected)

    if max_run_table_diff > float(args.run_table_tolerance_a):
        assert worst is not None
        step, name, got, expected = worst
        raise SystemExit(
            f"run_timeseries does not match replay table; worst step={step} column={name} "
            f"run={got:.12g} table={expected:.12g} diff={max_run_table_diff:.6g} A"
        )

    print(f"shot: {shot}")
    print(f"run_dir: {run_dir}")
    print(f"manifest: {manifest_path}")
    print(f"manifest_config_source: {config_path}")
    print(f"manifest_initial_state_source: {initial_state_path}")
    print(f"manifest_replay_path: {replay_path}")
    print(f"manifest_ip_csv: {ip_csv_path}")
    print(f"timeseries: {timeseries_path}")
    print(f"rows: {len(ideal_coils)}")
    print(f"idealized_coils_sha256: {_sha256(ideal_coils_path)}")
    print(f"idealized_ip_sha256: {_sha256(ideal_ip_path)}")
    print(f"manifest_replay_sha256: {manifest_replay_hash or '<not recorded by this old run>'}")
    print(f"manifest_ip_sha256: {manifest_ip_hash or '<not recorded by this old run>'}")
    print(f"trim50_coils_sha256: {_sha256(ref_coils_path)}")
    print(f"trim50_ip_sha256: {_sha256(ref_ip_path)}")
    print(f"idealized_coils_row0: {_format_row(ideal_coils[0])}")
    print(f"idealized_coils_row1: {_format_row(ideal_coils[1])}")
    print(f"trim50_coils_row0: {_format_row(ref_coils[0])}")
    print(f"trim50_coils_row1: {_format_row(ref_coils[1])}")
    print(f"max_idealized_current_deviation_a: {current_diff:.6g}")
    print(f"first{n}_ref_delta_a: {_format(ref_first_delta)}")
    print(f"first{n}_ideal_delta_a: {_format(ideal_first_delta)}")
    print(f"max_run_table_current_diff_a_first{n}: {max_run_table_diff:.6g}")
    print("verdict: replay table, initial alignment, and run_timeseries first-window currents are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
