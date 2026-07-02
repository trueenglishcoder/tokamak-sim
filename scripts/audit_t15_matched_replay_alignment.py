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
    data_root = Path(args.data_root)
    reference_root = Path(args.reference_root)
    run_root = Path(args.run_root)
    run_dir = Path(args.run_dir) if args.run_dir else _find_latest_run_dir(run_root, shot)

    ideal_coils = _read_numeric_table(data_root / "coils" / f"t15md_{shot}_coils.csv", delimiter=";")
    ref_coils = _read_numeric_table(reference_root / "coils" / f"t15md_{shot}_coils.csv", delimiter=";")
    ideal_ip = _read_numeric_table(data_root / "ip" / f"t15md_{shot}_ip.csv", delimiter=";")
    ref_ip = _read_numeric_table(reference_root / "ip" / f"t15md_{shot}_ip.csv", delimiter=";")

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
    print(f"timeseries: {timeseries_path}")
    print(f"rows: {len(ideal_coils)}")
    print(f"max_idealized_current_deviation_a: {current_diff:.6g}")
    print(f"first{n}_ref_delta_a: {_format(ref_first_delta)}")
    print(f"first{n}_ideal_delta_a: {_format(ideal_first_delta)}")
    print(f"max_run_table_current_diff_a_first{n}: {max_run_table_diff:.6g}")
    print("verdict: replay table, initial alignment, and run_timeseries first-window currents are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
