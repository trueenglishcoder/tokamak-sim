from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _latest_run_dir(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    runs = [p for p in parent.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "t", "yes"}


def _first_match(run_dir: Path, pattern: str) -> Path | None:
    matches = sorted(run_dir.glob(pattern))
    return matches[0] if matches else None


def _summarize_shot(runs_root: Path, shot: str) -> dict[str, object]:
    parent = runs_root / f"t15md_limited_replay_{shot}"
    run_dir = _latest_run_dir(parent)
    if run_dir is None:
        return {
            "shot": shot,
            "status": "missing_run_dir",
            "run_dir": "",
            "row_count": 0,
            "boundary_found_fraction": "",
            "missing_boundary_count": "",
            "ip_error_mean_a": "",
            "ip_error_max_a": "",
            "ip_error_final_a": "",
            "video_path": "",
        }

    ts_path = _first_match(run_dir, "run_timeseries*.csv")
    ev_path = _first_match(run_dir, "events*.csv")
    video_path = _first_match(run_dir, "video*.mp4") or _first_match(run_dir, "*.mp4")
    if ts_path is None or ev_path is None:
        return {
            "shot": shot,
            "status": "missing_artifacts",
            "run_dir": str(run_dir),
            "row_count": 0,
            "boundary_found_fraction": "",
            "missing_boundary_count": "",
            "ip_error_mean_a": "",
            "ip_error_max_a": "",
            "ip_error_final_a": "",
            "video_path": "" if video_path is None else str(video_path),
        }

    ts_rows = _read_csv_rows(ts_path)
    ev_rows = _read_csv_rows(ev_path)
    ip_errors = [abs(_as_float(row, "Ip") - _as_float(row, "Ip_ref")) for row in ts_rows]
    ip_errors = [x for x in ip_errors if x == x]
    boundary_values = [_truthy(row.get("boundary_found", "")) for row in ev_rows]
    boundary_found_count = sum(1 for x in boundary_values if x)
    boundary_total = len(boundary_values)

    return {
        "shot": shot,
        "status": "ok",
        "run_dir": str(run_dir),
        "row_count": len(ts_rows),
        "boundary_found_fraction": "" if boundary_total == 0 else boundary_found_count / boundary_total,
        "missing_boundary_count": "" if boundary_total == 0 else boundary_total - boundary_found_count,
        "ip_error_mean_a": "" if not ip_errors else sum(ip_errors) / len(ip_errors),
        "ip_error_max_a": "" if not ip_errors else max(ip_errors),
        "ip_error_final_a": "" if not ip_errors else ip_errors[-1],
        "video_path": "" if video_path is None else str(video_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize limited T15 replay runs.")
    parser.add_argument("--runs-root", default="runs", help="Root folder containing t15md_limited_replay_* outputs.")
    parser.add_argument("--out", default="runs/t15md_limited_replay_summary.csv", help="Summary CSV path.")
    parser.add_argument("--shots", nargs="+", required=True, help="Shot ids to summarize.")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    rows = [_summarize_shot(runs_root, str(shot)) for shot in args.shots]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "shot",
        "status",
        "run_dir",
        "row_count",
        "boundary_found_fraction",
        "missing_boundary_count",
        "ip_error_mean_a",
        "ip_error_max_a",
        "ip_error_final_a",
        "video_path",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
