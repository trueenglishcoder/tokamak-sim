#!/usr/bin/env python3
"""Create matched idealized T15 coil-current replay tables.

The canonical idealized set intentionally uses the same five trim50 shots as the
working replay-window RL pipeline.  It differs from the real trim50 data only in
the coil-current table: Ip is copied byte-for-byte and coil currents are replaced
by low-noise traces on the same time grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MATCHED_TRIM50_SHOTS = ("3856", "3857", "3858", "3863", "3864")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _discover_shots(root: Path) -> list[str]:
    ip_dir = root / "ip"
    coil_dir = root / "coils"
    shots: list[str] = []
    for ip_path in sorted(ip_dir.glob("t15md_*_ip.csv")):
        shot = ip_path.name.removeprefix("t15md_").removesuffix("_ip.csv")
        if (coil_dir / f"t15md_{shot}_coils.csv").exists():
            shots.append(shot)
    if not shots:
        raise FileNotFoundError(f"No paired t15md_*_ip.csv / t15md_*_coils.csv files found under {root}")
    return shots


def _load_table(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=";", dtype=float)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two rows and two columns")
    return data


def _format_table(path: Path, table: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, table, delimiter=";", fmt="%.12g")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _triangle_kernel(width: int) -> np.ndarray:
    if width <= 0 or width % 2 != 1:
        raise ValueError("smooth_window_steps must be a positive odd integer")
    half = width // 2
    up = np.arange(1, half + 2, dtype=float)
    kernel = np.concatenate([up, up[-2::-1]])
    return kernel / np.sum(kernel)


def _smooth_columns(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return np.array(values, copy=True)
    kernel = _triangle_kernel(width)
    pad = width // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(values, dtype=float)
    for col in range(values.shape[1]):
        out[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
    return out


def _idealize_currents(
    time_s: np.ndarray,
    currents: np.ndarray,
    *,
    method: str,
    knot_step_s: float,
    smooth_window_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "smooth_jdot":
        dt = np.diff(time_s)
        if np.any(dt <= 0.0):
            raise ValueError("time column must be strictly increasing")
        jdot = np.diff(currents, axis=0) / dt[:, None]
        smooth_jdot = _smooth_columns(jdot, int(smooth_window_steps))
        increments = smooth_jdot * dt[:, None]
        ideal = np.empty_like(currents, dtype=float)
        ideal[0] = currents[0]
        ideal[1:] = currents[0] + np.cumsum(increments, axis=0)

        # Preserve the final replay current exactly without reintroducing local jumps.
        drift = currents[-1] - ideal[-1]
        ideal += np.linspace(0.0, 1.0, currents.shape[0], dtype=float)[:, None] * drift[None, :]
        return ideal, np.arange(currents.shape[0], dtype=float)

    if method != "piecewise_linear":
        raise ValueError("method must be smooth_jdot or piecewise_linear")

    if knot_step_s <= 0.0:
        raise ValueError("knot_step_s must be positive")

    t0 = float(time_s[0])
    t1 = float(time_s[-1])
    knot_times = np.arange(t0, t1 + 0.5 * knot_step_s, knot_step_s, dtype=float)
    if knot_times.size == 0 or not np.isclose(knot_times[0], t0):
        knot_times = np.concatenate([[t0], knot_times])
    if knot_times[-1] < t1 or not np.isclose(knot_times[-1], t1):
        knot_times = np.concatenate([knot_times, [t1]])
    knot_times = np.unique(np.clip(knot_times, t0, t1))

    ideal = np.empty_like(currents)
    for col in range(currents.shape[1]):
        knot_values = np.interp(knot_times, time_s, currents[:, col])
        ideal[:, col] = np.interp(time_s, knot_times, knot_values)
    return ideal, knot_times


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def _jdot_rms(time_s: np.ndarray, currents: np.ndarray) -> float:
    dt = np.diff(time_s)
    if np.any(dt <= 0.0):
        raise ValueError("time column must be strictly increasing")
    return _rms(np.diff(currents, axis=0) / dt[:, None])


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "shot",
        "samples",
        "columns",
        "dt_median_s",
        "method",
        "knot_step_s",
        "smooth_window_steps",
        "knots",
        "max_abs_current_error_a",
        "rms_current_error_a",
        "orig_jdot_rms_aps",
        "ideal_jdot_rms_aps",
        "ip_path",
        "coil_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="data/t15_data_new_trim50")
    parser.add_argument("--output-root", default="data/t15_data_new_trim50_idealized_matched")
    parser.add_argument(
        "--shots",
        nargs="+",
        default=list(MATCHED_TRIM50_SHOTS),
        help="Shot ids, or 'auto' for every paired shot. Defaults to the matched working RL shot set.",
    )
    parser.add_argument(
        "--method",
        choices=("smooth_jdot", "piecewise_linear"),
        default="smooth_jdot",
        help="Canonical mode smooths current derivatives and reintegrates currents.",
    )
    parser.add_argument(
        "--knot-step-s",
        type=float,
        default=0.05,
        help="Spacing of piecewise-linear current knots when --method=piecewise_linear.",
    )
    parser.add_argument(
        "--smooth-window-steps",
        type=int,
        default=21,
        help="Odd triangular smoothing window for Jdot when --method=smooth_jdot.",
    )
    parser.add_argument("--summary-name", default="idealized_coil_summary.csv")
    args = parser.parse_args(argv)

    input_root = _resolve(args.input_root)
    output_root = _resolve(args.output_root)
    shots = _discover_shots(input_root) if len(args.shots) == 1 and args.shots[0].lower() == "auto" else [str(s) for s in args.shots]

    rows: list[dict[str, object]] = []
    for shot in shots:
        ip_in = input_root / "ip" / f"t15md_{shot}_ip.csv"
        coil_in = input_root / "coils" / f"t15md_{shot}_coils.csv"
        ip_out = output_root / "ip" / ip_in.name
        coil_out = output_root / "coils" / coil_in.name

        ip = _load_table(ip_in)
        coils = _load_table(coil_in)
        if ip.shape[0] != coils.shape[0]:
            raise ValueError(f"Shot {shot}: Ip rows ({ip.shape[0]}) != coil rows ({coils.shape[0]})")
        if not np.allclose(ip[:, 0], coils[:, 0], rtol=0.0, atol=1.0e-10):
            raise ValueError(f"Shot {shot}: Ip and coil time columns differ")

        time_s = coils[:, 0]
        currents = coils[:, 1:]
        ideal_currents, knot_times = _idealize_currents(
            time_s,
            currents,
            method=str(args.method),
            knot_step_s=float(args.knot_step_s),
            smooth_window_steps=int(args.smooth_window_steps),
        )
        ideal_table = np.column_stack([time_s, ideal_currents])

        ip_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ip_in, ip_out)
        _format_table(coil_out, ideal_table)

        diff = ideal_currents - currents
        rows.append(
            {
                "shot": shot,
                "samples": int(coils.shape[0]),
                "columns": int(coils.shape[1]),
                "dt_median_s": float(np.median(np.diff(time_s))),
                "method": str(args.method),
                "knot_step_s": float(args.knot_step_s),
                "smooth_window_steps": int(args.smooth_window_steps),
                "knots": int(knot_times.size),
                "max_abs_current_error_a": float(np.max(np.abs(diff))),
                "rms_current_error_a": _rms(diff),
                "orig_jdot_rms_aps": _jdot_rms(time_s, currents),
                "ideal_jdot_rms_aps": _jdot_rms(time_s, ideal_currents),
                "ip_path": _display_path(ip_out),
                "coil_path": _display_path(coil_out),
            }
        )

    summary_path = output_root / args.summary_name
    _write_summary(summary_path, rows)
    print(json.dumps({"output_root": str(output_root), "shots": shots, "summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
