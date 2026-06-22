#!/usr/bin/env python3
"""Create smoothed LQR boundary-reference files from replay-derived references."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import savgol_filter


REPO_ROOT = Path(__file__).resolve().parents[1]


def _odd_window(value: int, n: int) -> int:
    """Return an odd window in [3, n] suitable for temporal filters."""
    out = max(3, int(value))
    if out % 2 == 0:
        out += 1
    if out > n:
        out = n if n % 2 == 1 else n - 1
    return max(3, out)


def _fill_nan_linear(y: np.ndarray) -> np.ndarray:
    """Fill NaNs by linear interpolation, clamped at the edges."""
    arr = np.asarray(y, dtype=float).reshape(-1).copy()
    x = np.arange(arr.size, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr
    if np.all(finite):
        return arr
    arr[~finite] = np.interp(x[~finite], x[finite], arr[finite])
    return arr


def _hampel_replace(y: np.ndarray, *, window: int, sigma: float) -> tuple[np.ndarray, int]:
    """Replace isolated temporal outliers with the local median."""
    arr = _fill_nan_linear(y)
    out = arr.copy()
    half = int(window) // 2
    replacements = 0
    for idx in range(arr.size):
        lo = max(0, idx - half)
        hi = min(arr.size, idx + half + 1)
        local = arr[lo:hi]
        med = float(np.median(local))
        mad = float(np.median(np.abs(local - med)))
        scale = 1.4826 * mad
        if scale <= 1.0e-12:
            continue
        if abs(float(arr[idx]) - med) > float(sigma) * scale:
            out[idx] = med
            replacements += 1
    return out, replacements


def _smooth_radii(
    radii: np.ndarray,
    *,
    hampel_window: int,
    hampel_sigma: float,
    savgol_window: int,
    savgol_polyorder: int,
) -> tuple[np.ndarray, int, float]:
    """Smooth a T x A radius table independently along time for each angle."""
    raw = np.asarray(radii, dtype=float)
    if raw.ndim != 2:
        raise ValueError(f"radii_true must be a 2-D array, got {raw.shape}")
    t_count, angle_count = raw.shape
    if t_count < 3 or angle_count < 1:
        raise ValueError(f"radii_true is too small to smooth: {raw.shape}")

    hampel_w = _odd_window(hampel_window, t_count)
    savgol_w = _odd_window(savgol_window, t_count)
    poly = min(int(savgol_polyorder), savgol_w - 1)
    if poly < 1:
        raise ValueError("savgol_polyorder must be >= 1 after window adjustment")

    smoothed = np.empty_like(raw)
    replacement_count = 0
    for angle_idx in range(angle_count):
        cleaned, replacements = _hampel_replace(
            raw[:, angle_idx],
            window=hampel_w,
            sigma=float(hampel_sigma),
        )
        replacement_count += replacements
        smoothed[:, angle_idx] = savgol_filter(cleaned, window_length=savgol_w, polyorder=poly, mode="interp")
    max_delta = float(np.nanmax(np.abs(smoothed - raw))) if raw.size else 0.0
    return smoothed, replacement_count, max_delta


def _smooth_one(
    reference_path: Path,
    *,
    out_dir: Path,
    hampel_window: int,
    hampel_sigma: float,
    savgol_window: int,
    savgol_polyorder: int,
) -> dict[str, object]:
    """Smooth one replay reference and write NPZ/JSON sidecars."""
    with np.load(reference_path, allow_pickle=False) as src:
        files = set(src.files)
        required = {"shot", "t", "angles_rad", "radii_true", "initial_radii_true"}
        missing = sorted(required - files)
        if missing:
            raise ValueError(f"{reference_path} missing required fields: {', '.join(missing)}")
        shot = str(int(np.asarray(src["shot"]).reshape(-1)[0]))
        raw_radii = np.asarray(src["radii_true"], dtype=float)
        smoothed, replacements, max_delta = _smooth_radii(
            raw_radii,
            hampel_window=int(hampel_window),
            hampel_sigma=float(hampel_sigma),
            savgol_window=int(savgol_window),
            savgol_polyorder=int(savgol_polyorder),
        )
        finite_rows = np.all(np.isfinite(smoothed), axis=1)
        if np.any(finite_rows):
            initial_smoothed = smoothed[int(np.flatnonzero(finite_rows)[0])].copy()
        else:
            initial_smoothed = np.full((smoothed.shape[1],), np.nan, dtype=float)

        out_npz = out_dir / f"lqr_boundary_reference_{shot}_smoothed.npz"
        out_json = out_dir / f"lqr_boundary_reference_{shot}_smoothed.json"
        payload = {key: np.asarray(src[key]) for key in src.files}
        payload["radii_true_raw"] = raw_radii
        payload["initial_radii_true_raw"] = np.asarray(src["initial_radii_true"], dtype=float)
        payload["radii_true"] = smoothed
        payload["initial_radii_true"] = initial_smoothed
        payload["smoothing_hampel_window"] = np.asarray([int(hampel_window)], dtype=int)
        payload["smoothing_hampel_sigma"] = np.asarray([float(hampel_sigma)], dtype=float)
        payload["smoothing_savgol_window"] = np.asarray([int(savgol_window)], dtype=int)
        payload["smoothing_savgol_polyorder"] = np.asarray([int(savgol_polyorder)], dtype=int)
        np.savez_compressed(out_npz, **payload)

    sidecar = {
        "schema": "t15md_lqr_boundary_reference_v1_smoothed",
        "shot": shot,
        "source_reference_npz": str(reference_path),
        "reference_npz": str(out_npz),
        "hampel_window": int(hampel_window),
        "hampel_sigma": float(hampel_sigma),
        "savgol_window": int(savgol_window),
        "savgol_polyorder": int(savgol_polyorder),
        "hampel_replacement_count": int(replacements),
        "max_abs_radius_change_m": max_delta,
        "usage": {
            "follow_smoothed_recorded_boundary": "Use radii_true[t, angle_index] with angles_rad.",
            "raw_recorded_boundary": "Use radii_true_raw if the unsmoothed replay contour is needed.",
        },
    }
    out_json.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "shot": shot,
        "source_reference_npz": str(reference_path),
        "smoothed_reference_npz": str(out_npz),
        "smoothed_reference_json": str(out_json),
        "hampel_replacement_count": int(replacements),
        "max_abs_radius_change_m": max_delta,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smooth replay-derived LQR boundary references.")
    parser.add_argument("--dataset-root", default="runs/t15md_limited_replay_dataset")
    parser.add_argument("--hampel-window", type=int, default=21)
    parser.add_argument("--hampel-sigma", type=float, default=3.0)
    parser.add_argument("--savgol-window", type=int, default=81)
    parser.add_argument("--savgol-polyorder", type=int, default=3)
    args = parser.parse_args(argv)

    root = (REPO_ROOT / args.dataset_root).resolve() if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)
    refs = sorted(root.glob("lqr_boundary_reference_*.npz"))
    refs = [p for p in refs if not p.name.endswith("_smoothed.npz")]
    if not refs:
        raise SystemExit(f"No raw LQR boundary reference NPZ files found under {root}")

    rows = [
        _smooth_one(
            p,
            out_dir=root,
            hampel_window=int(args.hampel_window),
            hampel_sigma=float(args.hampel_sigma),
            savgol_window=int(args.savgol_window),
            savgol_polyorder=int(args.savgol_polyorder),
        )
        for p in refs
    ]
    summary = root / "smoothed_boundary_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "shot",
                "source_reference_npz",
                "smoothed_reference_npz",
                "smoothed_reference_json",
                "hampel_replacement_count",
                "max_abs_radius_change_m",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary": str(summary), "count": len(rows)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
