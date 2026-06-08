#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.geometry.boundary import BoundaryNotFoundError, find_plasma_boundary_with_status
from tokamak_control.geometry.boundary_equivalence import BoundaryEquivalenceTolerance, compare_boundary_results, write_boundary_equivalence_report
from tokamak_control.io.data_io import load_run


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check CPU/GPU boundary-finder equivalence on stored run psi snapshots.")
    ap.add_argument("--run-npz", required=True, help="Path to a tokamak-sim run*.npz artifact.")
    ap.add_argument("--out", required=True, help="Output JSON report path.")
    ap.add_argument("--gpu-device", default="cuda:0", help="CUDA device for GPU boundary mode.")
    ap.add_argument("--max-cases", type=int, default=0, help="Maximum number of psi snapshots to check. 0 checks all available snapshots.")
    ap.add_argument("--angle-count", type=int, default=128, help="Ray count used for shape equivalence.")
    ap.add_argument("--mean-radii-abs", type=float, default=5.0e-3, help="Allowed mean absolute radii difference in meters.")
    ap.add_argument("--max-radii-abs", type=float, default=1.5e-2, help="Allowed maximum absolute radii difference in meters.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    run = load_run(Path(args.run_npz))
    meta = run["meta"]
    grid = _grid_from_meta(meta)
    center = (float(meta.get("center", {}).get("R0", grid.r.center)), float(meta.get("center", {}).get("Z0", grid.z.center)))
    boundary_mode = str(meta.get("boundary", {}).get("mode", "limited"))
    limiter_shape = meta.get("limiter", {}).get("shape")
    limiter = None if limiter_shape is None else np.asarray(limiter_shape, dtype=float)
    psis = _psi_cases(run)
    if int(args.max_cases) > 0:
        psis = psis[: int(args.max_cases)]

    tolerance = BoundaryEquivalenceTolerance(
        angle_count=int(args.angle_count),
        mean_radii_abs=float(args.mean_radii_abs),
        max_radii_abs=float(args.max_radii_abs),
    )
    reports = []
    for psi in psis:
        cpu_poly, cpu_level, cpu_status = _find_one(psi, grid, center, limiter, boundary_mode, "cpu", "cuda:0")
        gpu_poly, gpu_level, gpu_status = _find_one(psi, grid, center, limiter, boundary_mode, "gpu", str(args.gpu_device))
        reports.append(
            compare_boundary_results(
                cpu_poly=cpu_poly,
                cpu_level=cpu_level,
                cpu_status=cpu_status,
                gpu_poly=gpu_poly,
                gpu_level=gpu_level,
                gpu_status=gpu_status,
                center=center,
                tolerance=tolerance,
            )
        )

    out = write_boundary_equivalence_report(Path(args.out), reports)
    ok = sum(1 for report in reports if report.equivalent)
    print(f"{ok}/{len(reports)} boundary cases equivalent")
    print(str(out))
    return 0 if ok == len(reports) else 1


def _find_one(psi: np.ndarray, grid: Grid2D, center: tuple[float, float], limiter: np.ndarray | None, boundary_mode: str, backend: str, gpu_device: str):
    try:
        poly, level, status = find_plasma_boundary_with_status(
            np.asarray(psi, dtype=float),
            grid,
            center,
            n_levels=80 if limiter is not None else 10,
            limiter_shape=limiter,
            boundary_mode=boundary_mode,  # type: ignore[arg-type]
            compute_backend=backend,
            gpu_device=gpu_device,
        )
        return poly, float(level), str(status)
    except BoundaryNotFoundError:
        return None, None, "not_found"


def _grid_from_meta(meta: dict) -> Grid2D:
    raw = meta.get("grid")
    if not isinstance(raw, dict):
        raise ValueError("run metadata has no grid table")
    return Grid2D(
        r=Grid1D(start=float(raw["r_start"]), step=float(raw["r_step"]), size=int(raw["r_size"]), center=float(raw["r_center"])),
        z=Grid1D(start=float(raw["z_start"]), step=float(raw["z_step"]), size=int(raw["z_size"]), center=float(raw["z_center"])),
    )


def _psi_cases(run: dict) -> list[np.ndarray]:
    cases: list[np.ndarray] = []
    if "psi_snaps" in run:
        snaps = np.asarray(run["psi_snaps"], dtype=float)
        if snaps.ndim == 3:
            cases.extend([np.asarray(snaps[i], dtype=float) for i in range(snaps.shape[0])])
    if not cases and "psi_final" in run:
        final = np.asarray(run["psi_final"], dtype=float)
        if final.ndim == 2 and final.size:
            cases.append(final)
    if not cases:
        raise ValueError("run artifact has no psi snapshots or final psi field")
    return cases


if __name__ == "__main__":
    raise SystemExit(main())
