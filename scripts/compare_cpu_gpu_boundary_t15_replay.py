#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.geometry.boundary import find_plasma_boundary_with_status
from tokamak_control.geometry.legacy_metrics import legacy_radii_at_angles


def _default_npz() -> Path:
    root = Path("runs/t15md_limited_replay_dataset")
    paths = sorted(root.glob("t15md_limited_replay_3857/*/run*.npz"))
    if paths:
        return paths[-1]
    all_paths = sorted(root.glob("t15md_limited_replay_*/*/run*.npz"))
    if not all_paths:
        raise FileNotFoundError(f"no replay npz files found under {root}")
    return all_paths[-1]


def _grid_from_meta(meta: dict) -> Grid2D:
    g = meta["grid"]
    return Grid2D(
        r=Grid1D(float(g["r_start"]), float(g["r_step"]), int(g["r_size"]), float(g["r_center"])),
        z=Grid1D(float(g["z_start"]), float(g["z_step"]), int(g["z_size"]), float(g["z_center"])),
    )


def _step_indices(total: int, requested: list[int] | None) -> list[int]:
    if requested:
        out = []
        for step in requested:
            idx = int(step)
            if idx < 0:
                idx = total + idx
            if idx < 0 or idx >= total:
                raise ValueError(f"step index {step} out of range 0..{total - 1}")
            out.append(idx)
        return out
    return sorted({0, total // 3, 2 * total // 3, total - 1})


def _fixed_angle_boundary(
    *,
    psi_batch: np.ndarray,
    grid: Grid2D,
    center: tuple[float, float],
    angles: np.ndarray,
    limiter: np.ndarray,
    boundary_mode: str,
    device: str,
    allow_cpu_emulation: bool,
):
    import torch
    import tokamak_control.geometry.boundary_gpu as boundary_gpu

    if device == "cpu":
        if not allow_cpu_emulation:
            raise RuntimeError("CPU emulation requires --allow-cpu-emulation")
        boundary_gpu.require_gpu_available = lambda _device: None
    elif not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but torch.cuda.is_available() is false")

    result = boundary_gpu.fixed_angle_boundary_gpu(
        psi=psi_batch,
        grid=grid,
        center=center,
        angles_rad=angles,
        limiter_shape=limiter,
        boundary_mode=boundary_mode,
        gpu_device=device,
        ray_samples=256,
    )
    return (
        result.points.detach().cpu().numpy(),
        result.radii.detach().cpu().numpy(),
        result.found.detach().cpu().numpy().astype(bool),
        result.level.detach().cpu().numpy(),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare CPU contour boundary with batched fixed-angle GPU boundary on a replay npz.")
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=Path("analysis_outputs/boundary_cpu_gpu_comparison/t15_replay_cpu_vs_gpu_boundary.png"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--allow-cpu-emulation", action="store_true")
    ap.add_argument("--steps", type=int, nargs="*", default=None)
    args = ap.parse_args()

    npz_path = args.npz or _default_npz()
    with np.load(npz_path, allow_pickle=True) as z:
        meta = json.loads(str(z["meta_json"].item()))
        psi_snaps = np.asarray(z["psi_snaps"], dtype=float)
        snap_steps = np.asarray(z["psi_snap_steps"], dtype=int)
        snap_t = np.asarray(z["psi_snap_t"], dtype=float)

    grid = _grid_from_meta(meta)
    center = (float(meta["center"]["R0"]), float(meta["center"]["Z0"]))
    limiter = np.asarray(meta["limiter"]["shape"], dtype=float).reshape(-1, 2)
    boundary_meta = meta.get("boundary", {})
    boundary_mode = str(boundary_meta.get("mode", "legacy_contour_limited"))
    if boundary_mode == "tracked_flux_contour":
        boundary_mode = str(boundary_meta.get("base_mode", "legacy_contour_limited"))
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False)
    chosen = _step_indices(int(psi_snaps.shape[0]), args.steps)

    psi_batch = psi_snaps[chosen]
    gpu_points, gpu_radii, gpu_found, gpu_levels = _fixed_angle_boundary(
        psi_batch=psi_batch,
        grid=grid,
        center=center,
        angles=angles,
        limiter=limiter,
        boundary_mode=boundary_mode,
        device=str(args.device),
        allow_cpu_emulation=bool(args.allow_cpu_emulation),
    )

    cpu_polys: list[np.ndarray | None] = []
    cpu_levels: list[float] = []
    cpu_points: list[np.ndarray] = []
    cpu_radii: list[np.ndarray] = []
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    for psi in psi_batch:
        try:
            poly, level, _status = find_plasma_boundary_with_status(
                psi,
                grid,
                center,
                limiter_shape=limiter,
                boundary_mode=boundary_mode,
                boundary_base_mode="legacy_contour_limited",
            )
            radii = legacy_radii_at_angles(poly, center, angles)
            points = np.asarray(center, dtype=float).reshape(1, 2) + radii.reshape(-1, 1) * dirs
        except Exception:
            poly = None
            level = float("nan")
            radii = np.full((angles.size,), np.nan, dtype=float)
            points = np.full((angles.size, 2), np.nan, dtype=float)
        cpu_polys.append(poly)
        cpu_levels.append(float(level))
        cpu_radii.append(radii)
        cpu_points.append(points)

    R, Z = grid.mesh()
    rows = len(chosen)
    fig, axes = plt.subplots(rows, 2, figsize=(13, max(3.2, 3.1 * rows)), squeeze=False)
    for row, idx in enumerate(chosen):
        psi = psi_snaps[idx]
        ax = axes[row, 0]
        ax.contour(R, Z, psi, levels=48, colors="0.72", linewidths=0.45)
        if limiter.size:
            ax.plot(limiter[:, 0], limiter[:, 1], "k-", lw=1.3, label="limiter")
        if cpu_polys[row] is not None:
            poly = cpu_polys[row]
            ax.plot(poly[:, 0], poly[:, 1], color="#1f77b4", lw=2.0, label="CPU contour")
            ax.scatter(cpu_points[row][:, 0], cpu_points[row][:, 1], s=14, color="#1f77b4", marker="o", label="CPU fixed-angle samples")
        if gpu_found[row]:
            pts = gpu_points[row]
            closed = np.vstack([pts, pts[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color="#ff7f0e", lw=1.5, ls="--", label=f"{args.device} fixed-angle")
            ax.scatter(pts[:, 0], pts[:, 1], s=18, color="#ff7f0e", marker="x")
        ax.scatter([center[0]], [center[1]], c="red", s=22, marker="+")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"step {int(snap_steps[idx])}, t={float(snap_t[idx]):.3f}s")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        if row == 0:
            ax.legend(loc="upper right", fontsize=8)

        diff = gpu_radii[row] - cpu_radii[row]
        ax2 = axes[row, 1]
        ax2.axhline(0.0, color="0.3", lw=0.8)
        ax2.plot(np.degrees(angles), cpu_radii[row], color="#1f77b4", label="CPU radius")
        ax2.plot(np.degrees(angles), gpu_radii[row], color="#ff7f0e", ls="--", label=f"{args.device} radius")
        ax2b = ax2.twinx()
        ax2b.plot(np.degrees(angles), 1000.0 * diff, color="#2ca02c", lw=1.2, alpha=0.85, label="GPU-CPU [mm]")
        finite = np.isfinite(diff)
        max_mm = float(np.nanmax(np.abs(diff)) * 1000.0) if np.any(finite) else float("nan")
        rms_mm = float(np.sqrt(np.nanmean(diff * diff)) * 1000.0) if np.any(finite) else float("nan")
        ax2.set_title(
            f"level CPU={cpu_levels[row]:.4g}, GPU={gpu_levels[row]:.4g}; "
            f"|diff|max={max_mm:.2f} mm, rms={rms_mm:.2f} mm"
        )
        ax2.set_xlabel("angle [deg]")
        ax2.set_ylabel("radius [m]")
        ax2b.set_ylabel("GPU-CPU [mm]")
        if row == 0:
            lines, labels = ax2.get_legend_handles_labels()
            lines_b, labels_b = ax2b.get_legend_handles_labels()
            ax2.legend(lines + lines_b, labels + labels_b, loc="upper right", fontsize=8)

    subtitle = "CPU emulation of GPU path" if str(args.device) == "cpu" else "CUDA GPU path"
    fig.suptitle(f"CPU contour vs {subtitle}\n{npz_path}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
