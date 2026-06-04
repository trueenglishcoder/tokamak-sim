"""Запустить расчет и сохранить графики, кадры и видео по запросу."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def _ensure_repo_root_on_path() -> None:
    """Добавить корень репозитория в путь импорта при прямом запуске скрипта."""
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from tokamak_control.cli.run_simulation import (
    parse_key_value_args,
    resolve_runtime_scenario,
    run as run_simulation,
)
from tokamak_control.control.registry import controller_names
from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.io.data_io import load_run
from tokamak_control.viz.plotting import (
    fig_boundary_from_poly,
    fig_time_series_from_npz,
    frames_to_video,
    save_run_frames,
)
from tokamak_control.geometry.limiters import get_limiter_shape, limiter_names


def _grid_from_meta(meta: dict) -> Grid2D:
    grid_meta = meta["grid"]
    return Grid2D(
        r=Grid1D(
            start=float(grid_meta["r_start"]),
            step=float(grid_meta["r_step"]),
            size=int(grid_meta["r_size"]),
            center=float(grid_meta.get("r_center", 0.0)),
        ),
        z=Grid1D(
            start=float(grid_meta["z_start"]),
            step=float(grid_meta["z_step"]),
            size=int(grid_meta["z_size"]),
            center=float(grid_meta.get("z_center", 0.0)),
        ),
    )


def _center_from_meta(meta: dict) -> tuple[float, float]:
    center_meta = meta["center"]
    return float(center_meta["R0"]), float(center_meta["Z0"])


def _coil_positions_from_meta(meta: dict) -> dict[str, np.ndarray] | None:
    coil_positions = meta.get("coil_positions", {})
    if not isinstance(coil_positions, dict):
        return None

    out: dict[str, np.ndarray] = {}

    pfc = coil_positions.get("pfc")
    if pfc is not None:
        arr = np.asarray(pfc, dtype=float)
        if arr.size != 0:
            out["pfc"] = arr

    sol = coil_positions.get("sol")
    if sol is not None:
        arr = np.asarray(sol, dtype=float)
        if arr.size != 0:
            out["sol"] = arr

    return out or None


def _limiter_shape_from_meta(meta: dict) -> np.ndarray | None:
    """Прочитать контур лимитера из metadata запуска."""
    limiter_meta = meta.get("limiter")
    if not isinstance(limiter_meta, dict):
        return None
    raw_shape = limiter_meta.get("shape")
    if raw_shape is not None:
        arr = np.asarray(raw_shape, dtype=float)
        if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] == 2:
            return arr
        raise ValueError(f"Invalid limiter shape in run metadata: {arr.shape}")
    raw_name = limiter_meta.get("name")
    if isinstance(raw_name, str) and raw_name.strip():
        return get_limiter_shape(raw_name)
    return None


def _polyline_from_padded_row(row: np.ndarray) -> np.ndarray:
    """Прочитать один NaN-заполненный контур из массива артефакта."""
    arr = np.asarray(row, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Boundary polyline row must have shape (N, 2), got {arr.shape}")
    valid = np.all(np.isfinite(arr), axis=1)
    poly = arr[valid]
    if poly.shape[0] < 3:
        raise RuntimeError("Stored boundary polyline is missing or too short")
    return poly


def main(argv: list[str] | None = None) -> int:
    """Разобрать CLI, выполнить расчет и записать артефакты визуализации."""
    ap = argparse.ArgumentParser(
        description="Run a simulation and save boundary and time-series plots, with optional frames and optional video."
    )
    ap.add_argument("--config", required=True, help="Path to TOML config.")
    ap.add_argument(
        "--initial-currents",
        default=None,
        help="Optional TOML file with active coil masks and initial currents.",
    )
    ap.add_argument("--steps", type=int, required=True, help="Number of steps.")
    ap.add_argument(
        "--out",
        default=None,
        help="Output root directory for generated run folders. Defaults to ./runs.",
    )
    ap.add_argument(
        "--controller",
        required=True,
        choices=controller_names(),
        help="Controller name.",
    )
    ap.add_argument(
        "--controller-arg",
        action="append",
        default=[],
        help="Controller parameter in key=value form. Repeat as needed.",
    )
    ap.add_argument("--angles", type=int, required=True, help="Boundary measurement angles.")
    ap.add_argument(
        "--scenario",
        required=True,
        choices=[
            "nominal",
            "boundary_step",
            "ip_ramp",
            "ip_flat_top",
            "ip_jet_like",
            "boundary_pulse",
            "joint_disturbance",
            "shot_follow",
            "ip_table",
            "ip_follow",
            "ip_crash",
        ],
        help="Scenario to run.",
    )
    ap.add_argument(
        "--scenario-arg",
        action="append",
        default=[],
        help="Scenario parameter in key=value form. Repeat as needed.",
    )
    ap.add_argument(
        "--realism",
        action="store_true",
        help="Enable measurement and actuation realism for this run.",
    )
    ap.add_argument(
        "--frames",
        action="store_true",
        help="Render frame_XXXX.png images into a frames directory.",
    )
    ap.add_argument(
        "--video",
        action="store_true",
        help="Render video.mp4 from saved frames using ffmpeg. Implies --frames.",
    )
    ap.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Video frames per second when --video is provided.",
    )
    ap.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Save/render one frame every N simulation steps. Higher values make video generation much faster.",
    )
    ap.add_argument(
        "--frame-dpi",
        type=int,
        default=160,
        help="DPI for saved frame PNGs. Lower values make frame and video generation faster.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print stage logs and periodic run status while the simulation executes.",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the live progress bar during the simulation loop.",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="Enable runtime profiling and write a profiling summary JSON into the run directory.",
    )
    ap.add_argument(
        "--profile-summary-every",
        type=int,
        default=0,
        help="Print profiling summaries every N timed steps. 0 disables periodic profiling log output.",
    )
    ap.add_argument(
        "--limiter",
        choices=limiter_names(),
        default=None,
        help="Override the configured limiter overlay on boundary plots and frames.",
    )

    args = ap.parse_args(list(argv) if argv is not None else None)

    try:
        controller_params = parse_key_value_args(args.controller_arg)
        scenario_params = parse_key_value_args(args.scenario_arg)
        scenario_name, scenario_params, disturbances = resolve_runtime_scenario(
            scenario_name=args.scenario,
            steps=args.steps,
            scenario_params=scenario_params,
        )
    except ValueError as e:
        ap.error(str(e))

    frame_stride = max(int(args.frame_stride), 1)
    frame_dpi = max(int(args.frame_dpi), 40)
    save_frames = bool(args.frames or args.video)
    snapshot_every = 1 if save_frames else args.steps

    result = run_simulation(
        config=Path(args.config),
        initial_currents_path=(Path(args.initial_currents) if args.initial_currents is not None else None),
        steps=args.steps,
        output_dir=(Path(args.out) if args.out is not None else None),
        controller_name=args.controller,
        controller_params=controller_params,
        M_angles=args.angles,
        scenario_name=scenario_name,
        scenario_params=scenario_params,
        snapshot_every=snapshot_every,
        disturbances=disturbances,
        realism_enabled=bool(args.realism),
        verbose=bool(args.verbose),
        show_progress=not bool(args.no_progress),
        profile=bool(args.profile),
        profile_summary_every=int(args.profile_summary_every),
    )

    out_dir = result.run_dir
    run_data = load_run(result.npz_path)
    meta = run_data["meta"]
    run_id = str(meta.get("run_id", ""))
    psi_snaps = np.asarray(run_data["psi_snaps"], dtype=float)
    psi_final_stored = np.asarray(run_data.get("psi_final"), dtype=float) if "psi_final" in run_data else np.zeros((0, 0), dtype=float)
    if psi_snaps.size == 0 and psi_final_stored.size == 0:
        print(str(result.run_dir))
        print(str(result.manifest_path))
        print(f"No physical plasma boundary was found: {result.stop_reason or 'no valid boundary state was written'}")
        return 0
    psi_final = np.asarray(psi_snaps[-1], dtype=float) if psi_snaps.size != 0 else psi_final_stored

    grid = _grid_from_meta(meta)
    center = _center_from_meta(meta)
    coil_positions = _coil_positions_from_meta(meta)
    limiter_shape = get_limiter_shape(args.limiter) if args.limiter is not None else _limiter_shape_from_meta(meta)
    boundary_polys = np.asarray(run_data.get("boundary_poly_true"), dtype=float) if "boundary_poly_true" in run_data else None
    if boundary_polys is None or boundary_polys.ndim != 3 or boundary_polys.shape[0] == 0:
        print(str(result.run_dir))
        print(str(result.manifest_path))
        print(f"No physical plasma boundary was found: {result.stop_reason or 'run artifact has no valid boundary polyline'}")
        return 0
    boundary_final = _polyline_from_padded_row(boundary_polys[-1])

    psi_boundary_path = out_dir / f"psi_boundary{run_id}.png"
    time_series_path = out_dir / f"time_series{run_id}.png"

    fig1 = fig_boundary_from_poly(
        psi=psi_final,
        grid=grid,
        center=center,
        poly=boundary_final,
        n_contours=60,
        title="ψ contours, heatmap, and boundary (final step)",
        coil_positions=coil_positions,
        limiter_shape=limiter_shape,
    )
    fig1.savefig(psi_boundary_path, dpi=160, bbox_inches="tight")

    fig2 = fig_time_series_from_npz(str(result.npz_path))
    fig2.savefig(time_series_path, dpi=160, bbox_inches="tight")

    frames_dir: Path | None = None
    video_path: Path | None = None
    if save_frames:
        frames_dir = out_dir / f"frames{run_id}"
        save_run_frames(
            npz_path=result.npz_path,
            frames_dir=frames_dir,
            n_levels_search=60,
            n_contours=60,
            coil_positions=coil_positions,
            limiter_shape=limiter_shape,
            frame_stride=frame_stride,
            dpi=frame_dpi,
        )

        if args.video:
            video_path = out_dir / f"video{run_id}.mp4"
            frames_to_video(frames_dir=frames_dir, video_path=video_path, fps=args.fps)

    print(str(out_dir))
    print(str(result.manifest_path))
    print(str(psi_boundary_path))
    print(str(time_series_path))
    if args.profile:
        print(str(out_dir / f"profile_summary{run_id}.json"))
    if frames_dir is not None:
        print(str(frames_dir))
    if video_path is not None:
        print(str(video_path))
    if result.stop_reason is not None:
        print(f"No physical plasma boundary at step {result.boundary_missing_step}: {result.stop_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
