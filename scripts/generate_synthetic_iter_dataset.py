# scripts/generate_synthetic_iter_dataset.py
"""
Generate synthetic ITER rising-segment Ip and coil-current shot tables for sigma/L recovery checks.

The script uses the real project plant model and the old-parity Zaitsev LQR controller. It creates
T15-like rising-only current-reference profiles, lets the controller generate coil
absolute next-current commands, records the plant's applied coil currents and resulting
Ip, and writes CSVs in the same headerless semicolon-separated format consumed by
`scripts/fit_sigma_L_grid.py`:

    <out-root>/ip/t15md_<shot>_ip.csv
        time_s;Ip

    <out-root>/coils/t15md_<shot>_coils.csv
        time_s;SOL_0;...;SOL_n;PFC_0;...;PFC_m

The output coil table stores applied currents from model.state, not controller
commands. That is the required format for replay-based parameter fitting.

The generated files contain only the rising Ip segment. There is no initial flat
section, no high-current hold, no fall-off section, and no tail in the saved data.
This matches the preprocessed T15 fitting examples, where only the rising segment is
kept before running the fitter.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def _ensure_repo_root_on_path() -> None:
    """Добавить корень репозитория в путь импорта при прямом запуске скрипта."""
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from tokamak_control.control.base import ControlAction
from tokamak_control.control.lqr_t15_zaitsev import LQRT15ZaitsevController
from tokamak_control.core.plasma_model import PlasmaModel
from tokamak_control.core.plasma_state import PlasmaState
from tokamak_control.io.config_io import apply_initial_state, load_config, load_initial_state, require_initial_state


@dataclass(frozen=True, slots=True)
class ShotProfile:
    shot_id: str
    ip_start: float
    ip_peak: float
    t_rise: float

    @property
    def duration(self) -> float:
        return self.t_rise


@dataclass(frozen=True, slots=True)
class ShotSummary:
    shot_id: str
    rows: int
    duration_s: float
    reference_ip_start: float
    reference_ip_peak: float
    actual_ip_start: float
    actual_ip_peak_abs: float
    actual_ip_end: float
    ip_csv: str
    coils_csv: str


def _finite_positive(name: str, value: float) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value}")
    return out


def _finite_nonnegative(name: str, value: float) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value}")
    return out


def _smoothstep(x: float) -> float:
    y = min(1.0, max(0.0, float(x)))
    return y * y * (3.0 - 2.0 * y)


def _ip_reference(profile: ShotProfile, t: float) -> float:
    a = _smoothstep(float(t) / float(profile.t_rise))
    return float((1.0 - a) * profile.ip_start + a * profile.ip_peak)


def _write_semicolon_csv(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(arr, dtype=float), delimiter=";", fmt="%.16g")


def _make_profile(
    *,
    shot_id: str,
    ip_base: float,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> ShotProfile:
    sign = 1.0 if float(ip_base) >= 0.0 else -1.0
    mag = abs(float(ip_base))
    if mag <= 0.0 or not np.isfinite(mag):
        raise ValueError(f"Initial-state Ip0 must be finite and nonzero, got {ip_base}")

    variation = float(args.profile_variation)

    def varied(value: float, *, min_value: float) -> float:
        if variation <= 0.0:
            return max(float(value), float(min_value))
        factor = 1.0 + rng.normal(0.0, variation)
        return max(float(value) * factor, float(min_value))

    ip_start_mag = varied(float(args.ip_start_frac) * mag, min_value=0.02 * mag)
    ip_peak_mag = varied(float(args.ip_peak_frac) * mag, min_value=ip_start_mag + 0.05 * mag)

    if ip_peak_mag <= ip_start_mag:
        ip_peak_mag = ip_start_mag + 0.05 * mag

    return ShotProfile(
        shot_id=str(shot_id),
        ip_start=sign * ip_start_mag,
        ip_peak=sign * ip_peak_mag,
        t_rise=varied(float(args.rise_s), min_value=1e-6),
    )


def _runtime_currents_from_model(model: PlasmaModel) -> tuple[np.ndarray, np.ndarray]:
    state = model.state
    if state is None:
        raise RuntimeError("PlasmaModel has no initialized state")
    return (
        np.asarray(state.pfc_currents, dtype=float).copy(),
        np.asarray(state.sol_currents, dtype=float).copy(),
    )


def _initialize_model_at_ip(
    model: PlasmaModel,
    *,
    ip0: float,
    pfc0: np.ndarray,
    sol0: np.ndarray,
) -> None:
    pfc0 = np.asarray(pfc0, dtype=float).copy()
    sol0 = np.asarray(sol0, dtype=float).copy()

    if pfc0.shape != (model.pfc.n_coils,):
        raise ValueError(f"pfc0 shape {pfc0.shape} != ({model.pfc.n_coils},)")
    if sol0.shape != (model.sol.n_coils,):
        raise ValueError(f"sol0 shape {sol0.shape} != ({model.sol.n_coils},)")

    psi0 = model._compose_psi(float(ip0), pfc0, sol0)
    model.Ip0 = float(ip0)
    model.state = PlasmaState(
        t=0.0,
        step=0,
        Ip=float(ip0),
        Ip0=float(ip0),
        psi=psi0,
        pfc_currents=pfc0,
        pfc_current_derivs=np.zeros((model.pfc.n_coils,), dtype=float),
        sol_currents=sol0,
        sol_current_derivs=np.zeros((model.sol.n_coils,), dtype=float),
    )


def _simulate_one_shot(
    *,
    cfg,
    profile: ShotProfile,
    controller_params: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial_state = require_initial_state(cfg)
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics, ip0=initial_state.ip0)
    pfc0, sol0 = _runtime_currents_from_model(model)
    _initialize_model_at_ip(model, ip0=float(profile.ip_start), pfc0=pfc0, sol0=sol0)

    controller = LQRT15ZaitsevController(**controller_params)
    controller.reset()

    dt = float(model.t_step)
    n_steps = int(math.ceil(profile.duration / dt))
    n_rows = n_steps + 1

    t = np.empty((n_rows,), dtype=float)
    ip = np.empty((n_rows,), dtype=float)
    pfc = np.empty((n_rows, model.pfc.n_coils), dtype=float)
    sol = np.empty((n_rows, model.sol.n_coils), dtype=float)

    state = model.state
    if state is None:
        raise RuntimeError("Model state was not initialized")

    t[0] = float(state.t)
    ip[0] = float(state.Ip)
    pfc[0] = np.asarray(state.pfc_currents, dtype=float)
    sol[0] = np.asarray(state.sol_currents, dtype=float)

    for k in range(n_steps):
        state = model.state
        if state is None:
            raise RuntimeError("Model state was lost during simulation")

        ip_ref = _ip_reference(profile, float(state.t))
        action = controller.compute_control(
            model=model,
            psi=model.state.psi,
            boundary_poly=None,
            center=(float(model.R0), float(model.Z0)),
            measure_angles=np.zeros((0,), dtype=float),
            ref_radii=np.zeros((0,), dtype=float),
            Ip_ref=ip_ref,
            scenario=None,
        )
        if not isinstance(action, ControlAction):
            raise TypeError(f"Controller returned {type(action)!r}, expected ControlAction")

        state = model.step_currents(
            pfc_currents_next=action.pfc_currents_next,
            sol_currents_next=action.sol_currents_next,
        )

        t[k + 1] = float(state.t)
        ip[k + 1] = float(state.Ip)
        pfc[k + 1] = np.asarray(state.pfc_currents, dtype=float)
        sol[k + 1] = np.asarray(state.sol_currents, dtype=float)

    return t, ip, np.column_stack([sol, pfc])


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.n_shots) <= 0:
        raise ValueError("--n-shots must be > 0")
    if int(args.shot_start) < 0:
        raise ValueError("--shot-start must be >= 0")

    _finite_positive("--ip-start-frac", args.ip_start_frac)
    _finite_positive("--ip-peak-frac", args.ip_peak_frac)
    if float(args.ip_peak_frac) <= float(args.ip_start_frac):
        raise ValueError("--ip-peak-frac must be greater than --ip-start-frac")

    _finite_positive("--rise-s", args.rise_s)
    _finite_nonnegative("--profile-variation", args.profile_variation)

    _finite_nonnegative("--q-ip", args.q_ip)
    _finite_nonnegative("--r-pfc", args.r_pfc)
    _finite_nonnegative("--r-sol", args.r_sol)
    _finite_nonnegative("--ridge", args.ridge)
    if args.u_clip is not None:
        _finite_nonnegative("--u-clip", args.u_clip)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Generate synthetic ITER rising-segment Ip/coil-current CSV shots by "
            "tracking T15-like rising Ip references with the old-parity Zaitsev LQR controller."
        )
    )
    ap.add_argument("--config", required=True, help="ITER TOML config used as the true synthetic plant")
    ap.add_argument("--initial-state", required=True, help="TOML file with explicit plasma Ip0 and initial coil currents.")
    ap.add_argument("--out-root", default="synthetic_iter", help="Output root containing ip/ and coils/ subfolders")
    ap.add_argument("--n-shots", type=int, default=6)
    ap.add_argument("--shot-start", type=int, default=900001)
    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--ip-start-frac", type=float, default=0.65, help="Starting Ip as a fraction of abs(initial-state Ip0)")
    ap.add_argument("--ip-peak-frac", type=float, default=1.00, help="Final target Ip as a fraction of abs(initial-state Ip0)")
    ap.add_argument("--rise-s", type=float, default=0.17, help="Duration of the saved rising segment in seconds")
    ap.add_argument("--profile-variation", type=float, default=0.035)

    ap.add_argument("--q-ip", type=float, default=1.0)
    ap.add_argument("--r-pfc", type=float, default=1e-10)
    ap.add_argument("--r-sol", type=float, default=1e-10)
    ap.add_argument("--ridge", type=float, default=1e-14)
    ap.add_argument("--u-clip", type=float, default=None)

    args = ap.parse_args()
    _validate_args(args)

    machine_cfg = load_config(Path(args.config))
    cfg = apply_initial_state(machine_cfg, load_initial_state(machine_cfg, Path(args.initial_state)))
    true_sigma = float(cfg.physics.sigma)
    true_L = float(cfg.physics.inductance_L)
    true_tau = true_sigma * true_L

    out_root = Path(args.out_root)
    ip_dir = out_root / "ip"
    coils_dir = out_root / "coils"
    ip_dir.mkdir(parents=True, exist_ok=True)
    coils_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    controller_params: dict[str, object] = {
        "boundary_weight": 0.0,
        "ip_weight": float(args.q_ip),
        "derivative_weight": max(float(args.r_pfc), float(args.r_sol), 0.0),
        "delta_derivative_weight": max(float(args.ridge), 1.0e-30),
        "derivative_scale_aps": 1.0e6 if args.u_clip is None else float(args.u_clip),
    }

    summaries: list[ShotSummary] = []
    profiles: list[ShotProfile] = []

    for i in range(int(args.n_shots)):
        shot_id = str(int(args.shot_start) + i)
        profile = _make_profile(shot_id=shot_id, ip_base=float(require_initial_state(cfg).ip0), rng=rng, args=args)
        profiles.append(profile)

        t, ip, coils = _simulate_one_shot(
            cfg=cfg,
            profile=profile,
            controller_params=controller_params,
        )

        ip_table = np.column_stack([t, ip])
        coils_table = np.column_stack([t, coils])

        ip_path = ip_dir / f"t15md_{shot_id}_ip.csv"
        coils_path = coils_dir / f"t15md_{shot_id}_coils.csv"
        _write_semicolon_csv(ip_path, ip_table)
        _write_semicolon_csv(coils_path, coils_table)

        summary = ShotSummary(
            shot_id=shot_id,
            rows=int(t.size),
            duration_s=float(t[-1] - t[0]),
            reference_ip_start=float(profile.ip_start),
            reference_ip_peak=float(profile.ip_peak),
            actual_ip_start=float(ip[0]),
            actual_ip_peak_abs=float(np.max(np.abs(ip))),
            actual_ip_end=float(ip[-1]),
            ip_csv=str(ip_path),
            coils_csv=str(coils_path),
        )
        summaries.append(summary)

        print(
            f"Saved synthetic rising shot {shot_id}: "
            f"rows={summary.rows}, duration={summary.duration_s:.6f}s, "
            f"Ip_start={summary.actual_ip_start:.6g}, "
            f"max_abs_Ip={summary.actual_ip_peak_abs:.6g}, "
            f"Ip_end={summary.actual_ip_end:.6g}"
        )

    meta_model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics, ip0=require_initial_state(cfg).ip0)
    if meta_model.state is None:
        raise RuntimeError("Metadata model has no initialized state")

    metadata = {
        "source_config": str(Path(args.config)),
        "true_sigma": true_sigma,
        "true_inductance_L": true_L,
        "true_tau": true_tau,
        "t_step": float(cfg.physics.t_step),
        "actuator_tau": float(cfg.physics.actuator_tau),
        "n_pfc": int(len(np.asarray(meta_model.state.pfc_currents))),
        "n_sol": int(len(np.asarray(meta_model.state.sol_currents))),
        "coil_csv_order": "time, SOL currents, PFC currents",
        "profile_shape": "rising_only",
        "controller": "LQRT15ZaitsevController",
        "controller_params": controller_params,
        "seed": int(args.seed),
        "profiles": [asdict(p) for p in profiles],
        "shots": [asdict(s) for s in summaries],
    }

    meta_path = out_root / "synthetic_iter_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    print(f"true_sigma={true_sigma:.16g}")
    print(f"true_inductance_L={true_L:.16g}")
    print(f"true_tau={true_tau:.16g}")
    print(f"ip_dir={ip_dir}")
    print(f"coils_dir={coils_dir}")
    print(f"metadata={meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
