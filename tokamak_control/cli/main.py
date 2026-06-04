# tokamak_control/cli/main.py
"""Thin command line wrapper over the canonical single-run simulation API."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from tokamak_control.cli.run_simulation import (
    parse_key_value_args,
    resolve_runtime_scenario,
    run as run_sim,
)
from tokamak_control.control.registry import controller_names


_SCENARIO_CHOICES = (
    "nominal",
    "boundary_step",
    "ip_ramp",
    "ip_flat_top",
    "ip_jet_like",
    "boundary_pulse",
    "joint_disturbance",
    "ip_table",
    "ip_crash",
)


def _add_simulate(p) -> None:
    """Register the simulate subcommand."""
    ap = p.add_parser("simulate", help="Run a single closed-loop simulation.")
    ap.add_argument("--config", required=True, help="Path to TOML config.")
    ap.add_argument(
        "--initial-currents",
        default=None,
        help="Optional TOML file with active coil masks and initial currents.",
    )
    ap.add_argument("--steps", type=int, required=True, help="Number of time steps.")
    ap.add_argument(
        "--out",
        default=None,
        help="Output root directory for generated run folders. Defaults to ./runs.",
    )
    ap.add_argument(
        "--controller",
        default="lqr_boundary",
        choices=controller_names(),
        help="Controller name.",
    )
    ap.add_argument(
        "--controller-arg",
        action="append",
        default=[],
        help="Controller parameter in key=value form. Repeat as needed.",
    )
    ap.add_argument("--angles", type=int, default=16, help="Number of measurement angles.")
    ap.add_argument(
        "--scenario",
        default="nominal",
        choices=_SCENARIO_CHOICES,
        help="Reference scenario or launch-time convenience scenario.",
    )
    ap.add_argument(
        "--scenario-arg",
        action="append",
        default=[],
        help="Scenario parameter in key=value form. Repeat as needed.",
    )
    ap.add_argument(
        "--snap-every",
        type=int,
        default=0,
        help="Store psi snapshot every N steps (0 = never).",
    )
    ap.add_argument(
        "--realism",
        action="store_true",
        help="Enable measurement and actuation realism for this run.",
    )

    def _run(args: argparse.Namespace) -> int:
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

        res = run_sim(
            config=Path(args.config),
            initial_currents_path=(Path(args.initial_currents) if args.initial_currents is not None else None),
            steps=args.steps,
            output_dir=(Path(args.out) if args.out is not None else None),
            controller_name=args.controller,
            controller_params=controller_params,
            M_angles=args.angles,
            scenario_name=scenario_name,
            scenario_params=scenario_params,
            snapshot_every=args.snap_every,
            disturbances=disturbances,
            realism_enabled=bool(args.realism),
        )
        print(str(res.run_dir))
        print(str(res.manifest_path))
        print(str(res.npz_path))
        print(str(res.events_path))
        return 0

    ap.set_defaults(func=_run)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and dispatch to the selected subcommand."""
    parser = argparse.ArgumentParser(
        prog="tokamakctl",
        description="Tokamak plasma boundary control simulation CLI",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_simulate(sub)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
