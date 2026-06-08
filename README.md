# Tokamak Sim

Pure-Python tokamak plasma simulation and control toolkit. The main workflow is running closed-loop simulations from local TOML machine configs and shot tables, writing reproducible run artifacts, and optionally plotting frames or video.

This repository is being prepared as source code. Local machine configs, private/large data tables, generated runs, and exploratory notes are intentionally kept out of Git.

## Quick Start

Install the project in a virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

For the optional CUDA boundary backend, install Torch as well:

```bash
python -m pip install -e ".[dev,gpu]"
```

The runnable examples below assume local config and data folders are present:

```text
configs/
data/
```

Those folders are ignored by Git. On a fresh clone or server, provide them separately before running simulations.

Run a short simulation with plots:

```bash
python scripts/run_simulation_artifacts.py \
  --config configs/T15MD_new_data.toml \
  --initial-currents configs/initial_currents/T15MD_new_data_3864.toml \
  --steps 20 \
  --controller lqr_boundary \
  --angles 32 \
  --scenario nominal \
  --out runs/quick_t15 \
  --no-progress
```

Run test collection or focused checks:

```bash
python -m pytest --collect-only -q tests/test_workflows.py
python -m pytest -q tests/test_workflows.py::test_split_t15md_boundary_uses_limiter_contact_by_default tests/test_workflows.py::test_boundary_search_uses_diverted_separatrix_rule
```

Some workflow tests require local ignored datasets under `data/` and local ignored config files under `configs/`.

## Current Layout

```text
README.md         Project overview and common commands
AGENTS.md         Coding rules for AI-assisted work
pyproject.toml    Package metadata and dependencies
docs/             Project documentation
scripts/          Directly runnable workflow scripts
tests/            Workflow regression tests
tokamak_control/  Importable simulation/control package
```

Local ignored folders:

```text
configs/          Local TOML machine configs and initial-current files
data/             Local shot tables and generated fitting datasets
runs/             Simulation run artifacts
output/           Fitting outputs and diagnostics
synthetic_iter/   Generated synthetic ITER tables
_local_archive/   Local notes, conversion scratch files, and old command logs
```

## Main Commands

T15MD replay simulation with video:

```bash
python scripts/run_simulation_artifacts.py \
  --config configs/T15MD_new_data.toml \
  --initial-currents configs/initial_currents/T15MD_new_data_3864.toml \
  --steps 1439 \
  --controller t15md_replay \
  --controller-arg replay_path=data/t15_data_new/coils/t15md_3864_coils.csv \
  --angles 32 \
  --scenario ip_follow \
  --scenario-arg ip_csv=data/t15_data_new/ip/t15md_3864_ip.csv \
  --video \
  --frame-stride 5 \
  --frame-dpi 90 \
  --fps 20 \
  --verbose
```

If the selected physical boundary rule cannot find a boundary, the run stops cleanly, writes partial artifacts, and logs a `boundary_missing` event. It does not draw, reuse, or invent a boundary.

## Compute Backend

The simulator defaults to CPU mode. CPU mode uses the established NumPy/SciPy/contourpy path and remains the reference implementation.

GPU mode is explicit:

```toml
[compute]
backend = "gpu"
gpu_device = "cuda:0"
boundary_equivalence_mode = "strict"
```

or from the CLI:

```bash
python scripts/run_simulation_artifacts.py \
  --config configs/T15MD_new_data.toml \
  --steps 20 \
  --controller lqr_boundary \
  --angles 32 \
  --scenario nominal \
  --compute-backend gpu \
  --gpu-device cuda:0
```

GPU mode requires `tokamak-sim[gpu]` and a working CUDA Torch installation. If GPU mode is requested and CUDA is unavailable, the run fails clearly before simulation work starts. Run artifacts record the selected backend, Torch version, CUDA availability, and GPU name.

To compare GPU boundary finding against the CPU reference on stored run artifacts:

```bash
python scripts/check_boundary_gpu_equivalence.py \
  --run-npz runs/example/run123.npz \
  --out output/boundary_gpu_equivalence.json \
  --gpu-device cuda:0
```

Synthetic fitting data:

```bash
python scripts/generate_synthetic_iter_dataset.py \
  --config configs/ITER.toml \
  --initial-currents configs/initial_currents/ITER_default_currents.toml \
  --out-root synthetic_iter \
  --n-shots 20
```

Synthetic T15-like Ip references for algorithmic controllers:

```bash
python scripts/generate_synthetic_ip_tables.py \
  --source-ip-dir data/t15_data_new_split/ip \
  --out-root data/t15_synthetic_ip \
  --out-initial-currents-dir configs/initial_currents \
  --initial-currents-prefix T15MD_new_data \
  --n-shots 12
```

Run a closed-loop simulation from one generated Ip table:

```bash
python scripts/run_simulation_artifacts.py \
  --config configs/T15MD_new_data.toml \
  --initial-currents configs/initial_currents/T15MD_new_data_950001.toml \
  --steps 1439 \
  --controller lqr_current \
  --angles 32 \
  --scenario ip_follow \
  --scenario-arg ip_csv=data/t15_synthetic_ip/ip/t15md_950001_ip.csv \
  --scenario-arg boundary_mode=t15_linear \
  --video \
  --verbose
```

Sigma/L grid fit:

```bash
python scripts/fit_sigma_L_grid.py \
  --config configs/T15MD_new_data.toml \
  --ip-dir data/t15_data_new/ip \
  --coils-dir data/t15_data_new/coils \
  --sigma-min 1.5e6 \
  --sigma-max 3.5e6 \
  --sigma-points 31 \
  --L-min 5e-7 \
  --L-max 5e-6 \
  --L-points 31 \
  --top-k 100 \
  --out output \
  --name t15_sigma_L_refined \
  --plot
```

Use `--plot`, not `-plot`; the fitter defines `--plot` as a long argparse flag.


## Programmatic Use

External tools can step the simulator directly through `tokamak_control.bridge.SimulationSession` without going through the plotting/artifact CLI. The bridge exposes loaded machine metadata, scenario references, active-coil current vectors, derivative commands, applied derivatives, sampled boundary radii, and clean boundary-failure termination status. It accepts physical active-coil current derivatives in A/s and keeps training/optimization-specific logic outside this repository.

Pure tracking and actuator diagnostics live in `tokamak_control.metrics`. These functions compute errors and margins only; they do not apply reward weights or controller logic.

## Docker

The Docker image contains source code and Python/runtime dependencies only. Local inputs and outputs are mounted at runtime:

```text
./configs -> /app/configs   read-only local machine configs
./data    -> /app/data      read-only local shot/data tables
./runs    -> /app/runs      writable simulation artifacts
./output  -> /app/output    writable fitting/diagnostic outputs
```

Build the image:

```bash
docker compose build
```

Show the artifact runner help:

```bash
docker compose run --rm tokamak-sim
```

Run a short simulation through Docker:

```bash
docker compose run --rm tokamak-sim \
  python scripts/run_simulation_artifacts.py \
    --config configs/T15MD_new_data.toml \
    --initial-currents configs/initial_currents/T15MD_new_data_3864.toml \
    --steps 20 \
    --controller lqr_boundary \
    --angles 32 \
    --scenario nominal \
    --out runs/docker_quick_t15 \
    --no-progress
```

Video generation uses `ffmpeg`, which is installed in the image.

## Boundary Notes

Boundary extraction supports two physical modes:

- `limited`: default mode; uses the configured limiter-contact flux surface.
- `diverted`: uses an X-point separatrix.

The magenta T15MD geometry in plots is the limiter. The boundary finder is not an optimization target and does not use largest-contour drawing as a physical boundary rule.

Run artifacts store true and measured boundary channels in `boundary_poly_true`/`boundary_poly_meas` and `radii_true`/`radii_meas`. When neutral realism is enabled, measured channels can include actuator and sensor nonidealities; plotting and frame rendering consume the stored polylines instead of recomputing a separate plotting boundary.

`tokamak_control.geometry.parametric_boundary` provides the T15 reference-boundary primitive used for synthetic training references. It validates the analytic `(R0, Z0, A0, kappa, delta)` boundary, includes replay-derived robust T15 bounds and smooth rate limits, generates deterministic rate-limited parameter trajectories, and converts generated shapes into `radii_ref`. `tokamak_control.config.ip_trajectories` provides the matching Ip reference primitive: it loads real `time;Ip` tables, generates seeded template perturbations with amplitude/duration/shape jitter, writes generated tables, and exposes clamped interpolation. The `t15_synthetic_follow` scenario combines these through the standard scenario interface using an explicit Ip ramp, a direct `ip_csv`, `ip_template_csv`, or `ip_template_dir`.

## Documentation

- [Repository Layout](docs/repository-layout.md)
- [Workflows](docs/workflows.md)
- [Architecture](docs/architecture.md)
- [Plasma Boundary Calculation](docs/plasma_boundary_calculation.txt)

## Notes For GitHub Preparation

- `.git` in this workspace is currently an empty directory, not an initialized repository.
- `configs/`, `data/`, `runs/`, `output/`, `synthetic_iter/`, and `_local_archive/` are ignored intentionally.
- Docker runtime files are included; provide `configs/` and `data/` as local mounts.
