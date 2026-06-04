# Workflows

This document records current user-facing workflows. Commands assume the local ignored folders `configs/` and `data/` are present.

## Run A Simulation With Artifacts

Use `scripts/run_simulation_artifacts.py` when you want the simulation plus plots and optional video:

```bash
python scripts/run_simulation_artifacts.py \
  --config configs/T15MD_new_data.toml \
  --initial-currents configs/initial_currents/T15MD_new_data_3864.toml \
  --steps 100 \
  --controller lqr_boundary \
  --angles 32 \
  --scenario nominal \
  --out runs/example_t15 \
  --no-progress
```

The script prints the run directory, manifest path, boundary plot path, time-series plot path, frame directory when frames are enabled, and video path when video is enabled.

If the selected physical boundary rule cannot find a boundary, the simulation stops cleanly, writes partial artifacts, records a `boundary_missing` event, and prints the no-boundary step/reason.

## T15MD Replay

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

`t15md_replay` disables actuator lag and plant-side current/derivative clipping at runtime so the simulation follows the supplied replay table as applied current data.

The T15MD config declares `[boundary] mode = "limited"` and `[limiter] name = "T15MD"`. Limited boundary extraction uses the limiter-contact flux surface. Diverted mode is available for X-point separatrix boundaries. Use `--limiter T15MD` only when overriding or adding a plotting limiter for a config that does not declare one.

## Generate Synthetic ITER Fitting Tables

```bash
python scripts/generate_synthetic_iter_dataset.py \
  --config configs/ITER.toml \
  --initial-currents configs/initial_currents/ITER_default_currents.toml \
  --out-root synthetic_iter \
  --n-shots 20
```

The generator writes:

```text
<out-root>/synthetic_iter_metadata.json
<out-root>/ip/t15md_<shot>_ip.csv
<out-root>/coils/t15md_<shot>_coils.csv
```

The generated coil tables store applied model currents, not controller derivative commands.

## Generate Synthetic T15-Like Ip Tables

Use `scripts/generate_synthetic_ip_tables.py` when you want controller-ready `Ip` references rather than replay-current fitting data.

```bash
python scripts/generate_synthetic_ip_tables.py \
  --source-ip-dir data/t15_data_new_split/ip \
  --out-root data/t15_synthetic_ip \
  --out-initial-currents-dir configs/initial_currents \
  --initial-currents-prefix T15MD_new_data \
  --n-shots 12
```

The generator writes:

```text
<out-root>/synthetic_ip_metadata.json
<out-root>/ip/t15md_<shot>_ip.csv
configs/initial_currents/T15MD_new_data_<shot>.toml
```

Each generated table is a two-column semicolon-separated file:

```text
time_s;Ip
```

Example closed-loop run with an algorithmic current controller:

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

`boundary_mode=t15_linear` is an `ip_follow` reference-shape option. It is separate from the physical boundary finder mode in `[boundary] mode`.

## Sigma/L Grid Fit

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

The fitter creates one output folder per invocation. The exact path is printed as `top_k_csv=...`.

You can also use the legacy path form:

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
  --out-csv output/t15_sigma_L_refined.csv \
  --plot
```

Use `--plot` for plots. `-plot` is not defined by argparse.

## Sigma/L Gradient Fit

`scripts/fit_sigma_L_gradient.py` is the SPSA-style alternative to the grid search. It uses the same input semantics and the same run-folder output convention as the grid fitter.


## Programmatic Simulation Session

Use `tokamak_control.bridge.SimulationSession` when another Python tool needs direct reset/step access without generating run artifacts:

```python
import numpy as np

from tokamak_control.bridge import DerivativeAction, SimulationSession

session = SimulationSession.from_paths(
    config_path="configs/T15MD_new_data.toml",
    initial_currents_path="configs/initial_currents/T15MD_new_data_3864.toml",
    scenario_name="nominal",
    scenario_args={},
    angles=32,
    steps=100,
)
reset = session.reset()
action = DerivativeAction(np.zeros(reset.machine.n_active_total, dtype=float))
step = session.step_derivatives(action)
```

The action vector is in physical A/s and follows the active actuator order reported by `reset.machine.active_order`. Normalization, search, training, or optimization logic should live outside this repository.

## Docker Runtime

The Docker setup is intended for CLI/server execution with local volumes. The image does not bake in `configs/`, `data/`, `runs/`, or `output/`.

Build:

```bash
docker compose build
```

Default help command:

```bash
docker compose run --rm tokamak-sim
```

Example simulation:

```bash
docker compose run --rm tokamak-sim \
  python scripts/run_simulation_artifacts.py \
    --config configs/T15MD_new_data.toml \
    --initial-currents configs/initial_currents/T15MD_new_data_3864.toml \
    --steps 100 \
    --controller lqr_boundary \
    --angles 32 \
    --scenario nominal \
    --out runs/docker_example_t15 \
    --no-progress
```

For a server deployment, clone the repository, provide `configs/` and `data/` as mounted directories, and keep `runs/` and `output/` writable.

## Tests

Run collection without requiring data fixtures:

```bash
python -m pytest --collect-only -q tests/test_workflows.py tests/test_bridge_and_metrics.py
```

Run focused boundary checks:

```bash
python -m pytest -q tests/test_workflows.py::test_split_t15md_boundary_uses_limiter_contact_by_default tests/test_workflows.py::test_boundary_search_uses_diverted_separatrix_rule
```

The full workflow suite includes tests that require local ignored datasets under `data/`.
