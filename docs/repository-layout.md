# Repository Layout

This document describes the maintained `tokamak-sim` source tree and local
artifact folders.

## Versioned Source Areas

```text
README.md         Project overview and common commands
CURRENT_PIPELINE.md Active plant/RL contract
AGENTS.md         AI-agent coding rules
pyproject.toml    Package metadata and dependencies
Dockerfile        Runtime image for CLI/server use
docker-compose.yml Local volume-mounted runtime
docs/             Human-facing project documentation
scripts/          Runnable workflow and diagnostic scripts
tests/            Regression and parity tests
tokamak_control/  Importable simulation/control package
```

## Protected Local Inputs

These folders are local inputs or preserved generated baselines:

```text
configs/          TOML machine configs and initial-current files
data/             T15 CSV datasets and fitting inputs
runs/             Simulation run folders and replay reference datasets
output/           Fitting/diagnostic outputs
```

For the final RL pipeline, the important local paths are:

```text
data/t15_data_new_trim50/
runs/t15md_trim50_plain_gpu_1e6_setup/
runs/t15md_limited_replay_dataset_trim50_gpu_plain_1e6/
```

## Package Layout

```text
tokamak_control/bridge/     programmatic reset/step API
tokamak_control/cli/        single-run orchestration and CLI helpers
tokamak_control/config/     settings dataclasses and scenario definitions
tokamak_control/control/    controllers, replay tables, learned-controller loader
tokamak_control/core/       grid, coils, Green functions, plant state/model
tokamak_control/geometry/   boundary finding, limiter polygons, GPU fixed-angle path
tokamak_control/io/         config loading, artifact IO, logging, profiling
tokamak_control/metrics/    pure tracking and actuator diagnostics
tokamak_control/realism/    optional neutral nonidealities
tokamak_control/viz/        plotting, frames, video helpers
```

## Maintained Scripts

```text
scripts/run_simulation_artifacts.py        run one simulation and write artifacts
scripts/run_t15md_limited_replay_dataset.py generate replay references
scripts/compare_cpu_gpu_boundary_t15_replay.py compare boundary extractors
scripts/fit_sigma_L_grid.py                sigma/L grid fit
scripts/fit_sigma_L_gradient.py            sigma/L gradient fit
scripts/fit_t15_boundary_parameters.py     boundary diagnostics
```

LittleScope audit docs and parity notes remain available under
`docs/tokamak-sim-0-audit/` as historical/diagnostic material, not active launch
instructions.
