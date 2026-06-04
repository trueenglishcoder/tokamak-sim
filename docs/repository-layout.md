# Repository Layout

This document describes the source-tree layout for GitHub preparation. The project keeps the existing `tokamak_control/` package structure and treats configs, data, and generated outputs as local ignored inputs/artifacts.

## Versioned Source Areas

```text
README.md         Project overview and common commands
AGENTS.md         AI-agent coding rules
pyproject.toml    Package metadata and dependencies
Dockerfile        Runtime image for CLI/server use
docker-compose.yml Local volume-mounted Docker runtime
.dockerignore     Build-context exclusions
docs/             Human-facing project documentation
scripts/          Directly runnable workflow scripts
tests/            Workflow regression tests
tokamak_control/  Importable Python package
```

## Ignored Local Areas

```text
configs/          Local TOML machine configs and initial-current files
data/             Local shot tables and generated fitting datasets
runs/             Simulation run folders
output/           Fitting/diagnostic outputs
synthetic_iter/   Generated synthetic ITER tables
_local_archive/   Local notes, old command logs, conversion scratch files
.venv/            Local virtual environment
.pytest_cache/    Pytest cache
```

`configs/`, `data/`, and generated outputs are not deleted. They remain usable in this workspace, but they are ignored so the future GitHub repository can be source-focused.

## Configs

`configs/` contains local TOML files used by simulation and fitting commands. Hard machine configs hold geometry, physics, optional boundary mode, and optional limiter names. Startup currents and active-coil masks live under `configs/initial_currents/`.

Use paths like:

```bash
--config configs/T15MD_new_data.toml
--initial-currents configs/initial_currents/T15MD_new_data_3864.toml
```

For Docker/server use, these files should be provided through a mounted volume or another explicit deployment step.

## Scripts

`scripts/` contains runnable entry points:

- `run_simulation_artifacts.py`: runs one simulation and writes plots, optional frames, and optional video.
- `generate_synthetic_iter_dataset.py`: generates synthetic ITER Ip/coil-current tables for fitting checks.
- `generate_synthetic_ip_tables.py`: generates synthetic T15-like Ip reference tables for algorithmic-controller runs.
- `fit_sigma_L_grid.py`: grid search for effective `sigma` and `inductance_L`.
- `fit_sigma_L_gradient.py`: SPSA-style gradient alternative for sigma/L fitting.

## Package

`tokamak_control/` is the importable package:

- `bridge/`: small programmatic reset/step API for external tools.
- `cli/`: canonical single-run orchestration and the package CLI.
- `config/`: settings dataclasses and scenario definitions.
- `control/`: controllers, replay tables, linearization, and controller registry.
- `core/`: grid, coils, Green functions, plasma state, and plasma model.
- `experiments/`: disturbances and realism injection.
- `geometry/`: boundary finding, limiter polygons, X-point detection, and coordinate utilities.
- `io/`: config loading, artifact IO, logging, profiling.
- `metrics/`: pure tracking, boundary, and actuator diagnostics.
- `viz/`: plotting, frames, and video helpers.

## Current Cleanup State

Loose local files that were not part of project functionality were moved to `_local_archive/` and ignored. The top-level source tree now contains only project metadata/docs plus source directories.
