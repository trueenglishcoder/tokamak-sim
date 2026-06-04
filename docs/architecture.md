# Architecture

This document summarizes the current runtime architecture without trying to be a full generated symbol index.

## Runtime Pipeline

The canonical simulation path lives in `tokamak_control/cli/run_simulation.py`.

```text
load config
build PlasmaModel
compute initial psi
find initial physical boundary
build scenario references
normalize controller through registry
construct controller
loop over steps:
  compute measured inputs from stored current boundary
  call controller
  apply optional realism
  advance model
  apply disturbances
  recompute psi
  update boundary tracker through the selected physical boundary rule
  write state/reference/radii/boundary/event records
  stop cleanly if no physical boundary exists
finalize RunWriter artifacts
```

`scripts/run_simulation_artifacts.py` wraps this API and adds final plots, optional frame rendering, and optional video. It consumes stored boundary polylines from the run artifact; it does not perform a separate plot-time boundary search.

## Programmatic Bridge

`tokamak_control/bridge/` exposes a small reset/step session for external tools that need simulator state without invoking the artifact CLI. `SimulationSession` loads the same TOML configs, builds the same `PlasmaModel`, uses the same scenario system, accepts physical active-coil current derivatives, advances the model, and updates the physical boundary through `tokamak_control/geometry/boundary.py`.

The bridge returns frozen dataclass snapshots with active-coil order, references, true `Ip`, active currents, commanded/applied derivatives, boundary polylines, sampled radii, and boundary-failure status. It does not own training algorithms, neural-network dependencies, policy loaders, plotting, or run-directory management.

`tokamak_control/metrics/` contains pure numerical diagnostics such as plasma-current error, sampled-radii RMSE, and actuator limit margins. These metrics are intentionally separate from rewards or controller objectives.

## Configuration

TOML configs are loaded through `tokamak_control/io/config_io.py` into `LoadedConfig`. In this GitHub-prep workspace, `configs/` is treated as a local ignored input directory rather than source code.

Important settings groups:

- grid dimensions and center coordinates
- PFC and SOL coil positions/currents
- optional boundary mode and limiter name for physical boundary extraction
- physical parameters such as `sigma`, `inductance_L`, `t_step`, `R0`, `Z0`
- actuator lag and optional current/derivative limits
- optional realism settings

Initial-current files live under `configs/initial_currents/`. They can define, per bank, both `active` masks and initial `currents`. Inactive actuators are removed from the runtime model dimensions, so controllers, replay tables, Green functions, and stored current vectors all see only active coils. If no initial-current file is provided, all configured coils are active with zero initial current.

Each config grid axis stores `start`, `end`, `size`, and `center`. The loader derives the uniform runtime step as `(end - start) / (size - 1)`, keeping the range and number of points as the config source of truth while preserving the `Grid1D.step` value used by the solver.

## Plant Model

`tokamak_control/core/plasma_model.py` owns the dynamic plant state. Controllers do not set coil currents directly; they return PFC/SOL current derivatives in a `ControlAction`. The model applies optional actuator lag, derivative/current limits, integrates currents, updates `Ip`, and composes the next `psi` field.

The current state is represented by `tokamak_control/core/plasma_state.py`.

## Boundary And Geometry

`tokamak_control/geometry/boundary.py` extracts a physical plasma boundary contour from `psi`. The default `limited` mode finds the magnetic axis, samples the configured limiter, and returns the flux surface that first contacts that limiter. The `diverted` mode finds an X-point and returns the closed separatrix contour at the X-point flux level.

If the selected rule cannot find a valid boundary, the finder raises `BoundaryNotFoundError`. During a simulation run this is treated as a clean physical stop: the runner records a `boundary_missing` event, finalizes partial artifacts, and reports the step/reason. The old largest-contour heuristic is not considered a plasma boundary.

`tokamak_control/geometry/limiters.py` stores named limiter polygons. T15MD configs use `[limiter] name = "T15MD"` so the limiter geometry is part of simulation behavior, not just a drawing overlay.

`tokamak_control/geometry/coordinates.py` converts boundary polylines into radii sampled at measurement angles. Controllers and metrics use these sampled radii, while artifacts also store the found boundary polylines.

## Scenarios

`tokamak_control/config/scenarios.py` builds references for supported scenarios such as:

- `nominal`
- `boundary_step`
- `ip_ramp`
- `ip_flat_top`
- `ip_jet_like`
- `boundary_pulse`
- `joint_disturbance`
- `shot_follow`
- `ip_table`
- `ip_follow`

`ip_crash` is resolved at launch time into a disturbance on top of a normal scenario.

## Controllers

Controllers live in `tokamak_control/control/` and implement `Controller.compute_control(...)`.

Controller families:

- boundary controllers: `lqr_boundary`, `hinf_boundary`
- current controllers: `lqr_current`, `hinf_current`
- joint controllers: `lqr_joint`, `hinf_joint`, `qp_joint`
- replay controllers: `coil_replay`, `t15md_replay`

`tokamak_control/control/registry.py` is the authoritative controller entry point. It owns controller names, launch-time parameter validation, and runtime argument filtering.

## Artifacts

`tokamak_control/io/data_io.py` writes and reads run artifacts:

- `run*.npz`
- `run_timeseries*.csv`
- `events*.csv`
- manifest JSON written by the runner
- optional profiling summary JSON

Important NPZ channels include:

- `psi_snaps`: optional stored psi snapshots
- `psi_final`: latest valid psi field
- `boundary_poly_true`: physical boundary polylines found during the run
- `boundary_poly_meas`: measured/noisy boundary polylines when realism is active
- `radii_true`, `radii_meas`, `radii_ref`: sampled boundary radii

Plotting helpers in `tokamak_control/viz/plotting.py` consume saved artifacts rather than reconstructing state from live Python objects or recomputing a separate plotting boundary.

