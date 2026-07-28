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
  apply optional actuator realism
  advance model
  apply disturbances
  recompute psi
  update boundary tracker through the selected physical boundary rule
  compute optional measured sensor channels
  write state/reference/radii/boundary/event records
  stop cleanly if no physical boundary exists
finalize RunWriter artifacts
```

`scripts/run_simulation_artifacts.py` wraps this API and adds final plots, optional frame rendering, and optional video. It consumes stored boundary polylines from the run artifact; it does not perform a separate plot-time boundary search.

## Programmatic Bridge

`tokamak_control/bridge/` exposes a small reset/step session for external tools that need simulator state without invoking the artifact CLI. `SimulationSession` loads the same TOML configs, builds the same `PlasmaModel`, uses the same scenario system, accepts physical active-coil current derivatives, advances the model, and updates the physical boundary through `tokamak_control/geometry/boundary.py`.

The bridge returns frozen dataclass snapshots with active-coil order, references, true and measured `Ip`, true and measured active currents, commanded/applied derivatives, dense boundary polylines, topology, axis, X-points, limiter contacts, separatrix branches, fixed-angle validity, sampled radii, quality metrics, and boundary-failure status. It does not own training algorithms, neural-network dependencies, policy loaders, plotting, or run-directory management.

`tokamak_control/metrics/` contains pure numerical diagnostics such as plasma-current error, sampled-radii RMSE, and actuator limit margins. These metrics are intentionally separate from rewards or controller objectives.

## Configuration

TOML configs are loaded through `tokamak_control/io/config_io.py` into `LoadedConfig`. In this GitHub-prep workspace, `configs/` is treated as a local ignored input directory rather than source code.

Important settings groups:

- grid dimensions and center coordinates
- PFC and SOL coil geometry, active masks, and limits
- optional boundary mode and limiter name for physical boundary extraction
- physical parameters such as `sigma`, `inductance_L`, `t_step`, `R0`, `Z0`
- actuator lag and optional current/derivative limits
- optional neutral `[realism]` settings for actuator and sensor nonidealities

Machine configs do not contain initial plasma current or initial coil currents. Runtime initial states live under `configs/initial_states/` or are supplied directly by replay, RL, or bridge callers. An initial-state TOML contains `[plasma].Ip0`, `[coils.pfc].currents`, and `[coils.sol].currents`; active masks remain in the machine config. Runs must provide an explicit initial state, so there is no silent fallback to machine defaults.

Each config grid axis stores `start`, `end`, `size`, and `center`. The loader derives the uniform runtime step as `(end - start) / (size - 1)`, keeping the range and number of points as the config source of truth while preserving the `Grid1D.step` value used by the solver.

## Plant Model

`tokamak_control/core/plasma_model.py` owns the dynamic plant state. The public plant command is absolute next-step coil currents; the plant derives `Jdot = (J_next - J_now) / dt`, updates `Ip`, and composes the next `psi` field. Current and derivative limits are controller/diagnostic metadata, not hidden plant-side initialization.

The current state is represented by `tokamak_control/core/plasma_state.py`.

## Boundary And Geometry

`tokamak_control/geometry/boundary.py` consumes a fully known `psi(R,Z)` field.
It does not reconstruct an equilibrium from diagnostic measurements. The active
`equilibrium_lcfs` mode automatically determines whether the connected primary
core is limiter-limited or bounded by one or more X-point separatrices.

The canonical CPU path is split into explicit components:

- `equilibrium_field.py`: bicubic field, gradient, Hessian, and level projection
- `critical_points.py`: subgrid O/X refinement and Hessian classification
- `level_set_graph.py`: topology-preserving level-set graph and branch cycles
- `lcfs.py`: wall/saddle candidate ordering, core validation, and result object
- `boundary_projection.py`: dense-boundary projection to fixed-angle RL radii

The dense LCFS is the geometric source of truth. X-points are explicit graph
nodes and no global smoothing spline is fitted through them. Fixed-angle radii
are accepted as a valid representation only when every ray has exactly one
forward boundary intersection.

The batched GPU path computes the derived fixed-angle signal, topology code,
boundary level, axis, and selected X-points. Single-lane artifact runs also use
the canonical CPU extractor so stored boundaries and videos contain the dense
physical contour.

If no candidate forms a physical boundary of the connected primary core, the
finder raises `BoundaryNotFoundError`. The runner records a physical boundary
loss and finalizes partial artifacts. It does not substitute a circle, cached
contour, largest polygon, or smoothed sparse-radius curve.

`tokamak_control/geometry/limiters.py` stores named material limiter polygons.
T15MD configs use `[limiter] name = "T15MD"`.

`tokamak_control/geometry/parametric_boundary.py` remains the analytic reference
boundary primitive for `(R0, Z0, A0, kappa, delta)`. It is separate from the
physical LCFS extractor.

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
- `t15_synthetic_follow`

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
- `Ip`, `Ip_meas`: true and measured plasma current when measured channels are recorded
- `pfc_currents`, `pfc_currents_meas`, `sol_currents`, `sol_currents_meas`: true and measured active currents when measured channels are recorded
- `radii_true`, `radii_meas`, `radii_ref`: sampled boundary radii
- `boundary_topology_code`, `boundary_axis`, `boundary_x_points`: topology and critical-point state
- `boundary_limiter_contacts`, `boundary_separatrix_branches`: material contacts and diverted branches
- `boundary_fixed_angle_valid`, `boundary_fixed_angle_counts`: explicit radial-representation contract
- `boundary_quality`: flux, closure, gradient, containment, and core-connectivity diagnostics

Plotting helpers in `tokamak_control/viz/plotting.py` consume saved artifacts rather than reconstructing state from live Python objects or recomputing a separate plotting boundary.


## Batched GPU boundary contract

`equilibrium_lcfs` has one physical definition. The CPU implementation owns the
dense LCFS graph, explicit X-point nodes, separatrix branches, and limiter
contacts. The batched GPU implementation computes the first-exit fixed-angle
projection of that same primary-core boundary together with topology, boundary
level, magnetic axis, and X-point tensors. It does not create a second smoothed
or fixed-ray physical boundary. Single-lane artifacts are always populated from
the canonical CPU dense extractor, and parity tests enforce agreement of the
derived tensor signal.
