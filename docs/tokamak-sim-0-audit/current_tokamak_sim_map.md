# Current Tokamak-Sim Subsystem Map

This file maps the current Python implementation as audited on 2026-06-21. Line references are to current working-tree files.

## Plant And Grid

| File | Lines | Behavior |
|---|---:|---|
| `tokamak_control/core/grid.py` | 40-76 | `Grid1D.coords()` uses the old center-half-cell shifted-start formula. This is intended parity with old C++ `grid_t`. |
| `tokamak_control/core/green.py` | 22-95 | Implements the old elliptic Green function and grouped actuator summation. `mu0` is applied outside the Green function, matching old C++. |
| `tokamak_control/core/plasma_model.py` | 100-110, 275-329 | CPU plant accepts derivative commands `Jdot`, integrates currents as `J_new = J_old + dt * Jdot`, recomputes passive `Ip0 * exp(-t/tau)`, and adds derivative coupling through `(mu0*sigma/R0) * g dot Jdot`. No plant-side lag/clipping is applied in `step`. |
| `tokamak_control/core/gpu_plasma_model.py` | 89-122 | Single-GPU plant mirrors CPU derivative-command plant. |
| `tokamak_control/core/batched_gpu_simulator.py` | 92-134 | Batched GPU plant mirrors CPU derivative-command plant. |
| `tokamak_control/core/plasma_model.py` | 238-246 | `get_ip_B_row()` returns derivative-command sensitivity with no extra `t_step`. |
| `tokamak_control/core/plasma_model.py` | 389-417 | CPU Green sampling for arbitrary points currently indexes from `grid.r.start`/`grid.z.start`, not from shifted coordinates. This is a likely parity bug after grid alignment changes. |
| `tokamak_control/core/torch_sampling.py` | 3-61 | GPU bilinear field sampling also indexes from raw `start`, not shifted coordinates. |

## Config Loading

| File | Lines | Behavior |
|---|---:|---|
| `tokamak_control/config/settings.py` | 11-55 | Physics settings describe old-style derivative-command plant, but keep actuator/current limits as metadata. |
| `tokamak_control/io/config_io.py` | 88-107, 311-331, 413-420 | Loads boundary mode, physics values, signs, limits, and `legacy_precision_index2`. |
| `configs/T15MD_new_data.toml` | 15-30 | Live T15 config uses fitted `sigma=3832562.947936214`, `inductance_L=3.36416228166149e-07`, `t_step=0.001`, `ip_coupling_sign=1.0`, `plasma_psi_sign=-1.0`, and default `boundary.mode="limited"`. |
| `configs/T15MD_new_data.toml` | 35-843 | T15 grouped PFC/SOL elements and weights. This is a T15-specific machine replacement for old generic config files. |

## Boundary And Metrics

| File | Lines | Behavior |
|---|---:|---|
| `tokamak_control/geometry/boundary.py` | 43-119 | Boundary dispatcher supports `limited`, `diverted`, and `legacy_contour`; routes legacy mode to CPU. |
| `tokamak_control/geometry/boundary_cpu.py` | 213-318 | Implements old-style `legacy_contour`: center-index conversion, search over a single contour level, closed contour accepted by center bounding box, limiter ignored. |
| `tokamak_control/geometry/boundary_cpu.py` | 355-639 | Modern `limited/diverted` extraction with limiter/x-point logic. |
| `tokamak_control/geometry/boundary_gpu.py` | 51-77 | GPU path supports only `limited/diverted`; `legacy_contour` must be CPU fallback. |
| `tokamak_control/core/batched_gpu_simulator.py` | 150-159 | Batched GPU path hardcodes `boundary_mode="limited"` inside `fixed_angle_boundary_gpu`. |
| `tokamak_control/geometry/coordinates.py` | 15-88 | Current boundary metrics use ray intersections at fixed angles. This is not the old MATLAB closest-point / `GetAngleBuddies` machinery. |
| `tokamak_control/geometry/distances.py` | 1-70 | Current closest-distance helpers, separate from old measurement-error state. |

## Runtime Loop And Artifacts

| File | Lines | Behavior |
|---|---:|---|
| `tokamak_control/cli/run_simulation.py` | 357-427 | Creates CPU/GPU plant, finds initial boundary, builds scenario from base radii. |
| `tokamak_control/cli/run_simulation.py` | 670-704 | Calls controller with runtime context and expects a `ControlAction` containing derivative commands. |
| `tokamak_control/cli/run_simulation.py` | 706-717 | Optional `RealismRuntime` can modify derivative commands before plant step. |
| `tokamak_control/cli/run_simulation.py` | 720-747 | Calls `model.step(pfc_current_derivs, sol_current_derivs)` and recomputes psi. |
| `tokamak_control/cli/run_simulation.py` | 749-786 | Updates boundary using selected boundary mode. |
| `tokamak_control/io/data_io.py` | 60-165, 503-587 | Run artifacts store currents, applied derivatives, command/effective derivatives, radii, boundaries, references, psi snapshots, CSV sidecars. |
| `tokamak_control/viz/plotting.py` | 569-656 | Time-series plot shows Ip/ref, boundary mean/RMSE, currents, derivative commands. |
| `scripts/run_simulation_artifacts.py` | 211-351 | Runs simulation, plots final boundary/time series, renders frames/video if requested. |

## Controllers

| File | Lines | Behavior |
|---|---:|---|
| `tokamak_control/control/base.py` | 10-63 | Controller interface returns derivative commands in A/s. It does not expose absolute current `J_new`. |
| `tokamak_control/control/lqr_t15_zaitsev.py` | 142-200 | New Zaitsev-style controller computes `delta_Jdot`, accumulates into a derivative command, clips derivative command, and returns derivative command to plant. |
| `tokamak_control/control/lqr_t15_zaitsev.py` | 202-333 | Builds a discrete LQR model using current Python boundary sensitivities and `get_ip_B_row()`. It is not a literal MATLAB `GetC1/Getg` finite-difference port. |
| `tokamak_control/control/lqr_boundary.py` | 61-116 | Legacy LQR boundary controller solves one-step derivative command directly; no delta-Jdot accumulation. |
| `tokamak_control/control/lqr_current.py` | 57-92 | Legacy Ip controller solves one-step derivative command directly. |
| `tokamak_control/control/lqr_joint.py` | 59-116 | Legacy joint controller solves one-step derivative command directly. |
| `tokamak_control/control/hinf_boundary.py`, `hinf_current.py`, `hinf_joint.py` | full files | One-step robust controllers; not old MATLAB Hinf synthesis/state. |
| `tokamak_control/control/qp_joint.py` | full file | Modern constrained one-step QP derivative controller. |
| `tokamak_control/control/coil_replay.py` | 163-210 | Converts target absolute currents to derivative commands `(target-current)/dt`. |
| `tokamak_control/control/t15md_replay.py` | 112-156 | Replays exact next-step target currents through derivative conversion. This is the clearest current equivalent of old absolute-current interface. |
| `tokamak_control/control/learned_magnetic_controller.py` | 40-186 | Learned controller supports several action contracts, including delta-Jdot variants; returns derivative commands to plant. |

## Bridge / RL-Facing Session

| File | Lines | Behavior |
|---|---:|---|
| `tokamak_control/bridge/simulation_session.py` | 192-258 | Programmatic `step_derivatives()` accepts active derivative commands and steps the plant. |
| `tokamak_control/bridge/simulation_session.py` | 325-412 | Snapshots expose current, derivative, boundary, reference, and margins to external callers. |
| `tokamak_control/bridge/types.py` | full file | Bridge dataclasses use derivative-action vocabulary. |

## Current Test Coverage Related To Parity

| File | Lines | Behavior |
|---|---:|---|
| `tests/test_plasma_model_parity.py` | 43-92 | Tests CPU old Ip formula and no plant clipping/lag. |
| `tests/test_plasma_model_parity.py` | 95-142 | Tests zero-command passive baseline, B row finite difference, GPU parity if CUDA exists. |
| `tests/test_legacy_boundary.py` | 16-118 | Tests legacy contour config, closed-contour success, bbox rejection, limiter independence. |
| `tests/test_lqr_t15_zaitsev.py` | 73-188 | Tests DARE gain, delta-Jdot accumulation/clipping, fallback gain, registry. |

## Current High-Level Data Flow

```text
config TOML
  -> load_config()
  -> PlasmaModel/GpuPlasmaModel
  -> initial boundary extraction
  -> scenario reference from base radii and Ip0
  -> controller receives measured state/reference
  -> controller returns derivative command Jdot [A/s]
  -> optional RealismRuntime changes Jdot
  -> model.step(Jdot)
  -> boundary extraction / radii metrics
  -> RunWriter artifacts / plots / video
```

This differs from tokamak-sim-0's runtime interface:

```text
MATLAB controller computes delta_Jdot
  -> accumulates Jdot
  -> computes absolute next currents J_new = J_old + dt*Jdot
  -> writes J_new files
  -> C++ computes Jdot = (J_new-J_old)/dt internally
```

The plant math can be equivalent, but the interface is not the same.
# Superseded Note

This file contains historical audit notes from before the active simulator API
was changed to `step_currents(...)` and before Ip was made a causal state. For
current Ip/Jdot/delta-Jdot semantics, use
`../ip_jdot_semantics_audit.md`.
