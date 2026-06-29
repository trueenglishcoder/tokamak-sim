# Tokamak-Sim-0 Audit Read Coverage

Date: 2026-06-21

This audit treats the current working tree as truth, including dirty and untracked files. Generated run outputs, virtualenvs, caches, and bulk data rows are excluded from manual line-by-line commentary unless they affect a parity formula. Bulk CSV data was reviewed by schema/usage path rather than row-by-row.

## Old Source Inputs

| File | Lines | Status | Notes |
|---|---:|---|---|
| `_local_archive/littlescope undode cpp files.txt` | 1183 | fully read | Contains `snowfed_plasma.cpp`, `LittleCppSCoPE.cpp`, `snowfed.cpp`, `working_directory.cpp`, `george_green.cpp`, `snowfed_array.cpp`. Main evidence for plant equations, grid alignment, Green function, file-lock runtime, and absolute-current plant interface. |
| `_local_archive/littlescope undone m files.txt` | 2719 | fully read | Contains `LoadInfo`, `ControlBoth`, `ControlBoundary`, `ControlCurrent`, `ControlLQRKernel`, `PlasmaBoundary`, `LineIsOk`, `GetC1`, `Getg`, `Getg2`, `GetNewPsiMatrix`, plotting/helpers. Main evidence for controller semantics and boundary metrics. |
| `_local_archive/MMTPEen_ZaitsevFS_2014_20210521.pdf` | PDF | excerpt read | Relevant text around section 2.3.1 and LQR/Hinf references was inspected through extracted text. Used as controller-structure context, not as source for tokamak-sim-0 runtime. |

## Current Source Files

`tokamak_control` Python files were read by subsystem, not keyword search.

| File | Lines | Status | Notes |
|---|---:|---|---|
| `tokamak_control/core/plasma_model.py` | 417 | fully read | CPU plant, Ip update, psi composition, sensitivity helpers, bilinear Green sampling. |
| `tokamak_control/core/gpu_plasma_model.py` | 158 | fully read | Single-GPU plant. |
| `tokamak_control/core/batched_gpu_simulator.py` | 186 | fully read | Batched RL/GPU plant and boundary dispatch. |
| `tokamak_control/core/grid.py` | 101 | fully read | Old half-cell grid alignment implementation. |
| `tokamak_control/core/green.py` | 128 | fully read | Green function and grouped actuator summation. |
| `tokamak_control/core/coils.py` | 152 | fully read | Grouped coils/actuators and weights. |
| `tokamak_control/core/plasma_state.py` | 68 | fully read | State representation and stale actuator-lag wording. |
| `tokamak_control/core/torch_sampling.py` | 61 | fully read | GPU field sampling; grid-origin mismatch noted. |
| `tokamak_control/geometry/boundary.py` | 136 | fully read | Boundary dispatch, CPU fallback for legacy contour. |
| `tokamak_control/geometry/boundary_cpu.py` | 895 | fully read | Limited/diverted and legacy contour extraction. |
| `tokamak_control/geometry/boundary_gpu.py` | 203 | fully read | GPU limited/diverted boundary path. |
| `tokamak_control/geometry/boundary_common.py` | 201 | fully read | Shared contour/limiter helpers. |
| `tokamak_control/geometry/coordinates.py` | 151 | fully read | Current ray/radius metrics. |
| `tokamak_control/geometry/distances.py` | 71 | fully read | Current closest-point distances. |
| `tokamak_control/geometry/limiters.py` | 53 | fully read | Built-in limiter shapes. |
| `tokamak_control/geometry/parametric_boundary.py` | 588 | reviewed | Modern reference generation and fitting, not tokamak-sim-0 plant math. |
| `tokamak_control/geometry/xpoints.py` | 84 | fully read | Current x-point diagnostics. |
| `tokamak_control/config/settings.py` | 107 | fully read | Physics/config schema. |
| `tokamak_control/io/config_io.py` | 549 | fully read | TOML loading, live config semantics, active masks, boundary mode. |
| `tokamak_control/config/scenarios.py` | 1368 | fully read | Scenario/reference generation, Ip table, shot follow, synthetic references. |
| `tokamak_control/config/ip_trajectories.py` | 424 | fully read | Synthetic Ip trajectory generation. |
| `tokamak_control/control/base.py` | 64 | fully read | Controller output contract. |
| `tokamak_control/control/registry.py` | 470 | fully read | Controller registry/runtime inputs and launch params. |
| `tokamak_control/control/linearization.py` | 149 | fully read | Current analytic boundary sensitivities. |
| `tokamak_control/control/lqr_boundary.py` | 118 | fully read | Legacy one-step derivative LQR. |
| `tokamak_control/control/lqr_current.py` | 92 | fully read | Legacy one-step Ip LQR. |
| `tokamak_control/control/lqr_joint.py` | 116 | fully read | Legacy one-step joint LQR. |
| `tokamak_control/control/lqr_t15_zaitsev.py` | 333 | fully read | New untracked Zaitsev-style controller. |
| `tokamak_control/control/hinf_boundary.py` | 149 | fully read | One-step Hinf boundary controller. |
| `tokamak_control/control/hinf_current.py` | 147 | fully read | One-step Hinf Ip controller. |
| `tokamak_control/control/hinf_joint.py` | 235 | fully read | One-step Hinf joint controller. |
| `tokamak_control/control/qp_joint.py` | 234 | fully read | One-step QP controller. |
| `tokamak_control/control/coil_replay.py` | 210 | fully read | Target-current replay to derivative command. |
| `tokamak_control/control/t15md_replay.py` | 156 | fully read | Exact applied-current replay via derivative conversion. |
| `tokamak_control/control/learned_magnetic_controller.py` | 409 | fully read | Learned-controller export runtime and action contracts. |
| `tokamak_control/control/replay_table.py` | 105 | reviewed | Numeric table interpolation helpers. |
| `tokamak_control/cli/run_simulation.py` | 1336 | fully read | Main runtime loop, controller call, plant stepping, artifacts. |
| `tokamak_control/bridge/simulation_session.py` | 561 | fully read | Programmatic stepping interface used by RL bridge. |
| `tokamak_control/bridge/types.py` | 126 | reviewed | Bridge dataclasses and derivative-action interface. |
| `tokamak_control/realism/runtime.py` | 293 | fully read | Optional actuation/sensor realism around plant. |
| `tokamak_control/realism/types.py` | 135 | reviewed | Realism schema. |
| `tokamak_control/io/data_io.py` | 661 | fully read | Artifact writer and run archive schema. |
| `tokamak_control/viz/plotting.py` | 830 | fully read | Plot/frame/video artifact generation. |
| `tokamak_control/metrics/*.py` | 97 | reviewed | Current/derivative/tracking metrics. |
| `tokamak_control/compute.py` | 71 | reviewed | Backend selection. |
| `tokamak_control/diagnostics.py` | 137 | reviewed | Diagnostics helpers. |
| `tokamak_control/experiments/disturbances.py` | 274 | reviewed | Optional perturbations outside old parity core. |
| `tokamak_control/cli/main.py` | 151 | reviewed | CLI entrypoint. |

## Current Configs, Scripts, Tests, Docs

| Path | Lines | Status | Notes |
|---|---:|---|---|
| `configs/T15MD_new_data.toml` | 843 | fully read | Live T15 plant config. Contains fitted `sigma/L`, signs, boundary mode, current/derivative limits, split coil weights. |
| `configs/T15MD_new_data_replay_start_3859.toml` | 843 | reviewed | Same structure with replay start. |
| `configs/T15MD_new_data_zero_start.toml` | 843 | reviewed | Same structure with zero start. |
| `configs/ITER*.toml`, `configs/JET.toml` | 6767 | reviewed | Non-T15 configs; not core to T15 parity, but schema compared. |
| `configs/initial_states/*.toml` | 1800+ | reviewed | Runtime initial-state files; T15 shot files contain explicit Ip0, PFC currents, and SOL currents. |
| `scripts/run_simulation_artifacts.py` | 360 | fully read | Artifact CLI and video path. |
| `scripts/fit_sigma_L_gradient.py`, `fit_sigma_L_grid.py` | 1980 | reviewed | T15 fitted sigma/L generation; explains intentional tau mismatch. |
| Other `scripts/*.py` and `run_t15md_replay_batch.sh` | 1247 | reviewed | Data/prep/artifact utilities; not old runtime math. |
| `tests/test_plasma_model_parity.py` | 142 | fully read | Current plant parity tests. |
| `tests/test_legacy_boundary.py` | 118 | fully read | Current legacy contour tests. |
| `tests/test_lqr_t15_zaitsev.py` | 188 | fully read | Current Zaitsev LQR tests. |
| Existing tracked tests | 1923 | reviewed | Workflow, learned-controller, Hinf, boundary split, compute backend. |
| `docs/architecture.md`, `workflows.md`, `repository-layout.md`, `plasma_boundary_calculation.txt` | 573 | reviewed | Current public docs; some lag behind parity edits. |
| Existing `docs/tokamak-sim-0-audit/*.md` | 591 | reviewed | Earlier audit notes; superseded by this package where conflicting. |

## Dirty And Untracked Files Included

Dirty files from `git status --short` were included, especially:

- `tokamak_control/core/plasma_model.py`
- `tokamak_control/core/gpu_plasma_model.py`
- `tokamak_control/core/batched_gpu_simulator.py`
- `tokamak_control/core/grid.py`
- `tokamak_control/geometry/boundary*.py`
- `tokamak_control/control/registry.py`
- `tokamak_control/control/lqr_t15_zaitsev.py`
- `tests/test_legacy_boundary.py`
- `tests/test_lqr_t15_zaitsev.py`
- `tests/test_plasma_model_parity.py`

## Coverage Caveats

- `__pycache__`, `.venv`, generated `runs/`, and historical output folders were excluded.
- Bulk CSV data was not manually annotated row by row. Its role was audited through loaders, configs, initial-current files, and formulas that consume those arrays.
- The audit did not modify simulator behavior. It only created documentation and mismatch classification.
