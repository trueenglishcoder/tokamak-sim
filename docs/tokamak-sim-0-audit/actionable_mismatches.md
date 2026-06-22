# Actionable Mismatches

This is the decision-ready list. It prioritizes mismatches that can plausibly change controller behavior or explain why current LQR/RL conclusions do not match old tokamak-sim-0 intuition.

## P0: The Plant Interface Difference Must Be Made Explicit Everywhere

**Verdict:** equivalent but different interface.

**Evidence**

- Old C++ plant reads absolute next-current arrays and computes the derivative internally from current differences: `_local_archive/littlescope undode cpp files.txt:297-346`.
- Old MATLAB `GetNewPsiMatrix` writes absolute `J_PFC`, `J_SOL` files to C++: `_local_archive/littlescope undone m files.txt:2569-2603`.
- Current Python plant/controller interface returns derivative commands: `tokamak_control/control/base.py:10-63`, `tokamak_control/cli/run_simulation.py:670-747`, `tokamak_control/bridge/simulation_session.py:192-258`.

**Impact**

The current plant can be mathematically equivalent only when callers use:

```text
Jdot = (J_new - J_old) / dt
```

Controllers that were ported mentally from the old MATLAB code but return `Jdot` directly can silently drift away from old semantics. This was the main thing that was not stated clearly enough before.

**Fix**

- Add a first-class `step_currents(J_pfc_new, J_sol_new)` API to `PlasmaModel`, `GpuPlasmaModel`, and `SimulationSession`.
- Implement it as the canonical old-parity interface, internally calling `step(Jdot=(J_new-J_old)/dt)`.
- Update docs and controller comments to distinguish:
  - plant derivative interface,
  - old absolute-current interface,
  - controller delta-Jdot accumulation.
- For old-parity controllers, use `step_currents` in tests and diagnostics.

**Tests**

- `step_currents(J_new)` equals `step(Jdot=(J_new-J_old)/dt)` for CPU/GPU.
- Replay of a known absolute-current sequence gives identical Ip/psi through both interfaces.

## P0: Grid-Origin Bug In Current Sampling After Old Grid Alignment

**Verdict:** likely bug.

**Evidence**

- Old grid coordinates are shifted by center-half-cell alignment: `_local_archive/littlescope undode cpp files.txt:388-409`.
- Current `Grid1D.coords()` implements that shift: `tokamak_control/core/grid.py:40-76`.
- CPU arbitrary Green sampling ignores shifted coords and indexes from raw `start`: `tokamak_control/core/plasma_model.py:389-417`.
- GPU sampling also indexes from raw `start`: `tokamak_control/core/torch_sampling.py:3-61`.
- GPU axis search uses raw `grid.r.start`/`grid.z.start`: `tokamak_control/geometry/boundary_gpu.py:121-146`.

**Impact**

Any sensitivity or boundary routine that samples fields at physical points can be shifted by a fraction of a cell. This can make LQR/Hinf/QP gains and GPU boundary results disagree with the old-parity grid. It directly affects `boundary_sensitivities()` through `model.sample_green_*()`.

**Fix**

- Replace raw `grid.r.start`/`grid.z.start` with `grid.r.coords()[0]`/`grid.z.coords()[0]` in:
  - `PlasmaModel._bilinear_sample_slice`
  - `torch_sampling.bilinear_sample_torch`
  - `torch_sampling.bilinear_sample_torch_points`
  - `boundary_gpu._axis_search`
- Audit any other direct coordinate-index math.

**Tests**

- Bilinear sample at `grid.coords()[i]` returns exact array value for CPU and GPU.
- CPU/GPU boundary and Green samples match on shifted grid.
- LQR sensitivity finite difference matches sample-based sensitivity on a shifted grid fixture.

## P0: Old Boundary Extraction Does Not Mean Old Boundary Control/Error Geometry

**Verdict:** mismatch.

**Evidence**

- Old startup picks measuring points as closest boundary points to PFC positions and builds measuring angles/radii: `_local_archive/littlescope undone m files.txt:3-93`.
- Old step loop updates `radiusMeasuring`, `radiusMeasurement`, and `Errors` through `GetAngleBuddies`: `_local_archive/littlescope undone m files.txt:405-471`, `2287-2337`.
- Current metrics/controllers use fixed-angle ray intersections and scenario `ref_radii`: `tokamak_control/geometry/coordinates.py:15-88`, `tokamak_control/cli/run_simulation.py:431-456`, `tokamak_control/control/lqr_t15_zaitsev.py:202-333`.

**Impact**

Switching `boundary.mode="legacy_contour"` changes which contour is selected, but it does not make the controller objective old-equivalent. LQR may look unchanged because its error vector is still current Python radius/reference machinery, not MATLAB's measurement geometry.

**Fix**

- Add an optional `legacy_boundary_metrics` mode:
  - choose old measuring points/angles from initial boundary and coil geometry,
  - compute errors through a port of `GetAngleBuddies`,
  - expose those errors to old-parity controllers.
- Keep modern radius metrics for RL/new tooling, but label them as modern.

**Tests**

- Port a small MATLAB angle/radius fixture.
- Verify old-style measured errors match known `GetAngleBuddies` output.

## P1: Batched GPU Ignores `legacy_contour`

**Verdict:** likely bug for parity/RL.

**Evidence**

- Boundary dispatcher routes `legacy_contour` to CPU for single-run paths: `tokamak_control/geometry/boundary.py:43-119`.
- Batched GPU simulator hardcodes `boundary_mode="limited"`: `tokamak_control/core/batched_gpu_simulator.py:150-159`.

**Impact**

Single-run tests can claim legacy contour parity while batched/RL rollouts still use modern limited-boundary extraction. That can invalidate comparisons between local tokamak-sim and RL environment behavior.

**Fix**

- Add `boundary_mode` and `legacy_precision_index2` to `BatchedGpuTokamakSimulator`.
- If mode is `legacy_contour`, either:
  - route per-lane boundary extraction to CPU for diagnostics/eval, or
  - explicitly reject legacy mode for batched training with a clear error.

**Tests**

- Batched simulator respects `boundary_mode` in metadata.
- Legacy mode in batched path either matches CPU for small batch or fails loudly.

## P1: `lqr_t15_zaitsev` Is Not A Full Port Of Old MATLAB LQR

**Verdict:** mismatch / partial port.

**Evidence**

- Old LQR uses finite-difference `GetC1`, `Getg`, `Getg2` through plant calls and boundary re-extraction: `_local_archive/littlescope undone m files.txt:2348-2375`, `2476-2545`.
- Old joint state and A/B assembly are in `ControlBoth`: `_local_archive/littlescope undone m files.txt:1594-1792`.
- Current `lqr_t15_zaitsev` uses analytic boundary sensitivities, current Python radius errors, and a simplified state: `tokamak_control/control/lqr_t15_zaitsev.py:202-333`.

**Impact**

The new controller can be useful, but it is not "the fixed old LQR" yet. If it behaves similarly to previous LQR, this is expected because much of the objective/sensitivity machinery remains modern.

**Fix**

- Create a separate `lqr_t15_sim0` or `lqr_t15_zaitsev_legacy_metrics` implementation.
- Build sensitivities with the old finite-difference method first, even if slower:
  - perturb absolute next currents,
  - advance a copy/clone through old-parity plant,
  - extract legacy boundary,
  - compute old-style measurement errors.
- Only optimize speed after behavior matches.

**Tests**

- One-step predicted error change vs finite-difference actual change for each actuator.
- Controller `delta_Jdot` accumulation exactly matches old equations on a fixture.

## P1: Legacy LQR/Hinf/QP Controllers Are Modern One-Step Derivative Controllers

**Verdict:** mismatch.

**Evidence**

- Old LQR/Hinf controller routines accumulate derivative command (`J_Derivatives += u`) and then write absolute currents: `_local_archive/littlescope undone m files.txt:1594-2218`.
- Current `lqr_boundary`, `lqr_current`, `lqr_joint`, `hinf_*`, `qp_joint` solve direct one-step derivative commands: `tokamak_control/control/lqr_joint.py:59-116`, `tokamak_control/control/hinf_joint.py:95-235`, `tokamak_control/control/qp_joint.py:1-234`.

**Impact**

Calling these "LQR/Hinf baselines" is fine, but calling them tokamak-sim-0-faithful baselines is wrong.

**Fix**

- Rename docs/user-facing labels to "legacy one-step Python LQR/Hinf/QP".
- Reserve old-parity labels for controllers that use delta-Jdot accumulation and old metric/sensitivity path.

**Tests**

- Registry classification test: old-parity controllers vs modern controllers.

## P1: Current T15 Config Still Defaults To Modern Limited Boundary

**Verdict:** mismatch unless intentionally selected.

**Evidence**

- `configs/T15MD_new_data.toml` uses `boundary.mode="limited"`: lines 29-30.
- Legacy contour exists but must be selected explicitly: `tokamak_control/geometry/boundary_cpu.py:213-318`.

**Impact**

Local "new boundary" runs must actually use a config override or TOML with `mode="legacy_contour"`. Otherwise they still run modern limiter-aware boundary selection.

**Fix**

- Add a `configs/T15MD_new_data_legacy_contour.toml` or CLI override flag.
- Write the active boundary mode in output title/manifest and final console output.

**Tests**

- Running with legacy config records `boundary.mode=legacy_contour` in manifest.

## P2: Stale Actuator-Lag/Clipping Language And Profiling Keys

**Verdict:** mismatch in docs/diagnostics, not math.

**Evidence**

- `PlasmaState` docstring still says derivatives may differ from commands because of actuator lag: `tokamak_control/core/plasma_state.py:19-22`.
- `PlasmaModel` profiling keys still mention clip/lag stages even though `step` no longer applies them: `tokamak_control/core/plasma_model.py:102-110`, `275-329`.

**Impact**

Confusing diagnostics and user reasoning. It can make it look as if the plant is still clipping/lagging commands.

**Fix**

- Update docstrings and profiling key names to say limits/lag are outside core plant.

## P2: Reproducibility Risk From Untracked Live Config And Parity Files

**Verdict:** process risk.

**Evidence**

- `git status --short` shows untracked parity tests and `lqr_t15_zaitsev.py`, plus dirty core files.
- Live configs are essential to math but some may be untracked or locally edited.

**Impact**

Server/local behavior can diverge silently.

**Fix**

- Commit parity-sensitive source/tests/configs together.
- For each run, copy full config and git status into run metadata.

## Recommended Fix Order

1. Add explicit `step_currents()` and tests proving absolute-current/derivative equivalence.
2. Fix shifted-grid sampling in CPU/GPU samplers.
3. Add old-style boundary error/measurement mode.
4. Make batched GPU boundary mode explicit or reject legacy mode.
5. Build a truly old-parity LQR controller using finite-difference old metrics.
6. Clean stale docs and run metadata.

Until items 1-3 are done, LQR/RL behavior should not be used to conclude that tokamak-sim-0-style control cannot work on T15.
# Superseded Note

This file is a historical mismatch list. Several items here have since been
implemented or deliberately superseded, including the active `step_currents(...)`
plant API and causal Ip state. For current Ip/Jdot/delta-Jdot semantics, use
`../ip_jdot_semantics_audit.md`.
