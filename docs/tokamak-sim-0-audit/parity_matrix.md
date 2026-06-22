# Tokamak-Sim-0 vs Current Tokamak-Sim Parity Matrix

Verdicts:

- `same`: mathematically equivalent.
- `intentional T15 change`: user-approved or machine-specific difference.
- `equivalent but different interface`: same math if a conversion is applied.
- `mismatch`: behavior differs from tokamak-sim-0.
- `likely bug`: mismatch is probably accidental and can affect results.
- `unclear`: needs a numeric probe or source clarification.

## Plant And Ip Evolution

| Topic | Tokamak-sim-0 evidence | Current evidence | Verdict | Notes |
|---|---|---|---|---|
| Plant command interface | Old C++ `plasma_t::act` reads absolute `J_PFC_new`/`J_SOL_new` and computes current change internally (`littlescope undode cpp files.txt:297-346`). MATLAB `GetNewPsiMatrix` writes absolute currents (`littlescope undone m files.txt:2569-2603`). | Current controller interface returns derivative commands (`control/base.py:10-63`), runtime calls `model.step(...derivs...)` (`cli/run_simulation.py:720-747`), bridge exposes derivative action (`bridge/simulation_session.py:192-258`). | equivalent but different interface | This was under-emphasized before. Equivalence requires `Jdot=(J_new-J_old)/dt`. Controllers that reason in absolute-current space must do this conversion explicitly. |
| Current integration | Old commits `J_new` directly after reading absolute next currents (`littlescope undode cpp files.txt:297-346`). | Current `PlasmaModel.step` integrates `J_new=J_old+dt*Jdot` (`plasma_model.py:275-329`). | equivalent but different interface | Equivalent if input derivative is exactly old current difference divided by dt. |
| Ip passive baseline | Old recomputes `Ip0 * exp(-t/(Sigma*L))` each step, not recursive previous-Ip decay (`littlescope undode cpp files.txt:297-346`). | Current CPU/GPU/batched recompute from reset `Ip0` and absolute next time (`plasma_model.py:301-309`, `gpu_plasma_model.py:101-110`, `batched_gpu_simulator.py:109-123`). | same | This parity edit is present. |
| Ip coil drive variable | Old coil drive is proportional to `(J_new-J_old)/t_step`, i.e. derivative/current-change (`littlescope undode cpp files.txt:297-346`). | Current coil drive uses applied derivative command directly (`plasma_model.py:301-316`). | equivalent but different interface | Same only if controller/runtime derivative equals old current change divided by dt. |
| Ip coupling sign | Old pasted C++ uses a negative sign in the coil-drive term (`littlescope undode cpp files.txt:297-346`). | Current sign is config field `ip_coupling_sign`, T15 config uses `+1.0` (`configs/T15MD_new_data.toml:26`). | intentional T15 change | User explicitly approved keeping configured signs. |
| `mu0` placement | Old Green has no `mu0`; `compute_Psi` multiplies whole psi sum by `mu0` (`littlescope undode cpp files.txt:349-385`, `881-943`). | Current Green function omits `mu0`, `_compose_psi` multiplies by `mu0` (`green.py:22-95`, `plasma_model.py:331-341`). | same | Good parity. |
| `Sigma * L` | Old `inductance_law` forces `Sigma=6e8`, `Sigma*L=0.03s` (`littlescope undode cpp files.txt:59-70`). | T15 config uses fitted `sigma` and `inductance_L`, giving about `1.29s` (`configs/T15MD_new_data.toml:17-18`). | intentional T15 change | User explicitly approved fitted T15 tau. |
| Plant-side actuator lag/clipping | Old C++ plant takes `J_new` and has no lag/current/derivative clipping in the core plant (`littlescope undode cpp files.txt:297-346`). | Current `step` does not apply lag/clipping; limits remain metadata (`plasma_model.py:275-329`, `gpu_plasma_model.py:89-122`). | same | Some stale helper methods/docstrings remain. |
| CPU arbitrary Green sampling grid origin | Old interpolation uses old shifted grid coordinates through `grid_t` (`littlescope undode cpp files.txt:388-477`). | `PlasmaModel._bilinear_sample_slice` indexes from `grid.r.start`/`grid.z.start`, not `grid.r.coords()[0]`/`grid.z.coords()[0]` (`plasma_model.py:389-417`). | likely bug | Affects sampled Green functions used by sensitivities/controllers after grid parity change. |
| GPU sampling grid origin | Old interpolation uses shifted grid coordinates (`littlescope undode cpp files.txt:388-477`). | `torch_sampling.py` and `boundary_gpu._axis_search` index from raw start (`torch_sampling.py:3-61`, `boundary_gpu.py:121-146`). | likely bug | CPU/GPU parity and limiter/ray sampling can differ after grid parity change. |

## Boundary Extraction And Boundary Metrics

| Topic | Tokamak-sim-0 evidence | Current evidence | Verdict | Notes |
|---|---|---|---|---|
| Old boundary contour selection | `PlasmaBoundary` + `LineIsOk`: single contour level search; first closed contour with bbox containing center; longest accepted contour retained (`littlescope undone m files.txt:1359-1398`, `2605-2630`). | `legacy_contour` implements this mode in CPU (`boundary_cpu.py:213-318`). | same for selected mode | Legacy mode is selectable, not the default T15 config. |
| Default T15 boundary mode | Old has no limiter-aware selection like current. | T15 config uses `boundary.mode="limited"` (`configs/T15MD_new_data.toml:29-30`). | mismatch unless intentionally selected | To run old-parity boundary extraction, config or runtime must use `legacy_contour`. |
| GPU legacy contour | Old boundary is contour-based. | Dispatcher falls back to CPU for `legacy_contour`; GPU function rejects legacy mode (`boundary.py:43-119`, `boundary_gpu.py:51-77`). | same for dispatcher path | Fine for single-run dispatch; batched GPU is separate. |
| Batched GPU boundary mode | Old does not have batched GPU. | `BatchedGpuTokamakSimulator` hardcodes `boundary_mode="limited"` in `fixed_angle_boundary_gpu` (`batched_gpu_simulator.py:150-159`). | likely bug for legacy parity | RL/batched runs cannot use old boundary extraction semantics today. |
| Boundary error measurement | Old chooses measurement points as closest initial-boundary points to PFC locations, converts to angles/radii, and uses `GetAngleBuddies` interpolation (`littlescope undone m files.txt:3-93`, `405-471`, `2287-2337`). | Current controllers/metrics use fixed angle ray intersections and scenario `ref_radii` (`coordinates.py:15-88`, `run_simulation.py:431-456`, `lqr_t15_zaitsev.py:202-333`). | mismatch | Extraction parity does not imply control/error parity. This is likely why new boundary mode may not change LQR behavior as expected. |
| Boundary sensitivity | Old `GetC1` perturbs absolute currents and re-runs C++/boundary extraction (`littlescope undone m files.txt:2348-2375`). | Current `boundary_sensitivities` uses local implicit contour gradient and Green samples (`linearization.py:82-149`). | mismatch | Faster, but not old mathematical behavior. |

## Control

| Topic | Tokamak-sim-0 evidence | Current evidence | Verdict | Notes |
|---|---|---|---|---|
| Old controller command variable | Old MATLAB LQR/Hinf computes `delta_Jdot`, accumulates `J_Derivatives += u`, then writes absolute `J_new` (`littlescope undone m files.txt:1594-1792`, `1841-1978`, `1982-2129`, `2132-2218`). | New `lqr_t15_zaitsev` computes delta, accumulates derivative command, clips, and returns derivative command (`lqr_t15_zaitsev.py:142-200`). | equivalent but different interface | The new controller is closer, but still feeds current Python plant derivative interface. |
| Legacy LQR/Hinf controllers | Old LQR/Hinf are state-space, multi-state, delta-Jdot controllers (`littlescope undone m files.txt:1594-2218`). | `lqr_boundary`, `lqr_current`, `lqr_joint`, `hinf_*` solve one-step derivative controls directly (`lqr_joint.py:59-116`, `hinf_joint.py:95-235`). | mismatch | These should be treated as modern legacy baselines, not tokamak-sim-0 equivalents. |
| `lqr_t15_zaitsev` state | Old joint state includes boundary errors, Ip error, `J_Derivatives*t_step`, current/drift terms (`littlescope undone m files.txt:1594-1792`). | New state is `h`, `dt*previous_command`, `drift`; uses current boundary radii and analytic sensitivities (`lqr_t15_zaitsev.py:202-333`). | mismatch / partial port | Book-inspired, not a line-by-line old MATLAB port. |
| Gain solve/reuse | Old code has `useold`, `KFREQ`, and variants for reusing or recomputing gains (`littlescope undone m files.txt:1594-1792`). | New controller recomputes DARE and falls back to cached gain on failure (`lqr_t15_zaitsev.py:73-93`, `142-200`). | mismatch | May be acceptable, but not old behavior. |
| Replay exact current interface | Old manual/file exchange writes absolute currents. | `t15md_replay` targets table current at `t+dt` and returns derivative `(target-current)/dt` (`t15md_replay.py:112-156`). | equivalent but different interface | This is the clearest current Python equivalent of old absolute-current stepping. |
| Learned controller action state | Old controller tracks accumulated `J_Derivatives`. | Learned controller supports delta-Jdot contracts and separate previous derivative command in v3 (`learned_magnetic_controller.py:40-186`). | modern extension | Not tokamak-sim-0, but can be made conceptually consistent. |

## Artifacts And Runtime

| Topic | Tokamak-sim-0 evidence | Current evidence | Verdict | Notes |
|---|---|---|---|---|
| File-lock runtime | Old C++/MATLAB communicate with files/locks (`littlescope undode cpp files.txt:675-748`, `littlescope undone m files.txt:2569-2603`). | Current Python has direct in-process API (`run_simulation.py:1062-1336`). | intentional implementation change | Does not affect math if formulas match. |
| Stored signals | Old writes separate text arrays/files for psi, currents, errors. | Current `RunWriter` writes NPZ/CSV with currents, derivatives, refs, boundaries, events (`data_io.py:60-587`). | equivalent but different interface | Current artifacts are richer. |
| Visualization | Old MATLAB GUI/plots. | Current scripts save PNG frames/video and time-series (`run_simulation_artifacts.py:211-351`, `plotting.py:569-830`). | intentional implementation change | Not a math parity issue. |
| Scenario references | Old references often GUI/current-table based; measurement points derived from initial boundary. | Current scenario system generates/runs table/synthetic refs (`scenarios.py`, `ip_trajectories.py`). | intentional extension | References can be old-equivalent only in specific table/replay cases. |
# Superseded Note

This table is a historical snapshot and contains entries from before the active
simulator API was changed to absolute next-current commands and before Ip was
made a causal state. For current Ip/Jdot/delta-Jdot semantics, use
`../ip_jdot_semantics_audit.md`.
