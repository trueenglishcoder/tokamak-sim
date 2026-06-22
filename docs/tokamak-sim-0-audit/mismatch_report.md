# Tokamak-Sim-0 Mismatch Report

This report lists actionable mismatches found in the reconstruction pass. It
does not patch behavior. Each item includes evidence and a proposed next check.

## P0: Ip Coil-Drive Time-Scaling Mismatch

**Verdict:** likely bug unless the current T15 coupling arrays were deliberately
re-fit to absorb a `1 / t_step` factor.

Old C++ computes the coil-driven Ip term from current change divided by
`t_step`:

- `_local_archive/littlescope undode cpp files.txt:319-325`

Current Python integrates derivative commands into current changes, then uses
raw `delta_current` in the Ip update:

- `tokamak_control/core/plasma_model.py:304-323`

Current one-step derivative sensitivity also multiplies by `t_step`:

- `tokamak_control/core/plasma_model.py:236-245`

**Impact:** With `t_step=0.001`, derivative-command authority over Ip may be
about `1000x` smaller than the old model for the same Green coupling. This can
explain controllers failing to track Ip or requiring unrealistic coil behavior.

**Next check:** Run a one-step parity probe:

1. Pick a reset state and one coil.
2. Apply `Jdot = 1 A/s` for one step.
3. Compare old formula `mu0*sigma/R0 * g * Jdot` to current model response.
4. Decide whether current `g/g2` should be divided by `dt`, whether step should
   use derivative directly, or whether T15 fitted couplings intentionally
   changed units.

## P0: Passive Ip Baseline Is Recursive Instead Of Original-Time Based

**Verdict:** mismatch.

Old C++ recomputes passive Ip from original `Ip0` and absolute time:

- `_local_archive/littlescope undode cpp files.txt:313-320`

Current Python decays from the current state:

- `tokamak_control/core/plasma_model.py:231-245`
- `tokamak_control/core/plasma_model.py:317-323`

**Impact:** The recursive model accumulates previous coil-driven Ip changes into
future passive decay. The old model does not; it adds the current step's
inductive effect on top of the original passive baseline. This changes long-run
Ip dynamics substantially.

**Next check:** Plot no-control Ip and one-pulse Ip under both recurrences for
the same `sigma`, `L`, `g`, and `dt`.

## P1: Grid Centering Semantics Changed

**Verdict:** mismatch.

Old `grid_t` shifts the internal grid start so the configured magnetic center
falls halfway between samples:

- `_local_archive/littlescope undode cpp files.txt:388-410`

Current config uses explicit ranges and center metadata. The active T15 config
declares center/sign/physics fields but does not reproduce the old start-shift
rule:

- `configs/T15MD_new_data.toml:17-27`

**Impact:** Green samples, contour extraction, and sensitivities can differ
even if machine dimensions appear equivalent.

**Next check:** Build old-style and current-style R/Z arrays for the same raw
config and compare plasma-center index, Green values, and contour radius.

## P1: Boundary Extraction Is Not Old-Parity

**Verdict:** resolved for selectable extraction mode.

Old MATLAB selects the longest acceptable contour at the center psi level:

- `_local_archive/littlescope undone m files.txt:1357-1399`

Current boundary extraction keeps the limiter/divertor-aware algorithms for
the default modes, and now also exposes `boundary.mode="legacy_contour"`:

- `tokamak_control/geometry/boundary_cpu.py`

That mode reproduces the old MATLAB `PlasmaBoundary + LineIsOk` contour search
in index space and intentionally ignores limiter geometry when accepting or
rejecting contours.

**Remaining exception:** boundary metrics/errors are still the current
angle/radius machinery, not the old MATLAB closest-point error logic.

## P1: LQR Sensitivities Use A Different Model

**Verdict:** mismatch.

Old MATLAB estimates sensitivities by experimental plant calls and observed
previous deltas:

- `_local_archive/littlescope undone m files.txt:1638-1660`
- `_local_archive/littlescope undone m files.txt:2476-2545`

Current `lqr_t15_zaitsev` builds an analytic local model from Green/contour
sensitivities:

- `tokamak_control/control/lqr_t15_zaitsev.py:202-332`
- `tokamak_control/control/linearization.py:143-148`

**Impact:** The new LQR may be book-shaped, but it is not using the same local
identification mechanism as the old working code. This matters when the
analytic model is badly conditioned.

**Next check:** Add a finite-difference sensitivity backend for LQR and compare
its `A/B` to the analytic backend on one reset state.

## P1: Actuator Lag And Clipping Are New Plant Dynamics

**Verdict:** intentional safety/realism change, not old parity.

Old C++ accepts absolute currents from MATLAB and commits them:

- `_local_archive/littlescope undode cpp files.txt:297-346`

Current Python clips derivative commands, applies lag, clips derivatives again,
then clips currents:

- `tokamak_control/core/plasma_model.py:274-307`

Active T15 lag:

- `configs/T15MD_new_data.toml:20`

**Impact:** Controllers tuned against the old plant can become too weak or too
late when lag and clipping are inserted. This also changes what "delta-Jdot"
means unless previous command state is handled carefully.

**Next check:** Run a zero-lag/no-current-clip compatibility mode for one
trajectory and compare with current default.

## P1: Sign Conventions Need Provenance

**Verdict:** unclear.

Old psi composition uses positive `Ip * G_Arr_0` and old Ip update subtracts the
inductive term:

- `_local_archive/littlescope undode cpp files.txt:319-325`
- `_local_archive/littlescope undode cpp files.txt:349-365`

Current active T15 config flips both conventions:

- `configs/T15MD_new_data.toml:26-27`
- `tokamak_control/core/plasma_model.py:317-354`

**Impact:** If signs were calibrated from T15 data, this is fine. If they were
introduced to compensate another unit mismatch, they can hide errors.

**Next check:** Document the calibration source for `ip_coupling_sign=1` and
`plasma_psi_sign=-1`, then verify one real T15 coil step direction.

## P2: Sigma/L Are No Longer Old Constants

**Verdict:** intentional T15 change, but should be explicitly justified.

Old runtime forces `Sigma=6e8` and `Sigma*L=0.03s`:

- `_local_archive/littlescope undode cpp files.txt:59-66`
- `_local_archive/littlescope undode cpp files.txt:297-307`

Current T15 config uses:

- `configs/T15MD_new_data.toml:17-18`

which gives a passive time constant of roughly `1.29s`.

**Impact:** No-control Ip decay is completely different from the old model. That
may be correct for T15, but it must not be treated as old parity.

**Next check:** Compare no-control current T15 Ip decay against real shot decay
segments.

## P2: Current Replay/Feedforward Paths From Old MATLAB Are Missing

**Verdict:** mismatch for old ControlBoundary/ControlCurrent modes.

Old MATLAB can replay measured `PFCderiv` or `SOLderiv` arrays while controlling
the other coil set:

- `_local_archive/littlescope undone m files.txt:1794-2130`

Current scenarios/controllers are cleaner, but do not reproduce those exact
hybrid feedforward modes by default.

**Impact:** A baseline comparison against old ControlBoundary/ControlCurrent is
not valid unless these feedforward paths are recreated.

**Next check:** Add explicit current/derivative table feedforward scenarios for
baseline work.

## P2: LQR Gain Reuse Differs

**Verdict:** mismatch/performance-risk.

Old MATLAB has practical reuse/caching branches for LQR/H-infinity gains:

- `_local_archive/littlescope undone m files.txt:1624-1686`

Current `lqr_t15_zaitsev` solves DARE inside `compute_control`:

- `tokamak_control/control/lqr_t15_zaitsev.py:142-188`

**Impact:** Re-solving can fail on ill-conditioned steps and can make behavior
depend on numerical DARE robustness rather than controller design.

**Next check:** Cache the last valid gain and optionally expose finite-difference
gain recomputation cadence.

## P2: Measurement-Point Error Semantics Need A Direct Probe

**Verdict:** unclear.

Old MATLAB computes signed error by target/actual closest points and center-side
logic:

- `_local_archive/littlescope undone m files.txt:2377-2518`

Current code has more structured boundary/radius metrics, but this audit has not
yet proven point-for-point equivalence.

**Impact:** Reward and LQR can optimize a different shape error than the old
controller.

**Next check:** On one psi field, export old-style measurement radii and current
metric radii for the same angles.

## Summary Of Fix Candidates

## Resolution Note

The plant parity pass resolves the top simulator-core mismatches by changing
CPU, single-GPU, and batched-GPU plant steps to the old Little SCoPE recurrence:
Ip is driven by applied `Jdot`, passive Ip is computed from reset `Ip0` at
absolute time, and plant-side lag/current/derivative clipping is no longer part
of the core model. The grid coordinate generator is also changed to the old
center-half-cell alignment. Boundary extraction, boundary metrics, fitted T15
`sigma/L`, and configured coupling signs remain intentionally different.

Recommended order for a later implementation pass:

1. Add an old-style boundary extractor behind a diagnostic flag if exact legacy
   contour selection is ever needed.
2. Add a finite-difference LQR sensitivity backend.
3. Document or re-fit sign conventions and Sigma/L from real T15 data.
