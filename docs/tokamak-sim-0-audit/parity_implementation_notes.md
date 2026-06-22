# Tokamak-Sim-0 Parity Implementation Notes

This pass makes the active simulator/control path use the tokamak-sim-0 style
mathematical contract.

## Resolved Items

- Public plant commands are absolute next coil currents.
  - Active runtime controllers return `ControlAction(pfc_currents_next, sol_currents_next)`.
  - CPU, GPU, batched GPU, and bridge/session paths call `step_currents(...)`.
  - The plant derives `Jdot = (J_next - J_now) / dt` internally for Ip coupling and diagnostics.

- Ip evolution now uses causal stateful replay physics.
  - The LittleScope scalar overwrite was found to produce a derivative-shaped
    pseudo-Ip for replay, so active simulation interprets the coil-drive term
    as part of `dIp/dt` rather than as Ip itself.
  - The active formula is
    `dIp/dt = -Ip/tau + K dot Jdot`; no free bias/intercept is used.
  - `Ip(t_next) = Ip(t) + dt * dIp/dt`, and `psi` is composed from the updated
    dynamic `Ip(t_next)`.
  - Current and derivative limits remain controller/diagnostic metadata, not
    plant-side clipping/lag mechanisms.

- Grid sampling uses old shifted grid coordinates.
  - CPU bilinear sampling, GPU torch sampling, and helper paths use `grid.coords()[0]`.

- Split SOL actuators are weighted volumetric actuators.
  - The active T15 model exposes three SOL runtime currents.
  - SOL0/SOL1/SOL2 are represented by 30/90/30 point elements with weights
    summing to 1 per actuator, so the commanded current is distributed across
    the split geometry rather than duplicated into every point.

- Boundary extraction now has explicit strict and tracked legacy modes.
  - `legacy_contour` is the strict LittleScope-style contour search with no limiter.
  - `legacy_contour_limited` is the same strict contour search with limiter containment.
  - `tracked_flux_contour` continues the previous flux-surface identity by tracking
    contour level and continuity, using a strict legacy base mode only for
    initialization/reset.
  - Active T15 config uses `tracked_flux_contour` with
    `base_mode = "legacy_contour_limited"`.
  - GPU/batched paths dispatch through CPU legacy contour extraction for correctness
    when boundary contours are needed.

- Old-style boundary measurement geometry is active for simulator control/metrics.
  - Initial measurement angles are selected from the reset contour by nearest PFC actuator centroid.
  - Runtime measurements use periodic angle/radius interpolation.

- Active controllers are restricted to old-parity-compatible paths.
  - Registry exposes replay controllers, `lqr_t15_zaitsev`, and the learned controller only.
  - Modern one-step LQR/Hinf/QP controllers are not exposed as parity baselines.
  - `lqr_t15_zaitsev` finite-differences the actual `step_currents(...)` transition and legacy boundary extraction.

- Learned-policy export/runtime contracts are invalidated and bumped.
  - New learned bundles must use `action_contract = "delta_jdot_derivative_command_v3"`.
  - Observation schema is `controller_state_v3`.
  - Old derivative-action bundles/checkpoints are rejected.

## Remaining Scope Notes

- Derivative diagnostics are still written to artifacts as derived quantities.
  This is intentional: the public command is absolute next current, while `Jdot`
  remains the physically useful derived signal.

- Legacy modern-controller source files may still exist in the tree, but they are
  not registered for active parity workflows and will fail under the new
  `ControlAction` signature if called directly.

- Existing run outputs, old learned exports, and old RL checkpoints are not
  comparable to runs generated after this parity pass.
