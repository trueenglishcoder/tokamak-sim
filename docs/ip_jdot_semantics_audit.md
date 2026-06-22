# Ip, Jdot, and Delta-Jdot Semantics Audit

This note records the current active semantics after the LittleScope Ip
misunderstanding was corrected. It supersedes older audit tables that described
the previous derivative-command API or the LittleScope algebraic Ip expression
as active behavior.

## Source Cross-Check

- `_local_archive/littlescope_ip_lqr_migration_note.md` states that the old
  LittleScope C++ `Ip0 * exp(-t/tau) + K * (J_new - J_old) / dt` expression is
  derivative-like and must not be treated as the corrected plasma-current state.
- The active plant now treats that coil-drive expression as a `dIp/dt`
  contribution and integrates true Ip as a state.
- The Zaitsev/LittleScope LQR controller idea is still delta-Jdot control, but
  its response model must be built against the corrected stateful plant.

## Active Semantics

| Area | Active meaning |
| --- | --- |
| Public plant command | Absolute next coil currents, `step_currents(J_next)` |
| Derived coil derivative | `Jdot = (J_next - J_now) / t_step`; diagnostics and Ip drive |
| Plasma current | Stateful `Ip_next = Ip_now + dt * (-Ip_now/tau + K dot Jdot)` |
| Poloidal flux | Composed from updated true `Ip_next` and next coil currents |
| LQR control variable | `delta_Jdot`, accumulated into `Jdot_cmd`, then converted to absolute `J_next` |
| RL control variable | Normalized requested delta-Jdot; environment accumulates derivative command and sends absolute `J_next` |
| Learned export contract | `delta_jdot_derivative_command_v3`; old exports are intentionally incompatible |

## File Classification

### Correct Active Plant Path

- `tokamak_control/core/plasma_model.py`: CPU `step_currents` derives Jdot,
  integrates stateful Ip, and composes psi from the updated Ip.
- `tokamak_control/core/gpu_plasma_model.py`: single-GPU mirror of CPU
  semantics.
- `tokamak_control/core/batched_gpu_simulator.py`: batched-GPU mirror; it accepts
  absolute active currents and internally derives Jdot.
- `tokamak_control/bridge/simulation_session.py`: programmatic stepping uses
  `step_currents`.
- `tokamak_control/cli/run_simulation.py`: controllers return next currents,
  realism operates on next-current commands, and the model is advanced via
  `step_currents`.

### Correct Controller/Replay Path

- `tokamak_control/control/base.py`: `ControlAction` carries absolute next PFC
  and SOL currents.
- `tokamak_control/control/t15md_replay.py`: exact T15 replay targets table
  currents at `t + dt`.
- `tokamak_control/control/coil_replay.py`: generic replay emits next-current
  commands; `u_clip` limits the current step, not the plant API.
- `tokamak_control/control/learned_magnetic_controller.py`: v3 exports map actor
  delta-Jdot to accumulated derivative command, then to absolute next currents.
- `tokamak_control/control/lqr_t15_zaitsev.py`: corrected in this pass to build
  a dense response through the stateful plant and to remove old pseudo-Ip
  controller shortcuts.

### Correct RL Path

- `tokamak_rl_v2/env/batch_env.py`: actor delta-Jdot is accumulated into an
  applied derivative command; the simulator receives absolute next currents via
  `step_currents`.
- `tokamak_rl_v2/export/cli.py` and `tokamak_rl_v2/export/policy.py`: exported
  policies use the v3 delta-Jdot contract expected by the learned controller.
- `tokamak_rl_v2/networks/critic.py`: critic action metadata rejects old
  requested-delta contracts.

### Historical Or Diagnostic References

- `ip_decay_baseline_at` is a diagnostic passive-decay helper only.
- `pfc_current_derivs` and `sol_current_derivs` are applied-derivative
  diagnostics derived from consecutive current states.
- `docs/tokamak-sim-0-audit/*` contains historical comparison notes. Older
  tables may mention the pre-repair derivative-command API; use this file as
  the current source of truth.

### Fixed/Stale Traps

- `scripts/demo_learned_controller_objective.py` previously used removed
  `action.pfc_derivs`/`model.step(...)`; it now derives physical Jdot from
  next-current commands and calls `step_currents`.
- `scripts/fit_sigma_L_grid.py` and `scripts/fit_sigma_L_gradient.py` now name
  `step_currents` in help text.
- `tokamak_control/core/plasma_state.py` now documents derivative fields as
  diagnostics, not plant input commands.

## Guard Expectations

- Active code must not compute real Ip as
  `Ip0 * exp(-t/tau) + K dot Jdot`.
- Active code must not call `PlasmaModel.step(...)`,
  `GpuPlasmaModel.step(...)`, or `BatchedGpuTokamakSimulator.step(...)` except
  in tests that assert the removed API fails.
- New controller and learned-policy code must return absolute next currents to
  the simulator.
- Any future LQR sensitivity must be taken through `step_currents` or through
  an analytic derivative of the same stateful plant, not through the old
  LittleScope algebraic pseudo-Ip.
