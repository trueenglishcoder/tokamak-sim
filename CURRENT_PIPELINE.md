# Current Tokamak-Sim Pipeline

`tokamak-sim` is the plant/runtime source for the current T15 RL work and for
local presentation/replay artifacts.

## Plant Contract

- Public simulator/control input is absolute next-step coil current:
  `step_currents(J_next)`.
- The simulator derives `Jdot = (J_next - J_now) / t_step` internally.
- `Ip` is a plant state advanced step by step; `psi` is composed from the
  updated `Ip` and coil contributions.
- Current and derivative limits are controller, reward, validation, and
  diagnostic metadata. Exact replay does not apply hidden plant-side clipping.

## Final RL Dataset Contract

The active learned-policy pipeline uses:

```text
data/t15_data_new_trim50/
runs/t15md_trim50_plain_gpu_1e6_setup/
runs/t15md_limited_replay_dataset_trim50_gpu_plain_1e6/
```

Boundary references are fixed-angle GPU measurements generated with:

```text
boundary mode = plain GPU fixed-angle
legacy_precision_index2 = 1e-6
angles = 32
```

These references feed `tokamak-rl-v2` replay-window/oracle targets. They are not
the old smoothed LQR boundary files.

## Learned Controller Contract

Exported learned policies use:

```text
observation_kind = controller_state_v6
action_contract = absolute_jdot_command_v1
```

`LearnedMagneticController` runs the actor, clips the normalized absolute Jdot
command to derivative limits, computes `J_next = J_now + dt * Jdot`, and returns
absolute next currents to the simulator.

Zaitsev/LQR diagnostics remain available and may use delta-Jdot internally, but
that is not the learned-policy contract.

## Canonical Replays

Exact T15 replay commands the next absolute current from the T15 coil table.
Replay outputs contain time-series CSVs, events, manifests, boundary plots, and
optional frames/video.

For training references, regenerate trim50 plain GPU 1e-6 replays first; then
build oracle targets from `tokamak-rl-v2`.
