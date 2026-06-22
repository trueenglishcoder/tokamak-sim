# Current Tokamak-Sim Pipeline

`tokamak-sim` is currently treated as the plant source of truth for T15 replay and
RL training.

## Plant Contract

- Public simulator/control input is absolute next-step coil current:
  `step_currents(J_next)`.
- The simulator derives `Jdot = (J_next - J_now) / t_step` internally.
- Plasma current `Ip` is a state advanced by the active plant update and `psi`
  is composed from the updated true `Ip`.
- Current and derivative limits are controller, reward, and diagnostic metadata;
  they are not hidden plant-side clipping in exact replay.

## Boundary Data

- The active replay-boundary dataset is:
  `runs/t15md_limited_replay_dataset/`.
- Smoothed per-shot boundary references are saved as
  `lqr_boundary_reference_<SHOT>_smoothed.npz`.
- Each reference contains 32 boundary radii samples, Ip, time, and
  boundary-found flags. RL consumes this dataset through the matching T15 shot id.

## Replay

- Exact T15 replay commands the next absolute current from the T15 coil table.
- Replay outputs should include video, time-series CSV, events CSV, and manifest
  files so the same boundary data can be reused as an RL reference source.
