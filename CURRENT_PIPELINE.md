# Current Tokamak-Sim Pipeline

`tokamak-sim` is the plant/runtime source for the current T15 RL experiment.

## Plant Contract

- Public simulator/control input is absolute next-step coil current:
  `step_currents(J_next)`.
- The simulator derives `Jdot = (J_next - J_now) / t_step` internally.
- `Ip` is advanced step by step. `psi` is composed from the updated `Ip` and
  coil contributions.
- Exact replay does not apply hidden plant-side current clipping.

## Active Boundary Experiment

The active experiment uses:

```text
data/t15_data_new_trim50_ip_calibrated/
boundary mode = suchkov_spline_contour
angles = 32
```

The new mode selects the outermost admissible closed `psi` surface and
represents its coordinates as periodic cubic splines `R(xi)` and `Z(xi)`.
The implementation is described in `docs/suchkov-spline-boundary.md`.

The old successful 100M run remains the baseline. It used
`legacy_contour_limited` with `legacy_precision_index2 = 1e-6` and must not be
rewritten or mixed with the new replay and oracle artifacts.

## Experiment Flow

1. Rebuild T15 replays for shots `3856`, `3857`, `3858`, `3863`, and `3864`
   from the calibrated `Ip +15%` data.
2. Build replay-window oracle targets using shots `3856`, `3857`, `3858`, and
   `3863` for training and shot `3864` as holdout.
3. Train the same 8-GPU MPO configuration and reward weights as the successful
   baseline for 50M environment steps.

All new replay, oracle, configuration, output, and W&B names use a distinct
`ip15_suchkov` prefix.
