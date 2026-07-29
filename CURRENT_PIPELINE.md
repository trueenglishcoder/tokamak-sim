# Current Tokamak-Sim Pipeline

`tokamak-sim` is the plant/runtime source for the current T15 RL experiment.

## Plant Contract

- Public simulator/control input is absolute next-step coil current: `step_currents(J_next)`.
- The simulator derives `Jdot = (J_next - J_now) / t_step` internally.
- `Ip` is advanced step by step and `psi` is composed from updated plasma and coil contributions.
- Exact replay does not apply hidden plant-side current clipping.

## Active Boundary Contract

The active T15 machine config uses:

```text
boundary mode = equilibrium_lcfs
angles = 32
```

`psi(R,Z)` is treated as the known equilibrium supplied by the simulator. The
canonical boundary is the dense last closed flux surface of the connected
primary core. The extractor finds subgrid O- and X-points, evaluates limiter and
saddle candidates outward from the primary axis, automatically classifies
limited or diverted topology, and preserves separatrix branches at X-points.

The production GPU extractor implements the same topology-preserving
level-set procedure as the CPU reference. It constructs marching-squares
segments, treats selected X-points as graph nodes, traces the primary-core cycle,
and projects the configured 32 radii from that selected geometry. The CPU
reference is used only in parity tests and is not called inside GPU replay or
training steps.

Single-lane replay materializes and stores the ordered dense cycle. Large RL
batches skip only that stored point sequence while retaining the same physical
candidate ordering and GPU level-set geometry. The configured radii are never
used as input to a spline or as a fallback physical boundary.

The full algorithm and artifact schema are documented in
`docs/plasma_boundary_calculation.txt`.

## Historical Artifact Separation

Existing directories and run names containing `ip15_suchkov` are historical
identifiers. They were generated with the removed `suchkov_spline_contour`
geometry and must not be mixed with new `equilibrium_lcfs` replay, oracle, or
training artifacts.
