# Tokamak-Sim-0 vs Current Tokamak-Sim Parity Comparison

Verdicts:

- `same`: semantics appear equivalent.
- `intentional T15 change`: different, but likely expected for the modern T15
  machine/data path.
- `unclear`: needs a numeric probe or provenance.
- `mismatch`: semantic drift from old model.
- `likely bug`: strong evidence that the current behavior is not the old model
  and is likely physically consequential.

| Axis | Tokamak-sim-0 behavior | Current tokamak-sim behavior | Verdict |
| --- | --- | --- | --- |
| Machine loading | Old C++ reads custom text config and writes `InitialData` (`littlescope undode cpp files.txt:77-135`). | TOML loader builds typed settings, grid, PFC/SOL, limiter, realism settings (`tokamak_control/io/config_io.py:318-330`, `470-482`). | intentional T15 change |
| Grid indexing | C++ shifts grid start so configured center is halfway between samples (`littlescope undode cpp files.txt:388-410`). | TOML grid uses configured range/size directly; active T15 config has `r/z` ranges and center fields (`configs/T15MD_new_data.toml:17-27`). | mismatch |
| Coil grouping | One current per group, multiple elements summed into one Green array (`littlescope undode cpp files.txt:479-554`). | Current T15 config has three runtime SOL actuators with 30/90/30 physical split points and fractional `element_weights` summing to 1 per actuator (`configs/T15MD_new_data.toml:78`, `688`). This means one SOL current is distributed across its volume samples. | intentional T15 physical-model change |
| Green formula | Elliptic-integral axisymmetric Green; no `mu0` inside function (`littlescope undode cpp files.txt:934-943`). | `green_axisymmetric` uses `ellipk/ellipe` with same formula class (`tokamak_control/core/green.py:9-27`). | same |
| `mu0` placement in psi | `compute_Psi` multiplies whole plasma+coil sum by `mu0` (`littlescope undode cpp files.txt:349-365`). | `_compose_psi` also multiplies the combined sum by `mu0` (`tokamak_control/core/plasma_model.py:344-354`). | same |
| Plasma psi sign | Old lumped psi is `+Ip * G_Arr_0` before `mu0` (`littlescope undode cpp files.txt:349-365`). | Active T15 config sets `plasma_psi_sign = -1.0` (`configs/T15MD_new_data.toml:26-27`). | unclear/intentional calibration |
| Ip passive decay baseline | Old C++ recomputes from original `Ip0 * exp(-t/(Sigma*L))` each step (`littlescope undode cpp files.txt:313-320`). | Current plant decays recursively from previous state `s.Ip * decay_factor(dt)` (`tokamak_control/core/plasma_model.py:317-323`). | mismatch |
| Ip coil-driven time scaling | Old term uses `(J_new - J_old) / t_step` through `mu0*Sigma/(R0*t_step)` (`littlescope undode cpp files.txt:319-325`). | Current `step` uses `dot(g, delta_current)` without `/dt` (`tokamak_control/core/plasma_model.py:317-323`); `get_ip_B_row` multiplies derivative sensitivity by `t_step` (`tokamak_control/core/plasma_model.py:236-245`). | likely bug unless `g` was re-fit |
| Sigma/L constants | Old runtime forces `Sigma=6e8` and `Sigma*L=0.03s` (`littlescope undode cpp files.txt:59-66`, `297-307`). | Active T15 config uses `sigma=3832562.947936214`, `inductance_L=3.36416228166149e-07`, time constant about `1.29s` (`configs/T15MD_new_data.toml:17-18`). | intentional T15 change, but needs provenance |
| Current interface to plant | MATLAB writes absolute next currents; derivatives are diagnostics (`littlescope undone m files.txt:2569-2603`). | Current plant accepts derivative commands, integrates them to currents (`tokamak_control/core/plasma_model.py:274-307`). | mismatch at plant API, equivalent only if controller integration is correct |
| Controller action semantics | Old LQR output is delta-Jdot; derivative command is accumulated, then current is integrated (`littlescope undone m files.txt:1600-1745`). | Current `lqr_t15_zaitsev` does delta-Jdot accumulation (`tokamak_control/control/lqr_t15_zaitsev.py:142-188`). | same for new LQR, not necessarily all controllers |
| Actuator lag | Old C++ applies absolute requested current immediately. No plant-side lag in `plasma_t::act` (`littlescope undode cpp files.txt:297-346`). | Current plant has first-order lag `alpha = exp(-dt/tau)` and active T15 has `actuator_tau=0.01` (`tokamak_control/core/plasma_model.py:269-298`, `configs/T15MD_new_data.toml:20`). | intentional realism or mismatch; not old parity |
| Derivative clipping | Old plant has no derivative clipping; MATLAB controllers may be tuned but C++ accepts written currents. | Current plant clips derivative commands before and after lag (`tokamak_control/core/plasma_model.py:285-298`). | intentional safety change |
| Current clipping | Old plant commits `J_PFC = J_PFC_new`, `J_SOL = J_SOL_new` with no clipping (`littlescope undode cpp files.txt:340-346`). | Current plant clips integrated currents to current limits (`tokamak_control/core/plasma_model.py:304-305`). | mismatch/intentional safety change |
| Boundary extraction | Old boundary is longest accepted contour accepted by `LineIsOk` during the MATLAB `PlasmaBoundary` index-space search (`littlescope undone m files.txt:1357-1399`, `2605-2655`). | Current default modes remain magnetic-axis/limiter/divertor aware, and `boundary.mode="legacy_contour"` now reproduces the old limiter-free contour selection (`tokamak_control/geometry/boundary_cpu.py`). | resolved by selectable legacy mode |
| Measurement radii/errors | Old errors come from matched contour points and `GetErrors` sign logic (`littlescope undone m files.txt:2495-2518`). | Current code uses boundary/target radii in scenario and metric layers. | unclear; needs direct point-by-point parity probe |
| LQR sensitivity construction | Old finite-differences through experimental C++ steps and previous observed deltas (`littlescope undone m files.txt:2476-2545`, `1638-1660`). | Current LQR uses analytic/static Green linearization and contour gradients (`tokamak_control/control/linearization.py:143-148`; `tokamak_control/control/lqr_t15_zaitsev.py:202-332`). | mismatch |
| LQR gain reuse | Old code sometimes computes and caches `control.K_both.LQR`, with comments/branches for reuse (`littlescope undone m files.txt:1624-1686`). | Current `lqr_t15_zaitsev` rebuilds system/gain inside `compute_control` unless implementation caching is added (`tokamak_control/control/lqr_t15_zaitsev.py:142-188`). | mismatch/performance-risk |
| H-infinity | Old MATLAB has H-infinity branches but they are partial and reuse the same state idea (`littlescope undone m files.txt:1687-1792`, `1980-2130`). | Current robust-control baseline is not yet source-faithful. | mismatch |
| Scenario/reference handling | Old references are MATLAB GUI arrays and sometimes replay derivative tables. | Current scenarios are explicit CLI objects and artifacts. | intentional infrastructure change |

## Most Important Parity Finding

The old Ip response to coil actuation is proportional to `dJ/dt`, because the
C++ term divides current change by `t_step`. The current Python plant first
integrates derivative commands into current changes and then applies the Ip
coupling to the raw current change. For `t_step = 0.001`, this is a factor of
about `1000` difference in one-step derivative authority if the same `g/g2`
couplings are used.

This is the strongest mismatch found in this pass. It directly affects whether
any controller, learned or LQR, has realistic authority over Ip.

## Split SOL Current Convention

The active T15 convention is deliberate: split SOL points are not separate
runtime channels. SOL0/SOL1/SOL2 each receive one current command, and their
physical point samples contribute by `element_weights` that sum to 1. An
unweighted interpretation of the same geometry would multiply SOL authority by
30/90/30 and is not the intended T15 model.
# Superseded Note

This file is a historical comparison from before the current causal-Ip,
absolute-next-current implementation. For current Ip/Jdot/delta-Jdot semantics,
use `../ip_jdot_semantics_audit.md`.
