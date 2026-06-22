# Tokamak-Sim-0 Runtime Model

This file reconstructs how `tokamak-sim-0` runs from the old pasted sources.
It is a semantic model, not a line-by-line translation.

## Ownership Split

`tokamak-sim-0` is split across MATLAB and C++:

- MATLAB loads user/control state, extracts boundaries, builds references,
  computes LQR/H-infinity commands, and writes next absolute coil currents.
- C++ owns the plasma stepper. It reads MATLAB current files, advances one
  time step, computes `Ip` and `PsiMatrix`, and writes those back.

The exchange is file based. `LittleCppSCoPE.cpp` waits on lock files and runs
normal, experimental, restart, or quit modes (`littlescope undode cpp files.txt:699-748`).
MATLAB triggers this through `GetNewPsiMatrix` (`littlescope undone m files.txt:2569-2603`).

## Initialization

1. C++ reads the old machine config, grid, PFC, SOL, and initial current files
   in `load_or_crash` (`littlescope undode cpp files.txt:77-135`).
2. It constructs Green arrays:
   - PFC/SOL group Green arrays for psi.
   - Plasma-center Green array `G_Arr_0`.
   - `g` and `g2` center coupling arrays for the induced-current update.
3. It writes initial data into `InitialData`.
4. MATLAB `LoadInfo` reads those files and builds a `plasma` struct with:
   - `PFC.J`, `SOL.J`;
   - zero initial `J_Derivatives`;
   - boundary and measurement points;
   - previous LQR state/action buffers
   (`littlescope undone m files.txt:3-115`).

## Grid Semantics

The old C++ grid does not use the requested lower bound directly. It shifts
the internal start so the configured center lands halfway between grid samples:

```text
h = (xmax - xmin) / (N - 1)
x_start = xmin + (fractpart((x0 - xmin) / h) - 0.5) * h
x(i) = x_start + i*h - h
```

Evidence: `grid_t::grid_t` and accessors
(`littlescope undode cpp files.txt:388-410`).

## Coil Groups

Each old coil group can contain several geometric elements, but all elements in
one group share one scalar group current. C++ accumulates element Green values
into the group Green array (`littlescope undode cpp files.txt:479-554`).

The center coupling arrays `g` and `g2` are also sums over group elements
(`littlescope undode cpp files.txt:661-672`).

## Green Function

The old Green function computes the standard axisymmetric vector-potential
kernel using elliptic integrals:

```text
sqrt(R*RP) * ((1 - 0.5*k^2) * K(k) - E(k)) / (pi*k)
```

where `mu0` is not included in the Green function itself. C++ multiplies by
`mu0` later in the psi and Ip equations. Evidence:
`littlescope undode cpp files.txt:934-943`.

## Plant Step

MATLAB writes absolute next currents, not derivative commands. `GetNewPsiMatrix`
also computes derivative diagnostics from the finite difference:

```text
J_Derivatives = (J_new - J_old) / t_step
```

Evidence: `littlescope undone m files.txt:2569-2603`.

C++ then advances:

```text
t += t_step
Ip = Ip0 * exp(-t / (Sigma * L))
Ip -= (mu0 * Sigma / (R0 * t_step))
      * (g dot (J_PFC_new - J_PFC_old) + g2 dot (J_SOL_new - J_SOL_old))
J = J_new
Psi = mu0 * (Ip * G_plasma + PFC_current_terms + SOL_current_terms)
```

Evidence: `plasma_t::act` and `compute_Psi`
(`littlescope undode cpp files.txt:297-385`).

Important consequences:

- The passive Ip baseline is always recomputed from the original `Ip0` and
  absolute simulation time, not recursively from the previous `Ip`.
- The coil-driven Ip term is proportional to current derivative because it uses
  `(J_new - J_old) / t_step`.
- Old C++ forces `Sigma=6e8` and `L=0.03/Sigma`, giving a fixed passive time
  constant of `0.03 s` (`littlescope undode cpp files.txt:59-66`, `297-307`).

## Psi Composition

For lumped plasma mode, old C++ computes:

```text
PsiMatrix = mu0 * (Ip * G_Arr_0 + sum(PFC_J * PFC_G_Arr) + sum(SOL_J * SOL_G_Arr))
```

Evidence: `littlescope undode cpp files.txt:349-365`.

There is a distributed-plasma branch guarded by `i_dist == 1`, but the normal
configuration path is the lumped `Ip * G_Arr_0` model.

## Boundary Extraction

Old MATLAB `PlasmaBoundary`:

1. Takes the psi value at a center point `O`.
2. Uses `contourc` at that single psi level.
3. Scans contour segments.
4. Keeps the longest segment accepted by `LineIsOk`.

Evidence: `littlescope undone m files.txt:1357-1399`.

This is not a limiter-aware LCFS/divertor classifier. It is a contour-selection
heuristic anchored on a chosen center point.

## Control Interface

The old LQR/H-infinity controllers do not directly choose absolute current and
do not directly choose absolute derivative. They choose a change in derivative:

```text
x = [
  boundary_errors,
  Ip - Ip0,
  PFC.J_Derivatives * t_step,
  SOL.J_Derivatives * t_step,
  boundary_drift,
  Ip_drift
]
u = K*x
J_Derivatives_next = J_Derivatives_prev + u
J_new = J_old + J_Derivatives_next * t_step
```

Evidence:

- `ControlBoth`: `littlescope undone m files.txt:1600-1745` and continuation
  through `1792`.
- `ControlBoundary`: `littlescope undone m files.txt:1794-1978`.
- `ControlCurrent`: `littlescope undone m files.txt:1980-2130`.
- `ControlLQRKernel`: `littlescope undone m files.txt:2132-2218`.

This is the key old action contract: the controller output is `delta_Jdot`.

## Sensitivities

The old MATLAB controller obtains sensitivities by experimental one-step calls
or by measured differences:

- `GetDeltaIp` runs a no-current-change experimental step and measures Ip drift
  (`littlescope undone m files.txt:2476-2493`).
- `Getg` and `Getg2` perturb current derivatives and measure the resulting
  response (`littlescope undone m files.txt:2520-2545`).
- Later LQR steps reuse previous observed errors and previous actions to infer
  diagonal sensitivity terms (`littlescope undone m files.txt:1638-1660`).

The old LQR model is therefore a finite-difference local model around the
current trajectory, not a purely analytic static Green linearization.

## LQR/H-Infinity Shape

For joint control, old MATLAB builds:

```text
A = [I C11 I;
     0 I   0;
     0 0   I]

B = [C22 0 0;
     dt*I 0 0;
     0    0 0]
```

Then it solves `dlqr(0.999*A, B, Q, R)`, with `control.K_both.LQR = -K`
(`littlescope undone m files.txt:1660-1686`).

The `0.999` stabilizer in front of `A` is part of the old practical tuning, not
just a display detail.

## Artifact Model

Old artifacts are mostly text arrays and debug files:

- `InitialData/*` generated by C++.
- `ExchangeData/*` for MATLAB/C++ communication.
- optional debug files such as `PFCderiv.txt` and `SOLderiv.txt`.

Modern videos, CSVs, structured run directories, and scenario objects are not
part of `tokamak-sim-0`; they are current-project infrastructure.
