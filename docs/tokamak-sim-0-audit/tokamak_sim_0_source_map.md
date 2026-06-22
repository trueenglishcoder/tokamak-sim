# Tokamak-Sim-0 Source Map

This file maps the old pasted source files into subsystems. Line references point to the paste files in `_local_archive/`.

## C++ Runtime

### `snowfed_plasma.cpp`

Evidence file: `_local_archive/littlescope undode cpp files.txt`.

| Lines | Component | Responsibility |
|---:|---|---|
| 1-47 | Includes and `config_error` | Runtime support and exception type. |
| 50 | `mu0` | Magnetic constant. Green functions do not include `mu0`; psi composition multiplies by `mu0`. |
| 59-70 | `inductance_law` | Forces `Sigma = 6e8`; returns `0.03 / Sigma`, so `Sigma * L = 0.03 s`. |
| 77-135 | `plasma_t::load_or_crash` | Loads grid, plasma center, initial `Ip0`, `Sigma`, coils, Green arrays, `g/g2`, initial currents, and computes initial psi. |
| 139-164 | `load_g` | Loads per-actuator Ip coupling vectors `g` and `g2`. |
| 166-197 | `load_green_array` | Loads plasma Green table and coil Green table. |
| 199-251 | `read_initial_config` | Loads runtime file paths and initial values. |
| 254-295 | `load_globals` | Loads grid dimensions, plasma center, initial `Ip`, and calls `inductance_law`. |
| 297-346 | `plasma_t::act` | Core plant update. Reads absolute next currents `J_PFC_new`, `J_SOL_new`; computes `Jdot` internally as `(J_new-J_old)/t_step`; updates `Ip`; commits currents. |
| 349-385 | `compute_Psi` | Composes psi as `mu0 * (Ip * G_plasma + sum(J_coil * G_coil))`. |
| 388-409 | `grid_t` | Old center-half-cell grid alignment formula. |
| 412-438 | `load_grids` | Loads grid metadata and builds `grid_t`. |
| 441-477 | `interpolate_green` | Bilinear interpolation over old grid coordinates. |
| 479-554 | `load_coils` | Loads grouped actuators, multiple physical elements, and weights. |
| 558-600 | `read_array` | Matrix reader. |
| 603-635 | `load_array` | Loads Green arrays and weights. |
| 637-644 | `save` | Writes arrays for MATLAB. |
| 647-659 | `get_time` | Runtime timer. |
| 661-672 | `Green_for_Eind` | Sums Green from plasma center to coil elements. |

The most important line block is `297-346`: tokamak-sim-0 plant input is **absolute next coil current**, not derivative. The derivative authority used in the Ip formula is derived from the current change:

```text
Jdot = (J_new - J_old) / t_step
Ip_next = Ip0 * exp(-t_next / (Sigma * L))
          - (mu0 * Sigma / (R0 * t_step)) * dot(g, J_new - J_old)
```

The sign is negative in the old pasted C++; current T15 configs may intentionally choose a different configured sign.

### `LittleCppSCoPE.cpp`

Evidence file: `_local_archive/littlescope undode cpp files.txt`.

| Lines | Component | Responsibility |
|---:|---|---|
| 675-748 | Main loop | File-lock runtime protocol with `NORMAL`, `EXPERIMENTAL`, and `RESTART` modes. Calls `plasma.act()` and writes output files. |

### `snowfed.cpp`, `working_directory.cpp`, `snowfed_array.cpp`

Evidence file: `_local_archive/littlescope undode cpp files.txt`.

| Lines | Component | Responsibility |
|---:|---|---|
| 751-793 | `snowfed.cpp` | Legacy path and lock helper functions. |
| 794-879 | `working_directory.cpp` | Path generation and file append helpers. |
| 946-1181 | `snowfed_array.cpp` | Matrix/vector save/load helpers. |

### `george_green.cpp`

Evidence file: `_local_archive/littlescope undode cpp files.txt`.

| Lines | Component | Responsibility |
|---:|---|---|
| 881-943 | `george_green` | Elliptic-integral Green function. It returns a geometric Green value without `mu0`; C++ psi composition applies `mu0` later. |

## MATLAB Runtime And Control

Evidence file: `_local_archive/littlescope undone m files.txt`.

### Startup And Runtime Loop

| Lines | Function | Responsibility |
|---:|---|---|
| 3-93 | `LoadInfo` | Starts C++ runtime, writes `MATLAB_Sigma` and `MATLAB_t_step`, reads initial currents/psi, computes first boundary and measurement points. |
| 315-384 | GUI automatic mode loop | Calls control routines and steps the plant through file exchange. |
| 405-471 | Step/update block | Calls `GetNewPsiMatrix`, recomputes boundary, measurement radii and errors. |
| 582-602 | Manual step | Writes absolute PFC/SOL currents from UI. |

### Boundary Extraction And Error Geometry

| Lines | Function | Responsibility |
|---:|---|---|
| 1359-1398 | `PlasmaBoundary` | Searches for a contour level around center `O`: starts at `P=O`, steps through psi levels, accepts closed contours whose bounding box contains `O`. |
| 2605-2630 | `LineIsOk` | Accepts the first closed contour whose min/max box contains the center point. |
| 2287-2337 | `GetAngleBuddies` | Periodic angle interpolation used to convert boundary points into radii at measuring angles. |
| 2348-2375 | `GetC1` | Finite-difference boundary sensitivity: perturb absolute current, run `GetNewPsiMatrix`, re-extract boundary, compare old measurement geometry. |
| 2476-2493 | `GetDeltaIp` | Experimental one-step Ip response through the same C++ plant. |

### LQR/Hinf Control

| Lines | Function | Responsibility |
|---:|---|---|
| 1594-1792 | `ControlBoth` | Joint boundary + Ip controller. State includes boundary errors, Ip error, `J_Derivatives * t_step`, drift/current terms. Computes LQR/Hinf gain and accumulates derivative command. |
| 1841-1978 | `ControlBoundary` | Boundary-only control, mostly PFC, with derivative accumulation. |
| 1982-2129 | `ControlCurrent` | Ip/current-only control, mostly SOL, with derivative accumulation. |
| 2132-2218 | `ControlLQRKernel` | Kernel form of the same joint control semantics. |
| 2632-2718 | `ReadAndConstructLQR` | Loads a matrix, constructs/solves a kernel LQR, and applies the same derivative increment semantics. |

The important control contract in these routines is:

```text
u = delta_Jdot
J_Derivatives = J_Derivatives + u
J_new = J_old + J_Derivatives * t_step
GetNewPsiMatrix writes J_new to C++
```

### Plant Exchange

| Lines | Function | Responsibility |
|---:|---|---|
| 2569-2603 | `GetNewPsiMatrix` | Converts absolute next currents into file exchange with C++; locally computes derivative for bookkeeping as `(J_new - J_old)/dt`; reads new `Ip` and `Psi`. |
| 2522-2545 | `Getg`, `Getg2` | Finite-difference Ip coupling from C++ plant. |

## Book Context

Zaitsev section 2.3.1 supports the state-space/control interpretation used by the old MATLAB routines: the controller is designed around a state equation and LQR/Hinf synthesis, with actuator increments appearing as the control variable. The pasted source is the stronger authority for implementation details because it shows exactly how the old simulator exchanged currents and accumulated derivative commands.
