# Tokamak-Sim-0 Source Map

This audit treats the two paste files in `_local_archive` as the source of truth
for the old simulator, called here `tokamak-sim-0`:

- `_local_archive/littlescope undode cpp files.txt`
- `_local_archive/littlescope undone m files.txt`

The old project is a MATLAB controller/GUI coupled to a C++ plasma stepper by
text files and lock files. MATLAB owns control, references, contour processing,
and LQR/H-infinity synthesis. C++ owns Green arrays, current-to-psi composition,
and the one-step plasma current update.

## C++ Sections

| Section | Lines | Role |
| --- | ---: | --- |
| `snowfed_plasma.cpp` | `littlescope undode cpp files.txt:1-674` | Main C++ plasma model. Defines constants, loads machine data, builds Green arrays, owns `plasma_t`, computes `Ip` and `PsiMatrix`. |
| `LittleCppSCoPE.cpp` | `littlescope undode cpp files.txt:675-750` | File-lock process loop. Reads MATLAB mode, runs normal/experimental/restart/quit commands. |
| `snowfed.cpp` | `littlescope undode cpp files.txt:751-793` | Small command-line entry point around the file-lock process. |
| `working_directory.cpp` | `littlescope undode cpp files.txt:794-880` | Working-directory and path helpers. |
| `george_green.cpp` | `littlescope undode cpp files.txt:881-944` | Axisymmetric Green function implementation. |
| `snowfed_array.cpp` | `littlescope undode cpp files.txt:946-1183` | Text matrix/vector load/save utilities. |

## Key C++ Functions And Data

| Symbol | Lines | Notes |
| --- | ---: | --- |
| `mu0` | `littlescope undode cpp files.txt:50` | `1.2566370614e-6`. |
| `inductance_law` | `littlescope undode cpp files.txt:59-66` | Ignores caller Sigma, forces `Sigma=6e8`, returns `0.03 / Sigma`. Old passive time constant is exactly `0.03 s`. |
| `load_or_crash` | `littlescope undode cpp files.txt:77-135` | Loads grid, plasma, PFC/SOL coil arrays, builds `PFC_G_Arr`, `SOL_G_Arr`, `G_Arr_0`, `g`, and `g2`, writes initial data for MATLAB. |
| `plasma_t::plasma_t` | `littlescope undode cpp files.txt:269-294` | Initializes `Ip`, absolute PFC/SOL currents, computes initial psi, writes `Cpp_PsiMatrix0` and `Cpp_Ip`. |
| `plasma_t::act` | `littlescope undode cpp files.txt:297-346` | Reads absolute next currents from MATLAB, advances time, updates `Ip`, commits currents. |
| `plasma_t::compute_Psi` | `littlescope undode cpp files.txt:349-385` | Computes psi as `mu0 * (Ip * G_Arr_0 + coil_current_terms)`, with a distributed-plasma branch. |
| `grid_t::grid_t` and accessors | `littlescope undode cpp files.txt:388-410` | Builds a shifted grid so the configured center lies halfway between samples. Accessors are 1-based. |
| `load_grid` | `littlescope undode cpp files.txt:429-477` | Reads R/Z grid bounds, sizes, and center from old config. |
| `load_coils` | `littlescope undode cpp files.txt:479-554` | Loads coil groups. Each group has multiple geometric elements but one current. |
| `Green_for_Eind` | `littlescope undode cpp files.txt:661-672` | Computes center Green coupling arrays `g` and `g2` for induced Ip update. |
| `compute_george_green` | `littlescope undode cpp files.txt:934-943` | Uses elliptic integrals. The Green result does not include `mu0`; callers multiply later. |
| file-lock loop | `littlescope undode cpp files.txt:699-748` | Normal mode updates current state, experimental mode runs a temporary step, restart resets to initial state. |

## MATLAB Sections

| Section | Lines | Role |
| --- | ---: | --- |
| `Loadinfo.m` | `littlescope undone m files.txt:1-115` | Creates MATLAB state, exchange paths, initial currents, boundary, measurement points, LQR buffers. |
| `StableBoundary.m` | `littlescope undone m files.txt:116-1356` | GUI and high-level run orchestration. |
| `PlasmaBoundary.m` | `littlescope undone m files.txt:1357-1399` | Extracts old plasma boundary from a psi contour. |
| `LoadInfo2.m` | `littlescope undone m files.txt:1400-1550` | Alternate loader. |
| `ClosestPointOfSegment.m` | `littlescope undone m files.txt:1551-1592` | Geometry helper. |
| `ControlBoth.m` | `littlescope undone m files.txt:1594-1792` | Joint boundary+Ip LQR/H-infinity control. |
| `ControlBoundary.m` | `littlescope undone m files.txt:1794-1978` | Boundary control with SOL derivative replay. |
| `ControlCurrent.m` | `littlescope undone m files.txt:1980-2130` | Ip/SOL control with PFC derivative replay. |
| `ControlLQRKernel.m` | `littlescope undone m files.txt:2132-2218` | Compact LQR kernel. |
| `DrawContourLines.m` | `littlescope undone m files.txt:2220-2274` | Plot helper. |
| `GetAnglesBuddies.m` | `littlescope undone m files.txt:2287-2337` | Measurement-angle pairing helper. |
| `GetC0.m` | `littlescope undone m files.txt:2339-2346` | Drift helper. |
| `GetC1.m` | `littlescope undone m files.txt:2348-2375` | Boundary sensitivity helper. |
| `GetClosestPoints*.m` | `littlescope undone m files.txt:2377-2474` | Boundary target/actual point matching. |
| `GetDeltaIp.m` | `littlescope undone m files.txt:2476-2493` | Experimental one-step Ip drift estimate. |
| `GetErrors.m` | `littlescope undone m files.txt:2495-2518` | Signed boundary-radius errors. |
| `Getg.m`, `Getg2.m` | `littlescope undone m files.txt:2520-2545` | Experimental derivative sensitivities for PFC/SOL. |
| `GetMeasuringAngles.m` | `littlescope undone m files.txt:2547-2567` | Measurement angle initialization. |
| `GetNewPsiMatrix.m` | `littlescope undone m files.txt:2569-2603` | Writes absolute currents to C++, reads new `Ip` and `PsiMatrix`; updates MATLAB derivative diagnostics. |
| `GetAndConstructLQR.m` | `littlescope undone m files.txt:2632-2718` | Reads externally supplied linearization, builds `dlqr`, returns command increments. |

## Current Files Used For Comparison

| Current file | Main comparison role |
| --- | --- |
| `tokamak_control/core/plasma_model.py` | Plant state, derivative command interface, actuator lag, current clipping, Ip update, psi composition. |
| `tokamak_control/core/green.py` | Current Green function. |
| `tokamak_control/config/settings.py` | Physics defaults, signs, limits, lag parameters. |
| `tokamak_control/io/config_io.py` | TOML machine/config loader. |
| `tokamak_control/geometry/boundary_cpu.py` | Current CPU boundary extraction. |
| `tokamak_control/geometry/boundary_gpu.py` | Current GPU boundary extraction. |
| `tokamak_control/control/linearization.py` | Current one-step sensitivity construction. |
| `tokamak_control/control/lqr_t15_zaitsev.py` | New book-style LQR implementation and delta-Jdot accumulation. |
| `tokamak_control/cli/run_simulation.py` | Runtime scenario, controller, artifact orchestration. |
| `configs/T15MD_new_data.toml` | Active T15 machine parameters and limits. |
