from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokamak_control.compute import ComputeSettings
from tokamak_control.config.settings import PhysicsSettings
from tokamak_control.core.coils import Coil, CoilActuator, CoilGroup
from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.geometry.boundary import BoundaryNotFoundError, find_plasma_boundary_with_status
from tokamak_control.geometry.boundary_equivalence import BoundaryEquivalenceTolerance, compare_boundary_results
from tokamak_control.io.config_io import dump_config, load_config


def test_config_loads_and_dumps_compute_settings(tmp_path: Path) -> None:
    path = tmp_path / "machine.toml"
    grid = Grid2D(Grid1D(0.0, 0.1, 8, 0.4), Grid1D(-0.4, 0.1, 8, 0.0))
    pfc = CoilGroup("pfc", [CoilActuator([Coil(0.2, 0.2)])], currents=np.array([0.0]))
    sol = CoilGroup("sol", [CoilActuator([Coil(0.6, -0.2)])], currents=np.array([0.0]))
    dump_config(
        path,
        grid=grid,
        pfc=pfc,
        sol=sol,
        physics=PhysicsSettings(Ip0=1.0e4, R0=0.4, Z0=0.0),
        compute=ComputeSettings(backend="gpu", gpu_device="cuda:0"),
        boundary_mode="limited",
    )

    cfg = load_config(path)

    assert cfg.compute.backend == "gpu"
    assert cfg.compute.gpu_device == "cuda:0"
    assert cfg.compute.boundary_equivalence_mode == "strict"


def test_gpu_boundary_matches_cpu_on_synthetic_limited_case() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    grid = Grid2D(
        r=Grid1D(start=0.0, step=0.02, size=121, center=1.0),
        z=Grid1D(start=-1.2, step=0.02, size=121, center=0.0),
    )
    R, Z = grid.mesh()
    center = (1.0, 0.0)
    psi = -((R - center[0]) ** 2 + (Z - center[1]) ** 2)
    limiter = np.array(
        [
            [0.35, -0.75],
            [1.65, -0.75],
            [1.65, 0.75],
            [0.35, 0.75],
            [0.35, -0.75],
        ],
        dtype=float,
    )

    cpu_poly, cpu_level, cpu_status = find_plasma_boundary_with_status(
        psi,
        grid,
        center,
        n_levels=32,
        limiter_shape=limiter,
        boundary_mode="limited",
        compute_backend="cpu",
    )
    try:
        gpu_poly, gpu_level, gpu_status = find_plasma_boundary_with_status(
            psi,
            grid,
            center,
            n_levels=32,
            limiter_shape=limiter,
            boundary_mode="limited",
            compute_backend="gpu",
            gpu_device="cuda:0",
        )
    except BoundaryNotFoundError as exc:
        pytest.fail(f"GPU backend did not find the CPU limited boundary: {exc}")

    report = compare_boundary_results(
        cpu_poly=cpu_poly,
        cpu_level=cpu_level,
        cpu_status=cpu_status,
        gpu_poly=gpu_poly,
        gpu_level=gpu_level,
        gpu_status=gpu_status,
        center=center,
        tolerance=BoundaryEquivalenceTolerance(mean_radii_abs=2.5e-2, max_radii_abs=5.0e-2),
    )
    assert report.equivalent, report.to_dict()
