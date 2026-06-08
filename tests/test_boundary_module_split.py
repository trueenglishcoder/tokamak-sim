from __future__ import annotations

import inspect

import numpy as np
import pytest

from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.geometry.boundary import find_plasma_boundary_with_status
from tokamak_control.geometry.boundary_cpu import find_plasma_boundary_cpu_with_status
from tokamak_control.geometry.boundary_gpu import find_plasma_boundary_gpu_with_status
from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def _case():
    grid = Grid2D(
        r=Grid1D(start=0.0, step=0.02, size=121, center=1.2),
        z=Grid1D(start=-1.2, step=0.02, size=121, center=0.0),
    )
    R, Z = grid.mesh()
    center = (1.2, 0.0)
    psi = -((R - center[0]) ** 2 + (Z - center[1]) ** 2)
    limiter = np.array(
        [
            [0.75, -0.45],
            [1.65, -0.45],
            [1.65, 0.45],
            [0.75, 0.45],
            [0.75, -0.45],
        ],
        dtype=float,
    )
    return psi, grid, center, limiter


def _radii(poly: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    return radii_from_polyline_ray_intersections(poly, center, angles)


def test_cpu_direct_api_matches_dispatcher_cpu_mode() -> None:
    psi, grid, center, limiter = _case()
    direct_poly, direct_level, direct_status = find_plasma_boundary_cpu_with_status(
        psi,
        grid,
        center,
        limiter_shape=limiter,
        boundary_mode="limited",
    )
    dispatch_poly, dispatch_level, dispatch_status = find_plasma_boundary_with_status(
        psi,
        grid,
        center,
        limiter_shape=limiter,
        boundary_mode="limited",
        compute_backend="cpu",
    )

    assert dispatch_status == direct_status
    assert dispatch_level == pytest.approx(direct_level)
    assert np.allclose(_radii(dispatch_poly, center), _radii(direct_poly, center), rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA is not available")
def test_gpu_direct_api_matches_dispatcher_gpu_mode() -> None:
    psi, grid, center, limiter = _case()
    direct_poly, direct_level, direct_status = find_plasma_boundary_gpu_with_status(
        psi,
        grid,
        center,
        limiter_shape=limiter,
        boundary_mode="limited",
    )
    dispatch_poly, dispatch_level, dispatch_status = find_plasma_boundary_with_status(
        psi,
        grid,
        center,
        limiter_shape=limiter,
        boundary_mode="limited",
        compute_backend="gpu",
    )

    assert dispatch_status == direct_status
    assert dispatch_level == pytest.approx(direct_level)
    assert np.allclose(_radii(dispatch_poly, center), _radii(direct_poly, center), rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA is not available")
def test_gpu_direct_api_matches_cpu_reference_radii() -> None:
    psi, grid, center, limiter = _case()
    cpu_poly, cpu_level, cpu_status = find_plasma_boundary_cpu_with_status(
        psi,
        grid,
        center,
        limiter_shape=limiter,
        boundary_mode="limited",
    )
    gpu_poly, gpu_level, gpu_status = find_plasma_boundary_gpu_with_status(
        psi,
        grid,
        center,
        limiter_shape=limiter,
        boundary_mode="limited",
    )

    assert gpu_status == cpu_status
    assert gpu_level == pytest.approx(cpu_level, rel=0.0, abs=1e-12)
    assert np.allclose(_radii(gpu_poly, center), _radii(cpu_poly, center), rtol=0.0, atol=5e-2)


def test_gpu_module_does_not_import_dispatcher_or_cpu_backend() -> None:
    import tokamak_control.geometry.boundary_gpu as boundary_gpu

    source = inspect.getsource(boundary_gpu)

    assert "tokamak_control.geometry.boundary import" not in source
    assert "tokamak_control.geometry.boundary_cpu import" not in source
    assert "contour_generator" not in source
    assert "psi_t.detach().cpu().numpy" not in source
