from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.test_bridge_and_metrics import _write_bridge_config
from tokamak_control.bridge import DerivativeAction, SimulationSession
from tokamak_control.config.settings import PhysicsSettings
from tokamak_control.core.coils import Coil, CoilActuator, CoilGroup
from tokamak_control.core.gpu_plasma_model import GpuPlasmaModel
from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.core.plasma_model import PlasmaModel


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


pytestmark = pytest.mark.skipif(not _cuda_available(), reason="CUDA is not available")


def _small_model_pair() -> tuple[PlasmaModel, GpuPlasmaModel]:
    grid = Grid2D(
        r=Grid1D(start=0.2, step=0.03, size=81, center=1.2),
        z=Grid1D(start=-1.0, step=0.03, size=81, center=0.0),
    )
    pfc = CoilGroup(
        name="pfc",
        coils=[CoilActuator([Coil(0.8, 0.8)]), CoilActuator([Coil(1.6, -0.8)])],
        currents=np.array([1000.0, -1500.0], dtype=float),
    )
    sol = CoilGroup(
        name="sol",
        coils=[CoilActuator([Coil(1.2, 1.0)])],
        currents=np.array([500.0], dtype=float),
    )
    physics = PhysicsSettings(
        Ip0=5.0e4,
        R0=1.2,
        Z0=0.0,
        sigma=2.0e6,
        inductance_L=2.0e-6,
        t_step=1.0e-3,
        actuator_tau=2.0e-3,
        pfc_current_limit=1.0e5,
        sol_current_limit=1.0e5,
        pfc_deriv_limit=1.0e6,
        sol_deriv_limit=1.0e6,
    )
    return (
        PlasmaModel.from_settings(grid=grid, pfc=pfc, sol=sol, settings=physics),
        GpuPlasmaModel.from_settings(grid=grid, pfc=pfc, sol=sol, settings=physics),
    )


def test_gpu_plasma_model_compute_psi_matches_cpu() -> None:
    cpu, gpu = _small_model_pair()

    assert np.allclose(gpu.compute_psi(), cpu.compute_psi(), rtol=1.0e-12, atol=1.0e-12)
    assert gpu.snapshot_state().Ip == pytest.approx(cpu.snapshot_state().Ip, rel=1.0e-14, abs=1.0e-14)


def test_gpu_plasma_model_step_matches_cpu_over_action_sequence() -> None:
    cpu, gpu = _small_model_pair()
    actions = [
        (np.array([1.0e5, -2.0e5]), np.array([5.0e4])),
        (np.array([-3.0e5, 1.0e5]), np.array([-1.5e5])),
        (np.array([7.5e5, 8.0e5]), np.array([9.0e5])),
    ]

    for pfc_derivs, sol_derivs in actions:
        cpu_state = cpu.step(pfc_derivs, sol_derivs)
        gpu_state = gpu.step(pfc_derivs, sol_derivs)
        assert gpu_state.t == pytest.approx(cpu_state.t)
        assert gpu_state.step == cpu_state.step
        assert gpu_state.Ip == pytest.approx(cpu_state.Ip, rel=1.0e-12, abs=1.0e-9)
        assert np.allclose(gpu_state.pfc_currents, cpu_state.pfc_currents, rtol=1.0e-12, atol=1.0e-9)
        assert np.allclose(gpu_state.sol_currents, cpu_state.sol_currents, rtol=1.0e-12, atol=1.0e-9)
        assert np.allclose(gpu_state.pfc_current_derivs, cpu_state.pfc_current_derivs, rtol=1.0e-12, atol=1.0e-9)
        assert np.allclose(gpu_state.sol_current_derivs, cpu_state.sol_current_derivs, rtol=1.0e-12, atol=1.0e-9)
        assert np.allclose(gpu_state.psi, cpu_state.psi, rtol=1.0e-12, atol=1.0e-12)


def test_gpu_simulation_session_reports_gpu_compute_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "bridge_machine.toml"
    _write_bridge_config(config_path)
    session = SimulationSession.from_paths(
        config_path=config_path,
        initial_currents_path=None,
        scenario_name="nominal",
        scenario_args={},
        angles=8,
        steps=2,
        compute_backend="gpu",
    )
    reset = session.reset(seed=10)

    assert reset.machine.compute_backend == "gpu"
    assert reset.episode_metadata["compute"]["plant_backend"] == "gpu"
    assert reset.episode_metadata["compute"]["boundary_backend"] == "gpu"
    assert reset.episode_metadata["compute"]["cuda_available"] is True

    result = session.step_derivatives(DerivativeAction(np.zeros((reset.machine.n_active_total,), dtype=float)))
    assert result.snapshot.step_index == 1
    assert result.snapshot.true_boundary_poly is not None
