from __future__ import annotations

import numpy as np
import pytest

from tokamak_control.config.settings import PhysicsSettings
from tokamak_control.core.coils import Coil, CoilActuator, CoilGroup
from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.core.plasma_model import PlasmaModel


def _small_grid() -> Grid2D:
    return Grid2D(
        r=Grid1D(start=0.0, step=0.2, size=16, center=1.0),
        z=Grid1D(start=-1.0, step=0.2, size=12, center=0.0),
    )


def _small_machine(*, actuator_tau: float = 0.0) -> PlasmaModel:
    pfc = CoilGroup(
        name="pfc",
        coils=[CoilActuator([Coil(0.8, 0.2)])],
        currents=np.array([500.0], dtype=float),
    )
    sol = CoilGroup(
        name="sol",
        coils=[CoilActuator([Coil(1.2, -0.2)])],
        currents=np.array([50.0], dtype=float),
    )
    physics = PhysicsSettings(
        Ip0=100.0,
        R0=1.0,
        Z0=0.0,
        sigma=4.0,
        inductance_L=2.0,
        t_step=0.1,
        ip_coupling_sign=1.0,
        plasma_psi_sign=1.0,
        actuator_tau=actuator_tau,
        pfc_current_limit=501.0,
        sol_current_limit=51.0,
        pfc_deriv_limit=1.0,
        sol_deriv_limit=1.0,
        ip_coupling_pfc=(0.25,),
        ip_coupling_sol=(-0.5,),
    )
    return PlasmaModel.from_settings(grid=_small_grid(), pfc=pfc, sol=sol, settings=physics)


def test_grid_uses_old_center_half_cell_alignment() -> None:
    axis = Grid1D(start=0.0, step=1.0, size=5, center=2.25)

    assert np.allclose(axis.coords(), np.array([-0.25, 0.75, 1.75, 2.75, 3.75], dtype=float))
    assert np.isclose(axis.coords()[2], axis.center - 0.5 * axis.step)
    assert np.isclose(axis.coords()[3], axis.center + 0.5 * axis.step)


def test_cpu_step_uses_causal_ip_dynamics_and_does_not_clip_or_lag() -> None:
    model = _small_machine(actuator_tau=999.0)
    pfc_jdot = np.array([10_000.0], dtype=float)
    sol_jdot = np.array([-20_000.0], dtype=float)

    state = model.step_currents(
        pfc_currents_next=model.state.pfc_currents + model.t_step * pfc_jdot,
        sol_currents_next=model.state.sol_currents + model.t_step * sol_jdot,
    )

    scale = model.ip_coupling_sign * (model.mu0 * model.sigma / model.R0)
    expected_dip_dt = -100.0 / (model.sigma * model.inductance_L)
    expected_dip_dt += scale * (0.25 * pfc_jdot[0] + -0.5 * sol_jdot[0])
    expected_ip = 100.0 + model.t_step * expected_dip_dt

    assert np.allclose(state.pfc_current_derivs, pfc_jdot)
    assert np.allclose(state.sol_current_derivs, sol_jdot)
    assert np.allclose(state.pfc_currents, np.array([1500.0], dtype=float))
    assert np.allclose(state.sol_currents, np.array([-1950.0], dtype=float))
    assert np.isclose(state.Ip, expected_ip)
    assert np.allclose(state.psi, model._compose_psi(expected_ip, state.pfc_currents, state.sol_currents))


def test_zero_command_uses_passive_loss_from_current_ip() -> None:
    model = _small_machine()
    first = model.step_currents(
        pfc_currents_next=model.state.pfc_currents + model.t_step * np.array([10_000.0]),
        sol_currents_next=model.state.sol_currents + model.t_step * np.array([-20_000.0]),
    )
    expected_ip = first.Ip * (1.0 - model.t_step / (model.sigma * model.inductance_L))
    state = model.step_currents(
        pfc_currents_next=model.state.pfc_currents.copy(),
        sol_currents_next=model.state.sol_currents.copy(),
    )
    assert np.isclose(state.Ip, expected_ip)


def test_active_ip_is_not_littlescope_algebraic_overwrite() -> None:
    """A zero-Jdot step must decay from current Ip, not reset to Ip0 baseline."""
    model = _small_machine()
    first = model.step_currents(
        pfc_currents_next=model.state.pfc_currents + model.t_step * np.array([10_000.0]),
        sol_currents_next=model.state.sol_currents + model.t_step * np.array([-20_000.0]),
    )
    before_second = float(first.Ip)
    second = model.step_currents(
        pfc_currents_next=model.state.pfc_currents.copy(),
        sol_currents_next=model.state.sol_currents.copy(),
    )

    reset_baseline = model.Ip0 * np.exp(-float(second.t) / model.time_constant())
    causal_baseline = before_second * model.decay_factor()

    assert np.isclose(second.Ip, causal_baseline)
    assert not np.isclose(second.Ip, reset_baseline)


def test_get_ip_b_row_matches_finite_difference_derivative_authority() -> None:
    model = _small_machine()
    eps = 123.0
    baseline = model.predict_Ip_decay_baseline_next()
    row = model.get_ip_B_row()

    state = model.step_currents(
        pfc_currents_next=model.state.pfc_currents + model.t_step * np.array([eps]),
        sol_currents_next=model.state.sol_currents.copy(),
    )

    assert row.shape == (2,)
    assert np.isclose((state.Ip - baseline) / eps, row[0])


def test_gpu_models_match_cpu_old_parity_step_if_cuda_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    from tokamak_control.core.batched_gpu_simulator import BatchedGpuTokamakSimulator
    from tokamak_control.core.gpu_plasma_model import GpuPlasmaModel

    cpu = _small_machine()
    gpu = GpuPlasmaModel.from_settings(grid=cpu.grid, pfc=cpu.pfc, sol=cpu.sol, settings=PhysicsSettings(
        Ip0=cpu.Ip0,
        R0=cpu.R0,
        Z0=cpu.Z0,
        sigma=cpu.sigma,
        inductance_L=cpu.inductance_L,
        t_step=cpu.t_step,
        ip_coupling_sign=cpu.ip_coupling_sign,
        plasma_psi_sign=cpu.plasma_psi_sign,
        actuator_tau=cpu.actuator_tau,
        pfc_current_limit=cpu.pfc_current_limit,
        sol_current_limit=cpu.sol_current_limit,
        pfc_deriv_limit=cpu.pfc_deriv_limit,
        sol_deriv_limit=cpu.sol_deriv_limit,
        ip_coupling_pfc=tuple(np.asarray(cpu.g, dtype=float)),
        ip_coupling_sol=tuple(np.asarray(cpu.g2, dtype=float)),
    ))
    batched = BatchedGpuTokamakSimulator(
        grid=cpu.grid,
        pfc=cpu.pfc,
        sol=cpu.sol,
        settings=gpu.settings,
        batch_size=2,
        angles_rad=np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False),
        limiter_shape=np.array([[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]], dtype=float),
    )

    pfc = np.array([1000.0], dtype=float)
    sol = np.array([-2000.0], dtype=float)
    cpu_state = cpu.step_currents(
        pfc_currents_next=cpu.state.pfc_currents + cpu.t_step * pfc,
        sol_currents_next=cpu.state.sol_currents + cpu.t_step * sol,
    )
    gpu_state = gpu.step_currents(
        pfc_currents_next=gpu.state.pfc_currents + gpu.t_step * pfc,
        sol_currents_next=gpu.state.sol_currents + gpu.t_step * sol,
    )
    batched_next = np.tile(np.concatenate([cpu.pfc.currents, cpu.sol.currents]) + cpu.t_step * np.concatenate([pfc, sol]), (2, 1))
    batched_state = batched.step_currents(batched_next).state

    assert np.isclose(gpu_state.Ip, cpu_state.Ip)
    assert np.allclose(gpu_state.pfc_currents, cpu_state.pfc_currents)
    assert np.allclose(gpu_state.sol_currents, cpu_state.sol_currents)
    assert np.allclose(batched_state.Ip.detach().cpu().numpy(), np.full((2,), cpu_state.Ip))
