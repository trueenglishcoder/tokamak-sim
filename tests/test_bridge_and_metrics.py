"""Проверки programmatic bridge API и чистых метрик."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tokamak_control.bridge import DerivativeAction, SimulationSession
from tokamak_control.config.settings import PhysicsSettings
from tokamak_control.core.coils import Coil, CoilActuator, CoilGroup
from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.io.config_io import dump_config
from tokamak_control.metrics import current_limit_margin, ip_abs_error, normalized_radii_rmse, radii_error


def _write_bridge_config(path: Path) -> None:
    """Записать маленькую физически валидную конфигурацию с T15MD limiter."""
    grid = Grid2D(
        r=Grid1D(start=0.2, step=0.02, size=121, center=1.2),
        z=Grid1D(start=-1.2, step=0.02, size=121, center=0.0),
    )
    pfc = CoilGroup(
        name="pfc",
        coils=[
            CoilActuator([Coil(0.8, 0.8)]),
            CoilActuator([Coil(1.6, -0.8)]),
        ],
        currents=np.array([1000.0, -1000.0], dtype=float),
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
        pfc_current_limit=1.0e5,
        sol_current_limit=1.0e5,
        pfc_deriv_limit=1.0e6,
        sol_deriv_limit=1.0e6,
    )
    dump_config(
        path,
        grid=grid,
        pfc=pfc,
        sol=sol,
        physics=physics,
        limiter_name="T15MD",
        boundary_mode="limited",
    )


def test_simulation_session_reset_and_step_shapes(tmp_path: Path) -> None:
    """Проверить reset/step bridge-сессии и стабильные формы основных массивов."""
    config_path = tmp_path / "bridge_machine.toml"
    _write_bridge_config(config_path)

    session = SimulationSession.from_paths(
        config_path=config_path,
        initial_currents_path=None,
        scenario_name="nominal",
        scenario_args={},
        angles=8,
        steps=2,
    )
    reset = session.reset()

    assert reset.machine.n_active_pfc == 2
    assert reset.machine.n_active_sol == 1
    assert reset.machine.n_active_total == 3
    assert reset.machine.active_order == ("pfc_0", "pfc_1", "sol_0")
    assert reset.observation_snapshot.true_radii is not None
    assert reset.observation_snapshot.true_radii.shape == (8,)

    action = DerivativeAction(np.array([1000.0, -2000.0, 500.0], dtype=float))
    step = session.step_derivatives(action)

    assert not step.terminated
    assert not step.truncated
    assert step.snapshot.step_index == 1
    assert step.snapshot.commanded_active_derivatives.shape == (3,)
    assert step.snapshot.applied_active_derivatives.shape == (3,)
    assert step.snapshot.previous_applied_active_derivatives.shape == (3,)
    assert step.snapshot.true_active_currents.shape == (3,)
    assert step.snapshot.reference.radii_ref.shape == (8,)
    assert step.snapshot.true_boundary_poly is not None
    assert step.snapshot.true_boundary_poly.shape[1] == 2


def test_simulation_session_truncates_at_configured_steps(tmp_path: Path) -> None:
    """Проверить, что bridge отличает нормальное окончание episode от failure."""
    config_path = tmp_path / "bridge_machine.toml"
    _write_bridge_config(config_path)
    session = SimulationSession.from_paths(config_path, None, "nominal", {}, 8, 1)
    machine = session.reset().machine

    result = session.step_derivatives(DerivativeAction(np.zeros((machine.n_active_total,), dtype=float)))

    assert result.truncated
    assert not result.terminated
    assert result.termination_reason is None


def test_metrics_are_physical_values_not_rewards() -> None:
    """Проверить чистые ошибки и actuator margins без весов reward-функций."""
    assert ip_abs_error(9.0, 12.5) == 3.5
    err = radii_error(np.array([1.0, 1.2]), np.array([0.9, 1.5]))
    assert np.allclose(err, np.array([0.1, -0.3]))
    assert np.isclose(normalized_radii_rmse(np.array([1.0, 1.2]), np.array([0.9, 1.5]), 2.0), np.sqrt(0.05) / 2.0)
    assert np.allclose(current_limit_margin(np.array([2.0, -5.0]), np.array([10.0, 10.0])), np.array([0.8, 0.5]))
