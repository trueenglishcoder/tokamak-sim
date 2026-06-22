"""Проверки programmatic bridge API и чистых метрик."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tokamak_control.bridge import CurrentAction, InitialStateOverride, SimulationSession
from tokamak_control.config.settings import PhysicsSettings
from tokamak_control.core.coils import Coil, CoilActuator, CoilGroup
from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.io.config_io import dump_config
from tokamak_control.metrics import current_limit_margin, ip_abs_error, normalized_radii_rmse, radii_error
from tokamak_control.realism import ActuatorRealismSettings, RealismRuntime, RealismSettings, SensorRealismSettings


def _write_bridge_config(path: Path, *, realism: RealismSettings | None = None) -> None:
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
        realism=realism,
        limiter_name="T15MD",
        boundary_mode="legacy_contour",
    )


def test_realism_runtime_identity_and_seeded_measurements() -> None:
    """Проверить identity-поведение и воспроизводимость шумов neutral realism."""
    boundary = np.array([[1.0, 0.0], [1.1, 0.0], [1.0, 0.1], [1.0, 0.0]], dtype=float)
    radii = np.array([0.1, 0.2, 0.3], dtype=float)
    currents = np.array([10.0, -20.0], dtype=float)

    identity = RealismRuntime(RealismSettings())
    measured = identity.measure(
        true_ip=123.0,
        true_active_currents=currents,
        true_boundary_poly=boundary,
        true_radii=radii,
    )
    assert measured.measured_ip == 123.0
    assert np.allclose(measured.measured_active_currents, currents)
    assert np.allclose(measured.measured_boundary_poly, boundary)
    assert np.allclose(measured.measured_radii, radii)

    noisy_settings = RealismSettings(
        seed=42,
        actuators=ActuatorRealismSettings(pfc_command_noise_sigma=1.0),
        sensors=SensorRealismSettings(ip_noise_sigma=1.0, active_current_noise_sigma=1.0, radii_noise_sigma=1.0),
    )
    a = RealismRuntime(noisy_settings)
    b = RealismRuntime(noisy_settings)
    ma = a.measure(true_ip=123.0, true_active_currents=currents, true_boundary_poly=boundary, true_radii=radii)
    mb = b.measure(true_ip=123.0, true_active_currents=currents, true_boundary_poly=boundary, true_radii=radii)

    assert ma.measured_ip != 123.0
    assert np.allclose(ma.measured_active_currents, mb.measured_active_currents)
    assert np.allclose(ma.measured_radii, mb.measured_radii)


def test_realism_runtime_does_not_invent_missing_boundary() -> None:
    """Проверить, что missing boundary не заменяется старым или синтетическим контуром."""
    runtime = RealismRuntime(RealismSettings(sensors=SensorRealismSettings(boundary_xy_noise_sigma=1.0, radii_noise_sigma=1.0)))
    measured = runtime.measure(
        true_ip=1.0,
        true_active_currents=np.array([0.0]),
        true_boundary_poly=None,
        true_radii=None,
    )

    assert measured.measured_boundary_poly is None
    assert measured.measured_radii is None


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
    assert reset.observation_snapshot.true_radii.shape == reset.machine.angles_rad.shape

    requested_derivs = np.array([1000.0, -2000.0, 500.0], dtype=float)
    action = CurrentAction(reset.observation_snapshot.true_active_currents + reset.machine.t_step * requested_derivs)
    step = session.step_currents(action)

    assert not step.terminated
    assert not step.truncated
    assert step.snapshot.step_index == 1
    assert step.snapshot.commanded_active_derivatives.shape == (3,)
    assert step.snapshot.applied_active_derivatives.shape == (3,)
    assert step.snapshot.previous_applied_active_derivatives.shape == (3,)
    assert step.snapshot.true_active_currents.shape == (3,)
    assert step.snapshot.reference.radii_ref.shape == reset.machine.angles_rad.shape
    assert step.snapshot.true_boundary_poly is not None
    assert step.snapshot.true_boundary_poly.shape[1] == 2


def test_simulation_session_accepts_zero_initial_state_override(tmp_path: Path) -> None:
    config_path = tmp_path / "bridge_machine_zero.toml"
    _write_bridge_config(config_path)
    session = SimulationSession.from_paths(
        config_path=config_path,
        initial_currents_path=None,
        scenario_name="t15_synthetic_follow",
        scenario_args={
            "duration_s": 0.05,
            "t_step": 1.0e-3,
            "ip_start": 0.0,
            "ip_end": 10_000.0,
            "ip_ramp_s": 0.05,
            "boundary_kind": "static_parameters",
            "boundary_parameters": {"R0": 1.2, "Z0": 0.0, "A0": 0.35, "kappa": 1.0, "delta": 0.0},
        },
        angles=8,
        steps=2,
        initial_state_override=InitialStateOverride(ip=0.0, coil_currents="zero", ip_scale=5.0e5),
    )

    reset = session.reset(seed=5)

    assert reset.observation_snapshot.true_ip == 0.0
    assert np.allclose(reset.observation_snapshot.true_active_currents, np.zeros((reset.machine.n_active_total,), dtype=float))
    assert reset.observation_snapshot.reference.ip_ref == 0.0
    assert reset.machine.ip_scale == 5.0e5
    assert reset.episode_metadata["initial_state_override"] == {
        "enabled": True,
        "ip": 0.0,
        "coil_currents": "zero",
        "ip_scale": 5.0e5,
    }

    step = session.step_currents(CurrentAction(reset.observation_snapshot.true_active_currents.copy()))

    assert not step.terminated
    assert step.snapshot.step_index == 1


def test_simulation_session_exposes_measured_channels_with_realism(tmp_path: Path) -> None:
    """Проверить, что bridge отдает measured channels из neutral realism."""
    config_path = tmp_path / "bridge_machine_realism.toml"
    realism = RealismSettings(
        seed=11,
        sensors=SensorRealismSettings(
            ip_bias=10.0,
            active_current_noise_sigma=5.0,
            radii_noise_sigma=0.001,
        ),
    )
    _write_bridge_config(config_path, realism=realism)

    session = SimulationSession.from_paths(
        config_path=config_path,
        initial_currents_path=None,
        scenario_name="nominal",
        scenario_args={},
        angles=8,
        steps=2,
        realism_enabled=True,
    )
    reset = session.reset(seed=11)

    snap = reset.observation_snapshot
    assert snap.measured_active_currents.shape == snap.true_active_currents.shape
    assert snap.measured_radii is not None
    assert snap.true_radii is not None
    assert snap.measured_boundary_poly is not None
    assert snap.measured_ip == snap.true_ip + 10.0
    assert not np.allclose(snap.measured_active_currents, snap.true_active_currents)
    assert not np.allclose(snap.measured_radii, snap.true_radii)


def test_simulation_session_accepts_realism_override_on_reset(tmp_path: Path) -> None:
    config_path = tmp_path / "bridge_machine_override.toml"
    _write_bridge_config(config_path)
    override = RealismSettings(enabled=True, sensors=SensorRealismSettings(ip_bias=17.0))

    session = SimulationSession.from_paths(
        config_path=config_path,
        initial_currents_path=None,
        scenario_name="nominal",
        scenario_args={},
        angles=8,
        steps=2,
        realism_enabled=False,
        realism_settings=override,
    )
    reset = session.reset(seed=4)

    snap = reset.observation_snapshot
    assert np.isclose(snap.measured_ip, snap.true_ip + 17.0)
    assert reset.episode_metadata["realism_active"] is True
    assert np.isclose(reset.episode_metadata["realism_settings"]["sensors"]["ip_bias"], 17.0)


def test_reference_at_time_matches_step_snapshot_reference(tmp_path: Path) -> None:
    config_path = tmp_path / "bridge_machine_reference.toml"
    _write_bridge_config(config_path)
    session = SimulationSession.from_paths(
        config_path=config_path,
        initial_currents_path=None,
        scenario_name="t15_synthetic_follow",
        scenario_args={
            "duration_s": 0.05,
            "t_step": 1.0e-3,
            "target_update_s": 0.01,
            "boundary_kind": "static_parameters",
            "boundary_parameters": {"R0": 1.2, "Z0": 0.0, "A0": 0.35, "kappa": 1.0, "delta": 0.0},
            "ip_start": 10_000.0,
            "ip_end": 20_000.0,
            "ip_ramp_s": 0.05,
        },
        angles=8,
        steps=4,
        realism_enabled=False,
    )
    reset = session.reset(seed=3)
    reset_ref = session.reference_at_time(reset.observation_snapshot.time_s)

    assert np.isclose(reset_ref.ip_ref, reset.observation_snapshot.reference.ip_ref)
    assert np.allclose(reset_ref.radii_ref, reset.observation_snapshot.reference.radii_ref)

    step = session.step_currents(CurrentAction(reset.observation_snapshot.true_active_currents.copy()))
    step_ref = session.reference_at_time(step.snapshot.time_s)

    assert np.isclose(step_ref.ip_ref, step.snapshot.reference.ip_ref)
    assert np.allclose(step_ref.radii_ref, step.snapshot.reference.radii_ref)


def test_simulation_session_truncates_at_configured_steps(tmp_path: Path) -> None:
    """Проверить, что bridge отличает нормальное окончание episode от failure."""
    config_path = tmp_path / "bridge_machine.toml"
    _write_bridge_config(config_path)
    session = SimulationSession.from_paths(config_path, None, "nominal", {}, 8, 1)
    reset = session.reset()
    machine = reset.machine

    result = session.step_currents(CurrentAction(reset.observation_snapshot.true_active_currents.copy()))

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
