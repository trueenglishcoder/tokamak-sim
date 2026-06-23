from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tokamak_control.config.scenarios import make_scenario
from tokamak_control.control.learned_magnetic_controller import (
    CONTROLLER_STATE_V4_FEATURE_ORDER,
    CONTROLLER_STATE_V5_FEATURE_ORDER,
    CONTROLLER_STATE_V6_FEATURE_ORDER,
)
from tokamak_control.control.registry import build_controller_runtime_call, controller_runtime_inputs, make_controller
from tokamak_control.core.plasma_model import PlasmaModel
from tokamak_control.geometry.boundary import find_plasma_boundary_with_status
from tokamak_control.geometry.legacy_metrics import legacy_radii_at_angles
from tokamak_control.io.config_io import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/T15MD_new_data.toml"
INITIAL = REPO_ROOT / "configs/initial_currents/T15MD_new_data_3864.toml"


def _feature_slices(
    *,
    action_dim: int,
    n_angles: int,
    preview_steps: int,
    feature_order: list[str] | None = None,
) -> tuple[dict[str, list[int]], int]:
    """Return v4 feature slices and total observation size."""
    sizes = {
        "step_norm": 1,
        "ip": 1,
        "ip_ref": 1,
        "ip_error": 1,
        "active_currents": action_dim,
        "active_current_derivs": action_dim,
        "measured_boundary_radii": n_angles,
        "ref_radii": n_angles,
        "boundary_radii_error": n_angles,
        "boundary_found": 1,
        "ip_ref_rate": 1,
        "boundary_ref_rate": n_angles,
        "ip_measured_rate": 1,
        "integral_ip_error": 1,
        "integral_boundary_radii_error": n_angles,
        "previous_action": action_dim,
        "target_preview": preview_steps * (2 + n_angles),
    }
    out: dict[str, list[int]] = {}
    cursor = 0
    order = CONTROLLER_STATE_V4_FEATURE_ORDER if feature_order is None else feature_order
    for name in order:
        size = int(sizes[name])
        out[name] = [cursor, cursor + size]
        cursor += size
    return out, cursor


def _write_v4_export(
    export: Path,
    *,
    model: PlasmaModel,
    n_angles: int,
    mean_bias: float = 0.0,
    action_contract: str = "absolute_jdot_command_v1",
) -> np.ndarray:
    """Write a minimal v4 learned-controller bundle and return derivative limits."""
    n_pfc = int(model.pfc.n_coils)
    n_sol = int(model.sol.n_coils)
    action_dim = n_pfc + n_sol
    slices, obs_dim = _feature_slices(action_dim=action_dim, n_angles=n_angles, preview_steps=0)
    derivative_scale = np.linspace(1.0e6, 9.0e6, action_dim, dtype=float)
    export.mkdir()
    schema = {
        "observation_kind": "controller_state_v4",
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "n_active_total": action_dim,
        "n_pfc": n_pfc,
        "n_sol": n_sol,
        "n_angles": n_angles,
        "angles_rad": np.linspace(-np.pi, np.pi, n_angles, endpoint=False, dtype=float).tolist(),
        "grid_shape": list(model.state.psi.shape),
        "feature_order": CONTROLLER_STATE_V4_FEATURE_ORDER,
        "feature_slices": slices,
        "target_preview_steps": 0,
        "target_preview_stride": 1,
        "action_contract": action_contract,
    }
    (export / "controller_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (export / "normalization.json").write_text(
        json.dumps(
            {
                "ip_scale": 5.0e5,
                "radius_scale": 1.0,
                "current_scale": [1.0e6] * action_dim,
                "derivative_scale": derivative_scale.tolist(),
            }
        ),
        encoding="utf-8",
    )
    (export / "metadata.json").write_text(json.dumps({"layer_norm_eps": 1.0e-5}), encoding="utf-8")
    hidden = 8
    np.savez(
        export / "policy_weights.npz",
        **{
            "input.weight": np.zeros((hidden, obs_dim), dtype=np.float32),
            "input.bias": np.zeros((hidden,), dtype=np.float32),
            "input_norm.weight": np.ones((hidden,), dtype=np.float32),
            "input_norm.bias": np.zeros((hidden,), dtype=np.float32),
            "hidden1.weight": np.zeros((hidden, hidden), dtype=np.float32),
            "hidden1.bias": np.zeros((hidden,), dtype=np.float32),
            "mean_head.weight": np.zeros((action_dim, hidden), dtype=np.float32),
            "mean_head.bias": np.full((action_dim,), float(mean_bias), dtype=np.float32),
        },
    )
    return derivative_scale


def _write_v5_export(
    export: Path,
    *,
    model: PlasmaModel,
    n_angles: int,
    mean_bias: float = 0.0,
    target_preview_steps: int = 0,
    target_preview_stride: int = 1,
) -> np.ndarray:
    """Write a minimal v5 learned-controller bundle and return derivative limits."""
    n_pfc = int(model.pfc.n_coils)
    n_sol = int(model.sol.n_coils)
    action_dim = n_pfc + n_sol
    slices, obs_dim = _feature_slices(
        action_dim=action_dim,
        n_angles=n_angles,
        preview_steps=target_preview_steps,
        feature_order=CONTROLLER_STATE_V5_FEATURE_ORDER,
    )
    derivative_scale = np.linspace(1.0e6, 9.0e6, action_dim, dtype=float)
    export.mkdir()
    schema = {
        "observation_kind": "controller_state_v5",
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "n_active_total": action_dim,
        "n_pfc": n_pfc,
        "n_sol": n_sol,
        "n_angles": n_angles,
        "angles_rad": np.linspace(-np.pi, np.pi, n_angles, endpoint=False, dtype=float).tolist(),
        "grid_shape": list(model.state.psi.shape),
        "feature_order": CONTROLLER_STATE_V5_FEATURE_ORDER,
        "feature_slices": slices,
        "target_preview_steps": target_preview_steps,
        "target_preview_stride": target_preview_stride,
        "ip_rate_scale_aps": 5.0e5,
        "boundary_rate_scale_mps": 1.0,
        "action_contract": "absolute_jdot_command_v1",
    }
    (export / "controller_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (export / "normalization.json").write_text(
        json.dumps(
            {
                "ip_scale": 5.0e5,
                "radius_scale": 1.0,
                "current_scale": [1.0e6] * action_dim,
                "derivative_scale": derivative_scale.tolist(),
                "ip_rate_scale_aps": 5.0e5,
                "boundary_rate_scale_mps": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (export / "metadata.json").write_text(json.dumps({"layer_norm_eps": 1.0e-5}), encoding="utf-8")
    hidden = 8
    np.savez(
        export / "policy_weights.npz",
        **{
            "input.weight": np.zeros((hidden, obs_dim), dtype=np.float32),
            "input.bias": np.zeros((hidden,), dtype=np.float32),
            "input_norm.weight": np.ones((hidden,), dtype=np.float32),
            "input_norm.bias": np.zeros((hidden,), dtype=np.float32),
            "hidden1.weight": np.zeros((hidden, hidden), dtype=np.float32),
            "hidden1.bias": np.zeros((hidden,), dtype=np.float32),
            "mean_head.weight": np.zeros((action_dim, hidden), dtype=np.float32),
            "mean_head.bias": np.full((action_dim,), float(mean_bias), dtype=np.float32),
        },
    )
    return derivative_scale


def _write_v6_export(
    export: Path,
    *,
    model: PlasmaModel,
    n_angles: int,
    mean_bias: float = 0.0,
    target_preview_steps: int = 0,
    target_preview_stride: int = 1,
) -> np.ndarray:
    """Write a minimal v6 learned-controller bundle and return derivative limits."""
    n_pfc = int(model.pfc.n_coils)
    n_sol = int(model.sol.n_coils)
    action_dim = n_pfc + n_sol
    slices, obs_dim = _feature_slices(
        action_dim=action_dim,
        n_angles=n_angles,
        preview_steps=target_preview_steps,
        feature_order=CONTROLLER_STATE_V6_FEATURE_ORDER,
    )
    derivative_scale = np.linspace(1.0e6, 9.0e6, action_dim, dtype=float)
    export.mkdir()
    schema = {
        "observation_kind": "controller_state_v6",
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "n_active_total": action_dim,
        "n_pfc": n_pfc,
        "n_sol": n_sol,
        "n_angles": n_angles,
        "angles_rad": np.linspace(-np.pi, np.pi, n_angles, endpoint=False, dtype=float).tolist(),
        "grid_shape": list(model.state.psi.shape),
        "feature_order": CONTROLLER_STATE_V6_FEATURE_ORDER,
        "feature_slices": slices,
        "target_preview_steps": target_preview_steps,
        "target_preview_stride": target_preview_stride,
        "ip_rate_scale_aps": 5.0e5,
        "boundary_rate_scale_mps": 1.0,
        "action_contract": "absolute_jdot_command_v1",
    }
    (export / "controller_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (export / "normalization.json").write_text(
        json.dumps(
            {
                "ip_scale": 5.0e5,
                "radius_scale": 1.0,
                "current_scale": [1.0e6] * action_dim,
                "derivative_scale": derivative_scale.tolist(),
                "ip_rate_scale_aps": 5.0e5,
                "boundary_rate_scale_mps": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (export / "metadata.json").write_text(json.dumps({"layer_norm_eps": 1.0e-5}), encoding="utf-8")
    hidden = 8
    np.savez(
        export / "policy_weights.npz",
        **{
            "input.weight": np.zeros((hidden, obs_dim), dtype=np.float32),
            "input.bias": np.zeros((hidden,), dtype=np.float32),
            "input_norm.weight": np.ones((hidden,), dtype=np.float32),
            "input_norm.bias": np.zeros((hidden,), dtype=np.float32),
            "hidden1.weight": np.zeros((hidden, hidden), dtype=np.float32),
            "hidden1.bias": np.zeros((hidden,), dtype=np.float32),
            "mean_head.weight": np.zeros((action_dim, hidden), dtype=np.float32),
            "mean_head.bias": np.full((action_dim,), float(mean_bias), dtype=np.float32),
        },
    )
    return derivative_scale


class _LinearReferenceScenario:
    def __init__(self, *, ip0: float, ip_rate: float, radii0: np.ndarray, boundary_rate: np.ndarray) -> None:
        self.ip0 = float(ip0)
        self.ip_rate = float(ip_rate)
        self.radii0 = np.asarray(radii0, dtype=float).reshape(-1)
        self.boundary_rate = np.asarray(boundary_rate, dtype=float).reshape(-1)

    def Ip_ref(self, t: float) -> float:
        return self.ip0 + self.ip_rate * float(t)

    def ref_radii(self, angles: np.ndarray, t: float) -> np.ndarray:
        del angles
        return self.radii0 + self.boundary_rate * float(t)


def _runtime_context(model: PlasmaModel, cfg, *, n_angles: int) -> dict[str, object]:
    """Build a learned-controller runtime context from the current model state."""
    angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False, dtype=float)
    poly, _level, _status = find_plasma_boundary_with_status(
        model.state.psi,
        model.grid,
        (model.R0, model.Z0),
        limiter_shape=cfg.limiter_shape,
        boundary_mode=cfg.boundary_mode,
    )
    ref_radii = legacy_radii_at_angles(poly, (model.R0, model.Z0), angles)
    scenario = make_scenario("nominal", ref_radii, model.Ip0, params={}, center=(model.R0, model.Z0))
    return {
        "model": model,
        "psi": model.compute_psi(),
        "boundary_poly": poly,
        "center": (model.R0, model.Z0),
        "measure_angles": angles,
        "ref_radii": ref_radii,
        "Ip_ref": model.Ip0,
        "scenario": scenario,
        "max_episode_steps": 10,
        "ignored_extra": object(),
    }


def test_learned_controller_runtime_inputs_are_current_contract() -> None:
    """Registry should expose only inputs needed by the active v3 controller."""
    runtime_inputs = controller_runtime_inputs("learned_magnetic_controller")
    assert "model" in runtime_inputs
    assert "psi" in runtime_inputs
    assert "boundary_poly" in runtime_inputs
    assert "max_episode_steps" in runtime_inputs
    assert "measured_ip" not in runtime_inputs
    assert "measured_radii" not in runtime_inputs


def test_learned_controller_v4_outputs_absolute_next_currents(tmp_path: Path) -> None:
    """The v4 controller maps actor output directly to absolute Jdot and next currents."""
    cfg = load_config(CONFIG, initial_currents_path=INITIAL)
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics)
    export = tmp_path / "export"
    derivative_scale = _write_v4_export(export, model=model, n_angles=8, mean_bias=0.5)
    controller = make_controller("learned_magnetic_controller", config={"export_dir": export})
    call = build_controller_runtime_call("learned_magnetic_controller", _runtime_context(model, cfg, n_angles=8))

    current_now = np.concatenate([model.state.pfc_currents, model.state.sol_currents]).astype(float)
    first = controller.compute_control(**call)
    first_next = np.concatenate([first.pfc_currents_next, first.sol_currents_next])
    first_jdot = (first_next - current_now) / float(model.t_step)

    requested_jdot = float(np.tanh(0.5))
    assert np.allclose(first_jdot, requested_jdot * derivative_scale, rtol=1.0e-6, atol=1.0e-6)
    assert np.allclose(controller._previous_action_norm, requested_jdot)

    second = controller.compute_control(**call)
    second_next = np.concatenate([second.pfc_currents_next, second.sol_currents_next])
    second_jdot = (second_next - current_now) / float(model.t_step)
    assert np.allclose(second_jdot, requested_jdot * derivative_scale, rtol=1.0e-6, atol=1.0e-6)


def test_learned_controller_zero_action_holds_currents(tmp_path: Path) -> None:
    """A zero actor output should command the same absolute currents again."""
    cfg = load_config(CONFIG, initial_currents_path=INITIAL)
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics)
    export = tmp_path / "export"
    _write_v4_export(export, model=model, n_angles=4, mean_bias=0.0)
    controller = make_controller("learned_magnetic_controller", config={"export_dir": export})
    call = build_controller_runtime_call("learned_magnetic_controller", _runtime_context(model, cfg, n_angles=4))
    action = controller.compute_control(**call)

    assert np.allclose(action.pfc_currents_next, model.state.pfc_currents)
    assert np.allclose(action.sol_currents_next, model.state.sol_currents)


def test_learned_controller_v5_observation_rates_match_training_units(tmp_path: Path) -> None:
    """v5 exports should receive Ip/reference rate features in the same units as training."""
    cfg = load_config(CONFIG, initial_currents_path=INITIAL)
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics)
    export = tmp_path / "export_v5"
    _write_v5_export(export, model=model, n_angles=6, mean_bias=0.0)
    controller = make_controller("learned_magnetic_controller", config={"export_dir": export})
    ctx = _runtime_context(model, cfg, n_angles=6)

    angles = np.asarray(ctx["measure_angles"], dtype=float)
    t0 = float(model.state.t)
    ref_now = np.asarray(ctx["ref_radii"], dtype=float)
    ip_rate = 2.5e5
    boundary_rate = np.linspace(-0.20, 0.30, angles.size, dtype=float)
    scenario = _LinearReferenceScenario(
        ip0=float(ctx["Ip_ref"]) - ip_rate * t0,
        ip_rate=ip_rate,
        radii0=ref_now - boundary_rate * t0,
        boundary_rate=boundary_rate,
    )
    measured_ip = float(model.state.Ip)
    controller._previous_ip = measured_ip - 250.0

    obs = controller._observation(
        model=ctx["model"],
        psi=ctx["psi"],
        boundary_poly=ctx["boundary_poly"],
        center=ctx["center"],
        measure_angles=angles,
        ref_radii=scenario.ref_radii(angles, t0),
        ip_ref=scenario.Ip_ref(t0),
        scenario=scenario,
        max_episode_steps=100,
    )
    slices = {name: tuple(bounds) for name, bounds in controller.schema["feature_slices"].items()}
    dt = float(model.t_step)

    start, stop = slices["ip_ref_rate"]
    assert np.allclose(obs[start:stop], np.asarray([ip_rate / 5.0e5], dtype=np.float32), rtol=1.0e-5, atol=1.0e-6)
    start, stop = slices["boundary_ref_rate"]
    assert np.allclose(obs[start:stop], boundary_rate.astype(np.float32), rtol=1.0e-5, atol=1.0e-6)
    start, stop = slices["ip_measured_rate"]
    expected_measured_rate = ((measured_ip - (measured_ip - 250.0)) / dt) / 5.0e5
    assert np.allclose(obs[start:stop], np.asarray([expected_measured_rate], dtype=np.float32), rtol=1.0e-5, atol=1.0e-6)


def test_learned_controller_v5_outputs_absolute_next_currents(tmp_path: Path) -> None:
    """The v5 controller keeps the same absolute-Jdot action contract as v4."""
    cfg = load_config(CONFIG, initial_currents_path=INITIAL)
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics)
    export = tmp_path / "export_v5"
    derivative_scale = _write_v5_export(export, model=model, n_angles=8, mean_bias=0.25)
    controller = make_controller("learned_magnetic_controller", config={"export_dir": export})
    call = build_controller_runtime_call("learned_magnetic_controller", _runtime_context(model, cfg, n_angles=8))

    current_now = np.concatenate([model.state.pfc_currents, model.state.sol_currents]).astype(float)
    first = controller.compute_control(**call)
    first_next = np.concatenate([first.pfc_currents_next, first.sol_currents_next])
    first_jdot = (first_next - current_now) / float(model.t_step)

    requested_jdot = float(np.tanh(0.25))
    assert np.allclose(first_jdot, requested_jdot * derivative_scale, rtol=1.0e-6, atol=1.0e-6)
    assert np.allclose(controller._previous_action_norm, requested_jdot)
    assert controller._previous_ip == pytest.approx(float(model.state.Ip))


def test_learned_controller_v6_outputs_absolute_next_currents(tmp_path: Path) -> None:
    """The v6 controller accepts current exports and keeps the absolute-Jdot contract."""
    cfg = load_config(CONFIG, initial_currents_path=INITIAL)
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics)
    export = tmp_path / "export_v6"
    derivative_scale = _write_v6_export(export, model=model, n_angles=8, mean_bias=0.25)
    controller = make_controller("learned_magnetic_controller", config={"export_dir": export})
    call = build_controller_runtime_call("learned_magnetic_controller", _runtime_context(model, cfg, n_angles=8))

    current_now = np.concatenate([model.state.pfc_currents, model.state.sol_currents]).astype(float)
    action = controller.compute_control(**call)
    next_current = np.concatenate([action.pfc_currents_next, action.sol_currents_next])
    jdot = (next_current - current_now) / float(model.t_step)

    requested_jdot = float(np.tanh(0.25))
    assert np.allclose(jdot, requested_jdot * derivative_scale, rtol=1.0e-6, atol=1.0e-6)
    assert np.allclose(controller._previous_action_norm, requested_jdot)
    assert controller._previous_ip == pytest.approx(float(model.state.Ip))


def test_learned_controller_v6_integral_features_match_training_units(tmp_path: Path) -> None:
    """v6 exports should receive accumulated Ip and boundary error features."""
    cfg = load_config(CONFIG, initial_currents_path=INITIAL)
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics)
    export = tmp_path / "export_v6"
    _write_v6_export(export, model=model, n_angles=6, mean_bias=0.0)
    controller = make_controller("learned_magnetic_controller", config={"export_dir": export})
    ctx = _runtime_context(model, cfg, n_angles=6)

    angles = np.asarray(ctx["measure_angles"], dtype=float)
    ref_now = np.asarray(ctx["ref_radii"], dtype=float)
    measured_ip = float(model.state.Ip)
    scenario = _LinearReferenceScenario(
        ip0=measured_ip + 15000.0,
        ip_rate=0.0,
        radii0=ref_now + 0.002,
        boundary_rate=np.zeros_like(ref_now),
    )
    slices = {name: tuple(bounds) for name, bounds in controller.schema["feature_slices"].items()}

    obs0 = controller._observation(
        model=ctx["model"],
        psi=ctx["psi"],
        boundary_poly=ctx["boundary_poly"],
        center=ctx["center"],
        measure_angles=angles,
        ref_radii=scenario.ref_radii(angles, float(model.state.t)),
        ip_ref=scenario.Ip_ref(float(model.state.t)),
        scenario=scenario,
        max_episode_steps=100,
    )
    start, stop = slices["integral_ip_error"]
    assert np.allclose(obs0[start:stop], np.zeros((1,), dtype=np.float32))
    start, stop = slices["integral_boundary_radii_error"]
    assert np.allclose(obs0[start:stop], np.zeros((angles.size,), dtype=np.float32))

    model.state.step += 1
    model.state.t += float(model.t_step)
    obs1 = controller._observation(
        model=ctx["model"],
        psi=ctx["psi"],
        boundary_poly=ctx["boundary_poly"],
        center=ctx["center"],
        measure_angles=angles,
        ref_radii=scenario.ref_radii(angles, float(model.state.t)),
        ip_ref=scenario.Ip_ref(float(model.state.t)),
        scenario=scenario,
        max_episode_steps=100,
    )
    start, stop = slices["integral_ip_error"]
    expected_ip = float(model.t_step) * 15000.0 / (15000.0 * 0.1)
    assert np.allclose(obs1[start:stop], np.asarray([expected_ip], dtype=np.float32), rtol=1.0e-5, atol=1.0e-6)
    start, stop = slices["integral_boundary_radii_error"]
    expected_boundary = np.full((angles.size,), float(model.t_step) * 0.002 / (0.02 * 0.1), dtype=np.float32)
    assert np.allclose(obs1[start:stop], expected_boundary, rtol=1.0e-5, atol=1.0e-6)


def test_learned_controller_can_use_rolling_training_horizon_norm(tmp_path: Path) -> None:
    """Full-shot deployment can keep the 0.1 s training step normalization."""
    cfg = load_config(CONFIG, initial_currents_path=INITIAL)
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics)
    model.state.step = 157
    export = tmp_path / "export"
    _write_v4_export(export, model=model, n_angles=4, mean_bias=0.0)
    controller = make_controller(
        "learned_magnetic_controller",
        config={"export_dir": export, "episode_norm_steps": 100, "rolling_episode_norm": True},
    )
    ctx = _runtime_context(model, cfg, n_angles=4)
    obs = controller._observation(
        model=ctx["model"],
        psi=ctx["psi"],
        boundary_poly=ctx["boundary_poly"],
        center=ctx["center"],
        measure_angles=ctx["measure_angles"],
        ref_radii=ctx["ref_radii"],
        ip_ref=float(ctx["Ip_ref"]),
        scenario=ctx["scenario"],
        max_episode_steps=1439,
    )
    slices, _obs_dim = _feature_slices(action_dim=model.pfc.n_coils + model.sol.n_coils, n_angles=4, preview_steps=0)
    start, stop = slices["step_norm"]
    assert np.allclose(obs[start:stop], np.asarray([57.0 / 100.0], dtype=np.float32))


def test_learned_controller_rejects_old_action_contract(tmp_path: Path) -> None:
    """Old learned exports must not silently run under the v4 plant contract."""
    cfg = load_config(CONFIG, initial_currents_path=INITIAL)
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics)
    export = tmp_path / "export_old"
    _write_v4_export(export, model=model, n_angles=4, action_contract="delta_jdot_derivative_command_v3")

    with pytest.raises(ValueError, match="absolute_jdot_command_v1"):
        make_controller("learned_magnetic_controller", config={"export_dir": export})
