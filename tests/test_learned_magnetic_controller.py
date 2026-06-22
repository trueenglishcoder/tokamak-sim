from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tokamak_control.config.scenarios import make_scenario
from tokamak_control.control.learned_magnetic_controller import CONTROLLER_STATE_V4_FEATURE_ORDER
from tokamak_control.control.registry import build_controller_runtime_call, controller_runtime_inputs, make_controller
from tokamak_control.core.plasma_model import PlasmaModel
from tokamak_control.geometry.boundary import find_plasma_boundary_with_status
from tokamak_control.geometry.legacy_metrics import legacy_radii_at_angles
from tokamak_control.io.config_io import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/T15MD_new_data.toml"
INITIAL = REPO_ROOT / "configs/initial_currents/T15MD_new_data_3864.toml"


def _feature_slices(*, action_dim: int, n_angles: int, preview_steps: int) -> tuple[dict[str, list[int]], int]:
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
        "previous_action": action_dim,
        "target_preview": preview_steps * (2 + n_angles),
    }
    out: dict[str, list[int]] = {}
    cursor = 0
    for name in CONTROLLER_STATE_V4_FEATURE_ORDER:
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


def test_learned_controller_rejects_old_action_contract(tmp_path: Path) -> None:
    """Old learned exports must not silently run under the v4 plant contract."""
    cfg = load_config(CONFIG, initial_currents_path=INITIAL)
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics)
    export = tmp_path / "export_old"
    _write_v4_export(export, model=model, n_angles=4, action_contract="delta_jdot_derivative_command_v3")

    with pytest.raises(ValueError, match="absolute_jdot_command_v1"):
        make_controller("learned_magnetic_controller", config={"export_dir": export})
