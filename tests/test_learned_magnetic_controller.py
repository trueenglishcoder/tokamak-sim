from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tokamak_control.config.scenarios import make_scenario
from tokamak_control.control.learned_magnetic_controller import COMPACT_JOINT_STATE_V2_FEATURE_ORDER, JOINT_STATE_V1_FEATURE_ORDER
from tokamak_control.control.registry import build_controller_runtime_call, controller_runtime_inputs, make_controller
from tokamak_control.core.plasma_model import PlasmaModel
from tokamak_control.geometry.boundary import find_plasma_boundary_with_status
from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections
from tokamak_control.io.config_io import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/T15MD_new_data.toml"
INITIAL = REPO_ROOT / "configs/initial_currents/T15MD_new_data_3864.toml"


def _feature_slices(*, action_dim: int, psi_size: int, n_angles: int, preview_steps: int) -> dict[str, list[int]]:
    sizes = [
        ("step_norm", 1),
        ("ip", 1),
        ("ip_ref", 1),
        ("ip_error", 1),
        ("active_currents", action_dim),
        ("active_current_derivs", action_dim),
        ("psi_flat", psi_size),
        ("measured_boundary_radii", n_angles),
        ("ref_radii", n_angles),
        ("boundary_radii_error", n_angles),
        ("boundary_found", 1),
        ("target_preview", preview_steps * (2 + n_angles)),
    ]
    out: dict[str, list[int]] = {}
    start = 0
    for name, size in sizes:
        out[name] = [start, start + int(size)]
        start += int(size)
    return out


def _write_joint_state_export(export: Path, *, obs_dim: int, action_dim: int, n_pfc: int, n_sol: int, n_angles: int, grid_shape: tuple[int, int], slices: dict[str, list[int]], normalization: dict[str, object] | None = None, mean_bias: np.ndarray | None = None) -> None:
    export.mkdir()
    schema = {
        "observation_kind": "joint_state_v1",
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "n_active_total": action_dim,
        "n_pfc": n_pfc,
        "n_sol": n_sol,
        "n_angles": n_angles,
        "angles_rad": np.linspace(-np.pi, np.pi, n_angles, endpoint=False, dtype=float).tolist(),
        "grid_shape": list(grid_shape),
        "feature_order": JOINT_STATE_V1_FEATURE_ORDER,
        "feature_slices": slices,
        "target_preview_steps": 0,
        "target_preview_stride": 1,
    }
    (export / "controller_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    norm = {
        "ip_scale": 5.0e5,
        "radius_scale": 1.0,
        "psi_scale": 1.0,
        "current_scale": [1.0e6] * action_dim,
        "derivative_scale": [1.0e6] * action_dim,
    }
    if normalization:
        norm.update(normalization)
    (export / "normalization.json").write_text(json.dumps(norm), encoding="utf-8")
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
            "mean_head.bias": np.zeros((action_dim,), dtype=np.float32) if mean_bias is None else np.asarray(mean_bias, dtype=np.float32).reshape(action_dim),
        },
    )


def test_learned_controller_uses_joint_boundary_state_inputs(tmp_path: Path) -> None:
    cfg = load_config(CONFIG, initial_currents_path=INITIAL)
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics)
    n_pfc = cfg.pfc.n_coils
    n_sol = cfg.sol.n_coils
    action_dim = n_pfc + n_sol
    n_angles = 4
    grid_shape = model.state.psi.shape
    slices = _feature_slices(action_dim=action_dim, psi_size=model.state.psi.size, n_angles=n_angles, preview_steps=0)
    obs_dim = slices["target_preview"][1]
    export = tmp_path / "export"
    _write_joint_state_export(export, obs_dim=obs_dim, action_dim=action_dim, n_pfc=n_pfc, n_sol=n_sol, n_angles=n_angles, grid_shape=grid_shape, slices=slices)

    runtime_inputs = controller_runtime_inputs("learned_magnetic_controller")
    assert "boundary_poly" in runtime_inputs
    assert "psi" in runtime_inputs
    assert "measured_radii" not in runtime_inputs
    assert "measured_ip" not in runtime_inputs
    assert "measured_active_currents" not in runtime_inputs

    controller = make_controller("learned_magnetic_controller", config={"export_dir": export})
    angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False, dtype=float)
    poly, _level, _status = find_plasma_boundary_with_status(
        model.state.psi,
        model.grid,
        (model.R0, model.Z0),
        limiter_shape=cfg.limiter_shape,
        boundary_mode=cfg.boundary_mode,
    )
    ref_radii = radii_from_polyline_ray_intersections(poly, (model.R0, model.Z0), angles) + 0.01
    scenario = make_scenario("nominal", ref_radii, model.Ip0, params={}, center=(model.R0, model.Z0))
    context = {
        "model": model,
        "psi": model.compute_psi(),
        "boundary_poly": poly,
        "center": (model.R0, model.Z0),
        "measure_angles": angles,
        "ref_radii": ref_radii,
        "Ip_ref": model.Ip0,
        "scenario": scenario,
        "max_episode_steps": 10,
        "measured_ip": model.state.Ip,
        "measured_active_currents": np.concatenate([model.state.pfc_currents, model.state.sol_currents]),
        "measured_radii": np.ones((n_angles,), dtype=float),
    }
    call = build_controller_runtime_call("learned_magnetic_controller", context)
    action = controller.compute_control(**call)
    assert action.pfc_derivs.shape == (n_pfc,)
    assert action.sol_derivs.shape == (n_sol,)
    assert np.allclose(action.pfc_derivs, 0.0)
    assert np.allclose(action.sol_derivs, 0.0)

    obs = controller._observation(
        model=call["model"],
        psi=call["psi"],
        boundary_poly=call["boundary_poly"],
        center=call["center"],
        measure_angles=call["measure_angles"],
        ref_radii=call["ref_radii"],
        ip_ref=call["Ip_ref"],
        scenario=call["scenario"],
        max_episode_steps=call["max_episode_steps"],
    )
    measured = radii_from_polyline_ray_intersections(poly, (model.R0, model.Z0), angles)
    i0, i1 = slices["measured_boundary_radii"]
    assert np.allclose(obs[i0:i1], measured.astype(np.float32))
    i0, i1 = slices["ref_radii"]
    assert np.allclose(obs[i0:i1], ref_radii.astype(np.float32))
    i0, i1 = slices["boundary_radii_error"]
    assert np.allclose(obs[i0:i1], (ref_radii - measured).astype(np.float32))
    i0, i1 = slices["boundary_found"]
    assert np.allclose(obs[i0:i1], np.array([1.0], dtype=np.float32))


def test_old_virtual_diagnostic_exports_are_rejected(tmp_path: Path) -> None:
    export = tmp_path / "old_export"
    export.mkdir()
    (export / "controller_schema.json").write_text(json.dumps({"obs_dim": 4, "action_dim": 1, "diagnostics": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="old virtual-diagnostic observation schema"):
        make_controller("learned_magnetic_controller", config={"export_dir": export})


def test_learned_controller_projects_derivatives_to_current_safety(tmp_path: Path) -> None:
    cfg = load_config(CONFIG, initial_currents_path=INITIAL)
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics)
    n_pfc = cfg.pfc.n_coils
    n_sol = cfg.sol.n_coils
    action_dim = n_pfc + n_sol
    n_angles = 4
    grid_shape = model.state.psi.shape
    slices = _feature_slices(action_dim=action_dim, psi_size=model.state.psi.size, n_angles=n_angles, preview_steps=0)
    obs_dim = slices["target_preview"][1]
    current_limits = np.full((action_dim,), 1.0e5, dtype=float)
    derivative_scale = np.full((action_dim,), 1.0e7, dtype=float)
    export = tmp_path / "export"
    _write_joint_state_export(
        export,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_pfc=n_pfc,
        n_sol=n_sol,
        n_angles=n_angles,
        grid_shape=grid_shape,
        slices=slices,
        normalization={
            "current_scale": current_limits.tolist(),
            "derivative_scale": derivative_scale.tolist(),
            "current_projection_enabled": True,
            "current_projection_margin_fraction": 0.02,
        },
        mean_bias=np.full((action_dim,), 8.0, dtype=np.float32),
    )
    model.state.pfc_currents[0] = 0.979 * current_limits[0]
    controller = make_controller("learned_magnetic_controller", config={"export_dir": export})
    angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False, dtype=float)
    poly, _level, _status = find_plasma_boundary_with_status(model.state.psi, model.grid, (model.R0, model.Z0), limiter_shape=cfg.limiter_shape, boundary_mode=cfg.boundary_mode)
    ref_radii = radii_from_polyline_ray_intersections(poly, (model.R0, model.Z0), angles)
    scenario = make_scenario("nominal", ref_radii, model.Ip0, params={}, center=(model.R0, model.Z0))
    action = controller.compute_control(model=model, psi=model.compute_psi(), boundary_poly=poly, center=(model.R0, model.Z0), measure_angles=angles, ref_radii=ref_radii, Ip_ref=model.Ip0, scenario=scenario, max_episode_steps=10)
    full_deriv = np.concatenate([action.pfc_derivs, action.sol_derivs])
    alpha = 0.0 if model.actuator_tau <= 0.0 else float(np.exp(-float(model.t_step) / float(model.actuator_tau)))
    applied = alpha * np.concatenate([model.state.pfc_current_derivs, model.state.sol_current_derivs]) + (1.0 - alpha) * full_deriv
    next_current = np.concatenate([model.state.pfc_currents, model.state.sol_currents]) + float(model.t_step) * applied
    assert np.max(np.abs(next_current) - current_limits) <= 0.0
    assert full_deriv[0] < derivative_scale[0] * 0.5


def test_learned_controller_accepts_compact_joint_state_export(tmp_path: Path) -> None:
    cfg = load_config(CONFIG, initial_currents_path=INITIAL)
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics)
    n_pfc = cfg.pfc.n_coils
    n_sol = cfg.sol.n_coils
    action_dim = n_pfc + n_sol
    n_angles = 4
    preview_steps = 0
    sizes = [
        ("step_norm", 1),
        ("ip", 1),
        ("ip_ref", 1),
        ("ip_error", 1),
        ("active_currents", action_dim),
        ("active_current_derivs", action_dim),
        ("measured_boundary_radii", n_angles),
        ("ref_radii", n_angles),
        ("boundary_radii_error", n_angles),
        ("boundary_found", 1),
        ("previous_action", action_dim),
        ("target_preview", preview_steps * (2 + n_angles)),
    ]
    slices: dict[str, list[int]] = {}
    cursor = 0
    for name, size in sizes:
        slices[name] = [cursor, cursor + int(size)]
        cursor += int(size)
    export = tmp_path / "compact_export"
    export.mkdir()
    schema = {
        "observation_kind": "controller_state_v2",
        "obs_dim": cursor,
        "action_dim": action_dim,
        "n_active_total": action_dim,
        "n_pfc": n_pfc,
        "n_sol": n_sol,
        "n_angles": n_angles,
        "angles_rad": np.linspace(-np.pi, np.pi, n_angles, endpoint=False, dtype=float).tolist(),
        "grid_shape": list(model.state.psi.shape),
        "feature_order": COMPACT_JOINT_STATE_V2_FEATURE_ORDER,
        "feature_slices": slices,
        "target_preview_steps": preview_steps,
        "target_preview_stride": 1,
    }
    (export / "controller_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (export / "normalization.json").write_text(json.dumps({"ip_scale": 5.0e5, "radius_scale": 1.0, "current_scale": [1.0e6] * action_dim, "derivative_scale": [1.0e6] * action_dim}), encoding="utf-8")
    (export / "metadata.json").write_text(json.dumps({"layer_norm_eps": 1.0e-5}), encoding="utf-8")
    hidden = 8
    np.savez(
        export / "policy_weights.npz",
        **{
            "input.weight": np.zeros((hidden, cursor), dtype=np.float32),
            "input.bias": np.zeros((hidden,), dtype=np.float32),
            "input_norm.weight": np.ones((hidden,), dtype=np.float32),
            "input_norm.bias": np.zeros((hidden,), dtype=np.float32),
            "hidden1.weight": np.zeros((hidden, hidden), dtype=np.float32),
            "hidden1.bias": np.zeros((hidden,), dtype=np.float32),
            "mean_head.weight": np.zeros((action_dim, hidden), dtype=np.float32),
            "mean_head.bias": np.zeros((action_dim,), dtype=np.float32),
        },
    )
    controller = make_controller("learned_magnetic_controller", config={"export_dir": export})
    angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False, dtype=float)
    poly, _level, _status = find_plasma_boundary_with_status(model.state.psi, model.grid, (model.R0, model.Z0), limiter_shape=cfg.limiter_shape, boundary_mode=cfg.boundary_mode)
    ref_radii = radii_from_polyline_ray_intersections(poly, (model.R0, model.Z0), angles)
    scenario = make_scenario("nominal", ref_radii, model.Ip0, params={}, center=(model.R0, model.Z0))
    action = controller.compute_control(model=model, psi=model.compute_psi(), boundary_poly=poly, center=(model.R0, model.Z0), measure_angles=angles, ref_radii=ref_radii, Ip_ref=model.Ip0, scenario=scenario, max_episode_steps=10)
    assert action.pfc_derivs.shape == (n_pfc,)
    assert action.sol_derivs.shape == (n_sol,)


def test_compact_controller_matches_torch_actor_math_and_derivative_scaling(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    cfg = load_config(CONFIG, initial_currents_path=INITIAL)
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics)
    n_pfc = cfg.pfc.n_coils
    n_sol = cfg.sol.n_coils
    action_dim = n_pfc + n_sol
    n_angles = 4
    preview_steps = 2
    sizes = [
        ("step_norm", 1),
        ("ip", 1),
        ("ip_ref", 1),
        ("ip_error", 1),
        ("active_currents", action_dim),
        ("active_current_derivs", action_dim),
        ("measured_boundary_radii", n_angles),
        ("ref_radii", n_angles),
        ("boundary_radii_error", n_angles),
        ("boundary_found", 1),
        ("previous_action", action_dim),
        ("target_preview", preview_steps * (2 + n_angles)),
    ]
    slices: dict[str, list[int]] = {}
    cursor = 0
    for name, size in sizes:
        slices[name] = [cursor, cursor + int(size)]
        cursor += int(size)
    export = tmp_path / "compact_parity_export"
    export.mkdir()
    schema = {
        "observation_kind": "controller_state_v2",
        "obs_dim": cursor,
        "action_dim": action_dim,
        "n_active_total": action_dim,
        "n_pfc": n_pfc,
        "n_sol": n_sol,
        "n_angles": n_angles,
        "angles_rad": np.linspace(-np.pi, np.pi, n_angles, endpoint=False, dtype=float).tolist(),
        "grid_shape": list(model.state.psi.shape),
        "feature_order": COMPACT_JOINT_STATE_V2_FEATURE_ORDER,
        "feature_slices": slices,
        "target_preview_steps": preview_steps,
        "target_preview_stride": 1,
    }
    (export / "controller_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    derivative_scale = np.linspace(1.0e5, 9.0e5, action_dim, dtype=float)
    (export / "normalization.json").write_text(json.dumps({"ip_scale": 5.0e5, "radius_scale": 1.0, "current_scale": [1.0e6] * action_dim, "derivative_scale": derivative_scale.tolist()}), encoding="utf-8")
    eps = 1.0e-5
    (export / "metadata.json").write_text(json.dumps({"layer_norm_eps": eps}), encoding="utf-8")
    rng = np.random.default_rng(123)
    hidden = 8
    weights = {
        "input.weight": rng.normal(0.0, 0.05, size=(hidden, cursor)).astype(np.float32),
        "input.bias": rng.normal(0.0, 0.02, size=(hidden,)).astype(np.float32),
        "input_norm.weight": rng.normal(1.0, 0.02, size=(hidden,)).astype(np.float32),
        "input_norm.bias": rng.normal(0.0, 0.02, size=(hidden,)).astype(np.float32),
        "hidden1.weight": rng.normal(0.0, 0.05, size=(hidden, hidden)).astype(np.float32),
        "hidden1.bias": rng.normal(0.0, 0.02, size=(hidden,)).astype(np.float32),
        "mean_head.weight": rng.normal(0.0, 0.05, size=(action_dim, hidden)).astype(np.float32),
        "mean_head.bias": rng.normal(0.0, 0.02, size=(action_dim,)).astype(np.float32),
    }
    np.savez(export / "policy_weights.npz", **weights)
    controller = make_controller("learned_magnetic_controller", config={"export_dir": export})
    angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False, dtype=float)
    poly, _level, _status = find_plasma_boundary_with_status(model.state.psi, model.grid, (model.R0, model.Z0), limiter_shape=cfg.limiter_shape, boundary_mode=cfg.boundary_mode)
    ref_radii = radii_from_polyline_ray_intersections(poly, (model.R0, model.Z0), angles) + 0.005
    scenario = make_scenario("nominal", ref_radii, model.Ip0 + 1000.0, params={}, center=(model.R0, model.Z0))
    obs = controller._observation(model=model, psi=model.compute_psi(), boundary_poly=poly, center=(model.R0, model.Z0), measure_angles=angles, ref_radii=ref_radii, ip_ref=model.Ip0 + 1000.0, scenario=scenario, max_episode_steps=10)
    numpy_action = controller._deterministic_action(obs.reshape(1, -1))[0]

    x = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32)
    x = x @ torch.as_tensor(weights["input.weight"]).T + torch.as_tensor(weights["input.bias"])
    mean = x.mean(dim=-1, keepdim=True)
    var = torch.mean((x - mean) ** 2, dim=-1, keepdim=True)
    x = (x - mean) / torch.sqrt(var + eps)
    x = x * torch.as_tensor(weights["input_norm.weight"]) + torch.as_tensor(weights["input_norm.bias"])
    x = torch.tanh(x)
    x = torch.nn.functional.elu(x @ torch.as_tensor(weights["hidden1.weight"]).T + torch.as_tensor(weights["hidden1.bias"]))
    torch_action = torch.tanh(x @ torch.as_tensor(weights["mean_head.weight"]).T + torch.as_tensor(weights["mean_head.bias"])).detach().numpy()[0]
    assert np.allclose(numpy_action, torch_action, atol=1.0e-6)

    action = controller.compute_control(model=model, psi=model.compute_psi(), boundary_poly=poly, center=(model.R0, model.Z0), measure_angles=angles, ref_radii=ref_radii, Ip_ref=model.Ip0 + 1000.0, scenario=scenario, max_episode_steps=10)
    physical = np.concatenate([action.pfc_derivs, action.sol_derivs])
    assert np.allclose(physical, numpy_action * derivative_scale, atol=1.0e-5)
    assert np.allclose(controller._previous_action_norm, numpy_action, atol=1.0e-6)
