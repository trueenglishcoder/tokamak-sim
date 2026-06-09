from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tokamak_control.config.scenarios import make_scenario
from tokamak_control.control.registry import build_controller_runtime_call, controller_runtime_inputs, make_controller
from tokamak_control.core.plasma_model import PlasmaModel
from tokamak_control.diagnostics import MagneticDiagnosticLayout
from tokamak_control.io.config_io import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/T15MD_new_data.toml"
INITIAL = REPO_ROOT / "configs/initial_currents/T15MD_new_data_3864.toml"


def test_learned_controller_uses_restricted_magnetic_inputs(tmp_path: Path) -> None:
    cfg = load_config(CONFIG, initial_currents_path=INITIAL)
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics)
    n_pfc = cfg.pfc.n_coils
    n_sol = cfg.sol.n_coils
    action_dim = n_pfc + n_sol
    n_angles = 4
    layout = MagneticDiagnosticLayout(
        flux_points=np.array([[1.35, 0.0], [1.55, 0.0]], dtype=float),
        field_points=np.array([[1.40, -0.1], [1.40, 0.1]], dtype=float),
        field_angles=np.array([0.0, np.pi / 2.0], dtype=float),
    )
    obs_dim = 4 + action_dim + layout.flux_count + layout.field_count + layout.flux_count + n_angles + action_dim
    export = tmp_path / "export"
    export.mkdir()
    schema = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "n_active_total": action_dim,
        "n_pfc": n_pfc,
        "n_sol": n_sol,
        "n_angles": n_angles,
        "target_preview_steps": 0,
        "target_preview_stride": 1,
        "diagnostics": {
            "flux_points": layout.flux_points.tolist(),
            "field_points": layout.field_points.tolist(),
            "field_angles": layout.field_angles.tolist(),
        },
    }
    (export / "controller_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (export / "normalization.json").write_text(
        json.dumps(
            {
                "ip_scale": 5.0e5,
                "radius_scale": 1.0,
                "flux_scale": 1.0,
                "field_scale": 1.0,
                "bdot_scale": 1.0,
                "current_scale": [1.0e6] * action_dim,
                "derivative_scale": [1.0e6] * action_dim,
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
            "mean_head.bias": np.zeros((action_dim,), dtype=np.float32),
        },
    )

    runtime_inputs = controller_runtime_inputs("learned_magnetic_controller")
    assert "measured_radii" not in runtime_inputs
    assert "boundary_poly" not in runtime_inputs
    assert "psi" in runtime_inputs

    controller = make_controller("learned_magnetic_controller", config={"export_dir": export})
    angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False, dtype=float)
    ref_radii = np.full((n_angles,), 0.5, dtype=float)
    scenario = make_scenario("nominal", ref_radii, model.Ip0, params={}, center=(model.R0, model.Z0))
    context = {
        "model": model,
        "psi": model.compute_psi(),
        "center": (model.R0, model.Z0),
        "measure_angles": angles,
        "ref_radii": ref_radii,
        "Ip_ref": model.Ip0,
        "scenario": scenario,
        "max_episode_steps": 10,
        "measured_ip": model.state.Ip,
        "measured_active_currents": np.concatenate([model.state.pfc_currents, model.state.sol_currents]),
        "measured_radii": np.ones((n_angles,), dtype=float),
        "boundary_poly": np.ones((3, 2), dtype=float),
    }
    call = build_controller_runtime_call("learned_magnetic_controller", context)
    action = controller.compute_control(**call)
    assert action.pfc_derivs.shape == (n_pfc,)
    assert action.sol_derivs.shape == (n_sol,)
    assert np.allclose(action.pfc_derivs, 0.0)
    assert np.allclose(action.sol_derivs, 0.0)
