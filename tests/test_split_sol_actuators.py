from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokamak_control.core.green import build_green_for_coils, build_green_for_eind
from tokamak_control.core.plasma_model import PlasmaModel
from tokamak_control.io.config_io import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
T15_CONFIG = REPO_ROOT / "configs/T15MD_new_data.toml"
T15_INITIAL = REPO_ROOT / "configs/initial_currents/T15MD_new_data_3864.toml"


def _require_t15_config() -> None:
    missing = [str(path.relative_to(REPO_ROOT)) for path in (T15_CONFIG, T15_INITIAL) if not path.exists()]
    if missing:
        pytest.skip("missing local T15 fixture(s): " + ", ".join(missing))


def test_t15_split_sol_green_fields_are_weighted_runtime_actuators() -> None:
    """
    T15 SOL split points are volume samples of three actuators, not 150 actuators.

    The 30/90/30 point sets must collectively behave as SOL0/SOL1/SOL2. With
    equal fractional weights, ignoring weights would multiply SOL authority by
    the split count.
    """
    _require_t15_config()
    cfg = load_config(T15_CONFIG, initial_currents_path=T15_INITIAL)

    counts = [group.shape[0] for group in cfg.sol.element_positions]
    assert cfg.sol.n_coils == 3
    assert cfg.sol.n_elements_total == 150
    assert counts == [30, 90, 30]
    for count, weights in zip(counts, cfg.sol.element_weights, strict=True):
        assert weights.shape == (count,)
        assert float(weights.sum()) == pytest.approx(1.0)
        assert np.allclose(weights, np.full((count,), 1.0 / float(count)))

    R, Z = cfg.grid.mesh()
    weighted_fields = build_green_for_coils(R, Z, cfg.sol.element_positions, cfg.sol.element_weights)
    unweighted_fields = build_green_for_coils(R, Z, cfg.sol.element_positions, None)
    assert weighted_fields.shape[0] == 3
    assert unweighted_fields.shape[0] == 3
    for idx, count in enumerate(counts):
        assert np.allclose(unweighted_fields[idx], float(count) * weighted_fields[idx], rtol=1e-12, atol=1e-14)

    weighted_coupling = build_green_for_eind(
        cfg.physics.R0,
        cfg.physics.Z0,
        cfg.sol.element_positions,
        cfg.sol.element_weights,
    )
    unweighted_coupling = build_green_for_eind(
        cfg.physics.R0,
        cfg.physics.Z0,
        cfg.sol.element_positions,
        None,
    )
    assert np.allclose(unweighted_coupling, weighted_coupling * np.asarray(counts, dtype=float), rtol=1e-12, atol=1e-14)


def test_t15_split_sol_step_currents_keeps_three_runtime_sol_channels() -> None:
    """The plant state and command interface expose SOL0/SOL1/SOL2 only."""
    _require_t15_config()
    cfg = load_config(T15_CONFIG, initial_currents_path=T15_INITIAL)
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics)

    sol0 = np.asarray(model.state.sol_currents, dtype=float).copy()
    pfc0 = np.asarray(model.state.pfc_currents, dtype=float).copy()
    sol_next = sol0 + np.asarray([100.0, -250.0, 400.0], dtype=float)
    state = model.step_currents(pfc_currents_next=pfc0, sol_currents_next=sol_next)

    assert state.sol_currents.shape == (3,)
    assert state.sol_current_derivs.shape == (3,)
    assert np.allclose(state.sol_currents, sol_next)
    assert np.allclose(state.sol_current_derivs, (sol_next - sol0) / float(cfg.physics.t_step))
