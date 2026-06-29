from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokamak_control.bridge import SimulationSession
from tokamak_control.compute import ComputeSettings, normalize_compute_backend, require_gpu_available
from tokamak_control.core.gpu_plasma_model import GpuPlasmaModel
from tokamak_control.core.plasma_model import PlasmaModel
from tokamak_control.io.config_io import apply_initial_state, dump_config, load_config, load_initial_state, require_initial_state


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/T15MD_new_data.toml"
INITIAL = REPO_ROOT / "configs/initial_states/T15MD_new_data_3864.toml"


def _load_with_initial():
    cfg = load_config(CONFIG)
    return apply_initial_state(cfg, load_initial_state(cfg, INITIAL))


def test_compute_settings_round_trip(tmp_path: Path) -> None:
    cfg = _load_with_initial()
    out = tmp_path / "config.toml"
    dump_config(
        out,
        grid=cfg.grid,
        pfc=cfg.pfc,
        sol=cfg.sol,
        physics=cfg.physics,
        compute=ComputeSettings(backend="gpu", gpu_device="cuda:0"),
        realism=cfg.realism,
        limiter_name=cfg.limiter_name,
        boundary_mode=cfg.boundary_mode,
    )
    loaded = load_config(out)
    assert loaded.compute.backend == "gpu"
    assert loaded.compute.gpu_device == "cuda:0"


def test_gpu_backend_has_no_cpu_fallback_on_invalid_device() -> None:
    with pytest.raises(RuntimeError, match="CUDA device"):
        require_gpu_available("cpu")
    session = SimulationSession.from_paths(
        CONFIG,
        INITIAL,
        "nominal",
        {},
        8,
        2,
        compute_backend="gpu",
        gpu_device="cpu",
    )
    with pytest.raises(RuntimeError, match="CUDA device"):
        session.reset()


def test_normalize_compute_backend_rejects_unknown() -> None:
    assert normalize_compute_backend(None) == "cpu"
    assert normalize_compute_backend("GPU") == "gpu"
    with pytest.raises(ValueError, match="compute backend"):
        normalize_compute_backend("automatic")


def test_gpu_plasma_model_matches_cpu_for_fixed_actions() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    cfg = _load_with_initial()
    initial = require_initial_state(cfg)
    cpu = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics, ip0=initial.ip0)
    gpu = GpuPlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics, ip0=initial.ip0, gpu_device="cuda:0")
    rng = np.random.default_rng(123)
    for _ in range(5):
        pfc = rng.normal(0.0, 2.0e5, size=cfg.pfc.n_coils)
        sol = rng.normal(0.0, 2.0e5, size=cfg.sol.n_coils)
        s_cpu = cpu.step_currents(
            pfc_currents_next=cpu.state.pfc_currents + cpu.t_step * pfc,
            sol_currents_next=cpu.state.sol_currents + cpu.t_step * sol,
        )
        s_gpu = gpu.step_currents(
            pfc_currents_next=gpu.state.pfc_currents + gpu.t_step * pfc,
            sol_currents_next=gpu.state.sol_currents + gpu.t_step * sol,
        )
        assert np.allclose(s_gpu.pfc_currents, s_cpu.pfc_currents, rtol=1e-10, atol=1e-6)
        assert np.allclose(s_gpu.sol_currents, s_cpu.sol_currents, rtol=1e-10, atol=1e-6)
        assert np.isclose(s_gpu.Ip, s_cpu.Ip, rtol=1e-10, atol=1e-6)
        assert np.allclose(s_gpu.psi, s_cpu.psi, rtol=1e-10, atol=1e-8)
