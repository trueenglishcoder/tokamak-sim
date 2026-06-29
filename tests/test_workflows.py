"""Регрессионные проверки основных пользовательских сценариев запуска."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tomllib

import numpy as np
import pytest

from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.core.plasma_model import PlasmaModel
from tokamak_control.config.scenarios import make_scenario
from tokamak_control.control.registry import normalize_controller_launch
from tokamak_control.control.t15md_replay import T15MDReplayController
from tokamak_control.geometry.boundary import find_plasma_boundary_with_status
from tokamak_control.io.config_io import apply_initial_state, dump_config, load_config, load_initial_state, require_initial_state


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = "configs/T15MD_new_data.toml"
SMOKE_INITIAL_STATE = "configs/initial_states/T15MD_new_data_3864.toml"
SMOKE_REPLAY_TABLE = "data/t15_data_new_trim50/coils/t15md_3864_coils.csv"


def _load_t15_with_initial():
    cfg = load_config(REPO_ROOT / SMOKE_CONFIG)
    return apply_initial_state(cfg, load_initial_state(cfg, REPO_ROOT / SMOKE_INITIAL_STATE))


def _require_local_paths(*relative_paths: str) -> None:
    """Пропустить workflow-проверку, если локальные ignored fixtures не подключены."""
    missing = [p for p in relative_paths if not (REPO_ROOT / p).exists()]
    if missing:
        pytest.skip("missing local ignored fixture(s): " + ", ".join(missing))


def _clean_env(tmp_path: Path) -> dict[str, str]:
    """Собрать окружение без внешнего PYTHONPATH для проверки прямого запуска."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    return env


def _run(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Запустить команду из корня репозитория и вернуть завершенный процесс."""
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=_clean_env(tmp_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def test_run_simulation_artifacts_uses_local_package_without_pythonpath(tmp_path: Path) -> None:
    """Проверить, что прямой запуск script-обертки берет пакет из текущего репозитория."""
    _require_local_paths(SMOKE_CONFIG, SMOKE_INITIAL_STATE, SMOKE_REPLAY_TABLE)
    out_root = tmp_path / "iter_run"
    result = _run(
        [
            sys.executable,
            "scripts/run_simulation_artifacts.py",
            "--config",
            SMOKE_CONFIG,
            "--initial-state",
            SMOKE_INITIAL_STATE,
            "--steps",
            "2",
            "--controller",
            "t15md_replay",
            "--controller-arg",
            f"replay_path={SMOKE_REPLAY_TABLE}",
            "--angles",
            "8",
            "--scenario",
            "nominal",
            "--out",
            str(out_root),
            "--no-progress",
        ],
        tmp_path,
    )

    first_line = result.stdout.splitlines()[0]
    run_dir = Path(first_line)
    assert run_dir.exists()
    assert any(run_dir.glob("run*.npz"))
    assert any(run_dir.glob("psi_boundary*.png"))


def test_generate_synthetic_iter_script_writes_expected_tables(tmp_path: Path) -> None:
    """Проверить прямой запуск генератора синтетических ITER-таблиц."""
    _require_local_paths(SMOKE_CONFIG, SMOKE_INITIAL_STATE)
    out_root = tmp_path / "synthetic_iter"
    _run(
        [
            sys.executable,
            "scripts/generate_synthetic_iter_dataset.py",
            "--config",
            SMOKE_CONFIG,
            "--initial-state",
            SMOKE_INITIAL_STATE,
            "--out-root",
            str(out_root),
            "--n-shots",
            "1",
        ],
        tmp_path,
    )

    assert (out_root / "synthetic_iter_metadata.json").exists()
    assert len(list((out_root / "ip").glob("t15md_*_ip.csv"))) == 1
    assert len(list((out_root / "coils").glob("t15md_*_coils.csv"))) == 1


def test_generate_synthetic_ip_tables_feed_algorithmic_controller(tmp_path: Path) -> None:
    """Проверить генерацию синтетического Ip и его использование в analytic controller run."""
    _require_local_paths(SMOKE_CONFIG, "data/t15_data_new_split/ip")
    out_root = tmp_path / "synthetic_ip"
    initial_dir = tmp_path / "initial_states"
    _run(
        [
            sys.executable,
            "scripts/generate_synthetic_ip_tables.py",
            "--source-ip-dir",
            "data/t15_data_new_split/ip",
            "--out-root",
            str(out_root),
            "--out-initial-state-dir",
            str(initial_dir),
            "--n-shots",
            "1",
            "--seed",
            "7",
        ],
        tmp_path,
    )

    ip_tables = sorted((out_root / "ip").glob("t15md_*_ip.csv"))
    assert len(ip_tables) == 1
    initial_tables = sorted(initial_dir.glob("T15MD_new_data_*.toml"))
    assert len(initial_tables) == 1
    assert (out_root / "synthetic_ip_metadata.json").exists()

    run_root = tmp_path / "controller_run"
    result = _run(
        [
            sys.executable,
            "scripts/run_simulation_artifacts.py",
            "--config",
            "configs/T15MD_new_data.toml",
            "--initial-state",
            str(initial_tables[0]),
            "--steps",
            "5",
            "--controller",
            "lqr_current",
            "--angles",
            "32",
            "--scenario",
            "ip_follow",
            "--scenario-arg",
            f"ip_csv={ip_tables[0]}",
            "--out",
            str(run_root),
            "--no-progress",
        ],
        tmp_path,
    )

    first_line = result.stdout.splitlines()[0]
    run_dir = Path(first_line)
    assert run_dir.exists()
    assert any(run_dir.glob("run*.npz"))


def test_sigma_l_fit_runs_on_generated_tables(tmp_path: Path) -> None:
    """Проверить минимальный цикл генерации и подбора sigma/L."""
    _require_local_paths(SMOKE_CONFIG)
    data_root = tmp_path / "synthetic_iter"
    out_csv = tmp_path / "sigma_fit.csv"

    _run(
        [
            sys.executable,
            "scripts/generate_synthetic_iter_dataset.py",
            "--config",
            SMOKE_CONFIG,
            "--initial-state",
            SMOKE_INITIAL_STATE,
            "--out-root",
            str(data_root),
            "--n-shots",
            "1",
        ],
        tmp_path,
    )
    result = _run(
        [
            sys.executable,
            "scripts/fit_sigma_L_grid.py",
            "--config",
            SMOKE_CONFIG,
            "--ip-dir",
            str(data_root / "ip"),
            "--coils-dir",
            str(data_root / "coils"),
            "--sigma-min",
            "6e8",
            "--sigma-max",
            "6e8",
            "--sigma-points",
            "1",
            "--L-min",
            "1e-6",
            "--L-max",
            "1e-6",
            "--L-points",
            "1",
            "--top-k",
            "1",
            "--out-csv",
            str(out_csv),
        ],
        tmp_path,
    )

    top_k_line = next(line for line in result.stdout.splitlines() if line.startswith("top_k_csv="))
    written_csv = Path(top_k_line.split("=", 1)[1])
    assert written_csv.exists()
    assert written_csv.parent.parent == tmp_path
    assert "mean_nrmse" in written_csv.read_text(encoding="utf-8").splitlines()[0]


def test_grid_config_uses_range_and_derives_original_step(tmp_path: Path) -> None:
    """Проверить, что TOML хранит диапазон сетки, а шаг вычисляется при загрузке."""
    _require_local_paths(SMOKE_CONFIG)
    cfg_path = REPO_ROOT / SMOKE_CONFIG
    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))

    assert "end" in raw["grid"]["r"]
    assert "step" not in raw["grid"]["r"]
    assert "end" in raw["grid"]["z"]
    assert "step" not in raw["grid"]["z"]

    cfg = load_config(cfg_path)
    expected_r_step = (raw["grid"]["r"]["end"] - raw["grid"]["r"]["start"]) / (raw["grid"]["r"]["size"] - 1)
    expected_z_step = (raw["grid"]["z"]["end"] - raw["grid"]["z"]["start"]) / (raw["grid"]["z"]["size"] - 1)
    assert cfg.grid.r.step == expected_r_step
    assert cfg.grid.z.step == expected_z_step

    out_path = tmp_path / "roundtrip.toml"
    dump_config(out_path, cfg.grid, cfg.pfc, cfg.sol, cfg.physics)
    dumped = tomllib.loads(out_path.read_text(encoding="utf-8"))
    assert "end" in dumped["grid"]["r"]
    assert "step" not in dumped["grid"]["r"]


def test_split_t15md_boundary_uses_limiter_contact_by_default() -> None:
    """Проверить, что T15MD по умолчанию использует лимитерный legacy/tracked контур."""
    _require_local_paths(SMOKE_CONFIG, SMOKE_INITIAL_STATE)
    cfg = _load_t15_with_initial()
    assert cfg.boundary_mode == "tracked_flux_contour"
    assert cfg.boundary_base_mode == "legacy_contour_limited"
    assert cfg.limiter_name == "T15MD"
    assert cfg.limiter_shape is not None

    model = PlasmaModel.from_settings(
        grid=cfg.grid,
        pfc=cfg.pfc,
        sol=cfg.sol,
        settings=cfg.physics,
        ip0=require_initial_state(cfg).ip0,
    )
    poly, _level, status = find_plasma_boundary_with_status(
        model.compute_psi(),
        model.grid,
        (model.R0, model.Z0),
        n_levels=80,
        limiter_shape=cfg.limiter_shape,
        boundary_mode=cfg.boundary_mode,
    )

    assert status in {"tracked_flux_contour_success", "tracked_flux_contour_reset"}
    assert poly.shape[0] >= 3


def test_boundary_search_rejects_removed_diverted_mode() -> None:
    """Проверить, что старый режим diverted больше не является активным API."""
    grid = Grid2D(
        r=Grid1D(start=-1.5, step=0.015, size=201, center=0.0),
        z=Grid1D(start=-1.5, step=0.015, size=201, center=0.0),
    )
    R, Z = grid.mesh()
    psi = (R * R + Z * Z) ** 2 - R * R + Z * Z

    with pytest.raises(ValueError, match="boundary_mode"):
        find_plasma_boundary_with_status(
            psi,
            grid,
            (0.7, 0.0),
            n_levels=20,
            boundary_mode="diverted",
        )


def test_machine_active_mask_and_initial_state_are_separate(tmp_path: Path) -> None:
    """active masks live in the machine config; initial-state TOMLs only carry Ip/currents."""
    machine_path = tmp_path / "machine.toml"
    machine_path.write_text(
        """
version = 1

[grid.r]
start = 0.5
end = 1.5
size = 8
center = 1.0

[grid.z]
start = -0.5
end = 0.5
size = 8
center = 0.0

[physics]
R0 = 1.0
Z0 = 0.0
t_step = 0.001

[coils.pfc]
name = "PFC"
positions = [[0.8, 0.2], [1.2, -0.2]]
active = [true, false]

[coils.sol]
name = "SOL"
positions = [[1.0, 0.4], [1.0, -0.4]]
active = [true, true]
""".strip(),
        encoding="utf-8",
    )
    initial_path = tmp_path / "initial.toml"
    initial_path.write_text(
        """
version = 1

[plasma]
Ip0 = 123.0

[coils.pfc]
currents = [10.0]

[coils.sol]
currents = [1.0, 2.0]
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(machine_path)
    cfg = apply_initial_state(cfg, load_initial_state(cfg, initial_path))
    assert cfg.pfc.n_coils == 1
    assert cfg.sol.n_coils == 2
    assert cfg.pfc.initial_currents.tolist() == [10.0]
    assert require_initial_state(cfg).ip0 == 123.0

    machine_with_ip0 = tmp_path / "machine_with_ip0.toml"
    machine_with_ip0.write_text(
        machine_path.read_text(encoding="utf-8").replace(
            "t_step = 0.001",
            "t_step = 0.001\nIp0 = 123.0",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Ip0"):
        load_config(machine_with_ip0)

    machine_with_currents = tmp_path / "machine_with_currents.toml"
    machine_with_currents.write_text(
        machine_path.read_text(encoding="utf-8").replace(
            "active = [true, false]",
            "active = [true, false]\ncurrents = [10.0]",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="initial-state"):
        load_config(machine_with_currents)

    initial_with_active = tmp_path / "initial_with_active.toml"
    initial_with_active.write_text(
        initial_path.read_text(encoding="utf-8").replace(
            "currents = [10.0]",
            "active = [true]\ncurrents = [10.0]",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="machine topology"):
        load_initial_state(cfg, initial_with_active)


def test_ip_follow_t15_linear_boundary_mode_changes_reference_shape() -> None:
    """Проверить, что ip_follow умеет строить линейную boundary reference от |Ip|."""
    _require_local_paths(SMOKE_CONFIG, SMOKE_INITIAL_STATE, "data/t15_data_new_split/ip/t15md_3864_ip.csv")
    cfg = _load_t15_with_initial()
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    base_radii = np.full((32,), 0.65, dtype=float)

    scenario = make_scenario(
        "ip_follow",
        base_radii,
        float(require_initial_state(cfg).ip0),
        params={
            "ip_csv": str(REPO_ROOT / "data/t15_data_new_split/ip/t15md_3864_ip.csv"),
            "boundary_mode": "t15_linear",
        },
    )
    radii_start = np.asarray(scenario.ref_radii(angles, 0.0), dtype=float)
    radii_late = np.asarray(scenario.ref_radii(angles, 1.0), dtype=float)

    assert radii_start.shape == (32,)
    assert radii_late.shape == (32,)
    assert float(np.max(np.abs(radii_late - radii_start))) > 1e-6


def test_t15md_replay_commands_exact_next_currents() -> None:
    """T15 replay must command exact table currents at t + dt, SOL first then PFC."""
    _require_local_paths(SMOKE_CONFIG, SMOKE_INITIAL_STATE, "data/t15_data_new/coils/t15md_3864_coils.csv")
    cfg = _load_t15_with_initial()
    assert cfg.sol.n_coils == 3
    assert cfg.sol.n_elements_total == 150
    assert [len(weights) for weights in cfg.sol.element_weights] == [30, 90, 30]
    assert [float(weights.sum()) for weights in cfg.sol.element_weights] == [1.0, 1.0, 1.0]

    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics, ip0=require_initial_state(cfg).ip0)
    replay_path = REPO_ROOT / "data/t15_data_new/coils/t15md_3864_coils.csv"
    controller = T15MDReplayController(replay_path=replay_path)
    action = controller.compute_control(model=model)
    table = np.loadtxt(replay_path, delimiter=";")

    assert action.sol_currents_next.shape == (3,)
    assert action.pfc_currents_next.shape == (6,)
    assert np.allclose(action.sol_currents_next, table[1, 1:4])
    assert np.allclose(action.pfc_currents_next, table[1, 4:10])


def test_t15md_replay_rejects_non_exact_u_clip_launch_arg() -> None:
    """T15 replay should not accept clipping because that is no longer exact replay."""
    _require_local_paths("data/t15_data_new/coils/t15md_3864_coils.csv")
    with pytest.raises(ValueError, match="does not accept"):
        normalize_controller_launch(
            "t15md_replay",
            {
                "replay_path": REPO_ROOT / "data/t15_data_new/coils/t15md_3864_coils.csv",
                "u_clip": 1.0,
            },
        )


def test_t15md_replay_rejects_wrong_table_width(tmp_path: Path) -> None:
    """Replay table width must match the loaded 3 SOL + 6 PFC plant."""
    _require_local_paths(SMOKE_CONFIG, SMOKE_INITIAL_STATE)
    cfg = _load_t15_with_initial()
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics, ip0=require_initial_state(cfg).ip0)
    replay_path = tmp_path / "bad_width.csv"
    replay_path.write_text("0.0;1;2\n0.001;3;4\n", encoding="utf-8")
    controller = T15MDReplayController(replay_path=replay_path)

    with pytest.raises(ValueError, match="actuator count"):
        controller.compute_control(model=model)
