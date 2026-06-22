from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tokamak_control.cli.run_simulation import _prepare
from tokamak_control.core.grid import Grid1D, Grid2D
from tokamak_control.geometry.boundary import BoundaryNotFoundError, find_plasma_boundary_with_status
from tokamak_control.geometry.boundary_cpu import _legacy_first_accepted_contour
from tokamak_control.geometry.legacy_metrics import legacy_measurement_angles_from_actuators
from tokamak_control.io.config_io import dump_config, load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _legacy_test_grid() -> Grid2D:
    return Grid2D(
        r=Grid1D(start=0.0, step=1.0, size=40, center=19.5),
        z=Grid1D(start=0.0, step=1.0, size=40, center=19.5),
    )


def _centered_quadratic_psi(grid: Grid2D) -> np.ndarray:
    R, Z = grid.mesh()
    return (R - 19.5) ** 2 + (Z - 19.5) ** 2


def test_legacy_measurement_angles_preserve_actuator_order() -> None:
    """LittleScope reorders interpolated errors back into original PFC order."""
    theta = np.linspace(-np.pi, np.pi, 257, endpoint=True)
    boundary = np.column_stack([np.cos(theta), np.sin(theta)])
    actuators = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )

    angles, radii = legacy_measurement_angles_from_actuators(boundary, (0.0, 0.0), actuators)

    assert angles.shape == (3,)
    assert radii.shape == (3,)
    assert np.allclose(angles, np.array([np.pi / 2.0, 0.0, -np.pi / 2.0]), atol=2.0e-2)
    assert np.allclose(radii, np.ones((3,), dtype=float), atol=1.0e-6)


def test_legacy_contour_config_round_trips(tmp_path: Path) -> None:
    cfg = load_config(REPO_ROOT / "configs/T15MD_new_data.toml")
    out_path = tmp_path / "legacy_contour.toml"
    dump_config(
        out_path,
        cfg.grid,
        cfg.pfc,
        cfg.sol,
        cfg.physics,
        compute=cfg.compute,
        realism=cfg.realism,
        limiter_name=cfg.limiter_name,
        boundary_mode="legacy_contour",
        boundary_legacy_precision_index2=2.5e-3,
    )

    loaded = load_config(out_path)

    assert loaded.boundary_mode == "legacy_contour"
    assert loaded.boundary_legacy_precision_index2 == pytest.approx(2.5e-3)
    assert loaded.limiter_shape is not None


def test_legacy_contour_limited_config_round_trips(tmp_path: Path) -> None:
    cfg = load_config(REPO_ROOT / "configs/T15MD_new_data.toml")
    assert cfg.boundary_mode == "tracked_flux_contour"
    assert cfg.boundary_base_mode == "legacy_contour_limited"

    out_path = tmp_path / "legacy_contour_limited.toml"
    dump_config(
        out_path,
        cfg.grid,
        cfg.pfc,
        cfg.sol,
        cfg.physics,
        compute=cfg.compute,
        realism=cfg.realism,
        limiter_name=cfg.limiter_name,
        boundary_mode="legacy_contour_limited",
    )

    loaded = load_config(out_path)

    assert loaded.boundary_mode == "legacy_contour_limited"
    assert loaded.limiter_shape is not None


def test_tracked_flux_contour_config_round_trips(tmp_path: Path) -> None:
    cfg = load_config(REPO_ROOT / "configs/T15MD_new_data.toml")

    assert cfg.boundary_mode == "tracked_flux_contour"
    assert cfg.boundary_base_mode == "legacy_contour_limited"
    assert cfg.boundary_level_smoothing_alpha == pytest.approx(0.6)
    assert cfg.boundary_level_search_span_fraction == pytest.approx(0.02)
    assert cfg.boundary_continuity_weight_radii == pytest.approx(1.0)
    assert cfg.boundary_continuity_weight_mean_radius == pytest.approx(0.3)
    assert cfg.boundary_continuity_weight_center == pytest.approx(0.2)
    assert cfg.boundary_continuity_weight_area == pytest.approx(0.2)
    assert cfg.boundary_continuity_weight_level == pytest.approx(0.1)

    out_path = tmp_path / "tracked_flux.toml"
    dump_config(
        out_path,
        cfg.grid,
        cfg.pfc,
        cfg.sol,
        cfg.physics,
        compute=cfg.compute,
        realism=cfg.realism,
        limiter_name=cfg.limiter_name,
        boundary_mode="tracked_flux_contour",
        boundary_base_mode="legacy_contour_limited",
        boundary_legacy_precision_index2=1.0e-3,
        boundary_level_smoothing_alpha=0.6,
        boundary_level_search_span_fraction=0.02,
        boundary_continuity_weight_radii=1.0,
        boundary_continuity_weight_mean_radius=0.3,
        boundary_continuity_weight_center=0.2,
        boundary_continuity_weight_area=0.2,
        boundary_continuity_weight_level=0.1,
    )

    loaded = load_config(out_path)

    assert loaded.boundary_mode == "tracked_flux_contour"
    assert loaded.boundary_base_mode == "legacy_contour_limited"
    assert loaded.boundary_level_smoothing_alpha == pytest.approx(0.6)
    assert loaded.boundary_level_search_span_fraction == pytest.approx(0.02)


def test_prepare_preserves_requested_32_reference_angles_for_generic_runs() -> None:
    """Generic scenario setup should use the requested uniform boundary samples."""
    cfg = load_config(REPO_ROOT / "configs/T15MD_new_data.toml")
    expected_angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)

    for mode in ("legacy_contour", "legacy_contour_limited"):
        active_cfg = replace(cfg, boundary_mode=mode)
        _model, angles, _scenario, base_radii = _prepare(
            active_cfg,
            M_angles=32,
            scenario_name="nominal",
            scenario_params={},
        )

        assert angles.shape == (32,)
        assert base_radii.shape == (32,)
        assert np.allclose(angles, expected_angles)
        assert np.all(np.isfinite(base_radii))


def test_prepare_uses_actuator_tied_reference_angles_for_zaitsev_lqr() -> None:
    """Zaitsev LQR setup should use old actuator-tied boundary samples."""
    cfg = load_config(REPO_ROOT / "configs/T15MD_new_data.toml")
    _model, angles, _scenario, base_radii = _prepare(
        cfg,
        M_angles=32,
        scenario_name="nominal",
        scenario_params={},
        controller_name="lqr_t15_zaitsev",
    )

    assert angles.shape == (cfg.pfc.n_coils,)
    assert base_radii.shape == (cfg.pfc.n_coils,)
    assert np.all(np.isfinite(angles))
    assert np.all(np.isfinite(base_radii))


def test_legacy_contour_finds_closed_contour_around_center() -> None:
    grid = _legacy_test_grid()
    psi = _centered_quadratic_psi(grid)

    poly, level, status = find_plasma_boundary_with_status(
        psi,
        grid,
        (19.5, 19.5),
        boundary_mode="legacy_contour",
        legacy_precision_index2=1.0e-3,
    )

    assert status == "legacy_contour_success"
    assert level > 0.0
    assert poly.shape[0] >= 4
    assert np.allclose(poly[0], poly[-1])
    assert float(np.min(poly[:, 0])) < 19.5 < float(np.max(poly[:, 0]))
    assert float(np.min(poly[:, 1])) < 19.5 < float(np.max(poly[:, 1]))


def test_legacy_line_is_ok_rejects_closed_contour_outside_center_bbox() -> None:
    x = np.arange(1, 41, dtype=float)
    y = np.arange(1, 41, dtype=float)
    X, Y = np.meshgrid(x, y)
    psi = (X - 10.0) ** 2 + (Y - 10.0) ** 2

    assert _legacy_first_accepted_contour(psi, 16.0, np.asarray([30.0, 30.0])) is None
    accepted = _legacy_first_accepted_contour(psi, 16.0, np.asarray([10.0, 10.0]))
    assert accepted is not None
    length, contour = accepted
    assert length == contour.shape[0]


def test_legacy_contour_ignores_limiter_shape() -> None:
    grid = _legacy_test_grid()
    psi = _centered_quadratic_psi(grid)
    off_center_limiter = np.asarray(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    poly_without_limiter, level_without_limiter, status_without_limiter = find_plasma_boundary_with_status(
        psi,
        grid,
        (19.5, 19.5),
        boundary_mode="legacy_contour",
        legacy_precision_index2=1.0e-3,
    )
    poly_with_limiter, level_with_limiter, status_with_limiter = find_plasma_boundary_with_status(
        psi,
        grid,
        (19.5, 19.5),
        limiter_shape=off_center_limiter,
        boundary_mode="legacy_contour",
        legacy_precision_index2=1.0e-3,
    )

    assert status_without_limiter == "legacy_contour_success"
    assert status_with_limiter == "legacy_contour_success"
    assert level_with_limiter == pytest.approx(level_without_limiter)
    assert np.allclose(poly_with_limiter, poly_without_limiter)


def test_legacy_contour_limited_rejects_contour_outside_limiter() -> None:
    grid = _legacy_test_grid()
    psi = _centered_quadratic_psi(grid)
    off_center_limiter = np.asarray(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    with pytest.raises(BoundaryNotFoundError):
        find_plasma_boundary_with_status(
            psi,
            grid,
            (19.5, 19.5),
            limiter_shape=off_center_limiter,
            boundary_mode="legacy_contour_limited",
            legacy_precision_index2=1.0e-3,
        )


def test_legacy_contour_limited_matches_legacy_inside_large_limiter() -> None:
    grid = _legacy_test_grid()
    psi = _centered_quadratic_psi(grid)
    large_limiter = np.asarray(
        [
            [0.0, 0.0],
            [39.0, 0.0],
            [39.0, 39.0],
            [0.0, 39.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    poly_legacy, level_legacy, status_legacy = find_plasma_boundary_with_status(
        psi,
        grid,
        (19.5, 19.5),
        boundary_mode="legacy_contour",
        legacy_precision_index2=1.0e-3,
    )
    poly_limited, level_limited, status_limited = find_plasma_boundary_with_status(
        psi,
        grid,
        (19.5, 19.5),
        limiter_shape=large_limiter,
        boundary_mode="legacy_contour_limited",
        legacy_precision_index2=1.0e-3,
    )

    assert status_legacy == "legacy_contour_success"
    assert status_limited == "legacy_contour_limited_success"
    assert level_limited == pytest.approx(level_legacy)
    assert np.allclose(poly_limited, poly_legacy)


def test_tracked_flux_contour_uses_previous_boundary_after_initial_reset() -> None:
    grid = _legacy_test_grid()
    psi = _centered_quadratic_psi(grid)
    large_limiter = np.asarray(
        [
            [0.0, 0.0],
            [39.0, 0.0],
            [39.0, 39.0],
            [0.0, 39.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    poly0, level0, status0 = find_plasma_boundary_with_status(
        psi,
        grid,
        (19.5, 19.5),
        limiter_shape=large_limiter,
        boundary_mode="tracked_flux_contour",
        boundary_base_mode="legacy_contour_limited",
        legacy_precision_index2=1.0e-3,
        level_smoothing_alpha=0.6,
        level_search_span_fraction=0.02,
    )
    poly1, level1, status1 = find_plasma_boundary_with_status(
        psi,
        grid,
        (19.5, 19.5),
        prev_level=level0,
        prev_poly=poly0,
        limiter_shape=large_limiter,
        boundary_mode="tracked_flux_contour",
        boundary_base_mode="legacy_contour_limited",
        legacy_precision_index2=1.0e-3,
        level_smoothing_alpha=0.6,
        level_search_span_fraction=0.02,
    )

    assert status0 == "tracked_flux_contour_reset"
    assert status1 == "tracked_flux_contour_success"
    assert level1 == pytest.approx(level0)
    assert np.allclose(poly1, poly0)
