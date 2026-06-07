"""Проверки аналитической reference-геометрии и траекторий формы."""

from __future__ import annotations

import numpy as np
import pytest

from tokamak_control.config.scenarios import make_scenario
from tokamak_control.geometry.limiters import get_limiter_shape
from tokamak_control.geometry.parametric_boundary import (
    BoundaryParameterBounds,
    BoundaryParameterRateLimits,
    BoundaryParameters,
    T15_REPLAY_ROBUST_BOUNDS,
    T15_REPLAY_SMOOTH_RATE_LIMITS,
    boundary_polyline_from_parameters,
    boundary_polyline_inside_limiter,
    boundary_polyline_self_intersects,
    evaluate_parametric_boundary,
    generate_boundary_parameter_trajectory,
    reference_radii_from_parameters,
    validate_boundary_parameters,
    validate_parameter_trajectory,
    validate_reference_boundary,
)


def test_parametric_boundary_formula_matches_characteristic_points() -> None:
    """Проверить формулу на правой, левой, верхней и нижней точках."""
    params = BoundaryParameters(R0=1.0, Z0=0.2, A0=0.5, kappa=2.0, delta=0.3)
    theta = np.array([0.0, np.pi, 0.5 * np.pi, -0.5 * np.pi], dtype=float)

    points = evaluate_parametric_boundary(theta, params)

    expected = np.array(
        [
            [1.5, 0.2],
            [0.5, 0.2],
            [0.85, 1.2],
            [0.85, -0.8],
        ],
        dtype=float,
    )
    assert np.allclose(points, expected)


def test_boundary_polyline_and_radii_from_circular_parameters() -> None:
    """Проверить замыкание полилинии и восстановление радиусов для круглой границы."""
    params = BoundaryParameters(R0=1.2, Z0=0.0, A0=0.4, kappa=1.0, delta=0.0)
    angles = np.linspace(-np.pi, np.pi, 16, endpoint=False, dtype=float)

    poly = boundary_polyline_from_parameters(params, theta_count=64)
    radii = reference_radii_from_parameters(params, center=(1.2, 0.0), angles=angles)

    assert poly.shape == (65, 2)
    assert np.allclose(poly[0], poly[-1])
    assert np.allclose(radii, np.full((16,), 0.4), atol=5.0e-5)


def test_reference_boundary_validation_rejects_bad_params_and_self_intersections() -> None:
    """Проверить отбраковку плохих параметров и самопересекающихся контуров."""
    with pytest.raises(ValueError, match="A0"):
        validate_boundary_parameters(BoundaryParameters(R0=1.0, Z0=0.0, A0=0.0, kappa=1.0, delta=0.0))

    bowtie = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype=float)
    assert boundary_polyline_self_intersects(bowtie)


def test_reference_boundary_limiter_containment_for_t15_shape() -> None:
    """Проверить, что консервативная T15-like reference-форма проходит проверку лимитера."""
    limiter = get_limiter_shape("T15MD")
    assert limiter is not None
    params = BoundaryParameters(R0=1.43, Z0=-0.02, A0=0.55, kappa=1.25, delta=0.20)

    poly = validate_reference_boundary(params, bounds=T15_REPLAY_ROBUST_BOUNDS, limiter_shape=limiter)

    assert not boundary_polyline_self_intersects(poly)
    assert boundary_polyline_inside_limiter(poly, limiter)


def test_parameter_trajectory_generation_is_seeded_bounded_and_rate_limited() -> None:
    """Проверить детерминизм, границы и скорости синтетической траектории параметров."""
    traj_a = generate_boundary_parameter_trajectory(step_count=200, t_step=1.0e-3, seed=7)
    traj_b = generate_boundary_parameter_trajectory(step_count=200, t_step=1.0e-3, seed=7)
    traj_c = generate_boundary_parameter_trajectory(step_count=200, t_step=1.0e-3, seed=8)

    mat_a = traj_a.parameter_matrix()
    mat_b = traj_b.parameter_matrix()
    mat_c = traj_c.parameter_matrix()

    assert np.allclose(mat_a, mat_b)
    assert not np.allclose(mat_a, mat_c)
    validate_parameter_trajectory(
        traj_a.parameters,
        traj_a.t,
        bounds=T15_REPLAY_ROBUST_BOUNDS,
        rate_limits=T15_REPLAY_SMOOTH_RATE_LIMITS,
    )
    assert traj_a.t.shape == (200,)
    assert mat_a.shape == (200, 5)


def test_parameter_trajectory_rejects_rate_violations() -> None:
    """Проверить явную ошибку при нарушении заданных скоростей параметров."""
    bounds = BoundaryParameterBounds(R0=(1.0, 2.0), Z0=(-1.0, 1.0), A0=(0.1, 1.0), kappa=(1.0, 3.0), delta=(0.0, 0.8))
    rates = BoundaryParameterRateLimits(R0=0.1, Z0=0.1, A0=0.1, kappa=0.1, delta=0.1)
    params = [
        BoundaryParameters(R0=1.1, Z0=0.0, A0=0.2, kappa=1.2, delta=0.2),
        BoundaryParameters(R0=1.9, Z0=0.0, A0=0.2, kappa=1.2, delta=0.2),
    ]

    with pytest.raises(ValueError, match="rate limits"):
        validate_parameter_trajectory(params, np.array([0.0, 1.0]), bounds=bounds, rate_limits=rates)


def test_t15_synthetic_follow_scenario_produces_seeded_reference_radii() -> None:
    """Проверить сценарий synthetic-follow через стандартный scenario-интерфейс."""
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    base_radii = np.full((32,), 0.6, dtype=float)
    params = {
        "seed": 11,
        "duration_s": 0.20,
        "t_step": 1.0e-3,
        "target_update_s": 0.05,
        "ip_start": 100.0,
        "ip_end": 300.0,
        "ip_ramp_s": 0.10,
    }

    scenario_a = make_scenario("t15_synthetic_follow", base_radii, 100.0, params=params, center=(1.2, 0.0))
    scenario_b = make_scenario("t15_synthetic_follow", base_radii, 100.0, params=params, center=(1.2, 0.0))
    scenario_c = make_scenario(
        "t15_synthetic_follow",
        base_radii,
        100.0,
        params={**params, "seed": 12},
        center=(1.2, 0.0),
    )

    radii_a0 = scenario_a.ref_radii(angles, 0.0)
    radii_b0 = scenario_b.ref_radii(angles, 0.0)
    radii_c0 = scenario_c.ref_radii(angles, 0.0)
    radii_a1 = scenario_a.ref_radii(angles, 0.12)

    assert scenario_a.Ip_ref(0.0) == pytest.approx(100.0)
    assert scenario_a.Ip_ref(0.05) == pytest.approx(200.0)
    assert scenario_a.Ip_ref(0.20) == pytest.approx(300.0)
    assert radii_a0.shape == (32,)
    assert np.all(np.isfinite(radii_a0))
    assert np.all(np.isfinite(radii_a1))
    assert np.allclose(radii_a0, radii_b0)
    assert not np.allclose(radii_a0, radii_c0)
    assert float(np.max(np.abs(radii_a1 - radii_a0))) > 0.0


def test_t15_synthetic_follow_static_circle_boundary_is_constant() -> None:
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    base_radii = np.full((32,), 0.6, dtype=float)
    params = {
        "reference_preset": "circle_static_boundary",
        "seed": 11,
        "duration_s": 0.20,
        "t_step": 1.0e-3,
        "boundary_kind": "static_parameters",
        "boundary_parameters": {"R0": 1.40, "Z0": 0.0, "A0": 0.55, "kappa": 1.0, "delta": 0.0},
        "ip_segmented": True,
        "ip_min": 100_000.0,
        "ip_max": 420_000.0,
        "ip_segment_min_steps": 10,
        "ip_segment_max_steps": 20,
        "ip_segment_count_min": 2,
        "ip_segment_count_max": 3,
        "ip_max_steps": 100,
        "ip_rate_limit": 8.0e6,
    }

    scenario = make_scenario("t15_synthetic_follow", base_radii, 100.0, params=params, center=(1.40, 0.0))
    radii_0 = scenario.ref_radii(angles, 0.0)
    radii_1 = scenario.ref_radii(angles, 0.15)

    assert np.allclose(radii_0, np.full((32,), 0.55), atol=5.0e-5)
    assert np.allclose(radii_0, radii_1)
    assert scenario.Ip_ref(0.0) >= 100_000.0
    assert scenario.Ip_ref(1.0) <= 420_000.0


def test_t15_training_circle_static_named_scenario_matches_rl_debug_reference() -> None:
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    base_radii = np.full((32,), 0.6, dtype=float)

    scenario = make_scenario("t15_training_circle_static", base_radii, 0.0, params={}, center=(1.40, 0.0))
    radii_0 = scenario.ref_radii(angles, 0.0)
    radii_1 = scenario.ref_radii(angles, 0.75)

    assert scenario.name == "t15_synthetic_follow"
    assert scenario.Ip_ref(0.0) == pytest.approx(0.0)
    assert 112_947.0 <= scenario.Ip_ref(0.50) <= 414_434.0
    assert np.allclose(radii_0, np.full((32,), 0.55), atol=5.0e-5)
    assert np.allclose(radii_0, radii_1)


def test_t15_training_circle_ip_scaled_expands_radius_with_ip_reference() -> None:
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    base_radii = np.full((32,), 0.6, dtype=float)

    scenario = make_scenario("t15_training_circle_ip_scaled", base_radii, 0.0, params={}, center=(1.40, 0.0))

    r0 = scenario.ref_radii(angles, 0.0)
    r_early = scenario.ref_radii(angles, 0.01)
    r_formed = scenario.ref_radii(angles, 0.10)
    ip_formed = scenario.Ip_ref(0.10)

    assert scenario.name == "t15_training_circle_ip_scaled"
    assert scenario.Ip_ref(0.0) == pytest.approx(0.0)
    assert np.allclose(r0, np.full((32,), 1.0e-6))
    assert np.allclose(r_early, r_early[0])
    assert np.allclose(r_formed, r_formed[0])
    assert r_early[0] > r0[0]
    assert r_formed[0] > r_early[0]
    assert r_formed[0] == pytest.approx(0.558330912696 + 2.03086959551e-7 * ip_formed)


def test_t15_synthetic_follow_accepts_initial_boundary_parameters() -> None:
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    base_radii = np.full((32,), 0.6, dtype=float)
    initial = {"R0": 1.40, "Z0": 0.0, "A0": 0.62, "kappa": 1.15, "delta": 0.12}

    scenario = make_scenario(
        "t15_synthetic_follow",
        base_radii,
        125_000.0,
        params={
            "duration_s": 0.2,
            "t_step": 1.0e-3,
            "boundary_kind": "generated_parameters",
            "boundary_initial_parameters": initial,
            "boundary_bounds": {
                "R0": {"min": 1.3, "max": 1.5},
                "Z0": {"min": -0.1, "max": 0.1},
                "A0": {"min": 0.5, "max": 0.7},
                "kappa": {"min": 1.0, "max": 1.5},
                "delta": {"min": 0.0, "max": 0.4},
            },
        },
        center=(1.40, 0.0),
    )

    radii_0 = scenario.ref_radii(angles, 0.0)

    expected = reference_radii_from_parameters(
        BoundaryParameters(**initial),
        center=(1.40, 0.0),
        angles=angles,
        theta_count=512,
    )
    assert np.nanmean(radii_0) == pytest.approx(np.nanmean(expected))



def test_t15_training_replay_start_3859_uses_replay_initial_state() -> None:
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    base_radii = np.full((32,), 0.6, dtype=float)

    scenario = make_scenario("t15_training_replay_start_3859", base_radii, 0.0, params={}, center=(1.40, 0.0))

    radii_0 = scenario.ref_radii(angles, 0.0)
    expected = reference_radii_from_parameters(
        BoundaryParameters(
            R0=1.411347297587252,
            Z0=-0.0034510165353685133,
            A0=0.6100682302704028,
            kappa=1.1120020187817363,
            delta=0.0993730470809749,
        ),
        center=(1.40, 0.0),
        angles=angles,
        theta_count=512,
    )

    assert scenario.Ip_ref(0.0) == pytest.approx(124844.69119195438)
    assert 112_947.0 <= scenario.Ip_ref(0.50) <= 414_434.0
    assert np.allclose(radii_0, expected)
    assert not np.allclose(radii_0, scenario.ref_radii(angles, 0.75))
