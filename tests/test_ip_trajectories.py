"""Проверки синтетических reference-траекторий Ip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokamak_control.config.ip_trajectories import (
    IpReferenceTrajectory,
    SegmentedIpConfig,
    SyntheticIpConfig,
    discover_ip_templates,
    generate_segmented_ip_reference,
    generate_ip_reference_from_template,
    generate_ip_reference_from_templates,
    load_semicolon_ip_table,
    make_ip_reference_from_table,
    write_semicolon_ip_table,
)
from tokamak_control.config.scenarios import make_scenario


def _write_ip_csv(path: Path, time_s: np.ndarray, ip_a: np.ndarray) -> None:
    table = np.column_stack([time_s, ip_a])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, table, delimiter=";", fmt="%.16g")


def test_load_and_interpolate_ip_reference_table(tmp_path: Path) -> None:
    """Проверить чтение time;Ip и clamped-интерполяцию."""
    path = tmp_path / "t15md_1234_ip.csv"
    _write_ip_csv(path, np.array([10.0, 10.5, 11.0]), np.array([0.0, 100.0, 50.0]))

    template = load_semicolon_ip_table(path)
    trajectory = make_ip_reference_from_table(path)
    offset_trajectory = make_ip_reference_from_table(path, time_offset=10.5)

    assert template.shot_id == "1234"
    assert np.allclose(template.time_s, np.array([0.0, 0.5, 1.0]))
    assert trajectory.at_time(-1.0) == pytest.approx(0.0)
    assert trajectory.at_time(0.25) == pytest.approx(50.0)
    assert trajectory.at_time(2.0) == pytest.approx(50.0)
    assert np.allclose(offset_trajectory.time_s, np.array([-0.5, 0.0, 0.5]))
    assert offset_trajectory.at_time(0.25) == pytest.approx(75.0)


def test_generate_ip_reference_from_template_is_seeded_and_bounded(tmp_path: Path) -> None:
    """Проверить детерминизм и базовые ограничения synthetic Ip."""
    path = tmp_path / "t15md_4321_ip.csv"
    _write_ip_csv(path, np.linspace(0.0, 1.0, 8), np.array([0.0, 10.0, 30.0, 60.0, 80.0, 70.0, 40.0, 20.0]))
    template = load_semicolon_ip_table(path)
    config = SyntheticIpConfig(amplitude_jitter=0.02, duration_jitter=0.03, shape_jitter=0.01)

    traj_a = generate_ip_reference_from_template(template, rng=np.random.default_rng(5), config=config)
    traj_b = generate_ip_reference_from_template(template, rng=np.random.default_rng(5), config=config)
    traj_c = generate_ip_reference_from_template(template, rng=np.random.default_rng(6), config=config)

    assert np.allclose(traj_a.time_s, traj_b.time_s)
    assert np.allclose(traj_a.ip_a, traj_b.ip_a)
    same_length = traj_a.ip_a.shape == traj_c.ip_a.shape
    assert (not same_length) or (not np.allclose(traj_a.ip_a, traj_c.ip_a))
    assert traj_a.rows == template.rows
    assert traj_a.duration_s > 0.0
    assert traj_a.amplitude_scale > 0.0
    assert traj_a.duration_scale > 0.0


def test_generate_ip_reference_from_template_can_force_zero_start(tmp_path: Path) -> None:
    path = tmp_path / "t15md_4322_ip.csv"
    _write_ip_csv(path, np.linspace(0.0, 1.0, 4), np.array([100.0, 200.0, 250.0, 300.0]))
    template = load_semicolon_ip_table(path)
    config = SyntheticIpConfig(amplitude_jitter=0.0, duration_jitter=0.0, shape_jitter=0.0, start_value=0.0)

    trajectory = generate_ip_reference_from_template(template, rng=np.random.default_rng(5), config=config)

    assert trajectory.ip_a[0] == pytest.approx(0.0)
    assert np.allclose(np.diff(trajectory.ip_a), np.diff(template.ip_a))


def test_discover_and_generate_from_template_directory(tmp_path: Path) -> None:
    """Проверить выбор шаблона из директории и запись generated table."""
    _write_ip_csv(tmp_path / "t15md_1001_ip.csv", np.array([0.0, 1.0]), np.array([0.0, 10.0]))
    _write_ip_csv(tmp_path / "t15md_1002_ip.csv", np.array([0.0, 1.0]), np.array([0.0, 20.0]))

    templates = discover_ip_templates(tmp_path)
    trajectory = generate_ip_reference_from_templates(
        templates,
        seed=9,
        config=SyntheticIpConfig(amplitude_jitter=0.0, duration_jitter=0.0, shape_jitter=0.0),
    )
    out_path = tmp_path / "out" / "t15md_9000_ip.csv"
    write_semicolon_ip_table(out_path, trajectory)
    reloaded = make_ip_reference_from_table(out_path)

    assert len(templates) == 2
    assert trajectory.template_shot_id in {"1001", "1002"}
    assert np.allclose(reloaded.time_s, trajectory.time_s)
    assert np.allclose(reloaded.ip_a, trajectory.ip_a)


def test_invalid_ip_reference_tables_are_rejected(tmp_path: Path) -> None:
    """Проверить явные ошибки для нефизичных таблиц Ip."""
    bad_time = tmp_path / "t15md_1111_ip.csv"
    _write_ip_csv(bad_time, np.array([0.0, -1.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="nondecreasing"):
        load_semicolon_ip_table(bad_time)

    zero_ip = tmp_path / "t15md_2222_ip.csv"
    _write_ip_csv(zero_ip, np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="nonzero"):
        load_semicolon_ip_table(zero_ip)

    with pytest.raises(ValueError, match="duration_scale"):
        IpReferenceTrajectory(np.array([0.0, 1.0]), np.array([1.0, 2.0]), duration_scale=0.0)


def test_t15_synthetic_follow_uses_template_generated_ip(tmp_path: Path) -> None:
    """Проверить, что scenario synthetic-follow использует template-generated Ip."""
    ip_path = tmp_path / "t15md_3333_ip.csv"
    _write_ip_csv(ip_path, np.array([0.0, 0.5, 1.0]), np.array([0.0, 100.0, 200.0]))
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    base_radii = np.full((32,), 0.6, dtype=float)
    params = {
        "seed": 3,
        "duration_s": 0.10,
        "t_step": 1.0e-3,
        "ip_template_csv": str(ip_path),
        "ip_seed": 44,
        "amplitude_jitter": 0.0,
        "duration_jitter": 0.0,
        "shape_jitter": 0.0,
    }

    scenario = make_scenario("t15_synthetic_follow", base_radii, 0.0, params=params, center=(1.2, 0.0))

    assert scenario.Ip_ref(0.25) == pytest.approx(50.0)
    assert scenario.Ip_ref(2.0) == pytest.approx(200.0)
    assert np.all(np.isfinite(scenario.ref_radii(angles, 0.05)))


def test_t15_synthetic_follow_rejects_multiple_ip_sources(tmp_path: Path) -> None:
    """Проверить запрет неоднозначных источников Ip."""
    ip_path = tmp_path / "t15md_3333_ip.csv"
    _write_ip_csv(ip_path, np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    base_radii = np.full((32,), 0.6, dtype=float)

    with pytest.raises(ValueError, match="only one Ip source"):
        make_scenario(
            "t15_synthetic_follow",
            base_radii,
            0.0,
            params={"ip_csv": str(ip_path), "ip_template_csv": str(ip_path)},
            center=(1.2, 0.0),
        )


def test_segmented_ip_reference_is_continuous_bounded_and_seeded() -> None:
    config = SegmentedIpConfig(
        value_bounds=(100_000.0, 420_000.0),
        segment_step_bounds=(10, 40),
        segment_count_bounds=(4, 7),
        max_steps=150,
        t_step=1.0e-3,
        rate_limit=8.0e6,
        hold_probability=0.25,
    )

    traj_a = generate_segmented_ip_reference(seed=77, config=config)
    traj_b = generate_segmented_ip_reference(seed=77, config=config)
    traj_c = generate_segmented_ip_reference(seed=78, config=config)

    assert np.allclose(traj_a.time_s, traj_b.time_s)
    assert np.allclose(traj_a.ip_a, traj_b.ip_a)
    same_length = traj_a.ip_a.shape == traj_c.ip_a.shape
    assert (not same_length) or (not np.allclose(traj_a.ip_a, traj_c.ip_a))
    assert traj_a.rows <= config.max_steps
    assert np.min(traj_a.ip_a) >= config.value_bounds[0]
    assert np.max(traj_a.ip_a) <= config.value_bounds[1]
    rates = np.diff(traj_a.ip_a) / np.diff(traj_a.time_s)
    assert np.max(np.abs(rates)) <= config.rate_limit + 1.0e-6
    assert traj_a.segments
    for previous, current in zip(traj_a.segments, traj_a.segments[1:], strict=False):
        assert current.start_step == previous.end_step
        assert current.start_value == pytest.approx(previous.end_value)


def test_segmented_ip_reference_zero_start_respects_rate_limit_before_bounds() -> None:
    config = SegmentedIpConfig(
        value_bounds=(100_000.0, 420_000.0),
        segment_step_bounds=(2, 2),
        segment_count_bounds=(1, 1),
        max_steps=3,
        t_step=1.0e-3,
        rate_limit=10_000.0,
        hold_probability=0.0,
        start_value=0.0,
    )

    trajectory = generate_segmented_ip_reference(seed=77, config=config)

    assert trajectory.ip_a[0] == pytest.approx(0.0)
    rates = np.diff(trajectory.ip_a) / np.diff(trajectory.time_s)
    assert np.max(np.abs(rates)) <= config.rate_limit + 1.0e-6
    assert trajectory.ip_a[-1] < config.value_bounds[0]
