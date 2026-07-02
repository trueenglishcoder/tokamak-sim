import numpy as np

from scripts.idealize_t15_coil_actions import _idealize_currents


def _jdot_jump_p90(time_s: np.ndarray, currents: np.ndarray) -> float:
    jdot = np.diff(currents, axis=0) / np.diff(time_s)[:, None]
    return float(np.percentile(np.abs(np.diff(jdot, axis=0)).max(axis=1), 90.0))


def test_smooth_jdot_idealizer_preserves_endpoints_and_reduces_jdot_jumps() -> None:
    time_s = np.arange(0.0, 0.5, 0.001)
    base = 2.0e5 * time_s[:, None] * np.array([[1.0, -0.7]])
    ripple = 1.0e3 * np.sin(2.0 * np.pi * 80.0 * time_s)[:, None] * np.array([[1.0, 0.4]])
    currents = base + ripple

    ideal, _ = _idealize_currents(
        time_s,
        currents,
        method="smooth_jdot",
        knot_step_s=0.05,
        smooth_window_steps=21,
    )

    np.testing.assert_allclose(ideal[0], currents[0])
    np.testing.assert_allclose(ideal[-1], currents[-1])
    assert _jdot_jump_p90(time_s, ideal) < 0.4 * _jdot_jump_p90(time_s, currents)
