from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import tokamak_control.control.lqr_t15_zaitsev as zaitsev
from tokamak_control.control.lqr_t15_zaitsev import LQRT15ZaitsevController, _dlqr_gain
from tokamak_control.control.registry import controller_names, make_controller, normalize_controller_launch


class _Scenario:
    """Минимальный сценарий с постоянными опорными значениями."""

    def __init__(self, *, ip_ref: float = 0.0) -> None:
        """Сохранить постоянный опорный ток плазмы."""
        self.ip_ref = float(ip_ref)

    def ref_radii(self, angles: np.ndarray, _t: float) -> np.ndarray:
        """Вернуть пустой профиль радиусов для тестов без границы."""
        return np.zeros_like(angles, dtype=float)

    def Ip_ref(self, _t: float) -> float:
        """Вернуть постоянный опорный ток плазмы."""
        return self.ip_ref


class _FakeModel:
    """Минимальная модель с одним PFC-каналом для проверки LQR-семантики."""

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        ip_b: float = 2.0,
        deriv_limit: float = 1.0,
        n_sol: int = 0,
        sol_ip_b: float = 0.0,
    ) -> None:
        """Создать модель с заданным запаздыванием и Ip-чувствительностью."""
        self.pfc = SimpleNamespace(n_coils=1)
        self.sol = SimpleNamespace(n_coils=int(n_sol))
        self.t_step = 0.1
        self.Ip0 = 0.0
        self.pfc_deriv_limit = float(deriv_limit)
        self.sol_deriv_limit = float(deriv_limit)
        self._alpha = float(alpha)
        self._ip_b = float(ip_b)
        self._sol_ip_b = float(sol_ip_b)
        self.step_current_deltas: list[np.ndarray] = []
        self.state = SimpleNamespace(
            t=0.0,
            step=0,
            Ip=0.0,
            Ip0=0.0,
            psi=np.zeros((2, 2), dtype=float),
            pfc_currents=np.array([0.0], dtype=float),
            sol_currents=np.zeros((int(n_sol),), dtype=float),
            pfc_current_derivs=np.array([0.0], dtype=float),
            sol_current_derivs=np.zeros((int(n_sol),), dtype=float),
        )

    def _actuator_alpha(self) -> float:
        """Вернуть заданный коэффициент запаздывания."""
        return self._alpha

    def get_ip_B_row(self) -> np.ndarray:
        """Вернуть чувствительность Ip к немедленно приложенной производной."""
        return np.concatenate(
            [
                np.array([self._ip_b], dtype=float),
                np.full((int(self.sol.n_coils),), self._sol_ip_b, dtype=float),
            ],
            axis=0,
        )

    def predict_Ip_decay_baseline_next(self) -> float:
        """Вернуть пассивный прогноз Ip без управляющего вклада."""
        return float(self.state.Ip)

    def snapshot_state(self):
        """Вернуть копию минимального состояния."""
        return SimpleNamespace(
            t=float(self.state.t),
            step=int(self.state.step),
            Ip=float(self.state.Ip),
            Ip0=float(self.state.Ip0),
            psi=np.asarray(self.state.psi, dtype=float).copy(),
            pfc_currents=np.asarray(self.state.pfc_currents, dtype=float).copy(),
            sol_currents=np.asarray(self.state.sol_currents, dtype=float).copy(),
            pfc_current_derivs=np.asarray(self.state.pfc_current_derivs, dtype=float).copy(),
            sol_current_derivs=np.asarray(self.state.sol_current_derivs, dtype=float).copy(),
        )

    def restore_state(self, state):
        """Восстановить копию минимального состояния."""
        self.state = SimpleNamespace(
            t=float(state.t),
            step=int(state.step),
            Ip=float(state.Ip),
            Ip0=float(state.Ip0),
            psi=np.asarray(state.psi, dtype=float).copy(),
            pfc_currents=np.asarray(state.pfc_currents, dtype=float).copy(),
            sol_currents=np.asarray(state.sol_currents, dtype=float).copy(),
            pfc_current_derivs=np.asarray(state.pfc_current_derivs, dtype=float).copy(),
            sol_current_derivs=np.asarray(state.sol_current_derivs, dtype=float).copy(),
        )
        return self.state

    def step_currents(self, *, pfc_currents_next=None, sol_currents_next=None):
        """Минимальная stateful Ip реакция на абсолютный следующий ток."""
        next_pfc = np.asarray(pfc_currents_next, dtype=float).reshape(1)
        n_sol = int(self.sol.n_coils)
        next_sol = np.zeros((n_sol,), dtype=float) if sol_currents_next is None else np.asarray(sol_currents_next, dtype=float).reshape(n_sol)
        self.step_current_deltas.append(
            np.concatenate([next_pfc - self.state.pfc_currents, next_sol - self.state.sol_currents], axis=0)
        )
        deriv = (next_pfc - self.state.pfc_currents) / float(self.t_step)
        sol_deriv = (next_sol - self.state.sol_currents) / float(self.t_step)
        self.state = SimpleNamespace(
            t=float(self.state.t) + float(self.t_step),
            step=int(self.state.step) + 1,
            Ip=float(self.state.Ip) + float(self._ip_b) * float(deriv[0]) + float(self._sol_ip_b) * float(np.sum(sol_deriv)),
            Ip0=float(self.Ip0),
            psi=np.zeros((2, 2), dtype=float),
            pfc_currents=next_pfc.copy(),
            sol_currents=next_sol.copy(),
            pfc_current_derivs=deriv.copy(),
            sol_current_derivs=sol_deriv.copy(),
        )
        return self.state


def test_dlqr_gain_matches_scalar_closed_form() -> None:
    """Проверить gain DARE на скалярном примере с аналитическим решением."""
    A = np.array([[0.9]], dtype=float)
    B = np.array([[1.0]], dtype=float)
    Q = np.array([[1.0]], dtype=float)
    R = np.array([[1.0]], dtype=float)
    gain = _dlqr_gain(A, B, Q, R)

    p = (0.81 + np.sqrt(0.81**2 + 4.0)) / 2.0
    expected = p * 0.9 / (p + 1.0)
    assert np.allclose(gain, np.array([[expected]], dtype=float))


def test_zaitsev_system_uses_delta_jdot_and_stateful_step() -> None:
    """Проверить знак и масштаб B для управления delta-Jdot."""
    model = _FakeModel(alpha=0.75, ip_b=20.0)
    controller = LQRT15ZaitsevController(ip_weight=1.0)
    system = controller._build_system(
        model=model,
        psi=np.zeros((2, 2), dtype=float),
        boundary_poly=None,
        center=(0.0, 0.0),
        measure_angles=np.zeros((0,), dtype=float),
        ref_radii=np.zeros((0,), dtype=float),
        Ip_ref=10.0,
        scenario=_Scenario(ip_ref=10.0),
        limiter_shape=None,
        boundary_mode="legacy_contour",
        legacy_precision_index2=1.0e-3,
    )

    assert system.h_size == 1
    assert system.n_u == 1
    assert np.allclose(system.B[0, 0], -20.0)
    assert np.allclose(system.B[1, 0], 0.1)
    assert np.allclose(system.A[0, 1], -200.0)


def test_zaitsev_lqr_normalizes_state_and_delta_jdot_r() -> None:
    """State errors and delta-Jdot input cost are normalized in physical units."""
    model = _FakeModel(alpha=0.0, ip_b=20.0)
    controller = LQRT15ZaitsevController(
        boundary_weight=2.0,
        ip_weight=3.0,
        derivative_weight=4.0,
        delta_derivative_weight=5.0,
        boundary_scale_m=0.05,
        ip_scale_a=20000.0,
        derivative_scale_aps=1.0e6,
        delta_derivative_scale_aps=2.0e5,
    )
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    theta = np.linspace(-np.pi, np.pi, 129, endpoint=True, dtype=float)
    boundary_poly = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    system = controller._build_system(
        model=model,
        psi=np.zeros((2, 2), dtype=float),
        boundary_poly=boundary_poly,
        center=(0.0, 0.0),
        measure_angles=angles,
        ref_radii=np.ones((32,), dtype=float),
        Ip_ref=0.0,
        scenario=_Scenario(ip_ref=0.0),
        limiter_shape=None,
        boundary_mode="legacy_contour",
        legacy_precision_index2=1.0e-3,
    )

    q = np.diag(system.Q)
    r = np.diag(system.R)
    assert np.isclose(q[0], 2.0 / 0.05**2)
    assert np.isclose(q[32], 3.0 / 20000.0**2)
    assert np.isclose(q[33], 4.0 / (model.t_step * 1.0e6) ** 2)
    assert np.isclose(r[0], 5.0 / 2.0e5**2)


def test_boundary_finite_difference_uses_absolute_current_perturbation(monkeypatch) -> None:
    """Boundary sensitivity should use LittleScope's finite -10 A perturbation."""
    model = _FakeModel(alpha=0.0, ip_b=0.0, deriv_limit=1000.0)

    def _radii_from_current(**kwargs):
        m = kwargs["model"]
        return np.array([float(m.state.pfc_currents[0])], dtype=float)

    monkeypatch.setattr(zaitsev, "_extract_legacy_boundary_radii", _radii_from_current)
    controller = LQRT15ZaitsevController(finite_difference_current_delta_a=10.0)
    system = controller._build_system(
        model=model,
        psi=np.zeros((2, 2), dtype=float),
        boundary_poly=np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]], dtype=float),
        center=(0.0, 0.0),
        measure_angles=np.array([0.0], dtype=float),
        ref_radii=np.array([0.0], dtype=float),
        Ip_ref=0.0,
        scenario=_Scenario(ip_ref=0.0),
        limiter_shape=None,
        boundary_mode="legacy_contour",
        legacy_precision_index2=1.0e-3,
    )

    assert any(np.allclose(delta, np.array([-10.0], dtype=float)) for delta in model.step_current_deltas)
    assert np.allclose(system.B[0, 0], -0.1)


def test_boundary_response_is_littlescope_diagonal_pfc_and_ip_uses_sol(monkeypatch) -> None:
    """Boundary rows follow ControlBoth PFC assignment; Ip control is SOL-only."""
    model = _FakeModel(alpha=0.0, ip_b=2.0, n_sol=1, sol_ip_b=4.0)

    def _radii_from_current(**kwargs):
        m = kwargs["model"]
        return np.array([float(m.state.pfc_currents[0])], dtype=float)

    monkeypatch.setattr(zaitsev, "_extract_legacy_boundary_radii", _radii_from_current)
    controller = LQRT15ZaitsevController(finite_difference_current_delta_a=10.0)
    system = controller._build_system(
        model=model,
        psi=np.zeros((2, 2), dtype=float),
        boundary_poly=np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]], dtype=float),
        center=(0.0, 0.0),
        measure_angles=np.array([0.0], dtype=float),
        ref_radii=np.array([0.0], dtype=float),
        Ip_ref=0.0,
        scenario=_Scenario(ip_ref=0.0),
        limiter_shape=None,
        boundary_mode="legacy_contour",
        legacy_precision_index2=1.0e-3,
    )

    assert any(np.allclose(delta, np.array([-10.0, 5.0], dtype=float)) for delta in model.step_current_deltas)
    assert not any(np.allclose(delta, np.array([0.0, -10.0], dtype=float)) for delta in model.step_current_deltas)
    assert np.allclose(system.B[0, 0], -0.1)
    assert np.allclose(system.B[0, 1], 0.0)
    assert np.allclose(system.B[1, 0], 0.0)
    assert np.allclose(system.B[1, 1], -4.0)


def test_boundary_error_produces_nonzero_lqr_action_after_scaling(monkeypatch) -> None:
    """A 5 cm 32-point boundary error should be visible to the controller."""
    model = _FakeModel(alpha=0.0, ip_b=20.0)

    def _radii_from_current(**kwargs):
        m = kwargs["model"]
        return np.full((32,), float(m.state.pfc_currents[0]), dtype=float)

    monkeypatch.setattr(zaitsev, "_extract_legacy_boundary_radii", _radii_from_current)
    controller = LQRT15ZaitsevController(
        boundary_weight=1.0,
        ip_weight=1.0,
        derivative_weight=0.0,
        delta_derivative_weight=1.0,
        boundary_scale_m=0.03,
        ip_scale_a=25000.0,
        delta_derivative_scale_aps=500000.0,
    )
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    theta = np.linspace(-np.pi, np.pi, 129, endpoint=True, dtype=float)
    boundary_poly = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    system = controller._build_system(
        model=model,
        psi=np.zeros((2, 2), dtype=float),
        boundary_poly=boundary_poly,
        center=(0.0, 0.0),
        measure_angles=angles,
        ref_radii=np.full((32,), 1.05, dtype=float),
        Ip_ref=0.0,
        scenario=_Scenario(ip_ref=0.0),
        limiter_shape=None,
        boundary_mode="legacy_contour",
        legacy_precision_index2=1.0e-3,
    )
    gain = _dlqr_gain(system.A, system.B, system.Q, system.R)

    boundary_only = np.zeros_like(system.x)
    boundary_only[:32] = 0.05
    ip_only = np.zeros_like(system.x)
    ip_only[32] = 50000.0
    boundary_action = -(gain @ boundary_only)
    ip_action = -(gain @ ip_only)

    assert np.max(np.abs(boundary_action)) > 0.1
    assert np.max(np.abs(boundary_action)) > np.max(np.abs(ip_action))


def test_zaitsev_system_uses_fixed_boundary_radius_errors() -> None:
    """Boundary state should be fixed reference radii minus measured radii."""
    model = _FakeModel(alpha=0.0, ip_b=0.0)
    controller = LQRT15ZaitsevController(boundary_weight=1.0, ip_weight=1.0)
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    theta = np.linspace(-np.pi, np.pi, 129, endpoint=True, dtype=float)
    measured_radius = 0.9
    boundary_poly = np.stack(
        [measured_radius * np.cos(theta), measured_radius * np.sin(theta)],
        axis=1,
    )
    ref_radii = np.ones((32,), dtype=float)

    system = controller._build_system(
        model=model,
        psi=np.zeros((2, 2), dtype=float),
        boundary_poly=boundary_poly,
        center=(0.0, 0.0),
        measure_angles=angles,
        ref_radii=ref_radii,
        Ip_ref=0.0,
        scenario=_Scenario(ip_ref=0.0),
        limiter_shape=None,
        boundary_mode="legacy_contour",
        legacy_precision_index2=1.0e-3,
    )

    assert system.h_size == 33
    assert np.allclose(system.x[:32], np.full((32,), 0.1, dtype=float), atol=1.0e-3)
    assert system.x[32] == 0.0


def test_missing_boundary_keeps_fixed_bad_boundary_state() -> None:
    """Boundary loss must not silently remove boundary state components."""
    model = _FakeModel(alpha=0.0, ip_b=0.0)
    controller = LQRT15ZaitsevController(boundary_weight=1.0, ip_weight=1.0)
    angles = np.linspace(-np.pi, np.pi, 6, endpoint=False, dtype=float)
    ref_radii = np.full((6,), 0.7, dtype=float)

    state = controller._build_state(
        model=model,
        psi=np.zeros((2, 2), dtype=float),
        boundary_poly=None,
        center=(0.0, 0.0),
        measure_angles=angles,
        ref_radii=ref_radii,
        Ip_ref=0.0,
        scenario=_Scenario(ip_ref=0.0),
        limiter_shape=None,
        boundary_mode="legacy_contour",
        legacy_precision_index2=1.0e-3,
        drift_override=None,
    )

    assert state.n_boundary == 6
    assert state.h_size == 7
    assert np.allclose(state.x[:6], np.ones((6,), dtype=float))


def test_command_is_accumulated_and_clipped_after_delta_jdot() -> None:
    """Проверить накопление команды и клиппинг накопленного Jdot."""
    model = _FakeModel(alpha=0.0, ip_b=10.0, deriv_limit=0.25)
    controller = LQRT15ZaitsevController(
        ip_weight=10.0,
        derivative_weight=0.0,
        delta_derivative_weight=1.0e-8,
    )

    first = controller.compute_control(
        model=model,
        psi=np.zeros((2, 2), dtype=float),
        boundary_poly=None,
        center=(0.0, 0.0),
        measure_angles=np.zeros((0,), dtype=float),
        ref_radii=np.zeros((0,), dtype=float),
        Ip_ref=100.0,
        scenario=_Scenario(ip_ref=100.0),
    )
    previous = np.asarray(first.pfc_currents_next, dtype=float).copy() / model.t_step
    assert np.allclose(previous, np.array([0.25], dtype=float))
    assert controller.last_derivative_clipping_fraction == 1.0

    second = controller.compute_control(
        model=model,
        psi=np.zeros((2, 2), dtype=float),
        boundary_poly=None,
        center=(0.0, 0.0),
        measure_angles=np.zeros((0,), dtype=float),
        ref_radii=np.zeros((0,), dtype=float),
        Ip_ref=100.0,
        scenario=_Scenario(ip_ref=100.0),
    )
    command = np.asarray(second.pfc_currents_next, dtype=float) / model.t_step
    assert np.allclose(command, np.array([0.25], dtype=float))
    assert np.allclose(controller.last_delta_jdot_applied, command - previous)


def test_cached_gain_is_used_when_later_dare_solve_fails(monkeypatch) -> None:
    """Проверить, что поздний сбой DARE не прерывает rollout при наличии K."""
    model = _FakeModel(alpha=0.0, ip_b=10.0, deriv_limit=1.0)
    controller = LQRT15ZaitsevController(
        ip_weight=10.0,
        derivative_weight=0.0,
        delta_derivative_weight=1.0e-4,
        gain_recompute_interval_steps=1,
    )

    first = controller.compute_control(
        model=model,
        psi=np.zeros((2, 2), dtype=float),
        boundary_poly=None,
        center=(0.0, 0.0),
        measure_angles=np.zeros((0,), dtype=float),
        ref_radii=np.zeros((0,), dtype=float),
        Ip_ref=10.0,
        scenario=_Scenario(ip_ref=10.0),
    )
    assert np.all(np.isfinite(first.pfc_currents_next))
    assert controller.last_gain is not None
    assert not controller.last_gain_fallback_used
    model.step_currents(pfc_currents_next=first.pfc_currents_next, sol_currents_next=first.sol_currents_next)

    def _fail_gain(*_args, **_kwargs):
        raise ValueError("forced DARE failure")

    monkeypatch.setattr(zaitsev, "_dlqr_gain", _fail_gain)
    second = controller.compute_control(
        model=model,
        psi=np.zeros((2, 2), dtype=float),
        boundary_poly=None,
        center=(0.0, 0.0),
        measure_angles=np.zeros((0,), dtype=float),
        ref_radii=np.zeros((0,), dtype=float),
        Ip_ref=10.0,
        scenario=_Scenario(ip_ref=10.0),
    )

    assert np.all(np.isfinite(second.pfc_currents_next))
    assert controller.last_gain_fallback_used
    assert controller.gain_fallback_count == 1
    assert controller.last_gain_failure == "forced DARE failure"


def test_lqr_gain_is_reused_between_recompute_intervals(monkeypatch) -> None:
    """The expensive DARE path should not run every control step."""
    model = _FakeModel(alpha=0.0, ip_b=10.0, deriv_limit=1.0)
    controller = LQRT15ZaitsevController(
        ip_weight=10.0,
        derivative_weight=0.0,
        delta_derivative_weight=1.0e-4,
        gain_recompute_interval_steps=25,
    )
    calls = {"count": 0}
    original = zaitsev._dlqr_gain

    def _counted_gain(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(zaitsev, "_dlqr_gain", _counted_gain)
    first = controller.compute_control(
        model=model,
        psi=np.zeros((2, 2), dtype=float),
        boundary_poly=None,
        center=(0.0, 0.0),
        measure_angles=np.zeros((0,), dtype=float),
        ref_radii=np.zeros((0,), dtype=float),
        Ip_ref=10.0,
        scenario=_Scenario(ip_ref=10.0),
    )
    assert calls["count"] == 1
    assert controller.last_gain_recomputed
    assert controller.last_gain_recompute_reason == "initial"

    model.step_currents(pfc_currents_next=first.pfc_currents_next, sol_currents_next=first.sol_currents_next)
    controller.compute_control(
        model=model,
        psi=np.zeros((2, 2), dtype=float),
        boundary_poly=None,
        center=(0.0, 0.0),
        measure_angles=np.zeros((0,), dtype=float),
        ref_radii=np.zeros((0,), dtype=float),
        Ip_ref=10.0,
        scenario=_Scenario(ip_ref=10.0),
    )

    assert calls["count"] == 1
    assert not controller.last_gain_recomputed
    assert controller.last_gain_age_steps == 1


def test_lqr_gain_recomputes_after_interval(monkeypatch) -> None:
    """A configured short interval should force a new gain synthesis."""
    model = _FakeModel(alpha=0.0, ip_b=10.0, deriv_limit=1.0)
    controller = LQRT15ZaitsevController(
        ip_weight=10.0,
        derivative_weight=0.0,
        delta_derivative_weight=1.0e-4,
        gain_recompute_interval_steps=1,
    )
    calls = {"count": 0}
    original = zaitsev._dlqr_gain

    def _counted_gain(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(zaitsev, "_dlqr_gain", _counted_gain)
    first = controller.compute_control(
        model=model,
        psi=np.zeros((2, 2), dtype=float),
        boundary_poly=None,
        center=(0.0, 0.0),
        measure_angles=np.zeros((0,), dtype=float),
        ref_radii=np.zeros((0,), dtype=float),
        Ip_ref=10.0,
        scenario=_Scenario(ip_ref=10.0),
    )
    model.step_currents(pfc_currents_next=first.pfc_currents_next, sol_currents_next=first.sol_currents_next)
    controller.compute_control(
        model=model,
        psi=np.zeros((2, 2), dtype=float),
        boundary_poly=None,
        center=(0.0, 0.0),
        measure_angles=np.zeros((0,), dtype=float),
        ref_radii=np.zeros((0,), dtype=float),
        Ip_ref=10.0,
        scenario=_Scenario(ip_ref=10.0),
    )

    assert calls["count"] == 2
    assert controller.last_gain_recomputed
    assert controller.last_gain_recompute_reason == "interval"
    assert controller.gain_recompute_count == 2


def test_lqr_t15_zaitsev_is_registered() -> None:
    """Проверить доступность нового контроллера через общий registry."""
    assert "lqr_t15_zaitsev" in controller_names()
    controller = make_controller(
        "lqr_t15_zaitsev",
        config={
            "ip_weight": 2.0,
            "boundary_scale_m": 0.04,
            "ip_scale_a": 30000.0,
            "delta_derivative_scale_aps": 250000.0,
            "finite_difference_current_delta_a": 5.0,
            "gain_recompute_interval_steps": 10,
            "gain_recompute_on_boundary_loss": False,
        },
    )
    assert isinstance(controller, LQRT15ZaitsevController)
    assert controller.gain_recompute_interval_steps == 10
    assert not controller.gain_recompute_on_boundary_loss


def test_lqr_t15_zaitsev_registry_rejects_invalid_scales() -> None:
    """Новые физические масштабы должны быть строго положительными."""
    with pytest.raises(ValueError):
        normalize_controller_launch("lqr_t15_zaitsev", {"boundary_scale_m": 0.0})
    with pytest.raises(ValueError):
        normalize_controller_launch("lqr_t15_zaitsev", {"ip_scale_a": -1.0})
    with pytest.raises(ValueError):
        normalize_controller_launch("lqr_t15_zaitsev", {"delta_derivative_scale_aps": 0.0})
    with pytest.raises(ValueError):
        normalize_controller_launch("lqr_t15_zaitsev", {"finite_difference_current_delta_a": 0.0})
    with pytest.raises(ValueError):
        normalize_controller_launch("lqr_t15_zaitsev", {"gain_recompute_interval_steps": 0})
