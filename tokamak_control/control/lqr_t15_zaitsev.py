from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, solve_discrete_are

from tokamak_control.control.base import ControlAction, Controller
from tokamak_control.geometry.boundary import find_plasma_boundary_with_status
from tokamak_control.geometry.boundary_common import BoundaryNotFoundError
from tokamak_control.geometry.legacy_metrics import legacy_radii_at_angles

_MISSING_BOUNDARY_ERROR_M = 1.0


@dataclass(frozen=True, slots=True)
class ZaitsevLqrSystem:
    """Линеаризованная система Зайцева для одного шага синтеза LQR."""

    A: np.ndarray
    B: np.ndarray
    x: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    command_limit: np.ndarray
    command_prev: np.ndarray
    h_size: int
    n_u: int


@dataclass(frozen=True, slots=True)
class ZaitsevLqrState:
    """Current LQR state that can be used with a cached gain."""

    x: np.ndarray
    command_limit: np.ndarray
    command_prev: np.ndarray
    h_size: int
    n_u: int
    n_boundary: int
    n_current: int


def _as_positive_finite(name: str, value: float, *, allow_zero: bool = False) -> float:
    """Проверить конечный положительный или неотрицательный параметр."""
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if allow_zero:
        if out < 0.0:
            raise ValueError(f"{name} must be >= 0")
    elif out <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return out


def _active_vector(model, pfc_values: np.ndarray, sol_values: np.ndarray) -> np.ndarray:
    """Собрать активный вектор в порядке PFC, затем SOL."""
    n_pfc = int(getattr(model.pfc, "n_coils", 0))
    n_sol = int(getattr(model.sol, "n_coils", 0))
    pfc = np.asarray(pfc_values, dtype=float).reshape(n_pfc)
    sol = np.asarray(sol_values, dtype=float).reshape(n_sol)
    return np.concatenate([pfc, sol], axis=0)


def _limit_vector(model, fallback_scale: float) -> np.ndarray:
    """Построить симметричные пределы команд производных для всех катушек."""
    n_pfc = int(getattr(model.pfc, "n_coils", 0))
    n_sol = int(getattr(model.sol, "n_coils", 0))
    fallback = float(fallback_scale)
    pfc_limit = getattr(model, "pfc_deriv_limit", None)
    sol_limit = getattr(model, "sol_deriv_limit", None)
    pfc = fallback if pfc_limit is None else float(pfc_limit)
    sol = fallback if sol_limit is None else float(sol_limit)
    return np.concatenate([
        np.full((n_pfc,), max(abs(pfc), 1e-12), dtype=float),
        np.full((n_sol,), max(abs(sol), 1e-12), dtype=float),
    ])


def _split_active(model, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_pfc = int(getattr(model.pfc, "n_coils", 0))
    n_sol = int(getattr(model.sol, "n_coils", 0))
    arr = np.asarray(active, dtype=float).reshape(n_pfc + n_sol)
    return arr[:n_pfc].copy(), arr[n_pfc:].copy()


def _sol_ip_compensation_for_channel(ip_sens: np.ndarray, n_pfc: int, pfc_index: int, pfc_delta_jdot: float) -> np.ndarray:
    """Return SOL delta-Jdot that cancels the Ip effect of one PFC perturbation.

    LittleScope's ``GetC1`` computes PFC boundary sensitivity while changing
    SOL enough to compensate the Ip change caused by the perturbed PFC coil.
    T15 has three SOL channels rather than LittleScope's single SOL channel,
    so the compensation is distributed in the least-norm direction of the SOL
    Ip-sensitivity vector.
    """
    sens = np.asarray(ip_sens, dtype=float).reshape(-1)
    n_pfc = int(n_pfc)
    n_sol = max(0, int(sens.size) - n_pfc)
    if n_sol == 0:
        return np.zeros((0,), dtype=float)
    sol_sens = sens[n_pfc:]
    denom = float(np.dot(sol_sens, sol_sens))
    if denom <= 1.0e-30 or not np.isfinite(denom):
        return np.zeros((n_sol,), dtype=float)
    pfc_effect = float(sens[int(pfc_index)]) * float(pfc_delta_jdot)
    return -(pfc_effect / denom) * sol_sens


def _littlescope_ip_control_row(model, ip_sens: np.ndarray) -> np.ndarray:
    """Return Ip sensitivity exposed to LQR inputs under ControlBoth semantics."""
    sens = np.asarray(ip_sens, dtype=float).reshape(-1)
    n_pfc = int(getattr(model.pfc, "n_coils", 0))
    n_sol = int(getattr(model.sol, "n_coils", 0))
    if n_sol <= 0:
        return sens.copy()
    row = np.zeros_like(sens)
    row[n_pfc:n_pfc + n_sol] = sens[n_pfc:n_pfc + n_sol]
    return row


def _actuator_alpha(model) -> float:
    """Вернуть plant-side lag coefficient.

    The parity plant matches Little SCoPE and applies the requested derivative
    command immediately. Actuator lag may exist as controller/safety metadata,
    but it is not part of the core plant transition.
    """
    return 0.0


def _dlqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Вычислить дискретный LQR gain через DARE с устойчивым fallback."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    try:
        P = solve_discrete_are(A, B, Q, R)
    except (LinAlgError, ValueError):
        R_reg = R + 1e-9 * np.eye(R.shape[0], dtype=float)
        try:
            P = solve_discrete_are(A, B, Q + 1e-12 * np.eye(Q.shape[0], dtype=float), R_reg)
            R = R_reg
        except (LinAlgError, ValueError):
            return _regularized_one_step_gain(A, B, Q, R_reg)
    H = B.T @ P @ B + R
    G = B.T @ P @ A
    return np.linalg.pinv(H) @ G


def _regularized_one_step_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Return a finite one-step LQR gain when infinite-horizon DARE is ill-conditioned."""
    H = B.T @ Q @ B + R
    G = B.T @ Q @ A
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.max(np.abs(H))) if H.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    H = H + (1.0e-9 * scale + 1.0e-30) * np.eye(H.shape[0], dtype=float)
    return np.linalg.pinv(H) @ G


def _extract_legacy_boundary_radii(
    *,
    model,
    center: tuple[float, float],
    measure_angles: np.ndarray,
    limiter_shape: np.ndarray | None,
    boundary_mode: str,
    legacy_precision_index2: float,
) -> np.ndarray | None:
    if np.asarray(measure_angles, dtype=float).size == 0:
        return np.zeros((0,), dtype=float)
    if not hasattr(model, "grid"):
        return None
    try:
        poly, _level, _status = find_plasma_boundary_with_status(
            model.state.psi,
            model.grid,
            center,
            limiter_shape=limiter_shape,
            boundary_mode=boundary_mode,
            legacy_precision_index2=legacy_precision_index2,
        )
    except BoundaryNotFoundError:
        return None
    return legacy_radii_at_angles(poly, center, measure_angles)


def _boundary_radius_error(ref_radii: np.ndarray, radii: np.ndarray | None) -> np.ndarray:
    """Return held-boundary radius error in the active measurement basis."""
    ref = np.asarray(ref_radii, dtype=float).reshape(-1)
    if radii is None:
        return np.full_like(ref, _MISSING_BOUNDARY_ERROR_M, dtype=float)
    measured = np.asarray(radii, dtype=float).reshape(-1)
    if ref.size == 0:
        return np.zeros((0,), dtype=float)
    if measured.shape != ref.shape:
        raise ValueError(f"boundary radii shape mismatch: ref={ref.shape}, measured={measured.shape}")
    return ref - measured


def _finite_difference_h_response(
    *,
    model,
    base_state,
    center: tuple[float, float],
    measure_angles: np.ndarray,
    ref_radii_next: np.ndarray,
    ip_ref_next: float,
    current_ref: np.ndarray | None,
    command_prev: np.ndarray,
    current_now: np.ndarray,
    h_now: np.ndarray,
    include_boundary: bool,
    limiter_shape: np.ndarray | None,
    boundary_mode: str,
    legacy_precision_index2: float,
    current_weight: float,
    finite_difference_current_delta_a: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the next-state response to delta-Jdot.

    LittleScope ``ControlBoth`` assigns one boundary measurement to each PFC
    coil and fills only the diagonal PFC boundary response.  During that
    boundary finite difference, ``GetC1`` compensates the PFC perturbation with
    a SOL change so the measured boundary response is not polluted by Ip
    motion.  The active Ip row is then exposed through SOL channels only.
    """
    dt = float(getattr(model, "t_step"))
    if dt <= 0.0:
        raise ValueError(f"model.t_step must be > 0, got {dt!r}")
    n_u = int(command_prev.size)

    def h_after(command: np.ndarray) -> tuple[np.ndarray, float]:
        model.restore_state(base_state)
        next_active = current_now + dt * np.asarray(command, dtype=float).reshape(n_u)
        pfc_next, sol_next = _split_active(model, next_active)
        model.step_currents(pfc_currents_next=pfc_next, sol_currents_next=sol_next)
        if include_boundary:
            radii = _extract_legacy_boundary_radii(
                model=model,
                center=center,
                measure_angles=measure_angles,
                limiter_shape=limiter_shape,
                boundary_mode=boundary_mode,
                legacy_precision_index2=legacy_precision_index2,
            )
        else:
            radii = None
        if include_boundary and radii is None:
            h_boundary = _boundary_radius_error(ref_radii_next, None)
        elif include_boundary:
            h_boundary = _boundary_radius_error(ref_radii_next, radii)
        else:
            h_boundary = np.zeros((0,), dtype=float)
        parts = [h_boundary, np.array([float(ip_ref_next) - float(model.state.Ip)], dtype=float)]
        if current_weight > 0.0 and current_ref is not None:
            current_after = _active_vector(model, model.state.pfc_currents, model.state.sol_currents)
            parts.append(np.asarray(current_ref, dtype=float).reshape(n_u) - current_after)
        return np.concatenate(parts, axis=0), float(model.state.Ip)

    h_next_prev, ip_after_prev = h_after(command_prev)
    del ip_after_prev
    B_h = np.zeros((h_now.size, n_u), dtype=float)

    try:
        ip_sens = np.asarray(model.get_ip_B_row(), dtype=float).reshape(n_u)
    except Exception:
        ip_sens = np.zeros((n_u,), dtype=float)

    eps = -float(finite_difference_current_delta_a) / dt
    if not np.isfinite(eps) or eps >= 0.0:
        raise ValueError("finite_difference_current_delta_a must produce a negative finite LittleScope Jdot perturbation")

    n_pfc = int(getattr(model.pfc, "n_coils", 0))
    use_littlescope_boundary_assignment = bool(include_boundary and ref_radii_next.size == n_pfc and n_pfc <= n_u)
    if use_littlescope_boundary_assignment:
        for k in range(n_pfc):
            pert = np.asarray(command_prev, dtype=float).copy()
            pert[k] += eps
            sol_comp = _sol_ip_compensation_for_channel(ip_sens, n_pfc, k, eps)
            if sol_comp.size:
                pert[n_pfc:n_pfc + sol_comp.size] += sol_comp
            h_pert, _ip_pert = h_after(pert)
            B_h[k, k] = (h_pert[k] - h_next_prev[k]) / eps
    else:
        for k in range(n_u):
            pert = np.asarray(command_prev, dtype=float).copy()
            pert[k] += eps
            h_pert, _ip_pert = h_after(pert)
            B_h[:, k] = (h_pert - h_next_prev) / eps

    ip_row = int(ref_radii_next.size) if include_boundary else 0
    if h_now.size > ip_row:
        B_h[ip_row, :] = -_littlescope_ip_control_row(model, ip_sens)

    current_start = ip_row + 1
    if current_weight > 0.0 and current_ref is not None and h_now.size >= current_start + n_u:
        B_h[current_start:current_start + n_u, :] = -dt * np.eye(n_u, dtype=float)

    model.restore_state(base_state)
    return h_next_prev, np.nan_to_num(B_h, nan=0.0, posinf=0.0, neginf=0.0)


class LQRT15ZaitsevController(Controller):
    """LQR-регулятор T15 по модели Зайцева с управлением приращением Jdot."""

    def __init__(
        self,
        *,
        boundary_weight: float = 1.0,
        ip_weight: float = 1.0,
        derivative_weight: float = 0.0,
        delta_derivative_weight: float = 1.0,
        drift_weight_fraction: float = 0.0,
        current_weight: float = 0.0,
        derivative_scale_aps: float = 1.0e6,
        delta_derivative_scale_aps: float = 5.0e5,
        boundary_scale_m: float = 0.03,
        ip_scale_a: float = 25000.0,
        finite_difference_current_delta_a: float = 10.0,
        gain_recompute_interval_steps: int = 25,
        gain_recompute_on_boundary_loss: bool = True,
    ) -> None:
        """Инициализировать LQR-задачу в нормированных физических координатах."""
        self.boundary_weight = _as_positive_finite("boundary_weight", boundary_weight, allow_zero=True)
        self.ip_weight = _as_positive_finite("ip_weight", ip_weight, allow_zero=True)
        self.derivative_weight = _as_positive_finite("derivative_weight", derivative_weight, allow_zero=True)
        self.delta_derivative_weight = _as_positive_finite("delta_derivative_weight", delta_derivative_weight)
        self.drift_weight_fraction = _as_positive_finite("drift_weight_fraction", drift_weight_fraction, allow_zero=True)
        self.current_weight = _as_positive_finite("current_weight", current_weight, allow_zero=True)
        self.derivative_scale_aps = _as_positive_finite("derivative_scale_aps", derivative_scale_aps)
        self.delta_derivative_scale_aps = _as_positive_finite("delta_derivative_scale_aps", delta_derivative_scale_aps)
        self.boundary_scale_m = _as_positive_finite("boundary_scale_m", boundary_scale_m)
        self.ip_scale_a = _as_positive_finite("ip_scale_a", ip_scale_a)
        self.finite_difference_current_delta_a = _as_positive_finite("finite_difference_current_delta_a", finite_difference_current_delta_a)
        self.gain_recompute_interval_steps = int(gain_recompute_interval_steps)
        if self.gain_recompute_interval_steps <= 0:
            raise ValueError("gain_recompute_interval_steps must be > 0")
        self.gain_recompute_on_boundary_loss = bool(gain_recompute_on_boundary_loss)
        self.reset()

    def reset(self) -> None:
        """Сбросить накопленную команду производной и диагностические поля."""
        self._command_prev: np.ndarray | None = None
        self._current_ref: np.ndarray | None = None
        self.last_delta_jdot_raw: np.ndarray | None = None
        self.last_delta_jdot_applied: np.ndarray | None = None
        self.last_jdot_command: np.ndarray | None = None
        self.last_derivative_clipping_fraction: float = 0.0
        self.last_gain: np.ndarray | None = None
        self.last_gain_fallback_used: bool = False
        self.last_gain_failure: str | None = None
        self.last_gain_recomputed: bool = False
        self.last_gain_recompute_reason: str | None = None
        self.last_gain_age_steps: int = 0
        self.gain_fallback_count: int = 0
        self.gain_recompute_count: int = 0
        self._cached_gain_step: int | None = None
        self._cached_h_size: int | None = None
        self._cached_n_u: int | None = None
        self._cached_drift: np.ndarray | None = None
        self.last_state: np.ndarray | None = None
        self.last_q_diag_summary: dict[str, float] = {}
        self.last_r_diag_summary: dict[str, float] = {}
        self.last_boundary_sensitivity_max: float = 0.0
        self.last_ip_sensitivity_max: float = 0.0
        self.last_response_condition: float = 0.0
        self.last_h_size: int = 0
        self.last_n_boundary: int = 0

    def compute_control(
        self,
        *,
        model,
        psi: np.ndarray,
        boundary_poly: np.ndarray | None,
        center: tuple[float, float],
        measure_angles: np.ndarray,
        ref_radii: np.ndarray,
        Ip_ref: float | None = None,
        scenario=None,
        limiter_shape: np.ndarray | None = None,
        boundary_mode: str = "legacy_contour",
        legacy_precision_index2: float = 1.0e-3,
    ) -> ControlAction:
        """Вычислить команду Jdot через приращение delta-Jdot."""
        state_view = self._build_state(
            model=model,
            psi=np.asarray(psi, dtype=float),
            boundary_poly=None if boundary_poly is None else np.asarray(boundary_poly, dtype=float),
            center=center,
            measure_angles=np.asarray(measure_angles, dtype=float).reshape(-1),
            ref_radii=np.asarray(ref_radii, dtype=float).reshape(-1),
            Ip_ref=Ip_ref,
            scenario=scenario,
            limiter_shape=limiter_shape,
            boundary_mode=boundary_mode,
            legacy_precision_index2=legacy_precision_index2,
            drift_override=self._cached_drift,
        )
        if state_view.n_u == 0:
            return ControlAction(pfc_currents_next=np.zeros((0,), dtype=float), sol_currents_next=np.zeros((0,), dtype=float))

        step = int(getattr(model.state, "step", 0))
        recompute_reason = self._gain_recompute_reason(state=state_view, step=step, boundary_poly=boundary_poly)
        gain_fallback_used = False
        gain_failure = None
        system_or_state: ZaitsevLqrSystem | ZaitsevLqrState = state_view
        if recompute_reason is not None:
            system = self._build_system(
                model=model,
                psi=np.asarray(psi, dtype=float),
                boundary_poly=None if boundary_poly is None else np.asarray(boundary_poly, dtype=float),
                center=center,
                measure_angles=np.asarray(measure_angles, dtype=float).reshape(-1),
                ref_radii=np.asarray(ref_radii, dtype=float).reshape(-1),
                Ip_ref=Ip_ref,
                scenario=scenario,
                limiter_shape=limiter_shape,
                boundary_mode=boundary_mode,
                legacy_precision_index2=legacy_precision_index2,
            )
            system_or_state = system
            try:
                K = _dlqr_gain(system.A, system.B, system.Q, system.R)
                self.last_gain = K.copy()
                self._cached_gain_step = step
                self._cached_h_size = system.h_size
                self._cached_n_u = system.n_u
                self._cached_drift = system.x[system.h_size + system.n_u:].copy()
                self.gain_recompute_count += 1
                self.last_gain_recomputed = True
                self.last_gain_recompute_reason = recompute_reason
            except (LinAlgError, ValueError) as exc:
                if self.last_gain is None or self.last_gain.shape != (system.n_u, system.A.shape[0]):
                    raise
                K = self.last_gain.copy()
                gain_fallback_used = True
                gain_failure = str(exc)
                self.gain_fallback_count += 1
                self.last_gain_recomputed = False
                self.last_gain_recompute_reason = "fallback_after_" + recompute_reason
        else:
            K = np.asarray(self.last_gain, dtype=float).copy()
            self.last_gain_recomputed = False
            self.last_gain_recompute_reason = None

        if self._cached_gain_step is None:
            self.last_gain_age_steps = 0
        else:
            self.last_gain_age_steps = max(0, step - int(self._cached_gain_step))

        delta_raw = -(K @ system_or_state.x)
        command_raw = system_or_state.command_prev + delta_raw
        command = np.clip(command_raw, -system_or_state.command_limit, system_or_state.command_limit)
        delta_applied = command - system_or_state.command_prev

        clipped = np.abs(command_raw - command) > 1e-9 * np.maximum(1.0, system_or_state.command_limit)
        self.last_delta_jdot_raw = delta_raw.copy()
        self.last_delta_jdot_applied = delta_applied.copy()
        self.last_jdot_command = command.copy()
        self.last_derivative_clipping_fraction = float(np.mean(clipped)) if clipped.size else 0.0
        self.last_gain_fallback_used = gain_fallback_used
        self.last_gain_failure = gain_failure
        self.last_state = system_or_state.x.copy()
        self._command_prev = command.copy()

        n_pfc = int(getattr(model.pfc, "n_coils", 0))
        current_now = _active_vector(model, model.state.pfc_currents, model.state.sol_currents)
        current_next = current_now + float(getattr(model, "t_step")) * command
        return ControlAction(
            pfc_currents_next=current_next[:n_pfc].copy(),
            sol_currents_next=current_next[n_pfc:].copy(),
        )

    def _gain_recompute_reason(self, *, state: ZaitsevLqrState, step: int, boundary_poly: np.ndarray | None) -> str | None:
        """Return a reason string when the cached gain must be rebuilt."""
        if self.last_gain is None:
            return "initial"
        if self.last_gain.shape != (state.n_u, state.x.size):
            return "shape"
        if self._cached_h_size != state.h_size or self._cached_n_u != state.n_u:
            return "state_layout"
        if self._cached_drift is None or self._cached_drift.shape != (state.h_size,):
            return "drift"
        if self.gain_recompute_on_boundary_loss and state.n_boundary and boundary_poly is None:
            return "boundary_loss"
        if self._cached_gain_step is None:
            return "missing_step"
        if step - int(self._cached_gain_step) >= self.gain_recompute_interval_steps:
            return "interval"
        return None

    def _build_state(
        self,
        *,
        model,
        psi: np.ndarray,
        boundary_poly: np.ndarray | None,
        center: tuple[float, float],
        measure_angles: np.ndarray,
        ref_radii: np.ndarray,
        Ip_ref: float | None,
        scenario,
        limiter_shape: np.ndarray | None,
        boundary_mode: str,
        legacy_precision_index2: float,
        drift_override: np.ndarray | None,
    ) -> ZaitsevLqrState:
        """Build the current error state without finite-difference sensitivities."""
        del psi, limiter_shape, boundary_mode, legacy_precision_index2
        state = model.state
        dt = float(getattr(model, "t_step"))
        if dt <= 0.0:
            raise ValueError(f"model.t_step must be > 0, got {dt!r}")

        n_pfc = int(getattr(model.pfc, "n_coils", 0))
        n_sol = int(getattr(model.sol, "n_coils", 0))
        n_u = n_pfc + n_sol
        command_limit = _limit_vector(model, self.derivative_scale_aps)
        if self._command_prev is None or self._command_prev.shape != (n_u,):
            self._command_prev = _active_vector(model, state.pfc_current_derivs, state.sol_current_derivs)
        command_prev = np.asarray(self._command_prev, dtype=float).reshape(n_u).copy()

        current_now = _active_vector(model, state.pfc_currents, state.sol_currents)
        if self._current_ref is None or self._current_ref.shape != (n_u,):
            self._current_ref = current_now.copy()

        ip_ref_now = float(Ip_ref) if Ip_ref is not None else float(getattr(model, "Ip0"))

        if boundary_poly is None:
            h_boundary = _boundary_radius_error(ref_radii, None)
        else:
            radii = legacy_radii_at_angles(boundary_poly, center, measure_angles)
            h_boundary = _boundary_radius_error(ref_radii, radii)

        h_parts = [h_boundary, np.array([ip_ref_now - float(state.Ip)], dtype=float)]
        if self.current_weight > 0.0 and n_u:
            h_current = self._current_ref - current_now
            h_parts.append(h_current)

        h = np.concatenate(h_parts, axis=0)
        h_size = int(h.size)
        drift = np.zeros((h_size,), dtype=float)
        if drift_override is not None and np.asarray(drift_override).shape == (h_size,):
            drift = np.asarray(drift_override, dtype=float).reshape(h_size).copy()
        x = np.concatenate([h, dt * command_prev, drift], axis=0)
        return ZaitsevLqrState(
            x=x,
            command_limit=command_limit,
            command_prev=command_prev,
            h_size=h_size,
            n_u=n_u,
            n_boundary=int(h_boundary.size),
            n_current=(n_u if self.current_weight > 0.0 else 0),
        )

    def _build_system(
        self,
        *,
        model,
        psi: np.ndarray,
        boundary_poly: np.ndarray | None,
        center: tuple[float, float],
        measure_angles: np.ndarray,
        ref_radii: np.ndarray,
        Ip_ref: float | None,
        scenario,
        limiter_shape: np.ndarray | None,
        boundary_mode: str,
        legacy_precision_index2: float,
    ) -> ZaitsevLqrSystem:
        """Собрать матрицы A, B, Q, R и состояние x для текущего шага."""
        state = model.state
        dt = float(getattr(model, "t_step"))
        if dt <= 0.0:
            raise ValueError(f"model.t_step must be > 0, got {dt!r}")

        state_view = self._build_state(
            model=model,
            psi=psi,
            boundary_poly=boundary_poly,
            center=center,
            measure_angles=measure_angles,
            ref_radii=ref_radii,
            Ip_ref=Ip_ref,
            scenario=scenario,
            limiter_shape=limiter_shape,
            boundary_mode=boundary_mode,
            legacy_precision_index2=legacy_precision_index2,
            drift_override=None,
        )
        n_u = state_view.n_u
        command_limit = state_view.command_limit
        command_prev = state_view.command_prev
        current_now = _active_vector(model, state.pfc_currents, state.sol_currents)

        ip_ref_now = float(Ip_ref) if Ip_ref is not None else float(getattr(model, "Ip0"))
        t_next = float(state.t) + dt
        ip_ref_next = float(scenario.Ip_ref(t_next)) if scenario is not None else ip_ref_now

        ref_next = ref_radii.copy()
        if scenario is not None:
            ref_next = np.asarray(scenario.ref_radii(measure_angles, t_next), dtype=float).reshape(ref_radii.shape)

        h = state_view.x[:state_view.h_size].copy()
        h_size = int(h.size)
        base_state = model.snapshot_state()
        h_next_prev, B_h = _finite_difference_h_response(
            model=model,
            base_state=base_state,
            center=center,
            measure_angles=measure_angles,
            ref_radii_next=ref_next,
            ip_ref_next=ip_ref_next,
            current_ref=self._current_ref,
            command_prev=command_prev,
            current_now=current_now,
            h_now=h,
            include_boundary=bool(state_view.n_boundary),
            limiter_shape=limiter_shape,
            boundary_mode=boundary_mode,
            legacy_precision_index2=legacy_precision_index2,
            current_weight=self.current_weight,
            finite_difference_current_delta_a=self.finite_difference_current_delta_a,
        )
        drift = h_next_prev - h - (B_h @ command_prev)

        C_hold = B_h / dt
        A = np.zeros((2 * h_size + n_u, 2 * h_size + n_u), dtype=float)
        A[:h_size, :h_size] = np.eye(h_size, dtype=float)
        A[:h_size, h_size:h_size + n_u] = C_hold
        A[:h_size, h_size + n_u:] = np.eye(h_size, dtype=float)
        A[h_size:h_size + n_u, h_size:h_size + n_u] = np.eye(n_u, dtype=float)
        A[h_size + n_u:, h_size + n_u:] = np.eye(h_size, dtype=float)
        B = np.zeros((A.shape[0], n_u), dtype=float)
        B[:h_size, :] = B_h
        B[h_size:h_size + n_u, :] = dt * np.eye(n_u, dtype=float)

        x = np.concatenate([h, dt * command_prev, drift], axis=0)
        Q = self._state_cost(h_size=h_size, n_u=n_u, n_boundary=state_view.n_boundary, n_current=state_view.n_current, dt=dt)
        R = self._input_cost(n_u=n_u)
        self.last_boundary_sensitivity_max = float(np.max(np.abs(B_h[: state_view.n_boundary, :]))) if state_view.n_boundary else 0.0
        ip_row_index = state_view.n_boundary
        self.last_ip_sensitivity_max = float(np.max(np.abs(B_h[ip_row_index:ip_row_index + 1, :]))) if h_size > ip_row_index else 0.0
        self.last_h_size = int(h_size)
        self.last_n_boundary = int(state_view.n_boundary)
        if B_h.size:
            try:
                self.last_response_condition = float(np.linalg.cond(B_h))
            except np.linalg.LinAlgError:
                self.last_response_condition = float("inf")
        else:
            self.last_response_condition = 0.0
        q_diag = np.diag(Q)
        r_diag = np.diag(R)
        self.last_q_diag_summary = {
            "boundary": float(q_diag[0]) if state_view.n_boundary else 0.0,
            "ip": float(q_diag[ip_row_index]) if q_diag.size > ip_row_index else 0.0,
            "derivative_command": float(q_diag[h_size]) if q_diag.size > h_size and n_u else 0.0,
        }
        self.last_r_diag_summary = {
            "delta_jdot": float(r_diag[0]) if r_diag.size else 0.0,
        }
        return ZaitsevLqrSystem(A=A, B=B, x=x, Q=Q, R=R, command_limit=command_limit, command_prev=command_prev, h_size=h_size, n_u=n_u)

    def _state_cost(self, *, h_size: int, n_u: int, n_boundary: int, n_current: int, dt: float) -> np.ndarray:
        """Построить диагональную матрицу штрафа состояния.

        Current boundary radii are in meters rather than the old GUI index
        units, so state errors are normalized into comparable coordinates.
        The control increment cost is normalized in ``_input_cost`` by the
        physical delta-Jdot scale; otherwise realistic increments in A/s are
        many orders of magnitude more expensive than boundary/Ip errors.
        """
        q_h = np.zeros((h_size,), dtype=float)
        idx = 0
        if n_boundary:
            q_h[idx:idx + n_boundary] = self.boundary_weight / (self.boundary_scale_m ** 2)
            idx += n_boundary
        q_h[idx] = self.ip_weight / (self.ip_scale_a ** 2)
        idx += 1
        if n_current:
            q_h[idx:idx + n_current] = self.current_weight

        command_state_scale = max(float(dt) * self.derivative_scale_aps, 1.0e-30)
        q_cmd = np.full((n_u,), self.derivative_weight / (command_state_scale ** 2), dtype=float)
        q_drift = self.drift_weight_fraction * q_h
        return np.diag(np.concatenate([q_h, q_cmd, q_drift], axis=0))

    def _input_cost(self, *, n_u: int) -> np.ndarray:
        """Построить матрицу штрафа для управляющего приращения delta-Jdot."""
        delta_scale = max(float(self.delta_derivative_scale_aps), 1.0e-30)
        return np.eye(n_u, dtype=float) * max(self.delta_derivative_weight, 1e-30) / (delta_scale ** 2)
