from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

from tokamak_control.geometry.boundary import BoundaryMode, BoundaryNotFoundError, find_plasma_boundary_with_status
from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections
from tokamak_control.realism.types import ActuatorRealismResult, RealismSettings, SensorRealismResult


class RealismRuntime:
    """Stateful runtime nonideality layer shared by CLI runs and bridge sessions."""

    def __init__(self, settings: RealismSettings, *, seed: int | None = None) -> None:
        settings.validate()
        self._settings = settings
        active_seed = settings.seed if seed is None else seed
        self._rng = np.random.default_rng(None if active_seed is None else int(active_seed))
        self._pfc_gain: np.ndarray | None = None
        self._sol_gain: np.ndarray | None = None
        self._pfc_bias: np.ndarray | None = None
        self._sol_bias: np.ndarray | None = None
        self._active_current_bias: np.ndarray | None = None
        self._radii_bias: np.ndarray | None = None
        self._ip_bias_sample: float | None = None
        self._pfc_cmd_buf: Deque[np.ndarray] | None = None
        self._sol_cmd_buf: Deque[np.ndarray] | None = None
        self._ip_buf: Deque[float] | None = None
        self._active_current_buf: Deque[np.ndarray] | None = None
        self._radii_buf: Deque[np.ndarray] | None = None
        self._boundary_buf: Deque[np.ndarray] | None = None

    @staticmethod
    def has_any_effect(settings: RealismSettings) -> bool:
        """Return True if any configured knob can change runtime behavior."""
        a = settings.actuators
        s = settings.sensors
        return any(
            (
                bool(settings.enabled),
                int(a.pfc_delay_steps) > 0,
                int(a.sol_delay_steps) > 0,
                float(a.pfc_gain_sigma) > 0.0,
                float(a.sol_gain_sigma) > 0.0,
                float(a.pfc_bias_sigma) > 0.0,
                float(a.sol_bias_sigma) > 0.0,
                float(a.pfc_command_noise_sigma) > 0.0,
                float(a.sol_command_noise_sigma) > 0.0,
                float(s.ip_noise_sigma) > 0.0,
                float(s.ip_bias) != 0.0,
                float(s.ip_bias_sigma) > 0.0,
                int(s.ip_delay_steps) > 0,
                float(s.active_current_noise_sigma) > 0.0,
                float(s.active_current_bias_sigma) > 0.0,
                int(s.active_current_delay_steps) > 0,
                float(s.radii_noise_sigma) > 0.0,
                float(s.radii_bias_sigma) > 0.0,
                int(s.radii_delay_steps) > 0,
                float(s.boundary_xy_noise_sigma) > 0.0,
                int(s.boundary_delay_steps) > 0,
                float(s.psi_noise_sigma) > 0.0,
            )
        )

    @property
    def active(self) -> bool:
        return self.has_any_effect(self._settings)

    def apply_actuation(self, pfc_commanded: np.ndarray, sol_commanded: np.ndarray) -> ActuatorRealismResult:
        """Apply actuator delays, fixed calibration errors, and command noise."""
        pfc_in = np.asarray(pfc_commanded, dtype=float).reshape(-1)
        sol_in = np.asarray(sol_commanded, dtype=float).reshape(-1)
        self._ensure_actuator_calibration(pfc_in.size, sol_in.size)

        a = self._settings.actuators
        pfc_delayed = self._delay_vector("pfc", pfc_in, int(a.pfc_delay_steps))
        sol_delayed = self._delay_vector("sol", sol_in, int(a.sol_delay_steps))

        assert self._pfc_gain is not None and self._pfc_bias is not None
        assert self._sol_gain is not None and self._sol_bias is not None
        pfc_out = self._pfc_gain * pfc_delayed + self._pfc_bias
        sol_out = self._sol_gain * sol_delayed + self._sol_bias

        if float(a.pfc_command_noise_sigma) > 0.0 and pfc_out.size:
            pfc_out = pfc_out + self._rng.normal(0.0, float(a.pfc_command_noise_sigma), size=pfc_out.shape)
        if float(a.sol_command_noise_sigma) > 0.0 and sol_out.size:
            sol_out = sol_out + self._rng.normal(0.0, float(a.sol_command_noise_sigma), size=sol_out.shape)

        return ActuatorRealismResult(
            pfc_commanded=pfc_in.copy(),
            sol_commanded=sol_in.copy(),
            pfc_applied=np.asarray(pfc_out, dtype=float).copy(),
            sol_applied=np.asarray(sol_out, dtype=float).copy(),
        )

    def measure(
        self,
        *,
        true_ip: float,
        true_active_currents: np.ndarray,
        true_boundary_poly: np.ndarray | None,
        true_radii: np.ndarray | None,
        true_psi: np.ndarray | None = None,
        model=None,
        center: tuple[float, float] | None = None,
        angles_rad: np.ndarray | None = None,
        limiter_shape: np.ndarray | None = None,
        boundary_mode: BoundaryMode = "limited",
    ) -> SensorRealismResult:
        """Create measured channels from true plant state without inventing missing data."""
        s = self._settings.sensors
        currents_true = np.asarray(true_active_currents, dtype=float).reshape(-1)
        ip_meas = self._measure_ip(float(true_ip))
        currents_meas = self._measure_active_currents(currents_true)

        psi_meas = None if true_psi is None else np.asarray(true_psi, dtype=float).copy()
        boundary_meas = None if true_boundary_poly is None else np.asarray(true_boundary_poly, dtype=float).copy()

        if (
            float(s.psi_noise_sigma) > 0.0
            and true_psi is not None
            and model is not None
            and center is not None
            and true_boundary_poly is not None
        ):
            psi_meas = np.asarray(true_psi, dtype=float) + self._rng.normal(0.0, float(s.psi_noise_sigma), size=np.asarray(true_psi).shape)
            try:
                boundary_recomputed, _level, _status = find_plasma_boundary_with_status(
                    psi_meas,
                    model.grid,
                    center,
                    n_levels=40,
                    limiter_shape=limiter_shape,
                    boundary_mode=boundary_mode,
                )
                boundary_meas = np.asarray(boundary_recomputed, dtype=float)
            except BoundaryNotFoundError:
                boundary_meas = None

        if boundary_meas is not None and float(s.boundary_xy_noise_sigma) > 0.0:
            boundary_meas = boundary_meas + self._rng.normal(0.0, float(s.boundary_xy_noise_sigma), size=boundary_meas.shape)
            if boundary_meas.shape[0] >= 2 and true_boundary_poly is not None and np.allclose(np.asarray(true_boundary_poly, dtype=float)[0], np.asarray(true_boundary_poly, dtype=float)[-1]):
                boundary_meas = boundary_meas.copy()
                boundary_meas[-1] = boundary_meas[0]

        if boundary_meas is not None:
            boundary_meas = self._delay_optional_polyline(boundary_meas, int(s.boundary_delay_steps))

        radii_true = None if true_radii is None else np.asarray(true_radii, dtype=float).reshape(-1)
        radii_meas = None
        if radii_true is not None:
            if boundary_meas is not None and center is not None and angles_rad is not None:
                radii_meas = radii_from_polyline_ray_intersections(boundary_meas, center, np.asarray(angles_rad, dtype=float))
            else:
                radii_meas = radii_true.copy()
            radii_meas = self._measure_radii(radii_meas)

        return SensorRealismResult(
            true_ip=float(true_ip),
            measured_ip=float(ip_meas),
            true_active_currents=currents_true.copy(),
            measured_active_currents=currents_meas,
            true_boundary_poly=None if true_boundary_poly is None else np.asarray(true_boundary_poly, dtype=float).copy(),
            measured_boundary_poly=None if boundary_meas is None else np.asarray(boundary_meas, dtype=float).copy(),
            true_radii=None if radii_true is None else radii_true.copy(),
            measured_radii=None if radii_meas is None else np.asarray(radii_meas, dtype=float).copy(),
            true_psi=None if true_psi is None else np.asarray(true_psi, dtype=float).copy(),
            measured_psi=None if psi_meas is None else np.asarray(psi_meas, dtype=float).copy(),
        )

    def _measure_ip(self, true_ip: float) -> float:
        s = self._settings.sensors
        if self._ip_bias_sample is None:
            self._ip_bias_sample = float(s.ip_bias)
            if float(s.ip_bias_sigma) > 0.0:
                self._ip_bias_sample += float(self._rng.normal(0.0, float(s.ip_bias_sigma)))
        out = float(true_ip) + float(self._ip_bias_sample)
        if float(s.ip_noise_sigma) > 0.0:
            out += float(self._rng.normal(0.0, float(s.ip_noise_sigma)))
        return self._delay_scalar(out, int(s.ip_delay_steps))

    def _measure_active_currents(self, true_currents: np.ndarray) -> np.ndarray:
        s = self._settings.sensors
        currents = np.asarray(true_currents, dtype=float).reshape(-1)
        if self._active_current_bias is None:
            self._active_current_bias = (
                self._rng.normal(0.0, float(s.active_current_bias_sigma), size=currents.shape)
                if float(s.active_current_bias_sigma) > 0.0
                else np.zeros_like(currents)
            )
        if self._active_current_bias.shape != currents.shape:
            raise ValueError("active current measurement dimension changed after realism initialization")
        out = currents + self._active_current_bias
        if float(s.active_current_noise_sigma) > 0.0:
            out = out + self._rng.normal(0.0, float(s.active_current_noise_sigma), size=out.shape)
        return self._delay_active_currents(out, int(s.active_current_delay_steps))

    def _measure_radii(self, radii: np.ndarray) -> np.ndarray:
        s = self._settings.sensors
        values = np.asarray(radii, dtype=float).reshape(-1)
        if self._radii_bias is None:
            self._radii_bias = (
                self._rng.normal(0.0, float(s.radii_bias_sigma), size=values.shape)
                if float(s.radii_bias_sigma) > 0.0
                else np.zeros_like(values)
            )
        if self._radii_bias.shape != values.shape:
            raise ValueError("radii measurement dimension changed after realism initialization")
        out = values + self._radii_bias
        if float(s.radii_noise_sigma) > 0.0:
            out = out + self._rng.normal(0.0, float(s.radii_noise_sigma), size=out.shape)
        return self._delay_radii(out, int(s.radii_delay_steps))

    def _ensure_actuator_calibration(self, n_pfc: int, n_sol: int) -> None:
        if self._pfc_gain is not None:
            if self._pfc_gain.shape != (n_pfc,) or self._sol_gain is None or self._sol_gain.shape != (n_sol,):
                raise ValueError("actuator dimensions changed after realism initialization")
            return
        a = self._settings.actuators
        self._pfc_gain = (
            1.0 + self._rng.normal(0.0, float(a.pfc_gain_sigma), size=(n_pfc,))
            if float(a.pfc_gain_sigma) > 0.0
            else np.ones((n_pfc,), dtype=float)
        )
        self._sol_gain = (
            1.0 + self._rng.normal(0.0, float(a.sol_gain_sigma), size=(n_sol,))
            if float(a.sol_gain_sigma) > 0.0
            else np.ones((n_sol,), dtype=float)
        )
        self._pfc_bias = (
            self._rng.normal(0.0, float(a.pfc_bias_sigma), size=(n_pfc,))
            if float(a.pfc_bias_sigma) > 0.0
            else np.zeros((n_pfc,), dtype=float)
        )
        self._sol_bias = (
            self._rng.normal(0.0, float(a.sol_bias_sigma), size=(n_sol,))
            if float(a.sol_bias_sigma) > 0.0
            else np.zeros((n_sol,), dtype=float)
        )

    def _delay_vector(self, bank: str, value: np.ndarray, delay_steps: int) -> np.ndarray:
        if delay_steps <= 0:
            return np.asarray(value, dtype=float).copy()
        buf = self._pfc_cmd_buf if bank == "pfc" else self._sol_cmd_buf
        if buf is None:
            buf = deque([np.asarray(value, dtype=float).copy() for _ in range(delay_steps + 1)], maxlen=delay_steps + 1)
            if bank == "pfc":
                self._pfc_cmd_buf = buf
            else:
                self._sol_cmd_buf = buf
        buf.append(np.asarray(value, dtype=float).copy())
        return np.asarray(buf[0], dtype=float).copy()

    def _delay_scalar(self, value: float, delay_steps: int) -> float:
        if delay_steps <= 0:
            return float(value)
        if self._ip_buf is None:
            self._ip_buf = deque([float(value) for _ in range(delay_steps + 1)], maxlen=delay_steps + 1)
        self._ip_buf.append(float(value))
        return float(self._ip_buf[0])

    def _delay_active_currents(self, value: np.ndarray, delay_steps: int) -> np.ndarray:
        if delay_steps <= 0:
            return np.asarray(value, dtype=float).copy()
        if self._active_current_buf is None:
            self._active_current_buf = deque([np.asarray(value, dtype=float).copy() for _ in range(delay_steps + 1)], maxlen=delay_steps + 1)
        self._active_current_buf.append(np.asarray(value, dtype=float).copy())
        return np.asarray(self._active_current_buf[0], dtype=float).copy()

    def _delay_radii(self, value: np.ndarray, delay_steps: int) -> np.ndarray:
        if delay_steps <= 0:
            return np.asarray(value, dtype=float).copy()
        if self._radii_buf is None:
            self._radii_buf = deque([np.asarray(value, dtype=float).copy() for _ in range(delay_steps + 1)], maxlen=delay_steps + 1)
        self._radii_buf.append(np.asarray(value, dtype=float).copy())
        return np.asarray(self._radii_buf[0], dtype=float).copy()

    def _delay_optional_polyline(self, value: np.ndarray, delay_steps: int) -> np.ndarray:
        if delay_steps <= 0:
            return np.asarray(value, dtype=float).copy()
        if self._boundary_buf is None:
            self._boundary_buf = deque([np.asarray(value, dtype=float).copy() for _ in range(delay_steps + 1)], maxlen=delay_steps + 1)
        self._boundary_buf.append(np.asarray(value, dtype=float).copy())
        return np.asarray(self._boundary_buf[0], dtype=float).copy()
