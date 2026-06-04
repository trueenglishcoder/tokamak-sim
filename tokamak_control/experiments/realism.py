# tokamak_control/experiments/realism.py
"""
Realism injector: measurement and actuation nonidealities applied by the simulation loop.

This module is intentionally not used by controllers directly. It provides:
- measurement corruption (psi noise + optional boundary re-extraction + boundary point noise + integer-step delays)
- actuation corruption (integer-step delays + per-coil gain/bias errors + per-step command noise)

All stochasticity is driven by a single NumPy RNG, optionally seeded for reproducibility.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from tokamak_control.config.settings import PhysicsSettings
from tokamak_control.geometry.boundary import BoundaryMode, find_plasma_boundary_with_status


class RealismInjector:
    @staticmethod
    def has_any_effect(settings: PhysicsSettings) -> bool:
        """Return True when any realism knob would change runtime behavior."""
        return any(
            (
                int(settings.pfc_cmd_delay_steps) > 0,
                int(settings.sol_cmd_delay_steps) > 0,
                int(settings.boundary_delay_steps) > 0,
                float(settings.pfc_gain_sigma) > 0.0,
                float(settings.sol_gain_sigma) > 0.0,
                float(settings.pfc_bias_sigma) > 0.0,
                float(settings.sol_bias_sigma) > 0.0,
                float(settings.pfc_cmd_noise_sigma) > 0.0,
                float(settings.sol_cmd_noise_sigma) > 0.0,
                float(settings.boundary_xy_noise_sigma) > 0.0,
                float(settings.psi_noise_sigma) > 0.0,
            )
        )

    """
    Applies configured nonidealities to measurements and actuator commands.

    Notes
    -----
    Per-coil gain/bias errors are sampled once and held fixed for the lifetime of the
    injector. Since coil counts are only known when commands are first seen, these
    vectors are sampled lazily on the first `actuation(...)` call and then reused.
    """

    def __init__(self, settings: PhysicsSettings):
        self._active = self.has_any_effect(settings)
        self._s = settings

        self._rng = np.random.default_rng(
            None if settings.realism_seed is None else int(settings.realism_seed)
        )

        self._pfc_gain: Optional[np.ndarray] = None
        self._sol_gain: Optional[np.ndarray] = None
        self._pfc_bias: Optional[np.ndarray] = None
        self._sol_bias: Optional[np.ndarray] = None

        self._pfc_cmd_buf: Optional[Deque[np.ndarray]] = None
        self._sol_cmd_buf: Optional[Deque[np.ndarray]] = None

        self._psi_buf: Optional[Deque[np.ndarray]] = None
        self._boundary_buf: Optional[Deque[np.ndarray]] = None

    @property
    def active(self) -> bool:
        return bool(self._active)

    def measurements(
        self,
        model,
        psi: np.ndarray,
        boundary_poly: np.ndarray,
        center: Tuple[float, float],
        limiter_shape: np.ndarray | None = None,
        boundary_mode: BoundaryMode = "limited",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Produce "measured" psi and boundary polyline.

        Pipeline:
          1) Optional psi noise.
          2) If psi was noised: recompute boundary from noised psi.
          3) Optional boundary XY noise.
          4) Optional integer-step delay applied to both psi and boundary.

        Parameters
        ----------
        model
            PlasmaModel-like object providing `grid`.
        psi : np.ndarray
            True psi field, shape (nz, nr).
        boundary_poly : np.ndarray
            True boundary polyline, shape (N, 2) in (R, Z).
        center : (float, float)
            (R0, Z0) used for boundary inference.
        limiter_shape
            Контур лимитера для физического определения границы.

        Returns
        -------
        (psi_meas, boundary_meas)
            Possibly corrupted and delayed measurements.
        """
        psi_in = np.asarray(psi, dtype=float)
        boundary_in = np.asarray(boundary_poly, dtype=float)

        psi_meas = psi_in
        boundary_meas = boundary_in

        psi_sigma = float(self._s.psi_noise_sigma)
        psi_noised = False
        if psi_sigma > 0.0:
            psi_meas = psi_in + self._rng.normal(0.0, psi_sigma, size=psi_in.shape)
            psi_noised = True

        if psi_noised:
            boundary_recomputed, _level, _status = find_plasma_boundary_with_status(
                psi_meas,
                model.grid,
                center,
                n_levels=40,
                limiter_shape=limiter_shape,
                boundary_mode=boundary_mode,
            )
            boundary_meas = np.asarray(boundary_recomputed, dtype=float)

        b_sigma = float(self._s.boundary_xy_noise_sigma)
        if b_sigma > 0.0:
            noise = self._rng.normal(0.0, b_sigma, size=boundary_meas.shape)
            boundary_meas = boundary_meas + noise

            if boundary_meas.shape[0] >= 2 and np.allclose(boundary_in[0], boundary_in[-1]):
                boundary_meas = boundary_meas.copy()
                boundary_meas[-1] = boundary_meas[0]

        delay = int(self._s.boundary_delay_steps)
        if delay <= 0:
            return psi_meas, boundary_meas

        self._ensure_measurement_buffers(psi_meas, boundary_meas, delay_steps=delay)

        assert self._psi_buf is not None
        assert self._boundary_buf is not None

        self._psi_buf.append(psi_meas.copy())
        self._boundary_buf.append(boundary_meas.copy())

        psi_out = self._psi_buf[0]
        boundary_out = self._boundary_buf[0]
        return psi_out, boundary_out

    def actuation(
        self, pfc_cmd: np.ndarray, sol_cmd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Produce "effective" derivative commands seen by the plant.

        Pipeline (per bank):
          1) Integer-step delay.
          2) Fixed per-coil gain/bias errors (sampled once).
          3) Per-step additive command noise.

        Parameters
        ----------
        pfc_cmd : np.ndarray
            Commanded PFC derivatives, shape (n_pfc,).
        sol_cmd : np.ndarray
            Commanded SOL derivatives, shape (n_sol,).

        Returns
        -------
        (pfc_cmd_eff, sol_cmd_eff)
            Corrupted effective commands.
        """
        pfc_in = np.asarray(pfc_cmd, dtype=float).reshape(-1)
        sol_in = np.asarray(sol_cmd, dtype=float).reshape(-1)

        self._ensure_calibration(pfc_in.size, sol_in.size)

        pfc_delayed = self._apply_cmd_delay(
            bank="pfc",
            cmd=pfc_in,
            delay_steps=int(self._s.pfc_cmd_delay_steps),
        )
        sol_delayed = self._apply_cmd_delay(
            bank="sol",
            cmd=sol_in,
            delay_steps=int(self._s.sol_cmd_delay_steps),
        )

        assert self._pfc_gain is not None and self._pfc_bias is not None
        assert self._sol_gain is not None and self._sol_bias is not None

        pfc_eff = self._pfc_gain * pfc_delayed + self._pfc_bias
        sol_eff = self._sol_gain * sol_delayed + self._sol_bias

        pfc_noise_sigma = float(self._s.pfc_cmd_noise_sigma)
        if pfc_noise_sigma > 0.0 and pfc_eff.size:
            pfc_eff = pfc_eff + self._rng.normal(0.0, pfc_noise_sigma, size=pfc_eff.shape)

        sol_noise_sigma = float(self._s.sol_cmd_noise_sigma)
        if sol_noise_sigma > 0.0 and sol_eff.size:
            sol_eff = sol_eff + self._rng.normal(0.0, sol_noise_sigma, size=sol_eff.shape)

        return pfc_eff, sol_eff

    def _ensure_calibration(self, n_pfc: int, n_sol: int) -> None:
        if self._pfc_gain is not None:
            if self._pfc_gain.size != n_pfc or (self._sol_gain is not None and self._sol_gain.size != n_sol):
                raise ValueError(
                    f"RealismInjector coil counts changed after initialization: "
                    f"expected (n_pfc={self._pfc_gain.size}, n_sol={self._sol_gain.size if self._sol_gain is not None else 'None'}), "
                    f"got (n_pfc={n_pfc}, n_sol={n_sol})"
                )
            return

        if n_pfc:
            gsig = float(self._s.pfc_gain_sigma)
            bsig = float(self._s.pfc_bias_sigma)
            self._pfc_gain = 1.0 + self._rng.normal(0.0, gsig, size=(n_pfc,)) if gsig > 0.0 else np.ones((n_pfc,))
            self._pfc_bias = self._rng.normal(0.0, bsig, size=(n_pfc,)) if bsig > 0.0 else np.zeros((n_pfc,))
        else:
            self._pfc_gain = np.ones((0,), dtype=float)
            self._pfc_bias = np.zeros((0,), dtype=float)

        if n_sol:
            gsig = float(self._s.sol_gain_sigma)
            bsig = float(self._s.sol_bias_sigma)
            self._sol_gain = 1.0 + self._rng.normal(0.0, gsig, size=(n_sol,)) if gsig > 0.0 else np.ones((n_sol,))
            self._sol_bias = self._rng.normal(0.0, bsig, size=(n_sol,)) if bsig > 0.0 else np.zeros((n_sol,))
        else:
            self._sol_gain = np.ones((0,), dtype=float)
            self._sol_bias = np.zeros((0,), dtype=float)

    def _apply_cmd_delay(self, bank: str, cmd: np.ndarray, delay_steps: int) -> np.ndarray:
        if delay_steps <= 0:
            return cmd

        if bank == "pfc":
            buf = self._pfc_cmd_buf
        elif bank == "sol":
            buf = self._sol_cmd_buf
        else:
            raise ValueError(f"Unknown bank: {bank!r}")

        if buf is None:
            buf = deque([cmd.copy() for _ in range(delay_steps + 1)], maxlen=delay_steps + 1)
            if bank == "pfc":
                self._pfc_cmd_buf = buf
            else:
                self._sol_cmd_buf = buf

        buf.append(cmd.copy())
        return buf[0]

    def _ensure_measurement_buffers(
        self, psi_meas: np.ndarray, boundary_meas: np.ndarray, delay_steps: int
    ) -> None:
        if self._psi_buf is not None:
            return

        self._psi_buf = deque([psi_meas.copy() for _ in range(delay_steps + 1)], maxlen=delay_steps + 1)
        self._boundary_buf = deque([boundary_meas.copy() for _ in range(delay_steps + 1)], maxlen=delay_steps + 1)
