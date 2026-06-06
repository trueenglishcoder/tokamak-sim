from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class ActuatorRealismSettings:
    """Runtime nonidealities between controller commands and plant actuation."""

    pfc_delay_steps: int = 0
    sol_delay_steps: int = 0
    pfc_gain_sigma: float = 0.0
    sol_gain_sigma: float = 0.0
    pfc_bias_sigma: float = 0.0
    sol_bias_sigma: float = 0.0
    pfc_command_noise_sigma: float = 0.0
    sol_command_noise_sigma: float = 0.0

    def validate(self) -> None:
        for name, steps in (
            ("pfc_delay_steps", self.pfc_delay_steps),
            ("sol_delay_steps", self.sol_delay_steps),
        ):
            if int(steps) != steps:
                raise ValueError(f"realism.actuators.{name} must be an int, got {steps!r}")
            if int(steps) < 0:
                raise ValueError(f"realism.actuators.{name} must be >= 0, got {steps!r}")
        for name, value in (
            ("pfc_gain_sigma", self.pfc_gain_sigma),
            ("sol_gain_sigma", self.sol_gain_sigma),
            ("pfc_bias_sigma", self.pfc_bias_sigma),
            ("sol_bias_sigma", self.sol_bias_sigma),
            ("pfc_command_noise_sigma", self.pfc_command_noise_sigma),
            ("sol_command_noise_sigma", self.sol_command_noise_sigma),
        ):
            _validate_nonnegative_finite(value, f"realism.actuators.{name}")


@dataclass(frozen=True, slots=True)
class SensorRealismSettings:
    """Runtime nonidealities between true plant state and measured channels."""

    ip_noise_sigma: float = 0.0
    ip_bias: float = 0.0
    ip_bias_sigma: float = 0.0
    ip_delay_steps: int = 0
    active_current_noise_sigma: float = 0.0
    active_current_bias_sigma: float = 0.0
    active_current_delay_steps: int = 0
    radii_noise_sigma: float = 0.0
    radii_bias_sigma: float = 0.0
    radii_delay_steps: int = 0
    boundary_xy_noise_sigma: float = 0.0
    boundary_delay_steps: int = 0
    psi_noise_sigma: float = 0.0

    def validate(self) -> None:
        for name, steps in (
            ("ip_delay_steps", self.ip_delay_steps),
            ("active_current_delay_steps", self.active_current_delay_steps),
            ("radii_delay_steps", self.radii_delay_steps),
            ("boundary_delay_steps", self.boundary_delay_steps),
        ):
            if int(steps) != steps:
                raise ValueError(f"realism.sensors.{name} must be an int, got {steps!r}")
            if int(steps) < 0:
                raise ValueError(f"realism.sensors.{name} must be >= 0, got {steps!r}")
        if not math.isfinite(float(self.ip_bias)):
            raise ValueError(f"realism.sensors.ip_bias must be finite, got {self.ip_bias!r}")
        for name, value in (
            ("ip_noise_sigma", self.ip_noise_sigma),
            ("ip_bias_sigma", self.ip_bias_sigma),
            ("active_current_noise_sigma", self.active_current_noise_sigma),
            ("active_current_bias_sigma", self.active_current_bias_sigma),
            ("radii_noise_sigma", self.radii_noise_sigma),
            ("radii_bias_sigma", self.radii_bias_sigma),
            ("boundary_xy_noise_sigma", self.boundary_xy_noise_sigma),
            ("psi_noise_sigma", self.psi_noise_sigma),
        ):
            _validate_nonnegative_finite(value, f"realism.sensors.{name}")


@dataclass(frozen=True, slots=True)
class RealismSettings:
    """Neutral runtime realism settings for CLI and programmatic sessions."""

    enabled: bool = False
    seed: int | None = None
    actuators: ActuatorRealismSettings = ActuatorRealismSettings()
    sensors: SensorRealismSettings = SensorRealismSettings()

    def validate(self) -> None:
        if self.seed is not None:
            if int(self.seed) != self.seed:
                raise ValueError(f"realism.seed must be an int if set, got {self.seed!r}")
            if int(self.seed) < 0:
                raise ValueError(f"realism.seed must be >= 0 if set, got {self.seed!r}")
        self.actuators.validate()
        self.sensors.validate()


@dataclass(frozen=True, slots=True)
class ActuatorRealismResult:
    """Command vectors before and after actuator nonidealities."""

    pfc_commanded: np.ndarray
    sol_commanded: np.ndarray
    pfc_applied: np.ndarray
    sol_applied: np.ndarray


@dataclass(frozen=True, slots=True)
class SensorRealismResult:
    """True and measured sensor channels for one runtime instant."""

    true_ip: float
    measured_ip: float
    true_active_currents: np.ndarray
    measured_active_currents: np.ndarray
    true_boundary_poly: np.ndarray | None
    measured_boundary_poly: np.ndarray | None
    true_radii: np.ndarray | None
    measured_radii: np.ndarray | None
    true_psi: np.ndarray | None = None
    measured_psi: np.ndarray | None = None


def _validate_nonnegative_finite(value: float, name: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if float(value) < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
