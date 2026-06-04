from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class PhysicsSettings:
    """
    Central physical and numerical settings for the simulation.

    ``sigma`` and ``inductance_L`` define the passive plasma-current time
    constant used by the plant:

        tau = sigma * inductance_L

    At each numerical step the plant decays the previous runtime ``Ip`` value by
    ``exp(-dt / tau)`` and then adds the coil-driven increment derived from
    actual actuator current increments over the step. The sign convention of
    that coupling is controlled by ``ip_coupling_sign``. The sign convention of
    the plasma-current contribution to rendered ``psi`` is controlled separately
    by ``plasma_psi_sign``. These are effective lumped parameters of the
    translated Little Scope model, not full profile-resolved tokamak quantities.
    """

    mu0: float = 4e-7 * math.pi
    sigma: float = 6.0e8
    inductance_L: float = 1.0e-6
    ip_coupling_sign: float = -1.0
    plasma_psi_sign: float = 1.0
    t_step: float = 1.0e-3
    actuator_tau: float = 0.0
    Ip0: float = 8.0
    R0: float = 1.2
    Z0: float = 0.0

    pfc_current_limit: float | None = None
    sol_current_limit: float | None = None

    ip_coupling_pfc: tuple[float, ...] | None = None
    ip_coupling_sol: tuple[float, ...] | None = None

    pfc_deriv_limit: float | None = None
    sol_deriv_limit: float | None = None

    pfc_cmd_delay_steps: int = 0
    sol_cmd_delay_steps: int = 0

    pfc_gain_sigma: float = 0.0
    sol_gain_sigma: float = 0.0
    pfc_bias_sigma: float = 0.0
    sol_bias_sigma: float = 0.0
    pfc_cmd_noise_sigma: float = 0.0
    sol_cmd_noise_sigma: float = 0.0

    boundary_xy_noise_sigma: float = 0.0
    boundary_delay_steps: int = 0
    psi_noise_sigma: float = 0.0

    realism_seed: int | None = None

    def validate(self) -> None:
        for name, value in (
            ("mu0", self.mu0),
            ("sigma", self.sigma),
            ("inductance_L", self.inductance_L),
            ("ip_coupling_sign", self.ip_coupling_sign),
            ("plasma_psi_sign", self.plasma_psi_sign),
            ("t_step", self.t_step),
            ("actuator_tau", self.actuator_tau),
            ("Ip0", self.Ip0),
            ("R0", self.R0),
            ("Z0", self.Z0),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")

        if self.mu0 <= 0.0:
            raise ValueError("mu0 must be > 0")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be > 0")
        if self.inductance_L <= 0.0:
            raise ValueError("inductance_L must be > 0")
        if self.t_step <= 0.0:
            raise ValueError("t_step must be > 0")
        if self.actuator_tau < 0.0:
            raise ValueError(f"actuator_tau must be >= 0, got {self.actuator_tau!r}")
        if self.R0 <= 0.0:
            raise ValueError("R0 must be > 0")

        s = float(self.ip_coupling_sign)
        if s not in (-1.0, 1.0):
            raise ValueError(f"ip_coupling_sign must be -1 or +1, got {self.ip_coupling_sign!r}")
        psi_s = float(self.plasma_psi_sign)
        if psi_s not in (-1.0, 1.0):
            raise ValueError(f"plasma_psi_sign must be -1 or +1, got {self.plasma_psi_sign!r}")

        for name, lim in (("pfc_current_limit", self.pfc_current_limit), ("sol_current_limit", self.sol_current_limit)):
            if lim is None:
                continue
            if not math.isfinite(float(lim)):
                raise ValueError(f"{name} must be finite if set, got {lim!r}")
            if float(lim) <= 0.0:
                raise ValueError(f"{name} must be > 0 if set, got {lim!r}")

        for name, coeffs in (("ip_coupling_pfc", self.ip_coupling_pfc), ("ip_coupling_sol", self.ip_coupling_sol)):
            if coeffs is None:
                continue
            if len(coeffs) == 0:
                raise ValueError(f"{name} cannot be an empty sequence")
            for c in coeffs:
                if not math.isfinite(float(c)):
                    raise ValueError(f"All entries in {name} must be finite, got {c!r}")

        for name, lim in (("pfc_deriv_limit", self.pfc_deriv_limit), ("sol_deriv_limit", self.sol_deriv_limit)):
            if lim is None:
                continue
            if not math.isfinite(float(lim)):
                raise ValueError(f"{name} must be finite if set, got {lim!r}")
            if float(lim) < 0.0:
                raise ValueError(f"{name} must be >= 0 if set, got {lim!r}")

        for name, steps in (("pfc_cmd_delay_steps", self.pfc_cmd_delay_steps), ("sol_cmd_delay_steps", self.sol_cmd_delay_steps), ("boundary_delay_steps", self.boundary_delay_steps)):
            if int(steps) != steps:
                raise ValueError(f"{name} must be an int, got {steps!r}")
            if steps < 0:
                raise ValueError(f"{name} must be >= 0, got {steps!r}")

        for name, s in (
            ("pfc_gain_sigma", self.pfc_gain_sigma),
            ("sol_gain_sigma", self.sol_gain_sigma),
            ("pfc_bias_sigma", self.pfc_bias_sigma),
            ("sol_bias_sigma", self.sol_bias_sigma),
            ("pfc_cmd_noise_sigma", self.pfc_cmd_noise_sigma),
            ("sol_cmd_noise_sigma", self.sol_cmd_noise_sigma),
            ("boundary_xy_noise_sigma", self.boundary_xy_noise_sigma),
            ("psi_noise_sigma", self.psi_noise_sigma),
        ):
            if not math.isfinite(float(s)):
                raise ValueError(f"{name} must be finite, got {s!r}")
            if float(s) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {s!r}")

        if self.realism_seed is not None:
            if int(self.realism_seed) != self.realism_seed:
                raise ValueError(f"realism_seed must be an int if set, got {self.realism_seed!r}")
            if self.realism_seed < 0:
                raise ValueError(f"realism_seed must be >= 0 if set, got {self.realism_seed!r}")
