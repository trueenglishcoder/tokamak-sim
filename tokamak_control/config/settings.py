from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class PhysicsSettings:
    """
    Central physical and numerical settings for the simulation.

    ``sigma`` and ``inductance_L`` define the passive plasma-current time
    constant:

        tau = sigma * inductance_L

    The plant advances Ip as a causal state using this passive decay and the
    LittleSCOPE coil-drive term as a dIp/dt contribution. Current/derivative
    limits and actuator time constants are retained as controller/diagnostic
    metadata; the core plant does not enforce them.
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

        for name, coeffs in (
            ("ip_coupling_pfc", self.ip_coupling_pfc),
            ("ip_coupling_sol", self.ip_coupling_sol),
        ):
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
