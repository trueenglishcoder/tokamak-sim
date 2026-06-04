# tokamak_control/core/plasma_state.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class PlasmaState:
    """
    Runtime container for the instantaneous plasma state.

    Notes
    -----
    ``PlasmaState`` is the sole owner of evolving coil currents during
    simulation. ``CoilGroup`` objects define static coil-bank specification plus
    initial-current values only.

    The coil derivative fields represent the *applied* derivatives used to
    integrate coil currents over the last step. If actuator lag is enabled in
    ``PlasmaModel``, these may differ from the commanded derivatives provided to
    ``PlasmaModel.step()``.

    Attributes
    ----------
    t : float
        Simulation time (s).
    step : int
        Discrete step index (>= 0).
    Ip : float
        Plasma current (A).
    Ip0 : float
        Initial plasma current at t=0 (A); used by the simple relaxation model.
    psi : np.ndarray
        Poloidal flux array with shape (nz, nr).
    pfc_currents : np.ndarray
        Runtime PFC coil currents (A), shape (n_pfc,).
    pfc_current_derivs : np.ndarray
        Applied PFC coil current time-derivatives (A/s), shape (n_pfc,).
    sol_currents : np.ndarray
        Runtime SOL coil currents (A), shape (n_sol,).
    sol_current_derivs : np.ndarray
        Applied SOL coil current time-derivatives (A/s), shape (n_sol,).
    """

    t: float
    step: int
    Ip: float
    Ip0: float
    psi: np.ndarray
    pfc_currents: np.ndarray
    pfc_current_derivs: np.ndarray
    sol_currents: np.ndarray
    sol_current_derivs: np.ndarray

    def copied(self) -> "PlasmaState":
        """Return a deep copy of the state arrays and scalar fields."""
        return PlasmaState(
            t=float(self.t),
            step=int(self.step),
            Ip=float(self.Ip),
            Ip0=float(self.Ip0),
            psi=np.asarray(self.psi, dtype=float).copy(),
            pfc_currents=np.asarray(self.pfc_currents, dtype=float).copy(),
            pfc_current_derivs=np.asarray(self.pfc_current_derivs, dtype=float).copy(),
            sol_currents=np.asarray(self.sol_currents, dtype=float).copy(),
            sol_current_derivs=np.asarray(self.sol_current_derivs, dtype=float).copy(),
        )
