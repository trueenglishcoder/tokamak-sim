# tokamak_control/control/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ControlAction:
    """
    Container for old-parity next-current commands.

    Attributes
    ----------
    pfc_currents_next : np.ndarray
        Absolute PFC currents for the next simulation step (A), shape (n_pfc,).
    sol_currents_next : np.ndarray
        Absolute SOL currents for the next simulation step (A), shape (n_sol,).
    """

    pfc_currents_next: np.ndarray
    sol_currents_next: np.ndarray


class Controller(ABC):
    """
    Abstract base class for controllers.

    Notes
    -----
    The simulation runner builds one superset runtime context for each control
    step. The controller registry owns the mapping from controller name to the
    runtime input names that controller actually consumes, and filters the
    superset context before calling the controller.

    Concrete controllers therefore declare only the keyword inputs they need.
    The base interface no longer pretends that every controller naturally
    shares one rigid ``compute_control(...)`` signature.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal controller state."""
        ...

    @abstractmethod
    def compute_control(self, **runtime_inputs: object) -> ControlAction:
        """
        Compute absolute next coil currents for one control step.

        Parameters
        ----------
        **runtime_inputs : object
            Registry-filtered runtime context entries for the concrete
            controller.

        Returns
        -------
        ControlAction
            Absolute next-step currents for PFC and SOL coils in A.
        """
        ...
