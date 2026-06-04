from __future__ import annotations

from pathlib import Path

import numpy as np

from tokamak_control.control.base import ControlAction, Controller
from tokamak_control.control.replay_table import (
    coalesce_near_duplicate_times,
    interp_columns_clamped,
    load_numeric_table,
)


_TIME_EPS = 1.0e-12


class T15MDReplayController(Controller):
    """
    Replay controller for preprocessed T15MD current tables.

    The controller expects a table whose first column is simulation time in
    seconds starting from 0, and whose remaining columns are target coil
    currents already expressed in the same current units used internally by the
    plant model. The controller is intended for exact applied-current replay:
    a step from ``t`` to ``t + dt`` targets the table current at ``t + dt``.

    Expected table layout
    ---------------------
    The input table must contain one time column followed by all runtime
    actuator-current channels. For the current T15MD 741 setup, the expected
    order is

        time_s, CS1, CS2, CS3, PF2, PF3, PF4, PF5

    which means the table lists the three central-solenoid channels first and
    the four PFC channels second. The controller validates that the loaded
    table width matches the currently loaded plant dimensions and maps the first
    ``n_sol`` current columns to the model SOL bank and the remaining ``n_pfc``
    columns to the model PFC bank.
    """

    def __init__(
        self,
        *,
        replay_path: str | Path,
        u_clip: float | None = None,
    ) -> None:
        self.replay_path = Path(replay_path)
        if self.replay_path.suffix.lower() not in {".csv", ".txt"}:
            raise ValueError(
                f"replay_path must point to a .csv or .txt table, got {self.replay_path!s}"
            )
        if not self.replay_path.exists():
            raise FileNotFoundError(f"Replay table not found: {self.replay_path}")

        if u_clip is not None:
            u_clip = float(u_clip)
            if not np.isfinite(u_clip):
                raise ValueError(f"u_clip must be finite if set, got {u_clip!r}")
            if u_clip < 0.0:
                raise ValueError("u_clip must be >= 0 if set")

        table = coalesce_near_duplicate_times(
            load_numeric_table(self.replay_path),
            time_eps=_TIME_EPS,
        )
        if table.ndim != 2 or table.shape[1] < 2:
            raise ValueError(
                f"Replay table must have at least two columns [time, currents...], got shape {table.shape!r}"
            )

        times = np.asarray(table[:, 0], dtype=float)
        if not np.all(np.isfinite(times)):
            raise ValueError("Replay table time column must be finite")
        if np.any(np.diff(times) < 0.0):
            raise ValueError("Replay table time column must be nondecreasing")
        if float(times[0]) < 0.0:
            raise ValueError("Replay table time column must start at t >= 0 seconds")

        currents = np.asarray(table[:, 1:], dtype=float)
        if not np.all(np.isfinite(currents)):
            raise ValueError("Replay table current columns must be finite")

        self._times = times
        self._currents = currents
        self.u_clip = u_clip

    def reset(self) -> None:
        return None

    def _target_currents_at(self, *, sim_t: float) -> np.ndarray:
        return interp_columns_clamped(float(sim_t), self._times, self._currents)

    def compute_control(self, *, model) -> ControlAction:
        if model.state is None:
            raise RuntimeError("T15MDReplayController requires an initialized model.state")

        n_pfc = int(model.pfc.n_coils)
        n_sol = int(model.sol.n_coils)
        n_total = n_pfc + n_sol

        if n_total == 0:
            return ControlAction(
                pfc_derivs=np.zeros((0,), dtype=float),
                sol_derivs=np.zeros((0,), dtype=float),
            )

        if self._currents.shape[1] != n_total:
            raise ValueError(
                f"Replay table actuator count {self._currents.shape[1]} does not match "
                f"loaded plant count {n_total} ({n_sol} SOL + {n_pfc} PFC)"
            )

        dt = float(model.t_step)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"model.t_step must be finite and > 0, got {dt!r}")

        target = self._target_currents_at(sim_t=float(model.state.t) + dt)
        target_sol = target[:n_sol] if n_sol else np.zeros((0,), dtype=float)
        target_pfc = target[n_sol:] if n_pfc else np.zeros((0,), dtype=float)

        curr_sol = (
            np.asarray(model.state.sol_currents, dtype=float).reshape(-1)
            if n_sol
            else np.zeros((0,), dtype=float)
        )
        curr_pfc = (
            np.asarray(model.state.pfc_currents, dtype=float).reshape(-1)
            if n_pfc
            else np.zeros((0,), dtype=float)
        )

        if curr_sol.size != n_sol:
            raise ValueError(
                f"Runtime SOL current vector has size {curr_sol.size}, expected {n_sol}"
            )
        if curr_pfc.size != n_pfc:
            raise ValueError(
                f"Runtime PFC current vector has size {curr_pfc.size}, expected {n_pfc}"
            )

        sol_derivs = (target_sol - curr_sol) / dt
        pfc_derivs = (target_pfc - curr_pfc) / dt

        if not np.all(np.isfinite(sol_derivs)) or not np.all(np.isfinite(pfc_derivs)):
            raise ValueError("Replay controller produced non-finite derivative commands")

        if self.u_clip is not None:
            sol_derivs = np.clip(sol_derivs, -self.u_clip, self.u_clip)
            pfc_derivs = np.clip(pfc_derivs, -self.u_clip, self.u_clip)

        return ControlAction(
            pfc_derivs=np.asarray(pfc_derivs, dtype=float),
            sol_derivs=np.asarray(sol_derivs, dtype=float),
        )
