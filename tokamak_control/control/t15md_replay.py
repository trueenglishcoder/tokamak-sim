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
    Exact replay controller for preprocessed T15MD current tables.

    The controller expects a table whose first column is source time in seconds,
    and whose remaining columns are target coil currents already expressed in
    the same current units used internally by the plant model. The source time
    origin is normalized to the first row so trimmed replay tables start
    immediately in simulator time. The controller is intended for exact
    applied-current replay: a step from ``t`` to ``t + dt`` targets the table
    current at ``t + dt`` relative to the first numeric row.

    Expected table layout
    ---------------------
    The input table must contain one time column followed by all runtime
    actuator-current channels. For the current T15MD new-data setup, the expected
    order is

        time_s, SOL0, SOL1, SOL2, PFC0, PFC1, PFC2, PFC3, PFC4, PFC5

    which means the table lists the three central-solenoid channels first and
    the six PFC channels second. The controller validates that the loaded
    table width matches the currently loaded plant dimensions and maps the first
    ``n_sol`` current columns to the model SOL bank and the remaining ``n_pfc``
    columns to the model PFC bank.

    This controller intentionally has no derivative/current clipping option:
    clipping would turn the run into a controller test instead of an exact
    replay of the real T15 current trajectory.
    """

    def __init__(
        self,
        *,
        replay_path: str | Path,
    ) -> None:
        self.replay_path = Path(replay_path)
        if self.replay_path.suffix.lower() not in {".csv", ".txt"}:
            raise ValueError(
                f"replay_path must point to a .csv or .txt table, got {self.replay_path!s}"
            )
        if not self.replay_path.exists():
            raise FileNotFoundError(f"Replay table not found: {self.replay_path}")

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
        if not np.isfinite(float(times[0])):
            raise ValueError("Replay table first timestamp must be finite")

        currents = np.asarray(table[:, 1:], dtype=float)
        if not np.all(np.isfinite(currents)):
            raise ValueError("Replay table current columns must be finite")

        self._source_time_origin = float(times[0])
        self._times = times - self._source_time_origin
        self._currents = currents

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
                pfc_currents_next=np.zeros((0,), dtype=float),
                sol_currents_next=np.zeros((0,), dtype=float),
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

        sol_delta = target_sol - curr_sol
        pfc_delta = target_pfc - curr_pfc

        if not np.all(np.isfinite(sol_delta)) or not np.all(np.isfinite(pfc_delta)):
            raise ValueError("Replay controller produced non-finite current commands")

        return ControlAction(
            pfc_currents_next=np.asarray(curr_pfc + pfc_delta, dtype=float),
            sol_currents_next=np.asarray(curr_sol + sol_delta, dtype=float),
        )
