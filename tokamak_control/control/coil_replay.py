from __future__ import annotations

from pathlib import Path

import numpy as np

from tokamak_control.control.base import ControlAction, Controller
from tokamak_control.control.replay_table import interp_columns_clamped, load_numeric_table


class CoilReplayController(Controller):
    """
    Generic replay controller for preprocessed coil-current tables.

    The controller expects a numeric table whose first column is source time in
    seconds and whose remaining columns are target runtime actuator currents.
    The target-current columns are mapped to the model's PFC and SOL banks by
    ``bank_order``. At runtime the controller interpolates target currents at
    the current simulation time and emits derivative commands that move the
    plant currents toward those targets in one simulation step.

    Time alignment
    --------------
    If ``time_offset`` is ``None``, the first source time in the table is used
    as the simulation origin. Otherwise, ``time_offset`` is subtracted from all
    source times. This lets raw experiment tables be replayed without rewriting
    the time column.

    Bank mapping
    ------------
    ``bank_order`` is a comma-separated list containing ``pfc`` and/or ``sol``.
    For example:

    - ``pfc`` maps all target-current columns to the PFC bank.
    - ``sol,pfc`` maps the first ``n_sol`` columns to SOL and the next
      ``n_pfc`` columns to PFC.
    - ``pfc,sol`` maps the first ``n_pfc`` columns to PFC and the next
      ``n_sol`` columns to SOL.

    Banks omitted from ``bank_order`` are left unchanged by emitting zero
    derivative commands for that bank.
    """

    def __init__(
        self,
        *,
        replay_path: str | Path,
        bank_order: str = "pfc,sol",
        time_offset: float | None = None,
        u_clip: float | None = None,
    ) -> None:
        self.replay_path = Path(replay_path)
        if self.replay_path.suffix.lower() not in {".csv", ".txt"}:
            raise ValueError(
                f"replay_path must point to a .csv or .txt table, got {self.replay_path!s}"
            )
        if not self.replay_path.exists():
            raise FileNotFoundError(f"Replay table not found: {self.replay_path}")

        self.bank_order = self._parse_bank_order(bank_order)

        if time_offset is not None:
            time_offset = float(time_offset)
            if not np.isfinite(time_offset):
                raise ValueError(f"time_offset must be finite if set, got {time_offset!r}")
        self.time_offset = time_offset

        if u_clip is not None:
            u_clip = float(u_clip)
            if not np.isfinite(u_clip):
                raise ValueError(f"u_clip must be finite if set, got {u_clip!r}")
            if u_clip < 0.0:
                raise ValueError("u_clip must be >= 0 if set")
        self.u_clip = u_clip

        table = load_numeric_table(self.replay_path)
        if table.ndim != 2 or table.shape[1] < 2:
            raise ValueError(
                f"Replay table must have at least two columns [time, currents...], got shape {table.shape!r}"
            )

        source_times = np.asarray(table[:, 0], dtype=float)
        if not np.all(np.isfinite(source_times)):
            raise ValueError("Replay table time column must be finite")
        if np.any(np.diff(source_times) < 0.0):
            raise ValueError("Replay table time column must be nondecreasing")

        origin = float(source_times[0]) if time_offset is None else float(time_offset)
        times = source_times - origin
        if np.any(np.diff(times) < 0.0):
            raise ValueError("Shifted replay table time column must be nondecreasing")

        currents = np.asarray(table[:, 1:], dtype=float)
        if not np.all(np.isfinite(currents)):
            raise ValueError("Replay table current columns must be finite")

        self._source_times = source_times
        self._times = times
        self._currents = currents

    def reset(self) -> None:
        return None

    @staticmethod
    def _parse_bank_order(bank_order: str) -> tuple[str, ...]:
        if not isinstance(bank_order, str):
            raise TypeError("bank_order must be a string")
        parts = tuple(part.strip().lower() for part in bank_order.split(",") if part.strip())
        if not parts:
            raise ValueError("bank_order must contain at least one bank name")
        allowed = {"pfc", "sol"}
        unknown = sorted(set(parts) - allowed)
        if unknown:
            raise ValueError(f"bank_order contains unsupported bank name(s): {', '.join(unknown)}")
        if len(set(parts)) != len(parts):
            raise ValueError("bank_order must not repeat a bank name")
        return parts

    def _target_currents_at(self, *, sim_t: float) -> np.ndarray:
        return interp_columns_clamped(float(sim_t), self._times, self._currents)

    def _split_targets(self, target: np.ndarray, *, n_pfc: int, n_sol: int) -> tuple[np.ndarray, np.ndarray]:
        target = np.asarray(target, dtype=float).reshape(-1)
        idx = 0
        target_pfc = np.zeros((n_pfc,), dtype=float)
        target_sol = np.zeros((n_sol,), dtype=float)

        for bank in self.bank_order:
            if bank == "pfc":
                width = int(n_pfc)
                if idx + width > target.size:
                    raise ValueError(
                        f"Replay table does not contain enough columns for bank_order={','.join(self.bank_order)!r}"
                    )
                target_pfc = target[idx : idx + width].astype(float)
                idx += width
            elif bank == "sol":
                width = int(n_sol)
                if idx + width > target.size:
                    raise ValueError(
                        f"Replay table does not contain enough columns for bank_order={','.join(self.bank_order)!r}"
                    )
                target_sol = target[idx : idx + width].astype(float)
                idx += width
            else:
                raise RuntimeError(f"Unsupported bank name after validation: {bank}")

        if idx != target.size:
            raise ValueError(
                f"Replay table has {target.size} current columns but bank_order={','.join(self.bank_order)!r} "
                f"consumes {idx} columns for the loaded plant ({n_pfc} PFC + {n_sol} SOL)"
            )

        return target_pfc, target_sol

    def compute_control(self, *, model) -> ControlAction:
        if model.state is None:
            raise RuntimeError("CoilReplayController requires an initialized model.state")

        n_pfc = int(model.pfc.n_coils)
        n_sol = int(model.sol.n_coils)
        n_total = n_pfc + n_sol

        if n_total == 0:
            return ControlAction(
                pfc_derivs=np.zeros((0,), dtype=float),
                sol_derivs=np.zeros((0,), dtype=float),
            )

        dt = float(model.t_step)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"model.t_step must be finite and > 0, got {dt!r}")

        target = self._target_currents_at(sim_t=float(model.state.t))
        target_pfc, target_sol = self._split_targets(target, n_pfc=n_pfc, n_sol=n_sol)

        curr_pfc = (
            np.asarray(model.state.pfc_currents, dtype=float).reshape(-1)
            if n_pfc
            else np.zeros((0,), dtype=float)
        )
        curr_sol = (
            np.asarray(model.state.sol_currents, dtype=float).reshape(-1)
            if n_sol
            else np.zeros((0,), dtype=float)
        )

        if curr_pfc.size != n_pfc:
            raise ValueError(
                f"Runtime PFC current vector has size {curr_pfc.size}, expected {n_pfc}"
            )
        if curr_sol.size != n_sol:
            raise ValueError(
                f"Runtime SOL current vector has size {curr_sol.size}, expected {n_sol}"
            )

        pfc_derivs = (target_pfc - curr_pfc) / dt
        sol_derivs = (target_sol - curr_sol) / dt

        if not np.all(np.isfinite(pfc_derivs)) or not np.all(np.isfinite(sol_derivs)):
            raise ValueError("Replay controller produced non-finite derivative commands")

        if self.u_clip is not None:
            pfc_derivs = np.clip(pfc_derivs, -self.u_clip, self.u_clip)
            sol_derivs = np.clip(sol_derivs, -self.u_clip, self.u_clip)

        return ControlAction(
            pfc_derivs=np.asarray(pfc_derivs, dtype=float),
            sol_derivs=np.asarray(sol_derivs, dtype=float),
        )
