from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


_ARTIFACT_VERSION = 3


def _nan_vec(n: int) -> np.ndarray:
    """Return a length-n float vector filled with NaNs."""
    return np.full((n,), np.nan, dtype=float)


def _stack_list_of_vectors(vecs: Sequence[np.ndarray]) -> np.ndarray:
    """Stack a list of 1D arrays into a 2D array with shape (T, N)."""
    if not vecs:
        return np.zeros((0, 0), dtype=float)
    n = vecs[0].shape[0]
    for v in vecs:
        if v.shape != (n,):
            raise ValueError("All vectors must have identical shape (N,)")
    return np.stack(vecs, axis=0)


def _stack_list_of_polylines(polylines: Sequence[np.ndarray]) -> np.ndarray:
    """Сложить контуры переменной длины в массив с NaN-заполнением."""
    if not polylines:
        return np.zeros((0, 0, 2), dtype=float)
    max_points = max(int(poly.shape[0]) for poly in polylines)
    out = np.full((len(polylines), max_points, 2), np.nan, dtype=float)
    for idx, poly in enumerate(polylines):
        arr = np.asarray(poly, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Boundary polyline must have shape (N, 2), got {arr.shape}")
        out[idx, : arr.shape[0], :] = arr
    return out


def _csv_cell(value: object) -> str:
    """Convert a Python value into a stable CSV cell string."""
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (int, bool, str)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


@dataclass(slots=True)
class RunWriter:
    """
    Structured run logger that accumulates arrays and writes run artifacts.

    Parameters
    ----------
    output_dir : str | Path
        Directory where run artifacts will be written.
    snapshot_every : int
        If > 0, store ψ snapshots every `snapshot_every` calls to `append`.
        If 0, ψ is only stored when explicitly provided to `append`.
    grid_shape : tuple[int, int] | None
        Optional `(nz, nr)` grid shape used to materialize an empty dense
        `psi_snaps` array when no snapshots were captured.
    metadata : dict[str, object] | None
        JSON-serializable metadata stored inside the run archive as `meta_json`.
    artifact_suffix : str
        Numeric or string suffix appended to artifact filenames, for example
        `run21312.npz`, `events21312.csv`, `run_timeseries21312.csv`.
    """

    output_dir: str | Path
    snapshot_every: int = 0
    grid_shape: tuple[int, int] | None = None
    metadata: dict[str, object] | None = None
    artifact_suffix: str = ""

    _t: list[float] = field(default_factory=list, init=False)
    _step: list[int] = field(default_factory=list, init=False)
    _Ip: list[float] = field(default_factory=list, init=False)

    _pfc_J: list[np.ndarray] = field(default_factory=list, init=False)
    _pfc_dJ_applied: list[np.ndarray] = field(default_factory=list, init=False)

    _sol_J: list[np.ndarray] = field(default_factory=list, init=False)
    _sol_dJ_applied: list[np.ndarray] = field(default_factory=list, init=False)

    _psi_snaps: list[np.ndarray] = field(default_factory=list, init=False)
    _psi_snap_steps: list[int] = field(default_factory=list, init=False)
    _psi_snap_t: list[float] = field(default_factory=list, init=False)
    _snap_counter: int = field(default=0, init=False)
    _psi_latest: np.ndarray | None = field(default=None, init=False)

    _track_pfc_cmd: bool = field(default=False, init=False)
    _track_sol_cmd: bool = field(default=False, init=False)
    _track_pfc_eff: bool = field(default=False, init=False)
    _track_sol_eff: bool = field(default=False, init=False)
    _track_pfc_current_cmd: bool = field(default=False, init=False)
    _track_sol_current_cmd: bool = field(default=False, init=False)
    _track_pfc_current_eff: bool = field(default=False, init=False)
    _track_sol_current_eff: bool = field(default=False, init=False)
    _track_radii: bool = field(default=False, init=False)
    _track_boundary_poly_true: bool = field(default=False, init=False)
    _track_boundary_poly_meas: bool = field(default=False, init=False)
    _track_Ip_ref: bool = field(default=False, init=False)
    _track_Ip_meas: bool = field(default=False, init=False)
    _track_pfc_currents_meas: bool = field(default=False, init=False)
    _track_sol_currents_meas: bool = field(default=False, init=False)
    _track_radii_ref: bool = field(default=False, init=False)

    _pfc_dJ_cmd: list[np.ndarray] = field(default_factory=list, init=False)
    _sol_dJ_cmd: list[np.ndarray] = field(default_factory=list, init=False)
    _pfc_dJ_eff: list[np.ndarray] = field(default_factory=list, init=False)
    _sol_dJ_eff: list[np.ndarray] = field(default_factory=list, init=False)
    _pfc_J_cmd: list[np.ndarray] = field(default_factory=list, init=False)
    _sol_J_cmd: list[np.ndarray] = field(default_factory=list, init=False)
    _pfc_J_eff: list[np.ndarray] = field(default_factory=list, init=False)
    _sol_J_eff: list[np.ndarray] = field(default_factory=list, init=False)

    _radii_true: list[np.ndarray] = field(default_factory=list, init=False)
    _radii_meas: list[np.ndarray] = field(default_factory=list, init=False)
    _boundary_poly_true: list[np.ndarray] = field(default_factory=list, init=False)
    _boundary_poly_meas: list[np.ndarray] = field(default_factory=list, init=False)
    _radii_ref: list[np.ndarray] = field(default_factory=list, init=False)
    _Ip_ref: list[float] = field(default_factory=list, init=False)
    _Ip_meas: list[float] = field(default_factory=list, init=False)
    _pfc_currents_meas: list[np.ndarray] = field(default_factory=list, init=False)
    _sol_currents_meas: list[np.ndarray] = field(default_factory=list, init=False)

    _n_pfc: int | None = field(default=None, init=False)
    _n_sol: int | None = field(default=None, init=False)
    _n_angles: int | None = field(default=None, init=False)

    _events: list[dict[str, object]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Create the output directory and validate constructor inputs."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_suffix = str(self.artifact_suffix or "")
        if self.grid_shape is not None:
            nz, nr = self.grid_shape
            if int(nz) <= 0 or int(nr) <= 0:
                raise ValueError(f"grid_shape must be positive, got {self.grid_shape!r}")
            self.grid_shape = (int(nz), int(nr))
        if self.metadata is not None:
            try:
                json.dumps(self.metadata, ensure_ascii=False)
            except TypeError as e:
                raise ValueError("metadata must be JSON-serializable") from e

    @property
    def npz_path(self) -> Path:
        return self.output_dir / f"run{self.artifact_suffix}.npz"

    @property
    def run_timeseries_csv_path(self) -> Path:
        return self.output_dir / f"run_timeseries{self.artifact_suffix}.csv"

    @property
    def events_csv_path(self) -> Path:
        return self.output_dir / f"events{self.artifact_suffix}.csv"

    def _ensure_len(self, buf: list[np.ndarray], n: int, fill: np.ndarray) -> None:
        """Ensure buf has length n by appending fill copies."""
        while len(buf) < n:
            buf.append(fill.copy())

    def append(
        self,
        t: float,
        Ip: float,
        pfc_currents: np.ndarray,
        pfc_derivs: np.ndarray,
        sol_currents: np.ndarray,
        sol_derivs: np.ndarray,
        psi: np.ndarray | None = None,
        **extra: object,
    ) -> None:
        """Append one time step of state, command, and optional snapshot data."""
        step_index = len(self._t)
        record_step = int(extra.get("step", step_index))

        pfc_currents = np.asarray(pfc_currents, dtype=float).copy()
        sol_currents = np.asarray(sol_currents, dtype=float).copy()
        pfc_derivs_applied = np.asarray(pfc_derivs, dtype=float).copy()
        sol_derivs_applied = np.asarray(sol_derivs, dtype=float).copy()

        if self._n_pfc is None:
            self._n_pfc = int(pfc_currents.shape[0])
        if self._n_sol is None:
            self._n_sol = int(sol_currents.shape[0])

        if pfc_currents.shape != (self._n_pfc,):
            raise ValueError(f"pfc_currents shape {pfc_currents.shape} != ({self._n_pfc},)")
        if sol_currents.shape != (self._n_sol,):
            raise ValueError(f"sol_currents shape {sol_currents.shape} != ({self._n_sol},)")
        if pfc_derivs_applied.shape != (self._n_pfc,):
            raise ValueError(f"pfc_derivs shape {pfc_derivs_applied.shape} != ({self._n_pfc},)")
        if sol_derivs_applied.shape != (self._n_sol,):
            raise ValueError(f"sol_derivs shape {sol_derivs_applied.shape} != ({self._n_sol},)")

        self._t.append(float(t))
        self._step.append(record_step)
        self._Ip.append(float(Ip))
        self._pfc_J.append(pfc_currents)
        self._pfc_dJ_applied.append(pfc_derivs_applied)
        self._sol_J.append(sol_currents)
        self._sol_dJ_applied.append(sol_derivs_applied)

        pfc_cmd = extra.get("pfc_derivs_cmd", None)
        sol_cmd = extra.get("sol_derivs_cmd", None)
        pfc_eff = extra.get("pfc_derivs_eff", None)
        sol_eff = extra.get("sol_derivs_eff", None)
        pfc_current_cmd = extra.get("pfc_currents_cmd", None)
        sol_current_cmd = extra.get("sol_currents_cmd", None)
        pfc_current_eff = extra.get("pfc_currents_eff", None)
        sol_current_eff = extra.get("sol_currents_eff", None)
        radii_true = extra.get("radii_true", None)
        radii_meas = extra.get("radii_meas", None)
        boundary_poly_true = extra.get("boundary_poly_true", None)
        boundary_poly_meas = extra.get("boundary_poly_meas", None)
        psi_latest = extra.get("psi_latest", None)
        radii_ref = extra.get("radii_ref", None)
        Ip_ref = extra.get("Ip_ref", None)
        Ip_meas = extra.get("Ip_meas", None)
        pfc_currents_meas = extra.get("pfc_currents_meas", None)
        sol_currents_meas = extra.get("sol_currents_meas", None)

        if pfc_cmd is not None and not self._track_pfc_cmd:
            self._track_pfc_cmd = True
            self._ensure_len(self._pfc_dJ_cmd, step_index, _nan_vec(self._n_pfc))
        if self._track_pfc_cmd:
            if pfc_cmd is None:
                self._pfc_dJ_cmd.append(_nan_vec(self._n_pfc))
            else:
                v = np.asarray(pfc_cmd, dtype=float).copy()
                if v.shape != (self._n_pfc,):
                    raise ValueError(f"pfc_derivs_cmd shape {v.shape} != ({self._n_pfc},)")
                self._pfc_dJ_cmd.append(v)

        if sol_cmd is not None and not self._track_sol_cmd:
            self._track_sol_cmd = True
            self._ensure_len(self._sol_dJ_cmd, step_index, _nan_vec(self._n_sol))
        if self._track_sol_cmd:
            if sol_cmd is None:
                self._sol_dJ_cmd.append(_nan_vec(self._n_sol))
            else:
                v = np.asarray(sol_cmd, dtype=float).copy()
                if v.shape != (self._n_sol,):
                    raise ValueError(f"sol_derivs_cmd shape {v.shape} != ({self._n_sol},)")
                self._sol_dJ_cmd.append(v)

        if pfc_eff is not None and not self._track_pfc_eff:
            self._track_pfc_eff = True
            self._ensure_len(self._pfc_dJ_eff, step_index, _nan_vec(self._n_pfc))
        if self._track_pfc_eff:
            if pfc_eff is None:
                self._pfc_dJ_eff.append(_nan_vec(self._n_pfc))
            else:
                v = np.asarray(pfc_eff, dtype=float).copy()
                if v.shape != (self._n_pfc,):
                    raise ValueError(f"pfc_derivs_eff shape {v.shape} != ({self._n_pfc},)")
                self._pfc_dJ_eff.append(v)

        if sol_eff is not None and not self._track_sol_eff:
            self._track_sol_eff = True
            self._ensure_len(self._sol_dJ_eff, step_index, _nan_vec(self._n_sol))
        if self._track_sol_eff:
            if sol_eff is None:
                self._sol_dJ_eff.append(_nan_vec(self._n_sol))
            else:
                v = np.asarray(sol_eff, dtype=float).copy()
                if v.shape != (self._n_sol,):
                    raise ValueError(f"sol_derivs_eff shape {v.shape} != ({self._n_sol},)")
                self._sol_dJ_eff.append(v)

        if pfc_current_cmd is not None and not self._track_pfc_current_cmd:
            self._track_pfc_current_cmd = True
            self._ensure_len(self._pfc_J_cmd, step_index, _nan_vec(self._n_pfc))
        if self._track_pfc_current_cmd:
            if pfc_current_cmd is None:
                self._pfc_J_cmd.append(_nan_vec(self._n_pfc))
            else:
                v = np.asarray(pfc_current_cmd, dtype=float).copy()
                if v.shape != (self._n_pfc,):
                    raise ValueError(f"pfc_currents_cmd shape {v.shape} != ({self._n_pfc},)")
                self._pfc_J_cmd.append(v)

        if sol_current_cmd is not None and not self._track_sol_current_cmd:
            self._track_sol_current_cmd = True
            self._ensure_len(self._sol_J_cmd, step_index, _nan_vec(self._n_sol))
        if self._track_sol_current_cmd:
            if sol_current_cmd is None:
                self._sol_J_cmd.append(_nan_vec(self._n_sol))
            else:
                v = np.asarray(sol_current_cmd, dtype=float).copy()
                if v.shape != (self._n_sol,):
                    raise ValueError(f"sol_currents_cmd shape {v.shape} != ({self._n_sol},)")
                self._sol_J_cmd.append(v)

        if pfc_current_eff is not None and not self._track_pfc_current_eff:
            self._track_pfc_current_eff = True
            self._ensure_len(self._pfc_J_eff, step_index, _nan_vec(self._n_pfc))
        if self._track_pfc_current_eff:
            if pfc_current_eff is None:
                self._pfc_J_eff.append(_nan_vec(self._n_pfc))
            else:
                v = np.asarray(pfc_current_eff, dtype=float).copy()
                if v.shape != (self._n_pfc,):
                    raise ValueError(f"pfc_currents_eff shape {v.shape} != ({self._n_pfc},)")
                self._pfc_J_eff.append(v)

        if sol_current_eff is not None and not self._track_sol_current_eff:
            self._track_sol_current_eff = True
            self._ensure_len(self._sol_J_eff, step_index, _nan_vec(self._n_sol))
        if self._track_sol_current_eff:
            if sol_current_eff is None:
                self._sol_J_eff.append(_nan_vec(self._n_sol))
            else:
                v = np.asarray(sol_current_eff, dtype=float).copy()
                if v.shape != (self._n_sol,):
                    raise ValueError(f"sol_currents_eff shape {v.shape} != ({self._n_sol},)")
                self._sol_J_eff.append(v)

        if (radii_true is not None or radii_meas is not None or radii_ref is not None) and not self._track_radii:
            self._track_radii = True
            r0 = radii_true if radii_true is not None else radii_meas
            if r0 is None:
                r0 = radii_ref
            rr = np.asarray(r0, dtype=float).ravel()
            self._n_angles = int(rr.shape[0])
            self._ensure_len(self._radii_true, step_index, _nan_vec(self._n_angles))
            self._ensure_len(self._radii_meas, step_index, _nan_vec(self._n_angles))

        if self._track_radii:
            if radii_true is None:
                self._radii_true.append(_nan_vec(self._n_angles or 0))
            else:
                v = np.asarray(radii_true, dtype=float).ravel().copy()
                if self._n_angles is None:
                    self._n_angles = int(v.shape[0])
                if v.shape != (self._n_angles,):
                    raise ValueError(f"radii_true shape {v.shape} != ({self._n_angles},)")
                self._radii_true.append(v)

            if radii_meas is None:
                self._radii_meas.append(_nan_vec(self._n_angles or 0))
            else:
                v = np.asarray(radii_meas, dtype=float).ravel().copy()
                if self._n_angles is None:
                    self._n_angles = int(v.shape[0])
                if v.shape != (self._n_angles,):
                    raise ValueError(f"radii_meas shape {v.shape} != ({self._n_angles},)")
                self._radii_meas.append(v)

        if boundary_poly_true is not None and not self._track_boundary_poly_true:
            self._track_boundary_poly_true = True
            self._ensure_len(self._boundary_poly_true, step_index, np.zeros((0, 2), dtype=float))
        if self._track_boundary_poly_true:
            if boundary_poly_true is None:
                self._boundary_poly_true.append(np.zeros((0, 2), dtype=float))
            else:
                v = np.asarray(boundary_poly_true, dtype=float).copy()
                if v.ndim != 2 or v.shape[1] != 2:
                    raise ValueError(f"boundary_poly_true shape {v.shape} must be (N, 2)")
                self._boundary_poly_true.append(v)

        if boundary_poly_meas is not None and not self._track_boundary_poly_meas:
            self._track_boundary_poly_meas = True
            self._ensure_len(self._boundary_poly_meas, step_index, np.zeros((0, 2), dtype=float))
        if self._track_boundary_poly_meas:
            if boundary_poly_meas is None:
                self._boundary_poly_meas.append(np.zeros((0, 2), dtype=float))
            else:
                v = np.asarray(boundary_poly_meas, dtype=float).copy()
                if v.ndim != 2 or v.shape[1] != 2:
                    raise ValueError(f"boundary_poly_meas shape {v.shape} must be (N, 2)")
                self._boundary_poly_meas.append(v)

        if Ip_ref is not None and not self._track_Ip_ref:
            self._track_Ip_ref = True
            while len(self._Ip_ref) < step_index:
                self._Ip_ref.append(np.nan)
        if self._track_Ip_ref:
            self._Ip_ref.append(float(Ip_ref) if Ip_ref is not None else np.nan)

        if Ip_meas is not None and not self._track_Ip_meas:
            self._track_Ip_meas = True
            while len(self._Ip_meas) < step_index:
                self._Ip_meas.append(np.nan)
        if self._track_Ip_meas:
            self._Ip_meas.append(float(Ip_meas) if Ip_meas is not None else np.nan)

        if pfc_currents_meas is not None and not self._track_pfc_currents_meas:
            self._track_pfc_currents_meas = True
            self._ensure_len(self._pfc_currents_meas, step_index, _nan_vec(self._n_pfc or 0))
        if self._track_pfc_currents_meas:
            if pfc_currents_meas is None:
                self._pfc_currents_meas.append(_nan_vec(self._n_pfc or 0))
            else:
                v = np.asarray(pfc_currents_meas, dtype=float).copy()
                if v.shape != (self._n_pfc,):
                    raise ValueError(f"pfc_currents_meas shape {v.shape} != ({self._n_pfc},)")
                self._pfc_currents_meas.append(v)

        if sol_currents_meas is not None and not self._track_sol_currents_meas:
            self._track_sol_currents_meas = True
            self._ensure_len(self._sol_currents_meas, step_index, _nan_vec(self._n_sol or 0))
        if self._track_sol_currents_meas:
            if sol_currents_meas is None:
                self._sol_currents_meas.append(_nan_vec(self._n_sol or 0))
            else:
                v = np.asarray(sol_currents_meas, dtype=float).copy()
                if v.shape != (self._n_sol,):
                    raise ValueError(f"sol_currents_meas shape {v.shape} != ({self._n_sol},)")
                self._sol_currents_meas.append(v)

        if radii_ref is not None and not self._track_radii_ref:
            self._track_radii_ref = True
            v0 = np.asarray(radii_ref, dtype=float).ravel()
            if self._n_angles is None:
                self._n_angles = int(v0.shape[0])
            if v0.shape != (self._n_angles,):
                raise ValueError(f"radii_ref shape {v0.shape} != ({self._n_angles},)")
            self._ensure_len(self._radii_ref, step_index, _nan_vec(self._n_angles))
        if self._track_radii_ref:
            if radii_ref is None:
                self._radii_ref.append(_nan_vec(self._n_angles or 0))
            else:
                v = np.asarray(radii_ref, dtype=float).ravel().copy()
                if self._n_angles is None:
                    self._n_angles = int(v.shape[0])
                if v.shape != (self._n_angles,):
                    raise ValueError(f"radii_ref shape {v.shape} != ({self._n_angles},)")
                self._radii_ref.append(v)

        take_snap = False
        if self.snapshot_every > 0:
            self._snap_counter += 1
            if self._snap_counter % self.snapshot_every == 0:
                take_snap = True
        if psi is not None:
            take_snap = True

        if take_snap:
            if psi is None:
                raise ValueError("Snapshot requested but no psi provided to append()")
            psi_arr = np.asarray(psi, dtype=float).copy()
            if psi_arr.ndim != 2:
                raise ValueError(f"psi snapshot must be 2D, got shape {psi_arr.shape}")
            if self.grid_shape is None:
                self.grid_shape = (int(psi_arr.shape[0]), int(psi_arr.shape[1]))
            if psi_arr.shape != self.grid_shape:
                raise ValueError(f"psi snapshot shape {psi_arr.shape} != {self.grid_shape}")
            self._psi_snaps.append(psi_arr)
            self._psi_snap_steps.append(record_step)
            self._psi_snap_t.append(float(t))
            self._psi_latest = psi_arr.copy()
        elif psi_latest is not None:
            psi_latest_arr = np.asarray(psi_latest, dtype=float).copy()
            if psi_latest_arr.ndim != 2:
                raise ValueError(f"psi_latest must be 2D, got shape {psi_latest_arr.shape}")
            if self.grid_shape is None:
                self.grid_shape = (int(psi_latest_arr.shape[0]), int(psi_latest_arr.shape[1]))
            if psi_latest_arr.shape != self.grid_shape:
                raise ValueError(f"psi_latest shape {psi_latest_arr.shape} != {self.grid_shape}")
            self._psi_latest = psi_latest_arr

    def log_event(self, record: Mapping[str, object]) -> None:
        """Accumulate one structured event record for `events.csv`."""
        payload = {}
        for key, value in dict(record).items():
            if isinstance(value, np.ndarray):
                payload[str(key)] = np.asarray(value).tolist()
            elif isinstance(value, np.generic):
                payload[str(key)] = value.item()
            else:
                payload[str(key)] = value
        self._events.append(payload)

    def _write_run_timeseries_csv(self, payload: Mapping[str, object]) -> Path:
        """Write a flat per-step CSV sidecar for the main time-series channels."""
        path = self.run_timeseries_csv_path

        t = np.asarray(payload["t"], dtype=float)
        T = int(t.shape[0])

        columns: list[tuple[str, np.ndarray]] = [
            ("step", np.asarray(payload.get("step", np.arange(T, dtype=int)), dtype=int)),
            ("t", t),
            ("Ip", np.asarray(payload["Ip"], dtype=float)),
        ]
        if "dIp_dt" in payload:
            columns.append(("dIp_dt", np.asarray(payload["dIp_dt"], dtype=float)))

        def add_vector_matrix(prefix: str, arr: np.ndarray) -> None:
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                columns.append((prefix, arr))
                return
            if arr.ndim != 2:
                raise ValueError(f"CSV channel {prefix!r} must be 1D or 2D, got {arr.shape}")
            for j in range(arr.shape[1]):
                columns.append((f"{prefix}_{j}", arr[:, j]))

        add_vector_matrix("pfc_current", np.asarray(payload["pfc_currents"], dtype=float))
        add_vector_matrix("sol_current", np.asarray(payload["sol_currents"], dtype=float))
        add_vector_matrix("pfc_deriv_applied", np.asarray(payload["pfc_derivs"], dtype=float))
        add_vector_matrix("sol_deriv_applied", np.asarray(payload["sol_derivs"], dtype=float))

        if "Ip_ref" in payload:
            columns.append(("Ip_ref", np.asarray(payload["Ip_ref"], dtype=float)))
        if "Ip_meas" in payload:
            columns.append(("Ip_meas", np.asarray(payload["Ip_meas"], dtype=float)))
        if "pfc_currents_meas" in payload:
            add_vector_matrix("pfc_current_meas", np.asarray(payload["pfc_currents_meas"], dtype=float))
        if "sol_currents_meas" in payload:
            add_vector_matrix("sol_current_meas", np.asarray(payload["sol_currents_meas"], dtype=float))
        if "pfc_derivs_cmd" in payload:
            add_vector_matrix("pfc_deriv_cmd", np.asarray(payload["pfc_derivs_cmd"], dtype=float))
        if "sol_derivs_cmd" in payload:
            add_vector_matrix("sol_deriv_cmd", np.asarray(payload["sol_derivs_cmd"], dtype=float))
        if "pfc_derivs_eff" in payload:
            add_vector_matrix("pfc_deriv_eff", np.asarray(payload["pfc_derivs_eff"], dtype=float))
        if "sol_derivs_eff" in payload:
            add_vector_matrix("sol_deriv_eff", np.asarray(payload["sol_derivs_eff"], dtype=float))
        if "pfc_currents_cmd" in payload:
            add_vector_matrix("pfc_current_cmd", np.asarray(payload["pfc_currents_cmd"], dtype=float))
        if "sol_currents_cmd" in payload:
            add_vector_matrix("sol_current_cmd", np.asarray(payload["sol_currents_cmd"], dtype=float))
        if "pfc_currents_eff" in payload:
            add_vector_matrix("pfc_current_eff", np.asarray(payload["pfc_currents_eff"], dtype=float))
        if "sol_currents_eff" in payload:
            add_vector_matrix("sol_current_eff", np.asarray(payload["sol_currents_eff"], dtype=float))
        if "radii_true" in payload:
            add_vector_matrix("radii_true", np.asarray(payload["radii_true"], dtype=float))
        if "radii_meas" in payload:
            add_vector_matrix("radii_meas", np.asarray(payload["radii_meas"], dtype=float))
        if "radii_ref" in payload:
            add_vector_matrix("radii_ref", np.asarray(payload["radii_ref"], dtype=float))

        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name for name, _ in columns])
            for i in range(T):
                row = [_csv_cell(series[i]) for _, series in columns]
                writer.writerow(row)

        return path

    def _write_events_csv(self) -> Path:
        """Write the accumulated structured event records as a flat CSV table."""
        path = self.events_csv_path

        if not self._events:
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["event_index"])
            return path

        fieldnames: list[str] = ["event_index"]
        seen = {"event_index"}
        for event in self._events:
            for key in event.keys():
                key = str(key)
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx, event in enumerate(self._events):
                row = {"event_index": idx}
                for key, value in event.items():
                    row[str(key)] = _csv_cell(value)
                writer.writerow(row)

        return path

    def finalize(self) -> Path:
        """Write run arrays to `run*.npz`, plus CSV sidecars, and return the NPZ path."""
        out = self.npz_path

        t = np.asarray(self._t, dtype=float)
        Ip = np.asarray(self._Ip, dtype=float)
        dIp_dt = np.full_like(Ip, np.nan, dtype=float)
        if Ip.size >= 2:
            dt = np.diff(t)
            valid = np.isfinite(dt) & (np.abs(dt) > 0.0)
            deriv = np.full((Ip.size - 1,), np.nan, dtype=float)
            deriv[valid] = np.diff(Ip)[valid] / dt[valid]
            dIp_dt[1:] = deriv
            dIp_dt[0] = deriv[0]

        pfc_J = _stack_list_of_vectors(self._pfc_J)
        pfc_dJ = _stack_list_of_vectors(self._pfc_dJ_applied)
        sol_J = _stack_list_of_vectors(self._sol_J)
        sol_dJ = _stack_list_of_vectors(self._sol_dJ_applied)

        if self.grid_shape is None:
            psi_snaps = np.zeros((0, 0, 0), dtype=float)
        elif self._psi_snaps:
            psi_snaps = np.stack(self._psi_snaps, axis=0).astype(float, copy=False)
        else:
            nz, nr = self.grid_shape
            psi_snaps = np.zeros((0, nz, nr), dtype=float)

        payload: dict[str, object] = {
            "step": np.asarray(self._step, dtype=int),
            "t": t,
            "Ip": Ip,
            "dIp_dt": dIp_dt,
            "pfc_currents": pfc_J,
            "pfc_derivs": pfc_dJ,
            "sol_currents": sol_J,
            "sol_derivs": sol_dJ,
            "psi_snaps": psi_snaps,
            "psi_snap_steps": np.asarray(self._psi_snap_steps, dtype=int),
            "psi_snap_t": np.asarray(self._psi_snap_t, dtype=float),
            "psi_final": np.asarray(self._psi_latest, dtype=float) if self._psi_latest is not None else np.zeros((0, 0), dtype=float),
            "version": np.array([_ARTIFACT_VERSION], dtype=int),
            "meta_json": np.array(json.dumps(self.metadata or {}, ensure_ascii=False), dtype=np.str_),
        }

        if self._track_pfc_cmd:
            payload["pfc_derivs_cmd"] = _stack_list_of_vectors(self._pfc_dJ_cmd)
        if self._track_sol_cmd:
            payload["sol_derivs_cmd"] = _stack_list_of_vectors(self._sol_dJ_cmd)
        if self._track_pfc_eff:
            payload["pfc_derivs_eff"] = _stack_list_of_vectors(self._pfc_dJ_eff)
        if self._track_sol_eff:
            payload["sol_derivs_eff"] = _stack_list_of_vectors(self._sol_dJ_eff)
        if self._track_pfc_current_cmd:
            payload["pfc_currents_cmd"] = _stack_list_of_vectors(self._pfc_J_cmd)
        if self._track_sol_current_cmd:
            payload["sol_currents_cmd"] = _stack_list_of_vectors(self._sol_J_cmd)
        if self._track_pfc_current_eff:
            payload["pfc_currents_eff"] = _stack_list_of_vectors(self._pfc_J_eff)
        if self._track_sol_current_eff:
            payload["sol_currents_eff"] = _stack_list_of_vectors(self._sol_J_eff)
        if self._track_radii:
            payload["radii_true"] = _stack_list_of_vectors(self._radii_true)
            payload["radii_meas"] = _stack_list_of_vectors(self._radii_meas)
        if self._track_boundary_poly_true:
            payload["boundary_poly_true"] = _stack_list_of_polylines(self._boundary_poly_true)
        if self._track_boundary_poly_meas:
            payload["boundary_poly_meas"] = _stack_list_of_polylines(self._boundary_poly_meas)
        if self._track_Ip_ref:
            payload["Ip_ref"] = np.asarray(self._Ip_ref, dtype=float)
        if self._track_Ip_meas:
            payload["Ip_meas"] = np.asarray(self._Ip_meas, dtype=float)
        if self._track_pfc_currents_meas:
            payload["pfc_currents_meas"] = _stack_list_of_vectors(self._pfc_currents_meas)
        if self._track_sol_currents_meas:
            payload["sol_currents_meas"] = _stack_list_of_vectors(self._sol_currents_meas)
        if self._track_radii_ref:
            payload["radii_ref"] = _stack_list_of_vectors(self._radii_ref)

        np.savez(out, **payload)
        self._write_run_timeseries_csv(payload)
        self._write_events_csv()
        return out


def load_run(npz_path: str | Path) -> dict[str, object]:
    """
    Load and normalize a run written by RunWriter.finalize().

    Returns
    -------
    dict[str, object]
        Canonical run payload, including parsed `meta`.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as d:
        files = set(d.files)
        version = int(np.asarray(d["version"]).reshape(-1)[0]) if "version" in files else 0
        if version != _ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported run artifact version {version}; expected {_ARTIFACT_VERSION}"
            )

        required = {
            "t",
            "Ip",
            "pfc_currents",
            "pfc_derivs",
            "sol_currents",
            "sol_derivs",
            "psi_snaps",
            "psi_snap_steps",
            "psi_snap_t",
            "meta_json",
        }
        missing = sorted(required - files)
        if missing:
            raise ValueError(f"Run artifact missing required keys: {', '.join(missing)}")

        meta_raw = np.asarray(d["meta_json"])
        meta_flat = meta_raw.reshape(-1)
        meta_str = str(meta_raw.item()) if meta_raw.shape == () else (str(meta_flat[0]) if meta_flat.size else "")
        meta = json.loads(meta_str) if meta_str else {}

        payload: dict[str, object] = {
            "step": np.asarray(d["step"], dtype=int) if "step" in files else np.arange(np.asarray(d["t"], dtype=float).shape[0], dtype=int),
            "t": np.asarray(d["t"], dtype=float),
            "Ip": np.asarray(d["Ip"], dtype=float),
            "dIp_dt": np.asarray(d["dIp_dt"], dtype=float) if "dIp_dt" in files else np.full_like(np.asarray(d["Ip"], dtype=float), np.nan),
            "pfc_currents": np.asarray(d["pfc_currents"], dtype=float),
            "pfc_derivs": np.asarray(d["pfc_derivs"], dtype=float),
            "sol_currents": np.asarray(d["sol_currents"], dtype=float),
            "sol_derivs": np.asarray(d["sol_derivs"], dtype=float),
            "psi_snaps": np.asarray(d["psi_snaps"], dtype=float),
            "psi_snap_steps": np.asarray(d["psi_snap_steps"], dtype=int),
            "psi_snap_t": np.asarray(d["psi_snap_t"], dtype=float),
            "psi_final": np.asarray(d["psi_final"], dtype=float) if "psi_final" in files else np.zeros((0, 0), dtype=float),
            "meta": meta,
            "version": version,
        }

        for key in (
            "pfc_derivs_cmd",
            "sol_derivs_cmd",
            "pfc_derivs_eff",
            "sol_derivs_eff",
            "pfc_currents_cmd",
            "sol_currents_cmd",
            "pfc_currents_eff",
            "sol_currents_eff",
            "radii_true",
            "radii_meas",
            "boundary_poly_true",
            "boundary_poly_meas",
            "Ip_ref",
            "Ip_meas",
            "pfc_currents_meas",
            "sol_currents_meas",
            "radii_ref",
        ):
            if key in files:
                payload[key] = np.asarray(d[key], dtype=float)

    return payload
