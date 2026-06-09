from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from tokamak_control.control.base import ControlAction, Controller
from tokamak_control.diagnostics import MagneticDiagnosticLayout, magnetic_diagnostics_numpy


class LearnedMagneticController(Controller):
    """Deterministic magnetic controller backed by an exported actor bundle.

    The observation is restricted to target references, virtual magnetic
    diagnostics, measured Ip, active coil currents, and previous action. It
    never consumes reconstructed measured boundary/radii.
    """

    def __init__(
        self,
        *,
        export_dir: str | Path,
        target_preview_stride: int | None = None,
        action_clip: float = 1.0,
    ) -> None:
        self.export_dir = Path(export_dir).expanduser()
        if not self.export_dir.is_dir():
            raise FileNotFoundError(f"learned controller export directory does not exist: {self.export_dir}")
        self.schema = _read_json(_first_existing(self.export_dir, ("controller_schema.json", "schema.json")))
        self.normalization = _read_json(self.export_dir / "normalization.json")
        self.metadata = _read_json(self.export_dir / "metadata.json")
        with np.load(self.export_dir / "policy_weights.npz", allow_pickle=False) as data:
            self.weights = {name: np.asarray(data[name], dtype=np.float32) for name in data.files}

        self.obs_dim = int(self.schema["obs_dim"])
        self.action_dim = int(self.schema["action_dim"])
        self.n_active_total = int(self.schema["n_active_total"])
        self.n_pfc = int(self.schema.get("n_pfc", -1))
        self.n_sol = int(self.schema.get("n_sol", -1))
        self.n_angles = int(self.schema["n_angles"])
        self.target_preview_steps = int(self.schema.get("target_preview_steps", 0))
        self.target_preview_stride = int(target_preview_stride if target_preview_stride is not None else self.schema.get("target_preview_stride", 1))
        if self.target_preview_steps < 0:
            raise ValueError("target_preview_steps must be >= 0")
        if self.target_preview_stride <= 0:
            raise ValueError("target_preview_stride must be > 0")

        diag = _diagnostic_layout_from_schema(self.schema)
        self.diagnostic_layout = diag
        self.flux_count = diag.flux_count
        self.field_count = diag.field_count
        self.layer_norm_eps = float(self.metadata.get("layer_norm_eps", 1.0e-5))
        self.ip_scale = _positive_scale(self.normalization.get("ip_scale", 1.0), "ip_scale")
        self.radius_scale = _positive_scale(self.normalization.get("radius_scale", 1.0), "radius_scale")
        self.flux_scale = _positive_scale(self.normalization.get("flux_scale", 1.0), "flux_scale")
        self.field_scale = _positive_scale(self.normalization.get("field_scale", 1.0), "field_scale")
        self.bdot_scale = _positive_scale(self.normalization.get("bdot_scale", 1.0), "bdot_scale")
        self.current_scale = _scale_array(self.normalization["current_scale"], self.n_active_total, "current_scale")
        self.derivative_scale = _scale_array(self.normalization["derivative_scale"], self.action_dim, "derivative_scale")
        self.action_clip = _positive_scale(action_clip, "action_clip")
        self._validate_weights()
        self.previous_action_norm = np.zeros((self.action_dim,), dtype=np.float32)
        self.previous_flux: np.ndarray | None = None

    def reset(self) -> None:
        self.previous_action_norm = np.zeros((self.action_dim,), dtype=np.float32)
        self.previous_flux = None

    def compute_control(
        self,
        *,
        model,
        psi: np.ndarray,
        center: tuple[float, float],
        measure_angles: np.ndarray,
        ref_radii: np.ndarray,
        Ip_ref: float,
        scenario,
        max_episode_steps: int,
        measured_ip: float,
        measured_active_currents: np.ndarray,
    ) -> ControlAction:
        del center
        n_pfc = int(model.pfc.n_coils)
        n_sol = int(model.sol.n_coils)
        if self.n_pfc >= 0 and n_pfc != self.n_pfc:
            raise ValueError(f"controller expected {self.n_pfc} PFC actuators, got {n_pfc}")
        if self.n_sol >= 0 and n_sol != self.n_sol:
            raise ValueError(f"controller expected {self.n_sol} SOL actuators, got {n_sol}")
        if n_pfc + n_sol != self.action_dim:
            raise ValueError(f"controller action_dim={self.action_dim} does not match machine active coils={n_pfc + n_sol}")
        obs = self._observation(
            model=model,
            psi=np.asarray(psi, dtype=float),
            measure_angles=measure_angles,
            ref_radii=ref_radii,
            ip_ref=float(Ip_ref),
            scenario=scenario,
            max_episode_steps=int(max_episode_steps),
            measured_ip=float(measured_ip),
            measured_active_currents=measured_active_currents,
        )
        action_norm = self._deterministic_action(obs.reshape(1, -1))[0]
        action_norm = np.clip(action_norm, -self.action_clip, self.action_clip).astype(np.float32, copy=False)
        self.previous_action_norm = action_norm.copy()
        physical = np.asarray(action_norm, dtype=float) * self.derivative_scale
        return ControlAction(pfc_derivs=physical[:n_pfc].copy(), sol_derivs=physical[n_pfc:].copy())

    def _observation(
        self,
        *,
        model,
        psi: np.ndarray,
        measure_angles: np.ndarray,
        ref_radii: np.ndarray,
        ip_ref: float,
        scenario,
        max_episode_steps: int,
        measured_ip: float,
        measured_active_currents: np.ndarray,
    ) -> np.ndarray:
        angles = np.asarray(measure_angles, dtype=float).reshape(-1)
        ref = np.asarray(ref_radii, dtype=float).reshape(-1)
        currents = np.asarray(measured_active_currents, dtype=float).reshape(-1)
        if angles.shape != (self.n_angles,):
            raise ValueError(f"controller expected {self.n_angles} reference angles, got {angles.shape[0]}")
        if ref.shape != (self.n_angles,):
            raise ValueError(f"controller expected {self.n_angles} target radii, got {ref.shape[0]}")
        if currents.shape != (self.n_active_total,):
            raise ValueError(f"controller expected {self.n_active_total} active currents, got {currents.shape[0]}")
        diagnostics = magnetic_diagnostics_numpy(
            psi=psi,
            grid=model.grid,
            layout=self.diagnostic_layout,
            previous_flux=self.previous_flux,
            dt=float(model.t_step),
        )
        self.previous_flux = np.asarray(diagnostics["flux"], dtype=float).copy()
        current_scale = np.where(self.current_scale > 0.0, self.current_scale, 1.0)
        parts = [
            np.array(
                [
                    float(model.state.step) / max(float(max_episode_steps), 1.0),
                    float(measured_ip) / self.ip_scale,
                    float(ip_ref) / self.ip_scale,
                    (float(measured_ip) - float(ip_ref)) / self.ip_scale,
                ],
                dtype=float,
            ),
            currents / current_scale,
            np.asarray(diagnostics["flux"], dtype=float).reshape(-1) / self.flux_scale,
            np.asarray(diagnostics["field"], dtype=float).reshape(-1) / self.field_scale,
            np.asarray(diagnostics["bdot"], dtype=float).reshape(-1) / self.bdot_scale,
            ref / self.radius_scale,
            self._reference_preview(model=model, scenario=scenario, angles=angles, max_episode_steps=max_episode_steps),
            np.clip(self.previous_action_norm.astype(float, copy=False), -1.0, 1.0),
        ]
        obs = np.concatenate(parts).astype(np.float32, copy=False)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"controller observation shape {obs.shape} != ({self.obs_dim},)")
        if not np.all(np.isfinite(obs)):
            raise ValueError("controller observation contains non-finite values")
        return obs

    def _reference_preview(self, *, model, scenario, angles: np.ndarray, max_episode_steps: int) -> np.ndarray:
        if self.target_preview_steps == 0:
            return np.zeros((0,), dtype=float)
        offsets = np.arange(1, self.target_preview_steps + 1, dtype=float) * float(self.target_preview_stride)
        times = float(model.state.t) + offsets * float(model.t_step)
        time_norm = offsets / max(float(max_episode_steps), 1.0)
        ip_preview = np.asarray([float(scenario.Ip_ref(float(t))) for t in times], dtype=float) / self.ip_scale
        radii_preview = np.stack([np.asarray(scenario.ref_radii(angles, float(t)), dtype=float).reshape(-1) for t in times], axis=0) / self.radius_scale
        return np.concatenate([time_norm, ip_preview, radii_preview.reshape(-1)], dtype=float)

    def _deterministic_action(self, observation: np.ndarray) -> np.ndarray:
        x = _linear(observation.astype(np.float32, copy=False), self.weights["input.weight"], self.weights["input.bias"])
        x = _layer_norm(x, self.weights["input_norm.weight"], self.weights["input_norm.bias"], eps=self.layer_norm_eps)
        x = np.tanh(x)
        hidden_index = 1
        while f"hidden{hidden_index}.weight" in self.weights:
            x = _elu(_linear(x, self.weights[f"hidden{hidden_index}.weight"], self.weights[f"hidden{hidden_index}.bias"]))
            hidden_index += 1
        mean = _linear(x, self.weights["mean_head.weight"], self.weights["mean_head.bias"])
        return np.tanh(mean).astype(np.float32, copy=False)

    def _validate_weights(self) -> None:
        required = {"input.weight", "input.bias", "input_norm.weight", "input_norm.bias", "hidden1.weight", "hidden1.bias", "mean_head.weight", "mean_head.bias"}
        missing = sorted(required - set(self.weights))
        if missing:
            raise ValueError(f"learned controller export is missing weights: {', '.join(missing)}")
        if self.weights["input.weight"].shape[1] != self.obs_dim:
            raise ValueError("controller input weight shape does not match obs_dim")
        if self.weights["mean_head.weight"].shape[0] != self.action_dim:
            raise ValueError("controller mean head shape does not match action_dim")


def _diagnostic_layout_from_schema(schema: Mapping[str, object]) -> MagneticDiagnosticLayout:
    diagnostics = schema.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("controller schema must include diagnostics geometry")
    flux_points = np.asarray(diagnostics["flux_points"], dtype=float).reshape(-1, 2)
    field_points = np.asarray(diagnostics["field_points"], dtype=float).reshape(-1, 2)
    field_angles = np.asarray(diagnostics["field_angles"], dtype=float).reshape(-1)
    if field_points.shape[0] != field_angles.shape[0]:
        raise ValueError("diagnostic field_points and field_angles size mismatch")
    return MagneticDiagnosticLayout(flux_points=flux_points, field_points=field_points, field_angles=field_angles)


def _first_existing(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"none of these files exist in {root}: {', '.join(names)}")


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _scale_array(value: object, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (int(size),):
        raise ValueError(f"{name} must have shape ({int(size)},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


def _positive_scale(value: object, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return out


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight.T + bias


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, *, eps: float) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + float(eps)) * weight + bias


def _elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x > 0.0, x, float(alpha) * (np.exp(x) - 1.0))
