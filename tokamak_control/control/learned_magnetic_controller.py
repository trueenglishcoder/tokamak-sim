from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tokamak_control.control.base import ControlAction, Controller
from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections


JOINT_STATE_V1_FEATURE_ORDER = [
    "step_norm",
    "ip",
    "ip_ref",
    "ip_error",
    "active_currents",
    "active_current_derivs",
    "psi_flat",
    "measured_boundary_radii",
    "ref_radii",
    "boundary_radii_error",
    "boundary_found",
    "target_preview",
]
CONTROLLER_STATE_V2_FEATURE_ORDER = [
    "step_norm",
    "ip",
    "ip_ref",
    "ip_error",
    "active_currents",
    "active_current_derivs",
    "measured_boundary_radii",
    "ref_radii",
    "boundary_radii_error",
    "boundary_found",
    "previous_action",
    "target_preview",
]
COMPACT_JOINT_STATE_V2_FEATURE_ORDER = CONTROLLER_STATE_V2_FEATURE_ORDER
OBSERVATION_KIND = "controller_state_v2"
EXPECTED_FEATURE_ORDER = CONTROLLER_STATE_V2_FEATURE_ORDER
SUPPORTED_FEATURE_ORDERS = {
    "joint_state_v1": JOINT_STATE_V1_FEATURE_ORDER,
    "compact_joint_state_v2": CONTROLLER_STATE_V2_FEATURE_ORDER,
    "controller_state_v2": CONTROLLER_STATE_V2_FEATURE_ORDER,
}


class LearnedMagneticController(Controller):
    """Deterministic learned magnetic controller backed by an exported actor bundle.

    The actor observation is the fixed-size tensor form of the state supplied to
    joint boundary controllers: full psi, reconstructed boundary radii, target
    radii, Ip target, active coil currents, and applied current derivatives.
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
        self._validate_observation_schema()
        self.action_contract = str(self.schema.get("action_contract", "requested_with_current_aware_saturation_v2"))
        if self.action_contract not in {"requested_with_current_aware_saturation_v2", "tcv_derivative_v1", "delta_jdot_derivative_command_v1", "delta_jdot_derivative_command_v2"}:
            raise ValueError(f"unsupported learned-controller action_contract: {self.action_contract!r}")
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
        grid_shape = np.asarray(self.schema["grid_shape"], dtype=int).reshape(-1)
        if grid_shape.shape != (2,) or np.any(grid_shape <= 0):
            raise ValueError("controller schema grid_shape must contain positive [nz, nr]")
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.target_preview_steps = int(self.schema.get("target_preview_steps", 0))
        self.target_preview_stride = int(target_preview_stride if target_preview_stride is not None else self.schema.get("target_preview_stride", 1))
        if self.target_preview_steps < 0:
            raise ValueError("target_preview_steps must be >= 0")
        if self.target_preview_stride <= 0:
            raise ValueError("target_preview_stride must be > 0")

        self.layer_norm_eps = float(self.metadata.get("layer_norm_eps", 1.0e-5))
        self.ip_scale = _positive_scale(self.normalization.get("ip_scale", 1.0), "ip_scale")
        self.radius_scale = _positive_scale(self.normalization.get("radius_scale", 1.0), "radius_scale")
        self.psi_scale = _positive_scale(self.normalization.get("psi_scale", 1.0), "psi_scale")
        self.current_scale = _scale_array(self.normalization["current_scale"], self.n_active_total, "current_scale")
        self.derivative_scale = _scale_array(self.normalization["derivative_scale"], self.action_dim, "derivative_scale")
        if self.action_contract == "delta_jdot_derivative_command_v2":
            if "delta_derivative_limits_aps" not in self.normalization:
                raise ValueError("delta_jdot_derivative_command_v2 requires delta_derivative_limits_aps in normalization.json")
            self.delta_derivative_limits_aps = _scale_array(self.normalization["delta_derivative_limits_aps"], self.action_dim, "delta_derivative_limits_aps")
            self.delta_derivative_scale_aps = float(np.max(self.delta_derivative_limits_aps))
        elif self.action_contract == "delta_jdot_derivative_command_v1":
            if "delta_derivative_scale_aps" not in self.normalization:
                raise ValueError("delta_jdot_derivative_command_v1 requires delta_derivative_scale_aps in normalization.json")
            self.delta_derivative_scale_aps = _positive_scale(self.normalization["delta_derivative_scale_aps"], "delta_derivative_scale_aps")
            self.delta_derivative_limits_aps = np.full((self.action_dim,), self.delta_derivative_scale_aps, dtype=float)
        else:
            raw_delta_scale = self.normalization.get("delta_derivative_scale_aps")
            self.delta_derivative_scale_aps = 0.0 if raw_delta_scale is None else _positive_scale(raw_delta_scale, "delta_derivative_scale_aps")
            self.delta_derivative_limits_aps = np.full((self.action_dim,), self.delta_derivative_scale_aps, dtype=float)
        raw_saturation_fraction = self.normalization.get("current_saturation_fraction")
        self.current_saturation_fraction = None if raw_saturation_fraction is None else float(raw_saturation_fraction)
        if self.current_saturation_fraction is not None and (
            not np.isfinite(self.current_saturation_fraction) or self.current_saturation_fraction < 1.0
        ):
            raise ValueError("current_saturation_fraction must be finite and >= 1")
        self.current_projection_enabled = bool(self.normalization.get("current_projection_enabled", False))
        self.current_projection_margin_fraction = float(self.normalization.get("current_projection_margin_fraction", 0.0))
        if not np.isfinite(self.current_projection_margin_fraction) or not 0.0 <= self.current_projection_margin_fraction < 1.0:
            raise ValueError("current_projection_margin_fraction must be finite and in [0, 1)")
        self.action_clip = _positive_scale(action_clip, "action_clip")
        self._validate_weights()
        self._previous_action_norm = np.zeros((self.action_dim,), dtype=np.float32)

    def reset(self) -> None:
        self._previous_action_norm = np.zeros((self.action_dim,), dtype=np.float32)
        return None

    def compute_control(
        self,
        *,
        model,
        psi: np.ndarray,
        boundary_poly: np.ndarray | None,
        center: tuple[float, float],
        measure_angles: np.ndarray,
        ref_radii: np.ndarray,
        Ip_ref: float,
        scenario,
        max_episode_steps: int,
    ) -> ControlAction:
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
            boundary_poly=boundary_poly,
            center=center,
            measure_angles=measure_angles,
            ref_radii=ref_radii,
            ip_ref=float(Ip_ref),
            scenario=scenario,
            max_episode_steps=int(max_episode_steps),
        )
        requested_action_norm = self._deterministic_action(obs.reshape(1, -1))[0]
        requested_action_norm = np.clip(requested_action_norm, -self.action_clip, self.action_clip).astype(np.float32, copy=False)
        command_norm = self._command_from_actor_action(requested_action_norm)
        physical = np.asarray(command_norm, dtype=float) * self.derivative_scale
        physical = self._project_physical_derivatives(model=model, physical=physical)
        derivative_scale = np.where(np.abs(self.derivative_scale) > 1.0e-12, self.derivative_scale, 1.0)
        applied_action_norm = np.clip(physical / derivative_scale, -1.0, 1.0)
        if self.action_contract == "tcv_derivative_v1":
            self._previous_action_norm = requested_action_norm.astype(np.float32, copy=True)
        else:
            self._previous_action_norm = applied_action_norm.astype(np.float32, copy=True)
        return ControlAction(pfc_derivs=physical[:n_pfc].copy(), sol_derivs=physical[n_pfc:].copy())

    def _command_from_actor_action(self, requested_action_norm: np.ndarray) -> np.ndarray:
        """Convert the actor output into the normalized derivative command sent to the plant."""
        if self.action_contract not in {"delta_jdot_derivative_command_v1", "delta_jdot_derivative_command_v2"}:
            return np.asarray(requested_action_norm, dtype=float)
        derivative_scale = np.where(np.abs(self.derivative_scale) > 1.0e-12, self.derivative_scale, 1.0)
        delta_norm = np.asarray(requested_action_norm, dtype=float) * self.delta_derivative_limits_aps / derivative_scale
        return np.clip(self._previous_action_norm.astype(float, copy=False) + delta_norm, -1.0, 1.0)

    def _project_physical_derivatives(self, *, model, physical: np.ndarray) -> np.ndarray:
        """Apply the exported action contract to physical derivative commands."""
        if self.action_contract == "tcv_derivative_v1":
            return np.asarray(physical, dtype=float)
        if self.current_saturation_fraction is not None:
            limits = np.maximum(self.current_scale * float(self.current_saturation_fraction), 1.0e-12)
            return self._clip_physical_to_current_envelope(model=model, physical=physical, limits=limits)
        if not self.current_projection_enabled:
            return np.asarray(physical, dtype=float)
        limits = np.maximum(self.current_scale * (1.0 - self.current_projection_margin_fraction), 1.0e-12)
        return self._clip_physical_to_current_envelope(model=model, physical=physical, limits=limits)

    def _clip_physical_to_current_envelope(self, *, model, physical: np.ndarray, limits: np.ndarray) -> np.ndarray:
        currents = np.concatenate([
            np.asarray(model.state.pfc_currents, dtype=float).reshape(-1),
            np.asarray(model.state.sol_currents, dtype=float).reshape(-1),
        ])
        previous_derivs = np.concatenate([
            np.asarray(model.state.pfc_current_derivs, dtype=float).reshape(-1),
            np.asarray(model.state.sol_current_derivs, dtype=float).reshape(-1),
        ])
        if currents.shape != (self.action_dim,) or previous_derivs.shape != (self.action_dim,):
            raise ValueError("controller current projection state shape does not match action_dim")
        dt = max(float(getattr(model, "t_step")), 1.0e-12)
        tau = float(getattr(model, "actuator_tau", 0.0))
        alpha = 0.0 if tau <= 0.0 else float(np.exp(-dt / tau))
        beta = max(1.0 - alpha, 1.0e-12)
        lower_applied = (-limits - currents) / dt
        upper_applied = (limits - currents) / dt
        lower_command = (lower_applied - alpha * previous_derivs) / beta
        upper_command = (upper_applied - alpha * previous_derivs) / beta
        lower = np.minimum(lower_command, upper_command)
        upper = np.maximum(lower_command, upper_command)
        return np.clip(np.asarray(physical, dtype=float), lower, upper)

    def _observation(
        self,
        *,
        model,
        psi: np.ndarray,
        boundary_poly: np.ndarray | None,
        center: tuple[float, float],
        measure_angles: np.ndarray,
        ref_radii: np.ndarray,
        ip_ref: float,
        scenario,
        max_episode_steps: int,
    ) -> np.ndarray:
        angles = np.asarray(measure_angles, dtype=float).reshape(-1)
        ref = np.asarray(ref_radii, dtype=float).reshape(-1)
        if angles.shape != (self.n_angles,):
            raise ValueError(f"controller expected {self.n_angles} reference angles, got {angles.shape[0]}")
        if ref.shape != (self.n_angles,):
            raise ValueError(f"controller expected {self.n_angles} target radii, got {ref.shape[0]}")
        psi_arr = np.asarray(psi, dtype=float)
        if "psi_flat" in self.schema.get("feature_order", []):
            if psi_arr.shape != self.grid_shape:
                raise ValueError(f"controller expected psi grid {self.grid_shape}, got {psi_arr.shape}")
        currents = np.concatenate([
            np.asarray(model.state.pfc_currents, dtype=float).reshape(-1),
            np.asarray(model.state.sol_currents, dtype=float).reshape(-1),
        ])
        derivs = np.concatenate([
            np.asarray(model.state.pfc_current_derivs, dtype=float).reshape(-1),
            np.asarray(model.state.sol_current_derivs, dtype=float).reshape(-1),
        ])
        if currents.shape != (self.n_active_total,):
            raise ValueError(f"controller expected {self.n_active_total} active currents, got {currents.shape[0]}")
        if derivs.shape != (self.action_dim,):
            raise ValueError(f"controller expected {self.action_dim} active current derivatives, got {derivs.shape[0]}")
        if boundary_poly is None:
            measured_radii = np.zeros((self.n_angles,), dtype=float)
            boundary_found = 0.0
        else:
            measured_radii = radii_from_polyline_ray_intersections(np.asarray(boundary_poly, dtype=float), center, angles)
            measured_radii = np.nan_to_num(measured_radii, nan=0.0, posinf=0.0, neginf=0.0)
            boundary_found = 1.0
        current_scale = np.where(self.current_scale > 0.0, self.current_scale, 1.0)
        derivative_scale = np.where(self.derivative_scale > 0.0, self.derivative_scale, 1.0)
        measured_ip = float(model.state.Ip)
        features = {
            "step_norm": np.array([float(model.state.step) / max(float(max_episode_steps), 1.0)], dtype=float),
            "ip": np.array([measured_ip / self.ip_scale], dtype=float),
            "ip_ref": np.array([float(ip_ref) / self.ip_scale], dtype=float),
            "ip_error": np.array([(measured_ip - float(ip_ref)) / self.ip_scale], dtype=float),
            "active_currents": currents / current_scale,
            "active_current_derivs": derivs / derivative_scale,
            "psi_flat": psi_arr.reshape(-1) / self.psi_scale,
            "measured_boundary_radii": measured_radii / self.radius_scale,
            "ref_radii": ref / self.radius_scale,
            "boundary_radii_error": (ref - measured_radii) / self.radius_scale,
            "boundary_found": np.array([boundary_found], dtype=float),
            "previous_action": self._previous_action_norm.astype(float, copy=False),
            "target_preview": self._reference_preview(model=model, scenario=scenario, angles=angles, max_episode_steps=max_episode_steps),
        }
        order = list(self.schema.get("feature_order", []))
        obs = np.concatenate([np.asarray(features[name], dtype=float).reshape(-1) for name in order]).astype(np.float32, copy=False)
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

    def _validate_observation_schema(self) -> None:
        kind = self.schema.get("observation_kind")
        if kind not in SUPPORTED_FEATURE_ORDERS:
            if "diagnostics" in self.schema:
                raise ValueError("learned_magnetic_controller export uses the old virtual-diagnostic observation schema; retrain and export with observation_kind='joint_state_v1'")
            raise ValueError(f"learned_magnetic_controller requires one of {sorted(SUPPORTED_FEATURE_ORDERS)}, got {kind!r}")
        order = list(self.schema.get("feature_order", []))
        if order != SUPPORTED_FEATURE_ORDERS[str(kind)]:
            raise ValueError(f"controller schema feature_order does not match {kind}")

    def _validate_weights(self) -> None:
        required = {"input.weight", "input.bias", "input_norm.weight", "input_norm.bias", "hidden1.weight", "hidden1.bias", "mean_head.weight", "mean_head.bias"}
        missing = sorted(required - set(self.weights))
        if missing:
            raise ValueError(f"learned controller export is missing weights: {', '.join(missing)}")
        if self.weights["input.weight"].shape[1] != self.obs_dim:
            raise ValueError("controller input weight shape does not match obs_dim")
        if self.weights["mean_head.weight"].shape[0] != self.action_dim:
            raise ValueError("controller mean head shape does not match action_dim")


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
