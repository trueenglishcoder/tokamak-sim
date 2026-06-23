from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tokamak_control.control.base import ControlAction, Controller
from tokamak_control.geometry.legacy_metrics import legacy_radii_at_angles


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
CONTROLLER_STATE_V4_FEATURE_ORDER = [
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
CONTROLLER_STATE_V5_FEATURE_ORDER = [
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
    "ip_ref_rate",
    "boundary_ref_rate",
    "ip_measured_rate",
    "previous_action",
    "target_preview",
]
CONTROLLER_STATE_V6_FEATURE_ORDER = [
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
    "ip_ref_rate",
    "boundary_ref_rate",
    "ip_measured_rate",
    "integral_ip_error",
    "integral_boundary_radii_error",
    "previous_action",
    "target_preview",
]
COMPACT_JOINT_STATE_V2_FEATURE_ORDER = CONTROLLER_STATE_V2_FEATURE_ORDER
OBSERVATION_KIND = "controller_state_v4"
EXPECTED_FEATURE_ORDER = CONTROLLER_STATE_V4_FEATURE_ORDER
SUPPORTED_FEATURE_ORDERS = {
    "controller_state_v4": CONTROLLER_STATE_V4_FEATURE_ORDER,
    "controller_state_v5": CONTROLLER_STATE_V5_FEATURE_ORDER,
    "controller_state_v6": CONTROLLER_STATE_V6_FEATURE_ORDER,
}


class LearnedMagneticController(Controller):
    """Deterministic learned magnetic controller backed by an exported actor bundle.

    The actor observation is the fixed-size tensor form of the state supplied to
    learned boundary controllers: reconstructed boundary radii, target radii,
    Ip target, active coil currents, applied current derivatives, and the
    previous clipped Jdot command.
    """

    def __init__(
        self,
        *,
        export_dir: str | Path,
        target_preview_stride: int | None = None,
        action_clip: float = 1.0,
        episode_norm_steps: int | None = None,
        rolling_episode_norm: bool = False,
    ) -> None:
        self.export_dir = Path(export_dir).expanduser()
        if not self.export_dir.is_dir():
            raise FileNotFoundError(f"learned controller export directory does not exist: {self.export_dir}")
        self.schema = _read_json(_first_existing(self.export_dir, ("controller_schema.json", "schema.json")))
        self._validate_observation_schema()
        self.observation_kind = str(self.schema.get("observation_kind"))
        self.action_contract = str(self.schema.get("action_contract", ""))
        if self.action_contract != "absolute_jdot_command_v1":
            raise ValueError(
                "learned-controller exports must use action_contract='absolute_jdot_command_v1'; "
                f"got {self.action_contract!r}"
            )
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
        self.episode_norm_steps = None if episode_norm_steps is None else int(episode_norm_steps)
        if self.episode_norm_steps is not None and self.episode_norm_steps <= 0:
            raise ValueError("episode_norm_steps must be > 0 when provided")
        self.rolling_episode_norm = bool(rolling_episode_norm)

        self.layer_norm_eps = float(self.metadata.get("layer_norm_eps", 1.0e-5))
        self.ip_scale = _positive_scale(self.normalization.get("ip_scale", 1.0), "ip_scale")
        self.radius_scale = _positive_scale(self.normalization.get("radius_scale", 1.0), "radius_scale")
        self.psi_scale = _positive_scale(self.normalization.get("psi_scale", 1.0), "psi_scale")
        self.ip_rate_scale = self._schema_or_normalization_scale("ip_rate_scale_aps", default=1.0)
        self.boundary_rate_scale = self._schema_or_normalization_scale("boundary_rate_scale_mps", default=1.0)
        self.current_scale = _scale_array(self.normalization["current_scale"], self.n_active_total, "current_scale")
        self.derivative_scale = _scale_array(self.normalization["derivative_scale"], self.action_dim, "derivative_scale")
        self.action_clip = _positive_scale(action_clip, "action_clip")
        self._validate_weights()
        self._previous_action_norm = np.zeros((self.action_dim,), dtype=np.float32)
        self._previous_ip: float | None = None
        self._integral_ip_error = 0.0
        self._integral_boundary_radii_error = np.zeros((self.n_angles,), dtype=np.float32)
        self._last_integral_step: int | None = None

    def reset(self) -> None:
        self._previous_action_norm = np.zeros((self.action_dim,), dtype=np.float32)
        self._previous_ip = None
        self._integral_ip_error = 0.0
        self._integral_boundary_radii_error = np.zeros((self.n_angles,), dtype=np.float32)
        self._last_integral_step = None
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
        measured_ip: float | None = None,
        measured_active_currents: np.ndarray | None = None,
        measured_radii: np.ndarray | None = None,
        boundary_found: bool | None = None,
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
            measured_ip=measured_ip,
            measured_active_currents=measured_active_currents,
            measured_radii=measured_radii,
            boundary_found=boundary_found,
            center=center,
            measure_angles=measure_angles,
            ref_radii=ref_radii,
            ip_ref=float(Ip_ref),
            scenario=scenario,
            max_episode_steps=int(max_episode_steps),
        )
        observed_ip = float(model.state.Ip) if measured_ip is None else float(measured_ip)
        requested_action_norm = self._deterministic_action(obs.reshape(1, -1))[0]
        requested_action_norm = np.clip(requested_action_norm, -self.action_clip, self.action_clip).astype(np.float32, copy=False)
        physical = np.asarray(requested_action_norm, dtype=float) * self.derivative_scale
        physical = self._project_physical_derivatives(model=model, physical=physical)
        derivative_scale = np.where(np.abs(self.derivative_scale) > 1.0e-12, self.derivative_scale, 1.0)
        applied_action_norm = np.clip(physical / derivative_scale, -1.0, 1.0)
        self._previous_action_norm = applied_action_norm.astype(np.float32, copy=True)
        self._previous_ip = observed_ip
        currents_now = np.concatenate([
            np.asarray(model.state.pfc_currents, dtype=float).reshape(-1),
            np.asarray(model.state.sol_currents, dtype=float).reshape(-1),
        ])
        currents_next = currents_now + float(getattr(model, "t_step")) * physical
        return ControlAction(
            pfc_currents_next=currents_next[:n_pfc].copy(),
            sol_currents_next=currents_next[n_pfc:].copy(),
        )

    def _project_physical_derivatives(self, *, model, physical: np.ndarray) -> np.ndarray:
        """Apply the exported action contract to physical derivative commands."""
        del model
        return np.asarray(physical, dtype=float)

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
        measured_ip: float | None = None,
        measured_active_currents: np.ndarray | None = None,
        measured_radii: np.ndarray | None = None,
        boundary_found: bool | None = None,
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
        observed_currents = currents if measured_active_currents is None else np.asarray(measured_active_currents, dtype=float).reshape(-1)
        derivs = np.concatenate([
            np.asarray(model.state.pfc_current_derivs, dtype=float).reshape(-1),
            np.asarray(model.state.sol_current_derivs, dtype=float).reshape(-1),
        ])
        if currents.shape != (self.n_active_total,):
            raise ValueError(f"controller expected {self.n_active_total} active currents, got {currents.shape[0]}")
        if observed_currents.shape != (self.n_active_total,):
            raise ValueError(f"controller expected {self.n_active_total} measured active currents, got {observed_currents.shape[0]}")
        if derivs.shape != (self.action_dim,):
            raise ValueError(f"controller expected {self.action_dim} active current derivatives, got {derivs.shape[0]}")
        if measured_radii is not None:
            measured_radii_arr = np.asarray(measured_radii, dtype=float).reshape(-1)
            if measured_radii_arr.shape != (self.n_angles,):
                raise ValueError(f"controller expected {self.n_angles} measured radii, got {measured_radii_arr.shape[0]}")
            measured_radii_arr = np.nan_to_num(measured_radii_arr, nan=0.0, posinf=0.0, neginf=0.0)
            found_value = 1.0 if bool(boundary_found) else 0.0
            if boundary_found is None:
                found_value = 1.0 if np.any(np.isfinite(np.asarray(measured_radii, dtype=float))) else 0.0
        elif boundary_poly is None:
            measured_radii_arr = np.zeros((self.n_angles,), dtype=float)
            found_value = 0.0
        else:
            measured_radii_arr = legacy_radii_at_angles(np.asarray(boundary_poly, dtype=float), center, angles)
            measured_radii_arr = np.nan_to_num(measured_radii_arr, nan=0.0, posinf=0.0, neginf=0.0)
            found_value = 1.0
        observed_ip = float(model.state.Ip) if measured_ip is None else float(measured_ip)
        self._sync_integral_errors(model=model, measured_ip=observed_ip, ip_ref=float(ip_ref), measured_radii=measured_radii_arr, ref_radii=ref)
        current_scale = np.where(self.current_scale > 0.0, self.current_scale, 1.0)
        derivative_scale = np.where(self.derivative_scale > 0.0, self.derivative_scale, 1.0)
        ip_ref_rate, boundary_ref_rate, ip_measured_rate = self._rate_features(
            model=model,
            scenario=scenario,
            angles=angles,
            ip_ref=float(ip_ref),
            ref_radii=ref,
            measured_ip=observed_ip,
        )
        features = {
            "step_norm": np.array([self._step_norm(model=model, max_episode_steps=max_episode_steps)], dtype=float),
            "ip": np.array([observed_ip / self.ip_scale], dtype=float),
            "ip_ref": np.array([float(ip_ref) / self.ip_scale], dtype=float),
            "ip_error": np.array([(observed_ip - float(ip_ref)) / self.ip_scale], dtype=float),
            "active_currents": observed_currents / current_scale,
            "active_current_derivs": derivs / derivative_scale,
            "psi_flat": psi_arr.reshape(-1) / self.psi_scale,
            "measured_boundary_radii": measured_radii_arr / self.radius_scale,
            "ref_radii": ref / self.radius_scale,
            "boundary_radii_error": (ref - measured_radii_arr) / self.radius_scale,
            "boundary_found": np.array([found_value], dtype=float),
            "ip_ref_rate": np.array([ip_ref_rate], dtype=float),
            "boundary_ref_rate": boundary_ref_rate,
            "ip_measured_rate": np.array([ip_measured_rate], dtype=float),
            "integral_ip_error": np.array([self._normalized_integral_ip_error()], dtype=float),
            "integral_boundary_radii_error": self._normalized_integral_boundary_radii_error(),
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

    def _rate_features(
        self,
        *,
        model,
        scenario,
        angles: np.ndarray,
        ip_ref: float,
        ref_radii: np.ndarray,
        measured_ip: float,
    ) -> tuple[float, np.ndarray, float]:
        """Return controller_state_v5/v6 rate features in the same normalized units as training."""
        if self.observation_kind not in {"controller_state_v5", "controller_state_v6"}:
            return 0.0, np.zeros((self.n_angles,), dtype=float), 0.0
        dt = max(float(getattr(model, "t_step")), 1.0e-12)
        t_next = float(model.state.t) + dt
        ip_ref_next = float(scenario.Ip_ref(t_next))
        ref_next = np.asarray(scenario.ref_radii(angles, t_next), dtype=float).reshape(-1)
        if ref_next.shape != (self.n_angles,):
            raise ValueError(f"controller expected {self.n_angles} next target radii, got {ref_next.shape[0]}")
        ip_ref_rate = ((ip_ref_next - float(ip_ref)) / dt) / self.ip_rate_scale
        boundary_ref_rate = ((ref_next - np.asarray(ref_radii, dtype=float).reshape(-1)) / dt) / self.boundary_rate_scale
        if self._previous_ip is None:
            ip_measured_rate = 0.0
        else:
            ip_measured_rate = ((float(measured_ip) - float(self._previous_ip)) / dt) / self.ip_rate_scale
        return float(ip_ref_rate), np.asarray(boundary_ref_rate, dtype=float), float(ip_measured_rate)

    def _sync_integral_errors(
        self,
        *,
        model,
        measured_ip: float,
        ip_ref: float,
        measured_radii: np.ndarray,
        ref_radii: np.ndarray,
    ) -> None:
        """Advance controller_state_v6 integral-error memory to the current model step."""
        if self.observation_kind != "controller_state_v6":
            return
        step = int(getattr(model.state, "step", 0))
        if self._last_integral_step is None:
            self._last_integral_step = step
            return
        delta_steps = max(step - int(self._last_integral_step), 0)
        if delta_steps <= 0:
            return
        dt = max(float(getattr(model, "t_step")), 1.0e-12) * float(delta_steps)
        self._integral_ip_error += dt * (float(ip_ref) - float(measured_ip))
        radii_error = np.asarray(ref_radii, dtype=float).reshape(self.n_angles) - np.nan_to_num(
            np.asarray(measured_radii, dtype=float).reshape(self.n_angles),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self._integral_boundary_radii_error = (self._integral_boundary_radii_error + dt * radii_error).astype(np.float32, copy=False)
        self._last_integral_step = step

    def _normalized_integral_ip_error(self) -> float:
        """Return the v6 normalized integral Ip error used by training."""
        scale = 15000.0 * 0.1
        return float(np.clip(float(self._integral_ip_error) / scale, -1.0, 1.0))

    def _normalized_integral_boundary_radii_error(self) -> np.ndarray:
        """Return the v6 normalized integral boundary-radius errors used by training."""
        scale = 0.02 * 0.1
        return np.clip(np.asarray(self._integral_boundary_radii_error, dtype=float) / scale, -1.0, 1.0)

    def _reference_preview(self, *, model, scenario, angles: np.ndarray, max_episode_steps: int) -> np.ndarray:
        if self.target_preview_steps == 0:
            return np.zeros((0,), dtype=float)
        offsets = np.arange(1, self.target_preview_steps + 1, dtype=float) * float(self.target_preview_stride)
        times = float(model.state.t) + offsets * float(model.t_step)
        horizon = self._normalization_horizon(max_episode_steps=max_episode_steps)
        time_norm = offsets / horizon
        ip_preview = np.asarray([float(scenario.Ip_ref(float(t))) for t in times], dtype=float) / self.ip_scale
        radii_preview = np.stack([np.asarray(scenario.ref_radii(angles, float(t)), dtype=float).reshape(-1) for t in times], axis=0) / self.radius_scale
        return np.concatenate([time_norm, ip_preview, radii_preview.reshape(-1)], dtype=float)

    def _normalization_horizon(self, *, max_episode_steps: int) -> float:
        if self.episode_norm_steps is not None:
            return float(self.episode_norm_steps)
        return max(float(max_episode_steps), 1.0)

    def _step_norm(self, *, model, max_episode_steps: int) -> float:
        horizon = self._normalization_horizon(max_episode_steps=max_episode_steps)
        step = float(model.state.step)
        if self.rolling_episode_norm and self.episode_norm_steps is not None:
            step = float(int(model.state.step) % int(self.episode_norm_steps))
        return step / horizon

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
                raise ValueError("learned_magnetic_controller export uses an old virtual-diagnostic observation schema; retrain under a supported controller_state schema")
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

    def _schema_or_normalization_scale(self, name: str, *, default: float) -> float:
        if name in self.schema:
            return _positive_scale(self.schema[name], name)
        if name in self.normalization:
            return _positive_scale(self.normalization[name], name)
        if self.observation_kind in {"controller_state_v5", "controller_state_v6"}:
            raise ValueError(f"{name} is required for {self.observation_kind} learned-controller exports")
        return _positive_scale(default, name)


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
