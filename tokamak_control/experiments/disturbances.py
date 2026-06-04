# tokamak_control/experiments/disturbances.py
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from tokamak_control.core.plasma_state import PlasmaState


@dataclass(frozen=True, slots=True)
class DisturbanceContext:
    """
    Per-step runtime context passed to plant disturbances.

    Disturbances are applied after the plant step and may return a modified
    PlasmaState. Scenarios define reference trajectories. Disturbances modify
    the actual plant state and are not part of the planned reference.
    """

    step_index: int
    t: float
    total_steps: int
    scenario_name: str


@dataclass(frozen=True, slots=True)
class DisturbanceResult:
    """
    Result of applying a disturbance to the current plant state.
    """

    state: PlasmaState
    applied: bool


class Disturbance(ABC):
    """
    Abstract interface for experiment-level plant disturbances.

    Disturbances are prepared once for a run and then applied after each plant
    step. They may modify the returned PlasmaState directly.
    """

    def prepare(self, total_steps: int) -> None:
        """
        Prepare the disturbance for a run of length total_steps.

        The default implementation does nothing.
        """
        return None

    @abstractmethod
    def apply(self, *, state: PlasmaState, context: DisturbanceContext) -> DisturbanceResult:
        """
        Apply the disturbance to the current plant state.
        """


def prepare_disturbances(
    disturbances: Sequence[Disturbance] | None,
    total_steps: int,
) -> list[Disturbance]:
    """
    Prepare fresh disturbance instances for a specific run length.

    Parameters
    ----------
    disturbances
        Disturbance templates or None.
    total_steps
        Total number of steps in the run.

    Returns
    -------
    list[Disturbance]
        Deep-copied, prepared disturbance instances ready for runtime use.
    """
    if int(total_steps) != total_steps or int(total_steps) <= 0:
        raise ValueError("total_steps must be a positive integer")

    prepared: list[Disturbance] = []
    for i, disturbance in enumerate(disturbances or ()): 
        if not isinstance(disturbance, Disturbance):
            raise TypeError(
                f"disturbances[{i}] must implement Disturbance, got {type(disturbance).__name__}"
            )
        d = copy.deepcopy(disturbance)
        d.prepare(int(total_steps))
        prepared.append(d)
    return prepared


def apply_prepared_disturbances(
    *,
    state: PlasmaState,
    step_index: int,
    total_steps: int,
    scenario_name: str,
    disturbances: Sequence[Disturbance] | None,
) -> tuple[PlasmaState, list[str]]:
    """
    Apply a prepared disturbance list using the shared post-step runtime timing.

    Disturbances are evaluated after the plant has advanced for the current step
    and before downstream code recomputes derived quantities such as ``psi`` and
    the plasma boundary. The returned state should therefore be written back into
    the model before any post-step geometry recomputation.
    """
    context = DisturbanceContext(
        step_index=int(step_index),
        t=float(state.t),
        total_steps=int(total_steps),
        scenario_name=str(scenario_name),
    )

    current = state
    applied: list[str] = []
    for disturbance in disturbances or ():
        result = disturbance.apply(state=current, context=context)
        current = result.state
        if result.applied:
            applied.append(disturbance.__class__.__name__)
    return current, applied


@dataclass(slots=True)
class IpCrash(Disturbance):
    """
    Disturbance that forces a disruption-like plasma current quench.

    This disturbance overrides Ip within a contiguous step window:
        [start_step, start_step + duration_steps)

    Within the window it produces a finite-time drop instead of an instantaneous clamp:

      1) Optional onset spike for `spike_steps`
      2) Smooth current-quench ramp over `ramp_steps`
      3) Optional hold at the low value for the remainder of the window
    """

    start_step: Optional[int] = 10
    duration_steps: int = 50

    # Final low value: Ip_low = baseline_Ip * drop_factor
    drop_factor: float = 0.6

    # Ramp to the low value over this many steps (0 => immediate drop)
    ramp_steps: int = 0

    # Optional onset spike: Ip_spike = baseline_Ip * spike_factor for spike_steps
    spike_factor: float = 1.0
    spike_steps: int = 0

    randomize_start: bool = False
    seed: Optional[int] = None

    _resolved_start: Optional[int] = field(init=False, default=None)
    _resolved_end: Optional[int] = field(init=False, default=None)
    _baseline_Ip: Optional[float] = field(init=False, default=None)

    def _validate_static_fields(self) -> None:
        if self.start_step is not None and int(self.start_step) != self.start_step:
            raise ValueError("start_step must be an int or None")
        if int(self.duration_steps) != self.duration_steps or self.duration_steps <= 0:
            raise ValueError("duration_steps must be a positive integer")
        if not np.isfinite(float(self.drop_factor)) or not (0.0 <= float(self.drop_factor) <= 1.0):
            raise ValueError("drop_factor must be finite and within [0, 1]")
        if not np.isfinite(float(self.spike_factor)) or float(self.spike_factor) < 0.0:
            raise ValueError("spike_factor must be finite and >= 0")
        if int(self.spike_steps) != self.spike_steps or self.spike_steps < 0:
            raise ValueError("spike_steps must be an integer >= 0")
        if int(self.ramp_steps) != self.ramp_steps or self.ramp_steps < 0:
            raise ValueError("ramp_steps must be an integer >= 0")
        if self.seed is not None and (int(self.seed) != self.seed or int(self.seed) < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        if self.spike_steps > self.duration_steps:
            raise ValueError("spike_steps cannot exceed duration_steps")
        if self.spike_steps + self.ramp_steps > self.duration_steps:
            raise ValueError("spike_steps + ramp_steps cannot exceed duration_steps")

    def prepare(self, total_steps: int) -> None:
        """
        Resolve the crash window for a run of given length.
        """
        self._validate_static_fields()
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")

        if self.randomize_start:
            max_start = max(0, total_steps - self.duration_steps)
            if max_start == 0:
                start = 0
            else:
                rng = np.random.default_rng(self.seed)
                start = int(rng.integers(0, max_start + 1))
        else:
            start = 0 if self.start_step is None else int(self.start_step)

        if start < 0 or start + self.duration_steps > total_steps:
            raise ValueError(
                f"IpCrash window [{start}, {start + self.duration_steps}) "
                f"out of bounds for total_steps={total_steps}"
            )

        self._resolved_start = start
        self._resolved_end = start + self.duration_steps
        self._baseline_Ip = None

    @staticmethod
    def _smoothstep01(p: float) -> float:
        """
        Smooth 0->1 transition (cosine) for p in [0, 1].
        """
        p = float(np.clip(p, 0.0, 1.0))
        return 0.5 - 0.5 * float(np.cos(np.pi * p))

    def apply(self, *, state: PlasmaState, context: DisturbanceContext) -> DisturbanceResult:
        """
        Apply the crash disturbance to the current plant state.

        Returns the original state unchanged when the crash is inactive.
        """
        if self._resolved_start is None or self._resolved_end is None:
            return DisturbanceResult(state=state, applied=False)

        if context.step_index < self._resolved_start or context.step_index >= self._resolved_end:
            return DisturbanceResult(state=state, applied=False)

        if self._baseline_Ip is None:
            self._baseline_Ip = float(state.Ip)

        rel = int(context.step_index - self._resolved_start)

        base = float(self._baseline_Ip)
        Ip_low = base * float(self.drop_factor)

        if self.spike_steps > 0 and rel < self.spike_steps:
            Ip_new = base * float(self.spike_factor)
            return DisturbanceResult(state=replace(state, Ip=float(Ip_new)), applied=True)

        Ip_start = base * float(self.spike_factor) if self.spike_steps > 0 else base

        if self.ramp_steps > 0:
            ramp_rel = rel - self.spike_steps
            if ramp_rel < self.ramp_steps:
                p = (ramp_rel + 1) / float(self.ramp_steps)
                s = self._smoothstep01(p)
                Ip_new = (1.0 - s) * Ip_start + s * Ip_low
                return DisturbanceResult(state=replace(state, Ip=float(Ip_new)), applied=True)

        return DisturbanceResult(state=replace(state, Ip=float(Ip_low)), applied=True)

    @classmethod
    def default_for_run(cls, steps: int) -> IpCrash:
        """
        Construct a default IpCrash configuration for a run.
        """
        if int(steps) != steps or int(steps) <= 0:
            raise ValueError("steps must be a positive integer")
        duration = max(1, int(steps) // 20)
        return cls(
            start_step=None,
            duration_steps=duration,
            drop_factor=0.2,
            ramp_steps=duration,
            spike_factor=1.0,
            spike_steps=0,
            randomize_start=True,
            seed=None,
        )
