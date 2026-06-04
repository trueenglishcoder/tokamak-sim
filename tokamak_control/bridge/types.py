from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class MachineSpec:
    """Стабильное описание активной runtime-машины для внешнего stepping-кода."""

    config_path: Path
    initial_currents_path: Path | None
    boundary_mode: str
    limiter_name: str | None
    t_step: float
    n_active_pfc: int
    n_active_sol: int
    n_active_total: int
    active_order: tuple[str, ...]
    pfc_active_mask: np.ndarray
    sol_active_mask: np.ndarray
    current_limits: np.ndarray
    derivative_limits: np.ndarray
    angles_rad: np.ndarray
    radius_scale: float
    ip_scale: float
    current_scale: np.ndarray
    derivative_scale: np.ndarray


@dataclass(frozen=True, slots=True)
class DerivativeAction:
    """Физическая команда производных токов по активным актуаторам, А/с."""

    active_current_derivatives: np.ndarray


@dataclass(frozen=True, slots=True)
class ReferenceFrame:
    """Опорные сигналы сценария на одном шаге моделирования."""

    time_s: float
    ip_ref: float
    radii_ref: np.ndarray
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class StepSnapshot:
    """Снимок состояния после reset или одного шага bridge-сессии."""

    step_index: int
    time_s: float
    reference: ReferenceFrame
    true_ip: float
    true_active_currents: np.ndarray
    commanded_active_derivatives: np.ndarray
    applied_active_derivatives: np.ndarray
    previous_applied_active_derivatives: np.ndarray
    true_boundary_poly: np.ndarray | None
    true_radii: np.ndarray | None
    psi_boundary_value: float | None
    boundary_found: bool
    boundary_reason: str | None
    current_limit_margin: np.ndarray | None
    derivative_limit_margin: np.ndarray | None


@dataclass(frozen=True, slots=True)
class ResetResult:
    """Результат reset для новой programmatic simulation session."""

    observation_snapshot: StepSnapshot
    machine: MachineSpec
    episode_metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class StepResult:
    """Результат одного шага bridge-сессии."""

    snapshot: StepSnapshot
    terminated: bool
    truncated: bool
    termination_reason: str | None
