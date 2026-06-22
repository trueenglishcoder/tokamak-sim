from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Coil:
    """
    Single physical coil element center.

    Attributes
    ----------
    R : float
        Major radius of the element center (m).
    Z : float
        Vertical position of the element center (m).
    """

    R: float
    Z: float


@dataclass(slots=True)
class CoilActuator:
    """
    One runtime actuator composed of one or more physical coil elements.

    Attributes
    ----------
    elements : list[Coil]
        Physical elements driven by one shared current.
    element_weights : np.ndarray | None
        Optional per-element contribution weights. The runtime current vector
        still has one value per actuator, not one value per physical element.
        For volumetric split coils, such as the T15 SOL coils, these weights
        are fractions that sum to 1 so the actuator's total current is
        distributed across the point elements. When omitted, each element
        contributes with weight 1.0.
    """

    elements: list[Coil]
    element_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Проверить элементы актуатора и, при наличии, веса split-элементов."""
        if not self.elements:
            raise ValueError("CoilActuator must contain at least one Coil element")
        for elem in self.elements:
            if not isinstance(elem, Coil):
                raise TypeError("CoilActuator.elements must contain Coil instances")
        if self.element_weights is None:
            return
        weights = np.asarray(self.element_weights, dtype=float).reshape(-1)
        if weights.shape != (len(self.elements),):
            raise ValueError("element_weights must have shape (n_elements,)")
        if not np.all(np.isfinite(weights)):
            raise ValueError("element_weights must contain only finite values")
        if np.any(weights < 0.0):
            raise ValueError("element_weights must be >= 0")
        if not float(np.sum(weights)) > 0.0:
            raise ValueError("element_weights must contain a positive total weight")
        self.element_weights = weights.copy()

    @property
    def centroid(self) -> tuple[float, float]:
        """Return the centroid of the actuator element centers."""
        R = np.array([c.R for c in self.elements], dtype=float)
        Z = np.array([c.Z for c in self.elements], dtype=float)
        return float(np.mean(R)), float(np.mean(Z))

    @property
    def positions(self) -> np.ndarray:
        """Return the actuator element-center coordinates as ``(n_elements, 2)``."""
        R = np.array([c.R for c in self.elements], dtype=float)
        Z = np.array([c.Z for c in self.elements], dtype=float)
        return np.stack([R, Z], axis=1)

    @property
    def weights(self) -> np.ndarray:
        """Return physical-element weights for this one runtime actuator."""
        if self.element_weights is None:
            return np.ones((len(self.elements),), dtype=float)
        return np.asarray(self.element_weights, dtype=float).copy()


@dataclass(slots=True)
class CoilGroup:
    """
    Static specification of a named coil bank.

    ``positions`` is metadata for plotting and manifests. Plant physics must use
    ``element_positions`` so grouped actuators are modeled from their physical
    element geometry rather than centroid surrogates.
    """

    name: str
    coils: list[CoilActuator]
    currents: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.coils)
        if self.currents is None:
            self.currents = np.zeros(n, dtype=float)
        else:
            self.currents = np.asarray(self.currents, dtype=float)

        if self.currents.shape != (n,):
            raise ValueError("currents must have shape (n_actuators,)")
        if not np.all(np.isfinite(self.currents)):
            raise ValueError("currents must contain only finite values")
        for actuator in self.coils:
            if not isinstance(actuator, CoilActuator):
                raise TypeError("coils must contain CoilActuator instances")

    @property
    def initial_currents(self) -> np.ndarray:
        """Return the configured initial-current vector."""
        return self.currents

    @property
    def n_coils(self) -> int:
        """Return the number of runtime actuators in this group."""
        return len(self.coils)

    @property
    def positions(self) -> np.ndarray:
        """
        Return one representative position per actuator.

        The representative point is the centroid of that actuator's physical
        element centers. This is intended for metadata and plotting. Plant
        physics should use ``element_positions``.
        """
        if not self.coils:
            return np.zeros((0, 2), dtype=float)
        centroids = np.array([act.centroid for act in self.coils], dtype=float)
        return centroids.reshape(len(self.coils), 2)

    @property
    def element_positions(self) -> list[np.ndarray]:
        """Return grouped element-center coordinates for physics use."""
        return [act.positions.copy() for act in self.coils]

    @property
    def element_weights(self) -> list[np.ndarray]:
        """Return one element-weight vector per runtime actuator."""
        return [act.weights for act in self.coils]

    @property
    def n_elements_total(self) -> int:
        """Return the total number of physical elements across all actuators."""
        return int(sum(len(act.elements) for act in self.coils))
