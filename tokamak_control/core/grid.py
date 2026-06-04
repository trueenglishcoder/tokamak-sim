from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class Grid1D:
    """
    One-dimensional uniform grid.

    Attributes
    ----------
    start : float
        Nominal coordinate near the first sample.
    step : float
        Positive spacing between samples.
    size : int
        Number of samples (>= 2).
    center : float
        Physical coordinate of the plasma center along this axis.

    Notes
    -----
    The runtime uses ``center`` as the primary geometric anchor and derives grid
    coordinates from the nearest grid index to that center. This makes the grid
    semantics depend on ``center`` rather than treating ``start`` as the sole
    source of truth.
    """

    start: float
    step: float
    size: int
    center: float

    def __post_init__(self) -> None:
        if self.step <= 0.0:
            raise ValueError("Grid1D.step must be > 0")
        if self.size < 2:
            raise ValueError("Grid1D.size must be >= 2")
        if not np.isfinite(float(self.start)):
            raise ValueError("Grid1D.start must be finite")
        if not np.isfinite(float(self.center)):
            raise ValueError("Grid1D.center must be finite")
        idx = (float(self.center) - float(self.start)) / float(self.step)
        if not np.isfinite(idx):
            raise ValueError("Grid1D center alignment is not finite")
        if idx < -0.5 or idx > (self.size - 1) + 0.5:
            raise ValueError("Grid1D.center must lie within half a grid step of the domain")

    @property
    def center_index(self) -> int:
        """Return the nearest integer grid index associated with ``center``."""
        idx = int(round((float(self.center) - float(self.start)) / float(self.step)))
        return int(min(max(idx, 0), self.size - 1))

    @property
    def aligned_start(self) -> float:
        """Return the effective first coordinate implied by ``center`` and ``center_index``."""
        return float(self.center) - float(self.center_index) * float(self.step)

    def coord(self, i: int) -> float:
        """Return the coordinate value at 0-based index ``i``."""
        if i < 0 or i >= self.size:
            raise IndexError("Grid1D index out of bounds")
        return float(self.center) + (int(i) - self.center_index) * float(self.step)

    def coords(self) -> np.ndarray:
        """Return coordinates for all indices."""
        idx = np.arange(self.size, dtype=float)
        return float(self.center) + (idx - float(self.center_index)) * float(self.step)

    def nearest_index(self, x: float) -> int:
        """Return the nearest valid index to coordinate ``x``."""
        idx = int(round((float(x) - float(self.center)) / float(self.step) + float(self.center_index)))
        return int(min(max(idx, 0), self.size - 1))


@dataclass(frozen=True, slots=True)
class Grid2D:
    """
    Rectangular (R,Z) grid composed of two Grid1D axes.

    Notes
    -----
    Array fields on this grid follow shape ``(nz, nr)``, with Z as the first
    axis and R as the second.
    """

    r: Grid1D
    z: Grid1D

    @property
    def shape(self) -> tuple[int, int]:
        return (self.z.size, self.r.size)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        R = self.r.coords()[None, :].repeat(self.z.size, axis=0)
        Z = self.z.coords()[:, None].repeat(self.r.size, axis=1)
        return R, Z
