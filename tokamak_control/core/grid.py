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
    Coordinates follow the original Little SCoPE grid convention: the configured
    center is placed halfway between two neighboring samples by shifting the
    effective first coordinate by the fractional part of the center offset.
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
        """Return the lower bracketing index for the configured center."""
        q = (float(self.center) - float(self.start)) / float(self.step)
        idx = int(np.floor(q))
        return int(min(max(idx, 0), self.size - 1))

    @property
    def aligned_start(self) -> float:
        """Return the old-model shifted first coordinate."""
        q = (float(self.center) - float(self.start)) / float(self.step)
        frac = q - float(np.floor(q))
        return float(self.start) + (frac - 0.5) * float(self.step)

    def coord(self, i: int) -> float:
        """Return the coordinate value at 0-based index ``i``."""
        if i < 0 or i >= self.size:
            raise IndexError("Grid1D index out of bounds")
        return self.aligned_start + int(i) * float(self.step)

    def coords(self) -> np.ndarray:
        """Return coordinates for all indices."""
        idx = np.arange(self.size, dtype=float)
        return self.aligned_start + idx * float(self.step)

    def nearest_index(self, x: float) -> int:
        """Return the nearest valid index to coordinate ``x``."""
        idx = int(round((float(x) - self.aligned_start) / float(self.step)))
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
