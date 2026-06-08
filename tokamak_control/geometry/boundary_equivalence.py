from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np

from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections


@dataclass(frozen=True, slots=True)
class BoundaryEquivalenceTolerance:
    """Numerical tolerances for CPU/GPU boundary equivalence."""

    level_abs: float = 1.0e-8
    level_rel: float = 1.0e-8
    mean_radii_abs: float = 5.0e-3
    max_radii_abs: float = 1.5e-2
    angle_count: int = 128


@dataclass(frozen=True, slots=True)
class BoundaryEquivalenceReport:
    """Result of comparing two boundary-finder outputs."""

    equivalent: bool
    reason: str | None
    cpu_status: str
    gpu_status: str
    cpu_found: bool
    gpu_found: bool
    level_abs_error: float | None
    level_rel_error: float | None
    mean_radii_abs_error: float | None
    max_radii_abs_error: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compare_boundary_results(
    *,
    cpu_poly: np.ndarray | None,
    cpu_level: float | None,
    cpu_status: str,
    gpu_poly: np.ndarray | None,
    gpu_level: float | None,
    gpu_status: str,
    center: tuple[float, float],
    tolerance: BoundaryEquivalenceTolerance = BoundaryEquivalenceTolerance(),
) -> BoundaryEquivalenceReport:
    """Compare CPU and GPU boundary results without relying on vertex order."""
    cpu_found = cpu_poly is not None and str(cpu_status) != "not_found"
    gpu_found = gpu_poly is not None and str(gpu_status) != "not_found"
    if cpu_found != gpu_found:
        return _report(False, "found/not-found mismatch", cpu_status, gpu_status, cpu_found, gpu_found)
    if not cpu_found and not gpu_found:
        return _report(True, None, cpu_status, gpu_status, False, False)
    if str(cpu_status) != str(gpu_status):
        return _report(False, "status mismatch", cpu_status, gpu_status, True, True)
    if cpu_level is None or gpu_level is None:
        return _report(False, "missing level", cpu_status, gpu_status, True, True)

    level_abs = abs(float(cpu_level) - float(gpu_level))
    level_rel = level_abs / max(abs(float(cpu_level)), abs(float(gpu_level)), 1.0)
    if level_abs > tolerance.level_abs and level_rel > tolerance.level_rel:
        return BoundaryEquivalenceReport(
            equivalent=False,
            reason="level mismatch",
            cpu_status=str(cpu_status),
            gpu_status=str(gpu_status),
            cpu_found=True,
            gpu_found=True,
            level_abs_error=float(level_abs),
            level_rel_error=float(level_rel),
            mean_radii_abs_error=None,
            max_radii_abs_error=None,
        )

    angles = np.linspace(-np.pi, np.pi, int(tolerance.angle_count), endpoint=False, dtype=float)
    cpu_radii = radii_from_polyline_ray_intersections(np.asarray(cpu_poly, dtype=float), center, angles)
    gpu_radii = radii_from_polyline_ray_intersections(np.asarray(gpu_poly, dtype=float), center, angles)
    valid = np.isfinite(cpu_radii) & np.isfinite(gpu_radii)
    if not bool(np.any(valid)):
        return BoundaryEquivalenceReport(False, "no comparable finite radii", str(cpu_status), str(gpu_status), True, True, float(level_abs), float(level_rel), None, None)
    errors = np.abs(cpu_radii[valid] - gpu_radii[valid])
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    ok = mean_error <= float(tolerance.mean_radii_abs) and max_error <= float(tolerance.max_radii_abs)
    return BoundaryEquivalenceReport(
        equivalent=bool(ok),
        reason=None if ok else "radii mismatch",
        cpu_status=str(cpu_status),
        gpu_status=str(gpu_status),
        cpu_found=True,
        gpu_found=True,
        level_abs_error=float(level_abs),
        level_rel_error=float(level_rel),
        mean_radii_abs_error=mean_error,
        max_radii_abs_error=max_error,
    )


def write_boundary_equivalence_report(path: str | Path, reports: list[BoundaryEquivalenceReport]) -> Path:
    """Write a JSON report for a boundary-equivalence run."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_count": len(reports),
        "equivalent_count": sum(1 for report in reports if report.equivalent),
        "reports": [report.to_dict() for report in reports],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _report(
    equivalent: bool,
    reason: str | None,
    cpu_status: str,
    gpu_status: str,
    cpu_found: bool,
    gpu_found: bool,
) -> BoundaryEquivalenceReport:
    return BoundaryEquivalenceReport(
        equivalent=bool(equivalent),
        reason=reason,
        cpu_status=str(cpu_status),
        gpu_status=str(gpu_status),
        cpu_found=bool(cpu_found),
        gpu_found=bool(gpu_found),
        level_abs_error=None,
        level_rel_error=None,
        mean_radii_abs_error=None,
        max_radii_abs_error=None,
    )
