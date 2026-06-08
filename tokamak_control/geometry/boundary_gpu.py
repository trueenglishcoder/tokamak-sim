from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from tokamak_control.core.grid import Grid2D
from tokamak_control.geometry.boundary import (
    BoundaryMode,
    BoundaryNotFoundError,
    BoundaryStatus,
    _MagneticAxis,
    _close_poly,
    _encloses_center,
    _is_closed_poly,
    _limited_candidate_levels,
    _ordered_limiter_contact_levels,
    _point_to_polyline_distance,
    _points_in_or_on_polygon,
    _poly_area,
    _poly_fits_limiter,
    _polyline_to_points_distance,
    _prepare_limiter_shape,
    _psi_at_nearest_grid_point,
    _sample_limiter_points,
    _time_block,
)


@dataclass(frozen=True, slots=True)
class _TorchRuntime:
    torch: object
    device: object


def find_plasma_boundary_gpu_with_status(
    psi: np.ndarray,
    grid: Grid2D,
    center: tuple[float, float],
    n_levels: int = 10,
    prev_level: float | None = None,
    prev_poly: np.ndarray | None = None,
    local_n_levels: int = 7,
    local_span_frac: float = 0.02,
    target_mean_radius: float | None = None,
    target_switch_ratio: float = 1.15,
    target_switch_abs_delta: float = 0.10,
    local_bbox_pad_r: float | None = None,
    local_bbox_pad_z: float | None = None,
    limiter_shape: np.ndarray | None = None,
    boundary_mode: BoundaryMode = "limited",
    gpu_device: str = "cuda:0",
) -> tuple[np.ndarray, float, BoundaryStatus]:
    """Find the plasma boundary with the CUDA/Torch backend.

    The physical selection rules intentionally mirror the CPU boundary finder.
    CPU fallback is not allowed here: requesting GPU mode requires a usable CUDA
    device.
    """
    _ = (
        prev_level,
        prev_poly,
        local_n_levels,
        local_span_frac,
        target_mean_radius,
        target_switch_ratio,
        target_switch_abs_delta,
        local_bbox_pad_r,
        local_bbox_pad_z,
    )
    runtime = _require_torch_runtime(gpu_device)
    torch = runtime.torch

    with _time_block("boundary_total"):
        psi_t = _psi_tensor(psi, grid=grid, runtime=runtime)
        psi_arr = psi_t.detach().cpu().numpy().astype(float, copy=True)
        limiter_poly = _prepare_limiter_shape(limiter_shape)
        limiter_tol = 0.5 * min(abs(float(grid.r.step)), abs(float(grid.z.step)))
        with _time_block("axis_search"):
            axis = _find_magnetic_axis_gpu(
                psi_t=psi_t,
                grid=grid,
                center=center,
                limiter_poly=limiter_poly,
                limiter_tol=limiter_tol,
                runtime=runtime,
            )

        if boundary_mode == "limited":
            found = _limited_boundary_gpu(
                psi_t=psi_t,
                psi_np=psi_arr,
                grid=grid,
                axis=axis,
                limiter_poly=limiter_poly,
                limiter_tol=limiter_tol,
                n_levels=n_levels,
                runtime=runtime,
            )
            if found is not None:
                poly, level = found
                return _close_poly(poly), float(level), "limited_success"
            raise BoundaryNotFoundError("No limited plasma boundary found inside limiter")

        found = _separatrix_boundary_gpu(
            psi_t=psi_t,
            psi_np=psi_arr,
            grid=grid,
            axis=axis.point,
            limiter_poly=limiter_poly,
            limiter_tol=limiter_tol,
            runtime=runtime,
        )
        if found is not None:
            poly, level = found
            return _close_poly(poly), float(level), "separatrix_success"
        raise BoundaryNotFoundError("No diverted plasma separatrix boundary found")


def _psi_tensor(psi, *, grid: Grid2D, runtime: _TorchRuntime):
    torch = runtime.torch
    if hasattr(psi, "detach") and hasattr(psi, "device"):
        psi_t = psi.detach().to(device=runtime.device, dtype=torch.float64)
    else:
        psi_t = torch.as_tensor(np.asarray(psi, dtype=float), dtype=torch.float64, device=runtime.device)
    if tuple(psi_t.shape) != tuple(grid.shape):
        raise ValueError(f"psi shape {tuple(psi_t.shape)} != grid shape {grid.shape}")
    return psi_t


def _require_torch_runtime(gpu_device: str) -> _TorchRuntime:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("GPU compute backend requires tokamak-sim[gpu] with torch installed") from exc

    device = torch.device(str(gpu_device))
    if device.type != "cuda":
        raise RuntimeError(f"GPU compute backend requires a CUDA device, got {gpu_device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU compute backend requested, but torch.cuda.is_available() is False")
    try:
        torch.empty((1,), device=device)
    except Exception as exc:  # pragma: no cover - depends on host CUDA setup
        raise RuntimeError(f"GPU compute backend could not initialize device {gpu_device!r}") from exc
    return _TorchRuntime(torch=torch, device=device)


def _grid_tensors(grid: Grid2D, runtime: _TorchRuntime):
    torch = runtime.torch
    r = torch.as_tensor(np.asarray(grid.r.coords(), dtype=float), dtype=torch.float64, device=runtime.device)
    z = torch.as_tensor(np.asarray(grid.z.coords(), dtype=float), dtype=torch.float64, device=runtime.device)
    return r, z


def _limiter_mask_gpu(grid: Grid2D, limiter_poly: np.ndarray, tol: float, runtime: _TorchRuntime):
    torch = runtime.torch
    r, z = _grid_tensors(grid, runtime)
    zz, rr = torch.meshgrid(z, r, indexing="ij")
    points = torch.stack((rr.reshape(-1), zz.reshape(-1)), dim=1)
    mask = _points_in_or_on_polygon_gpu(points, limiter_poly, tol, runtime)
    return mask.reshape(grid.shape)


def _points_in_or_on_polygon_gpu(points, polygon: np.ndarray, tol: float, runtime: _TorchRuntime):
    torch = runtime.torch
    poly_np = _close_poly(np.asarray(polygon, dtype=float))
    poly = torch.as_tensor(poly_np, dtype=torch.float64, device=runtime.device)
    x = points[:, 0]
    y = points[:, 1]
    inside = torch.zeros((points.shape[0],), dtype=torch.bool, device=runtime.device)
    for idx in range(poly.shape[0] - 1):
        x0 = poly[idx, 0]
        y0 = poly[idx, 1]
        x1 = poly[idx + 1, 0]
        y1 = poly[idx + 1, 1]
        crosses = (y0 > y) != (y1 > y)
        x_cross = x0 + (y - y0) * (x1 - x0) / ((y1 - y0) + 1e-30)
        inside = torch.logical_xor(inside, crosses & (x_cross > x))

    if float(tol) <= 0.0:
        return inside

    on_edge = torch.zeros_like(inside)
    tol2 = float(tol) * float(tol)
    for idx in range(poly.shape[0] - 1):
        a = poly[idx]
        b = poly[idx + 1]
        ab = b - a
        denom = torch.sum(ab * ab)
        if float(denom.detach().cpu()) <= 0.0:
            d2 = torch.sum((points - a) ** 2, dim=1)
        else:
            u = torch.clamp(torch.sum((points - a) * ab, dim=1) / denom, 0.0, 1.0)
            nearest = a + u[:, None] * ab
            d2 = torch.sum((points - nearest) ** 2, dim=1)
        on_edge |= d2 <= tol2
    return inside | on_edge


def _find_magnetic_axis_gpu(
    *,
    psi_t,
    grid: Grid2D,
    center: tuple[float, float],
    limiter_poly: np.ndarray | None,
    limiter_tol: float,
    runtime: _TorchRuntime,
) -> _MagneticAxis:
    torch = runtime.torch
    finite = torch.isfinite(psi_t)
    if tuple(psi_t.shape) != tuple(grid.shape) or not bool(torch.any(finite).item()):
        raise BoundaryNotFoundError("No finite psi values available for magnetic axis search")

    allowed = finite.clone()
    if limiter_poly is not None:
        allowed &= _limiter_mask_gpu(grid, limiter_poly, limiter_tol, runtime)
        if not bool(torch.any(allowed).item()):
            raise BoundaryNotFoundError("No finite psi values inside limiter for magnetic axis search")

    nz, nr = psi_t.shape
    r, z = _grid_tensors(grid, runtime)
    candidates: list[tuple[float, float, float, str, int, int]] = []
    if nz >= 3 and nr >= 3:
        import torch.nn.functional as F

        x = psi_t[None, None, :, :]
        max3 = F.max_pool2d(torch.where(torch.isfinite(x), x, torch.full_like(x, -torch.inf)), 3, stride=1, padding=1)[0, 0]
        min3 = -F.max_pool2d(torch.where(torch.isfinite(x), -x, torch.full_like(x, -torch.inf)), 3, stride=1, padding=1)[0, 0]
        interior = torch.zeros_like(allowed)
        interior[1:-1, 1:-1] = True
        maxima = allowed & interior & (psi_t >= max3)
        minima = allowed & interior & (psi_t <= min3)
        for kind, mask in (("maximum", maxima), ("minimum", minima)):
            idx = torch.nonzero(mask, as_tuple=False).detach().cpu().numpy()
            for j, i in idx:
                candidates.append((float(r[int(i)].detach().cpu()), float(z[int(j)].detach().cpu()), float(psi_t[int(j), int(i)].detach().cpu()), kind, int(j), int(i)))

    if candidates:
        return _select_axis_candidate_gpu(candidates, center)

    masked = torch.where(allowed, psi_t, torch.full_like(psi_t, torch.nan))
    finite_masked = torch.isfinite(masked)
    if not bool(torch.any(finite_masked).item()):
        raise BoundaryNotFoundError("No finite psi values available for magnetic axis search")
    max_flat = int(torch.nanargmax(masked).detach().cpu())
    min_flat = int(torch.nanargmin(masked).detach().cpu())
    max_j, max_i = divmod(max_flat, nr)
    min_j, min_i = divmod(min_flat, nr)
    fallback = [
        (float(r[max_i].detach().cpu()), float(z[max_j].detach().cpu()), float(masked[max_j, max_i].detach().cpu()), "maximum", max_j, max_i),
        (float(r[min_i].detach().cpu()), float(z[min_j].detach().cpu()), float(masked[min_j, min_i].detach().cpu()), "minimum", min_j, min_i),
    ]
    return _select_axis_candidate_gpu(fallback, center)


def _select_axis_candidate_gpu(candidates: list[tuple[float, float, float, str, int, int]], center: tuple[float, float]) -> _MagneticAxis:
    c = np.asarray(center, dtype=float)
    best = min(candidates, key=lambda item: (float(np.linalg.norm(np.asarray(item[:2], dtype=float) - c)), item[4], item[5], item[3]))
    return _MagneticAxis(point=(best[0], best[1]), level=float(best[2]), kind=best[3])  # type: ignore[arg-type]


def _limited_boundary_gpu(
    *,
    psi_t,
    psi_np: np.ndarray,
    grid: Grid2D,
    axis: _MagneticAxis,
    limiter_poly: np.ndarray | None,
    limiter_tol: float,
    n_levels: int,
    runtime: _TorchRuntime,
) -> tuple[np.ndarray, float] | None:
    if limiter_poly is None:
        raise RuntimeError("Limited boundary mode requires limiter geometry")
    contact = _limited_limiter_contact_boundary_gpu(
        psi_t=psi_t,
        grid=grid,
        axis=axis,
        limiter_poly=limiter_poly,
        limiter_tol=limiter_tol,
        runtime=runtime,
    )
    if contact is not None:
        return contact
    return _outermost_limited_boundary_gpu(
        psi_t=psi_t,
        psi_np=psi_np,
        grid=grid,
        axis=axis,
        limiter_poly=limiter_poly,
        limiter_tol=limiter_tol,
        n_levels=n_levels,
        runtime=runtime,
    )


def _limited_limiter_contact_boundary_gpu(
    *,
    psi_t,
    grid: Grid2D,
    axis: _MagneticAxis,
    limiter_poly: np.ndarray,
    limiter_tol: float,
    runtime: _TorchRuntime,
) -> tuple[np.ndarray, float] | None:
    with _time_block("limited_contact_levels"):
        limiter_points = _sample_limiter_points(limiter_poly, grid)
        limiter_psi = _sample_psi_bilinear_gpu(psi_t, grid, limiter_points, runtime)
        valid = np.isfinite(limiter_psi)
    if not bool(np.any(valid)):
        return None

    points = limiter_points[valid]
    values = limiter_psi[valid]
    rounded_values = np.round(values, 14)
    touch_tol = 2.5 * float(max(abs(float(grid.r.step)), abs(float(grid.z.step)), float(limiter_tol)))
    levels = _ordered_limiter_contact_levels(values, axis)

    for level_chunk in _level_chunks(levels):
        contours_by_level = _contours_at_levels_gpu(psi_t, grid, level_chunk, runtime)
        for contact_level, contours in zip(level_chunk, contours_by_level, strict=True):
            result = _select_limited_contact_candidate(
                contours=contours,
                contact_level=float(contact_level),
                points=points,
                rounded_values=rounded_values,
                axis=axis,
                limiter_poly=limiter_poly,
                limiter_tol=limiter_tol,
                touch_tol=touch_tol,
            )
            if result is not None:
                return result
    return None


def _select_limited_contact_candidate(
    *,
    contours: list[np.ndarray],
    contact_level: float,
    points: np.ndarray,
    rounded_values: np.ndarray,
    axis: _MagneticAxis,
    limiter_poly: np.ndarray,
    limiter_tol: float,
    touch_tol: float,
) -> tuple[np.ndarray, float] | None:
        contact_points = points[rounded_values == float(contact_level)]
        if contact_points.shape[0] == 0:
            return None
        candidates: list[tuple[np.ndarray, float]] = []
        for poly in contours:
            with _time_block("candidate_filtering"):
                if not _is_closed_poly(poly):
                    continue
                if not _encloses_center(poly, axis.point):
                    continue
                closed = _close_poly(poly)
                if not _poly_fits_limiter(closed, limiter_poly, tol=limiter_tol):
                    continue
                contact_dist = _polyline_to_points_distance(closed, contact_points)
                if contact_dist > touch_tol:
                    continue
                candidates.append((closed, float(contact_dist)))
        if not candidates:
            return None
        best_poly, _dist = min(candidates, key=lambda item: item[1])
        return best_poly, float(contact_level)


def _outermost_limited_boundary_gpu(
    *,
    psi_t,
    psi_np: np.ndarray,
    grid: Grid2D,
    axis: _MagneticAxis,
    limiter_poly: np.ndarray,
    limiter_tol: float,
    n_levels: int,
    runtime: _TorchRuntime,
) -> tuple[np.ndarray, float] | None:
    levels = _limited_candidate_levels(
        psi=psi_np,
        grid=grid,
        axis=axis,
        limiter_poly=limiter_poly,
        limiter_tol=limiter_tol,
        n_levels=n_levels,
    )
    if levels.size == 0:
        return None
    candidates: list[tuple[np.ndarray, float, float]] = []
    for level_chunk in _level_chunks(levels):
        contours_by_level = _contours_at_levels_gpu(psi_t, grid, level_chunk, runtime)
        for level, contours in zip(level_chunk, contours_by_level, strict=True):
            for poly in contours:
                with _time_block("candidate_filtering"):
                    if not _is_closed_poly(poly):
                        continue
                    if not _encloses_center(poly, axis.point):
                        continue
                    closed = _close_poly(poly)
                    if not _poly_fits_limiter(closed, limiter_poly, tol=limiter_tol):
                        continue
                    area = abs(_poly_area(closed))
                    if np.isfinite(area) and area > 0.0:
                        candidates.append((closed, float(level), float(area)))
    if not candidates:
        return None
    best_poly, best_level, _area = max(candidates, key=lambda item: item[2])
    return best_poly, best_level


def _separatrix_boundary_gpu(
    *,
    psi_t,
    psi_np: np.ndarray,
    grid: Grid2D,
    axis: tuple[float, float],
    limiter_poly: np.ndarray | None,
    limiter_tol: float,
    runtime: _TorchRuntime,
) -> tuple[np.ndarray, float] | None:
    min_sep = 2.0 * float(max(abs(float(grid.r.step)), abs(float(grid.z.step))))
    with _time_block("xpoint_search"):
        x_points = _find_x_points_gpu(psi_t, grid, max_points=8, min_separation=min_sep, runtime=runtime)
    if x_points.shape[0] == 0:
        return None
    candidates: list[tuple[np.ndarray, float, float]] = []
    for point in x_points:
        level = _psi_at_nearest_grid_point(psi_np, grid, (float(point[0]), float(point[1])))
        if not np.isfinite(level):
            continue
        for poly in _contours_at_level_gpu(psi_t, grid, float(level), runtime):
            with _time_block("candidate_filtering"):
                if not _is_closed_poly(poly):
                    continue
                if not _encloses_center(poly, axis):
                    continue
                closed = _close_poly(poly)
                if _point_to_polyline_distance(np.asarray(point, dtype=float), closed) > min_sep:
                    continue
                if limiter_poly is not None and not _poly_fits_limiter(closed, limiter_poly, tol=limiter_tol):
                    continue
                area = abs(_poly_area(closed))
                if np.isfinite(area) and area > 0.0:
                    candidates.append((closed, float(level), float(area)))
    if not candidates:
        return None
    best_poly, best_level, _area = max(candidates, key=lambda item: item[2])
    return best_poly, best_level


def _sample_psi_bilinear_gpu(psi_t, grid: Grid2D, points: np.ndarray, runtime: _TorchRuntime) -> np.ndarray:
    torch = runtime.torch
    pts = torch.as_tensor(np.asarray(points, dtype=float), dtype=torch.float64, device=runtime.device)
    r, z = _grid_tensors(grid, runtime)
    out = torch.full((pts.shape[0],), torch.nan, dtype=torch.float64, device=runtime.device)
    in_bounds = (pts[:, 0] >= r[0]) & (pts[:, 0] <= r[-1]) & (pts[:, 1] >= z[0]) & (pts[:, 1] <= z[-1])
    if not bool(torch.any(in_bounds).item()):
        return out.detach().cpu().numpy()
    idx = torch.nonzero(in_bounds, as_tuple=False).reshape(-1)
    p = pts[idx]
    i0 = torch.searchsorted(r, p[:, 0].contiguous(), right=True) - 1
    j0 = torch.searchsorted(z, p[:, 1].contiguous(), right=True) - 1
    i0 = torch.clamp(i0, 0, grid.r.size - 2)
    j0 = torch.clamp(j0, 0, grid.z.size - 2)
    r0 = r[i0]
    r1 = r[i0 + 1]
    z0 = z[j0]
    z1 = z[j0 + 1]
    valid = (r1 > r0) & (z1 > z0)
    if bool(torch.any(valid).item()):
        vi = idx[valid]
        i = i0[valid]
        j = j0[valid]
        ar = (p[valid, 0] - r0[valid]) / (r1[valid] - r0[valid])
        az = (p[valid, 1] - z0[valid]) / (z1[valid] - z0[valid])
        q00 = psi_t[j, i]
        q10 = psi_t[j, i + 1]
        q01 = psi_t[j + 1, i]
        q11 = psi_t[j + 1, i + 1]
        out[vi] = (1.0 - ar) * (1.0 - az) * q00 + ar * (1.0 - az) * q10 + (1.0 - ar) * az * q01 + ar * az * q11
    return out.detach().cpu().numpy()


def _contours_at_level_gpu(psi_t, grid: Grid2D, level: float, runtime: _TorchRuntime) -> list[np.ndarray]:
    with _time_block("contours_at_level"):
        segments = _marching_square_segments_gpu(psi_t, grid, float(level), runtime)
        return _stitch_segments_to_polylines(segments)


def _contours_at_levels_gpu(psi_t, grid: Grid2D, levels: np.ndarray, runtime: _TorchRuntime) -> list[list[np.ndarray]]:
    levels_arr = np.ascontiguousarray(np.asarray(levels, dtype=float).reshape(-1))
    if levels_arr.size == 0:
        return []
    if levels_arr.size == 1:
        return [_contours_at_level_gpu(psi_t, grid, float(levels_arr[0]), runtime)]
    with _time_block("contours_at_level"):
        grouped_segments = _marching_square_segments_for_levels_gpu(psi_t, grid, levels_arr, runtime)
        return [_stitch_segments_to_polylines(grouped_segments[idx]) for idx in range(levels_arr.size)]


def _level_chunks(levels: np.ndarray, *, chunk_size: int = 64):
    arr = np.ascontiguousarray(np.asarray(levels, dtype=float).reshape(-1))
    for start in range(0, arr.size, int(chunk_size)):
        yield arr[start : start + int(chunk_size)]


def _marching_square_segments_gpu(psi_t, grid: Grid2D, level: float, runtime: _TorchRuntime) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    torch = runtime.torch
    r, z = _grid_tensors(grid, runtime)
    bl = psi_t[:-1, :-1]
    br = psi_t[:-1, 1:]
    tr = psi_t[1:, 1:]
    tl = psi_t[1:, :-1]
    finite = torch.isfinite(bl) & torch.isfinite(br) & torch.isfinite(tr) & torch.isfinite(tl)
    above = [bl >= level, br >= level, tr >= level, tl >= level]
    case = above[0].to(torch.int16) + 2 * above[1].to(torch.int16) + 4 * above[2].to(torch.int16) + 8 * above[3].to(torch.int16)
    active = finite & (case != 0) & (case != 15)
    indices = torch.nonzero(active, as_tuple=False)
    if indices.numel() == 0:
        return []
    idx_np = indices.detach().cpu().numpy()
    case_np = case[active].detach().cpu().numpy().astype(int)
    bl_np = bl[active].detach().cpu().numpy()
    br_np = br[active].detach().cpu().numpy()
    tr_np = tr[active].detach().cpu().numpy()
    tl_np = tl[active].detach().cpu().numpy()
    r_np = r.detach().cpu().numpy()
    z_np = z.detach().cpu().numpy()
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for (j_raw, i_raw), c, v_bl, v_br, v_tr, v_tl in zip(idx_np, case_np, bl_np, br_np, tr_np, tl_np, strict=True):
        j = int(j_raw)
        i = int(i_raw)
        pairs = _case_edge_pairs(c, float(v_bl), float(v_br), float(v_tr), float(v_tl), float(level))
        if not pairs:
            continue
        pts = {
            0: _interp_edge((r_np[i], z_np[j]), (r_np[i + 1], z_np[j]), float(v_bl), float(v_br), float(level)),
            1: _interp_edge((r_np[i + 1], z_np[j]), (r_np[i + 1], z_np[j + 1]), float(v_br), float(v_tr), float(level)),
            2: _interp_edge((r_np[i], z_np[j + 1]), (r_np[i + 1], z_np[j + 1]), float(v_tl), float(v_tr), float(level)),
            3: _interp_edge((r_np[i], z_np[j]), (r_np[i], z_np[j + 1]), float(v_bl), float(v_tl), float(level)),
        }
        for ea, eb in pairs:
            segments.append((pts[ea], pts[eb]))
    return segments


def _marching_square_segments_for_levels_gpu(psi_t, grid: Grid2D, levels: np.ndarray, runtime: _TorchRuntime) -> list[list[tuple[tuple[float, float], tuple[float, float]]]]:
    torch = runtime.torch
    levels_arr = np.ascontiguousarray(np.asarray(levels, dtype=float).reshape(-1))
    level_t = torch.as_tensor(levels_arr, dtype=torch.float64, device=runtime.device)
    r, z = _grid_tensors(grid, runtime)
    bl = psi_t[:-1, :-1]
    br = psi_t[:-1, 1:]
    tr = psi_t[1:, 1:]
    tl = psi_t[1:, :-1]
    finite = torch.isfinite(bl) & torch.isfinite(br) & torch.isfinite(tr) & torch.isfinite(tl)
    bl_above = bl[None, :, :] >= level_t[:, None, None]
    br_above = br[None, :, :] >= level_t[:, None, None]
    tr_above = tr[None, :, :] >= level_t[:, None, None]
    tl_above = tl[None, :, :] >= level_t[:, None, None]
    case = bl_above.to(torch.int16) + 2 * br_above.to(torch.int16) + 4 * tr_above.to(torch.int16) + 8 * tl_above.to(torch.int16)
    active = finite[None, :, :] & (case != 0) & (case != 15)
    indices = torch.nonzero(active, as_tuple=False)
    grouped: list[list[tuple[tuple[float, float], tuple[float, float]]]] = [[] for _ in range(levels_arr.size)]
    if indices.numel() == 0:
        return grouped

    l_idx = indices[:, 0]
    j_idx = indices[:, 1]
    i_idx = indices[:, 2]
    case_np = case[l_idx, j_idx, i_idx].detach().cpu().numpy().astype(int)
    l_np = l_idx.detach().cpu().numpy().astype(int)
    j_np = j_idx.detach().cpu().numpy().astype(int)
    i_np = i_idx.detach().cpu().numpy().astype(int)
    bl_np = bl[j_idx, i_idx].detach().cpu().numpy()
    br_np = br[j_idx, i_idx].detach().cpu().numpy()
    tr_np = tr[j_idx, i_idx].detach().cpu().numpy()
    tl_np = tl[j_idx, i_idx].detach().cpu().numpy()
    r_np = r.detach().cpu().numpy()
    z_np = z.detach().cpu().numpy()

    for l, j, i, c, v_bl, v_br, v_tr, v_tl in zip(l_np, j_np, i_np, case_np, bl_np, br_np, tr_np, tl_np, strict=True):
        level = float(levels_arr[int(l)])
        pairs = _case_edge_pairs(int(c), float(v_bl), float(v_br), float(v_tr), float(v_tl), level)
        if not pairs:
            continue
        pts = {
            0: _interp_edge((r_np[i], z_np[j]), (r_np[i + 1], z_np[j]), float(v_bl), float(v_br), level),
            1: _interp_edge((r_np[i + 1], z_np[j]), (r_np[i + 1], z_np[j + 1]), float(v_br), float(v_tr), level),
            2: _interp_edge((r_np[i], z_np[j + 1]), (r_np[i + 1], z_np[j + 1]), float(v_tl), float(v_tr), level),
            3: _interp_edge((r_np[i], z_np[j]), (r_np[i], z_np[j + 1]), float(v_bl), float(v_tl), level),
        }
        for ea, eb in pairs:
            grouped[int(l)].append((pts[ea], pts[eb]))
    return grouped


def _case_edge_pairs(case_id: int, bl: float, br: float, tr: float, tl: float, level: float) -> tuple[tuple[int, int], ...]:
    lookup: dict[int, tuple[tuple[int, int], ...]] = {
        1: ((3, 0),),
        2: ((0, 1),),
        3: ((3, 1),),
        4: ((1, 2),),
        6: ((0, 2),),
        7: ((3, 2),),
        8: ((2, 3),),
        9: ((0, 2),),
        11: ((1, 2),),
        12: ((1, 3),),
        13: ((0, 1),),
        14: ((3, 0),),
    }
    if case_id in lookup:
        return lookup[case_id]
    if case_id in {5, 10}:
        center = 0.25 * (float(bl) + float(br) + float(tr) + float(tl))
        center_above = center >= float(level)
        if case_id == 5:
            return ((3, 2), (0, 1)) if center_above else ((3, 0), (1, 2))
        return ((0, 3), (1, 2)) if center_above else ((0, 1), (2, 3))
    return ()


def _interp_edge(a: tuple[float, float], b: tuple[float, float], va: float, vb: float, level: float) -> tuple[float, float]:
    denom = float(vb) - float(va)
    if abs(denom) <= 1e-30:
        t = 0.5
    else:
        t = (float(level) - float(va)) / denom
    t = min(max(float(t), 0.0), 1.0)
    return (float(a[0]) + t * (float(b[0]) - float(a[0])), float(a[1]) + t * (float(b[1]) - float(a[1])))


def _stitch_segments_to_polylines(segments: list[tuple[tuple[float, float], tuple[float, float]]]) -> list[np.ndarray]:
    if not segments:
        return []
    adjacency: dict[tuple[int, int], list[int]] = defaultdict(list)
    keys: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for idx, (a, b) in enumerate(segments):
        ka = _point_key(a)
        kb = _point_key(b)
        keys.append((ka, kb))
        adjacency[ka].append(idx)
        adjacency[kb].append(idx)

    used = np.zeros((len(segments),), dtype=bool)
    polylines: list[np.ndarray] = []
    for start_idx in range(len(segments)):
        if used[start_idx]:
            continue
        used[start_idx] = True
        a, b = segments[start_idx]
        ka, kb = keys[start_idx]
        coords = [a, b]
        _extend_polyline(coords, kb, adjacency, keys, segments, used, append=True)
        _extend_polyline(coords, ka, adjacency, keys, segments, used, append=False)
        arr = np.asarray(coords, dtype=float)
        if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] == 2:
            polylines.append(arr)
    return polylines


def _extend_polyline(
    coords: list[tuple[float, float]],
    current_key: tuple[int, int],
    adjacency: dict[tuple[int, int], list[int]],
    keys: list[tuple[tuple[int, int], tuple[int, int]]],
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    used: np.ndarray,
    *,
    append: bool,
) -> None:
    while True:
        next_idx = None
        for candidate in adjacency.get(current_key, []):
            if not bool(used[candidate]):
                next_idx = int(candidate)
                break
        if next_idx is None:
            return
        used[next_idx] = True
        ka, kb = keys[next_idx]
        a, b = segments[next_idx]
        if ka == current_key:
            next_point = b
            current_key = kb
        else:
            next_point = a
            current_key = ka
        if append:
            coords.append(next_point)
        else:
            coords.insert(0, next_point)


def _point_key(point: tuple[float, float]) -> tuple[int, int]:
    return (int(round(float(point[0]) * 1.0e12)), int(round(float(point[1]) * 1.0e12)))


def _find_x_points_gpu(psi_t, grid: Grid2D, *, max_points: int, min_separation: float | None, runtime: _TorchRuntime) -> np.ndarray:
    torch = runtime.torch
    if psi_t.shape[0] < 3 or psi_t.shape[1] < 3:
        return np.zeros((0, 2), dtype=float)
    dz = float(grid.z.step)
    dr = float(grid.r.step)
    dpsi_dz = torch.gradient(psi_t, spacing=(dz, dr), edge_order=2)[0]
    dpsi_dr = torch.gradient(psi_t, spacing=(dz, dr), edge_order=2)[1]
    d2psi_dz2, d2psi_dzdr = torch.gradient(dpsi_dz, spacing=(dz, dr), edge_order=2)
    d2psi_drdz, d2psi_dr2 = torch.gradient(dpsi_dr, spacing=(dz, dr), edge_order=2)
    grad_norm = torch.hypot(dpsi_dr, dpsi_dz)
    det_hessian = d2psi_dr2 * d2psi_dz2 - d2psi_drdz * d2psi_dzdr

    import torch.nn.functional as F

    x = grad_norm[None, None, :, :]
    local_min = grad_norm <= (-F.max_pool2d(torch.where(torch.isfinite(x), -x, torch.full_like(x, -torch.inf)), 3, stride=1, padding=1)[0, 0])
    interior = torch.zeros_like(local_min)
    interior[1:-1, 1:-1] = True
    mask = interior & torch.isfinite(grad_norm) & torch.isfinite(det_hessian) & (det_hessian < 0.0) & local_min
    idx = torch.nonzero(mask, as_tuple=False)
    if idx.numel() == 0:
        return np.zeros((0, 2), dtype=float)
    scores = grad_norm[mask]
    order = torch.argsort(scores)
    idx_np = idx[order].detach().cpu().numpy()
    score_np = scores[order].detach().cpu().numpy()
    r = np.asarray(grid.r.coords(), dtype=float)
    z = np.asarray(grid.z.coords(), dtype=float)
    candidates = [(float(score), float(r[int(i)]), float(z[int(j)])) for score, (j, i) in zip(score_np, idx_np, strict=True)]
    return _select_separated_xpoints(candidates, max_points=max_points, min_separation=min_separation)


def _select_separated_xpoints(candidates: list[tuple[float, float, float]], *, max_points: int, min_separation: float | None) -> np.ndarray:
    limit = max(int(max_points), 0)
    if limit == 0 or not candidates:
        return np.zeros((0, 2), dtype=float)
    sep = 0.0 if min_separation is None else max(float(min_separation), 0.0)
    selected: list[tuple[float, float]] = []
    for _score, r_value, z_value in candidates:
        point = (float(r_value), float(z_value))
        if sep > 0.0 and any(np.hypot(point[0] - old[0], point[1] - old[1]) < sep for old in selected):
            continue
        selected.append(point)
        if len(selected) >= limit:
            break
    return np.asarray(selected, dtype=float).reshape(-1, 2)
