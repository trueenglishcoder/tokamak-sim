"""Exact spline primitives for the batched topology-first GPU LCFS extractor.

The equilibrium in the simulator is a linear combination of static basis
fields.  At simulator construction time each basis field is converted to the
same cell-local bicubic polynomial used by the CPU ``RectBivariateSpline``.
At runtime values, gradients and Hessians are evaluated directly on the target
Torch device from the current basis amplitudes.  No CPU extraction or ray-first
boundary search is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RectBivariateSpline

from tokamak_control.core.grid import Grid2D


@dataclass(frozen=True, slots=True, repr=True)
class ExactSplineGpuGeometry:
    """Static exact bicubic representation of the equilibrium basis fields."""

    coefficients: object
    limiter_segment_basis_values: object
    r0: float
    z0: float
    dr: float
    dz: float
    nr: int
    nz: int
    topology_r_lower: float
    topology_r_upper: float
    topology_z_lower: float
    topology_z_upper: float
    limiter_vertices: object
    limiter_segment_starts: object
    limiter_segment_vectors: object
    limiter_sample_t: object
    limiter_sample_points: object
    limiter_sample_valid: object
    native_points: object


@dataclass(frozen=True, slots=True, repr=True)
class ExactSplineGpuField:
    """Per-step batched spline coefficients and limiter values.

    The simulator equilibrium is a linear combination of static basis fields.
    Combining those basis coefficients once per simulation step avoids repeating
    the same ``B x K x cells`` contraction in every Newton, root-refinement and
    topology evaluation.
    """

    geometry: ExactSplineGpuGeometry
    coefficients: object
    limiter_segment_values: object


def combine_exact_spline_gpu_field(
    *,
    amplitudes: object,
    geometry: ExactSplineGpuGeometry,
) -> ExactSplineGpuField:
    """Combine static basis splines once for one batched equilibrium state."""
    import torch

    amp = torch.as_tensor(
        amplitudes,
        dtype=geometry.coefficients.dtype,
        device=geometry.coefficients.device,
    )
    if amp.ndim != 2:
        raise ValueError(f"amplitudes must have shape (B,K), got {tuple(amp.shape)}")
    if int(amp.shape[1]) != int(geometry.coefficients.shape[0]):
        raise ValueError("amplitude count does not match exact spline basis")
    coefficient_flat = geometry.coefficients.reshape(
        int(geometry.coefficients.shape[0]),
        -1,
        4,
        4,
    )
    return ExactSplineGpuField(
        geometry=geometry,
        coefficients=torch.einsum("bk,kcij->bcij", amp, coefficient_flat),
        limiter_segment_values=torch.einsum(
            "bk,ksm->bsm",
            amp,
            geometry.limiter_segment_basis_values,
        ),
    )


def repeat_exact_spline_gpu_field(
    field: ExactSplineGpuField,
    repeats: int,
) -> ExactSplineGpuField:
    """Repeat every batch lane for a small parallel candidate block."""
    count = int(repeats)
    if count <= 0:
        raise ValueError("repeats must be positive")
    if count == 1:
        return field
    return ExactSplineGpuField(
        geometry=field.geometry,
        coefficients=field.coefficients.repeat_interleave(count, dim=0),
        limiter_segment_values=field.limiter_segment_values.repeat_interleave(count, dim=0),
    )


def prepare_exact_spline_gpu_geometry(
    *,
    grid: Grid2D,
    basis_fields: np.ndarray,
    limiter_samples: np.ndarray,
    limiter_poly: np.ndarray | None = None,
    device: object,
    dtype: object,
) -> ExactSplineGpuGeometry:
    """Convert static basis fields to exact local bicubic power coefficients."""
    import torch

    basis = np.asarray(basis_fields, dtype=np.float64)
    if basis.ndim != 3 or tuple(basis.shape[1:]) != tuple(grid.shape):
        raise ValueError(
            f"basis_fields must have shape (K, {grid.shape[0]}, {grid.shape[1]}), "
            f"got {basis.shape}"
        )
    limiter = np.asarray(limiter_samples, dtype=np.float64).reshape(-1, 2)
    physical_limiter = np.asarray(
        limiter if limiter_poly is None else limiter_poly,
        dtype=np.float64,
    ).reshape(-1, 2)
    if physical_limiter.shape[0] < 3:
        raise ValueError("exact equilibrium LCFS requires physical limiter vertices")
    if not np.allclose(physical_limiter[0], physical_limiter[-1], rtol=0.0, atol=1.0e-12):
        physical_limiter = np.vstack((physical_limiter, physical_limiter[0]))
    r = np.asarray(grid.r.coords(), dtype=np.float64)
    z = np.asarray(grid.z.coords(), dtype=np.float64)
    if r.size < 4 or z.size < 4:
        raise ValueError("exact equilibrium LCFS requires at least a 4x4 grid")

    sample = np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=np.float64)
    vandermonde = np.vander(sample, N=4, increasing=True)
    inverse = np.linalg.inv(vandermonde)
    cell_r = r[:-1][None, :, None, None] + float(grid.r.step) * sample[None, None, :, None]
    cell_z = z[:-1][:, None, None, None] + float(grid.z.step) * sample[None, None, None, :]
    sample_r = np.broadcast_to(cell_r, (z.size - 1, r.size - 1, 4, 4)).reshape(-1)
    sample_z = np.broadcast_to(cell_z, (z.size - 1, r.size - 1, 4, 4)).reshape(-1)

    coefficients = np.empty(
        (basis.shape[0], z.size - 1, r.size - 1, 4, 4),
        dtype=np.float64,
    )
    splines: list[RectBivariateSpline] = []
    for index, field in enumerate(basis):
        spline = RectBivariateSpline(r, z, field.T, kx=3, ky=3, s=0.0)
        splines.append(spline)
        values = np.asarray(spline.ev(sample_r, sample_z), dtype=np.float64).reshape(
            z.size - 1,
            r.size - 1,
            4,
            4,
        )
        coefficients[index] = np.einsum(
            "pa,ijab,qb->ijpq",
            inverse,
            values,
            inverse,
            optimize=True,
        )

    grid_scale = max(abs(float(grid.r.step)), abs(float(grid.z.step)))
    segment_starts = physical_limiter[:-1]
    segment_vectors = physical_limiter[1:] - physical_limiter[:-1]
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    segment_sample_counts = np.maximum(
        np.ceil(segment_lengths / max(0.25 * grid_scale, 1.0e-12)).astype(np.int64),
        12,
    )
    maximum_segment_samples = int(np.max(segment_sample_counts)) + 1
    segment_sample_t = np.zeros((segment_starts.shape[0], maximum_segment_samples), dtype=np.float64)
    segment_sample_points = np.zeros((segment_starts.shape[0], maximum_segment_samples, 2), dtype=np.float64)
    segment_sample_valid = np.zeros((segment_starts.shape[0], maximum_segment_samples), dtype=bool)
    for segment_index, sample_count in enumerate(segment_sample_counts):
        values = np.linspace(0.0, 1.0, int(sample_count) + 1, dtype=np.float64)
        count = values.size
        segment_sample_t[segment_index, :count] = values
        segment_sample_points[segment_index, :count] = (
            segment_starts[segment_index, None, :]
            + values[:, None] * segment_vectors[segment_index, None, :]
        )
        segment_sample_valid[segment_index, :count] = True

    flat_segment_points = segment_sample_points.reshape(-1, 2)
    segment_basis_values = np.empty(
        (basis.shape[0], segment_starts.shape[0], maximum_segment_samples),
        dtype=np.float64,
    )
    for index, spline in enumerate(splines):
        segment_basis_values[index] = np.asarray(
            spline.ev(flat_segment_points[:, 0], flat_segment_points[:, 1]),
            dtype=np.float64,
        ).reshape(segment_starts.shape[0], maximum_segment_samples)

    R_native, Z_native = grid.mesh()
    native_points = np.column_stack((R_native.reshape(-1), Z_native.reshape(-1)))
    return ExactSplineGpuGeometry(
        coefficients=torch.as_tensor(coefficients, dtype=dtype, device=device),
        limiter_segment_basis_values=torch.as_tensor(
            segment_basis_values,
            dtype=dtype,
            device=device,
        ),
        r0=float(r[0]),
        z0=float(z[0]),
        dr=float(grid.r.step),
        dz=float(grid.z.step),
        nr=int(r.size),
        nz=int(z.size),
        topology_r_lower=max(float(r[0]), float(np.min(physical_limiter[:, 0])) - 1.25 * grid_scale),
        topology_r_upper=min(float(r[-1]), float(np.max(physical_limiter[:, 0])) + 1.25 * grid_scale),
        topology_z_lower=max(float(z[0]), float(np.min(physical_limiter[:, 1])) - 1.25 * grid_scale),
        topology_z_upper=min(float(z[-1]), float(np.max(physical_limiter[:, 1])) + 1.25 * grid_scale),
        limiter_vertices=torch.as_tensor(physical_limiter, dtype=dtype, device=device),
        limiter_segment_starts=torch.as_tensor(segment_starts, dtype=dtype, device=device),
        limiter_segment_vectors=torch.as_tensor(segment_vectors, dtype=dtype, device=device),
        limiter_sample_t=torch.as_tensor(segment_sample_t, dtype=dtype, device=device),
        limiter_sample_points=torch.as_tensor(segment_sample_points, dtype=dtype, device=device),
        limiter_sample_valid=torch.as_tensor(segment_sample_valid, dtype=torch.bool, device=device),
        native_points=torch.as_tensor(native_points, dtype=dtype, device=device),
    )


def _exact_spline_cells(
    *,
    field: ExactSplineGpuField,
    points: object,
) -> tuple[object, object, object, object, bool]:
    """Gather local bicubic coefficients and normalized cell coordinates."""
    import torch

    coefficients = torch.as_tensor(field.coefficients)
    geometry = field.geometry
    pts = torch.as_tensor(points, dtype=coefficients.dtype, device=coefficients.device)
    squeeze = False
    if pts.ndim == 2:
        pts = pts[:, None, :]
        squeeze = True
    if pts.ndim != 3 or int(pts.shape[2]) != 2:
        raise ValueError(f"points must have shape (B,N,2) or (B,2), got {tuple(pts.shape)}")
    if int(coefficients.shape[0]) != int(pts.shape[0]):
        raise ValueError("combined spline and points batch dimensions differ")

    R = pts[:, :, 0]
    Z = pts[:, :, 1]
    r_max = geometry.r0 + geometry.dr * float(geometry.nr - 1)
    z_max = geometry.z0 + geometry.dz * float(geometry.nz - 1)
    contains = (
        torch.isfinite(R)
        & torch.isfinite(Z)
        & (R >= geometry.r0)
        & (R <= r_max)
        & (Z >= geometry.z0)
        & (Z <= z_max)
    )
    ii = torch.clamp(
        torch.floor((R - geometry.r0) / geometry.dr).to(torch.int64),
        0,
        geometry.nr - 2,
    )
    jj = torch.clamp(
        torch.floor((Z - geometry.z0) / geometry.dz).to(torch.int64),
        0,
        geometry.nz - 2,
    )
    u = torch.clamp(
        (R - (geometry.r0 + ii.to(coefficients.dtype) * geometry.dr)) / geometry.dr,
        0.0,
        1.0,
    )
    v = torch.clamp(
        (Z - (geometry.z0 + jj.to(coefficients.dtype) * geometry.dz)) / geometry.dz,
        0.0,
        1.0,
    )
    flat = jj * (geometry.nr - 1) + ii
    local = torch.gather(
        coefficients,
        1,
        flat[:, :, None, None].expand(-1, -1, 4, 4),
    )
    return local, u, v, contains, squeeze


def _cubic_derivative_basis(value: object, maximum_order: int) -> object:
    """Return power-basis rows for orders zero through ``maximum_order``."""
    import torch

    x = torch.as_tensor(value)
    x2 = x * x
    powers = torch.stack((torch.ones_like(x), x, x2, x2 * x), dim=-1)
    order = int(maximum_order)
    if order == 0:
        return powers.unsqueeze(-2)
    basis = torch.zeros(
        powers.shape[:-1] + (order + 1, 4),
        dtype=powers.dtype,
        device=powers.device,
    )
    basis[..., 0, :] = powers
    basis[..., 1, 1:] = powers[..., :3] * torch.as_tensor(
        (1.0, 2.0, 3.0), dtype=powers.dtype, device=powers.device
    )
    if order >= 2:
        basis[..., 2, 2:] = powers[..., :2] * torch.as_tensor(
            (2.0, 6.0), dtype=powers.dtype, device=powers.device
        )
    return basis


def _evaluate_local_bicubic(
    local: object,
    u: object,
    v: object,
    *,
    derivative_order: int,
) -> object:
    """Evaluate all requested local derivatives with two batched 4x4 products."""
    import torch

    u_basis = _cubic_derivative_basis(u, derivative_order)
    v_basis = _cubic_derivative_basis(v, derivative_order).transpose(-1, -2)
    return torch.matmul(torch.matmul(u_basis, torch.as_tensor(local)), v_basis)


def evaluate_exact_spline_value(
    *,
    field: ExactSplineGpuField,
    points: object,
) -> tuple[object, object]:
    """Evaluate only psi from the exact local bicubic polynomial."""
    import torch

    local, u, v, contains, squeeze = _exact_spline_cells(field=field, points=points)
    derivatives = _evaluate_local_bicubic(local, u, v, derivative_order=0)
    value = derivatives[:, :, 0, 0]
    value = torch.where(contains, value, torch.full_like(value, float("nan")))
    if squeeze:
        return value[:, 0], contains[:, 0]
    return value, contains


def evaluate_exact_spline_value_gradient(
    *,
    field: ExactSplineGpuField,
    points: object,
) -> tuple[object, object, object]:
    """Evaluate psi and gradient from one pair of batched matrix products."""
    import torch

    local, u, v, contains, squeeze = _exact_spline_cells(field=field, points=points)
    geometry = field.geometry
    derivatives = _evaluate_local_bicubic(local, u, v, derivative_order=1)
    value = derivatives[:, :, 0, 0]
    gradient = torch.stack(
        (
            derivatives[:, :, 1, 0] / geometry.dr,
            derivatives[:, :, 0, 1] / geometry.dz,
        ),
        dim=2,
    )
    value = torch.where(contains, value, torch.full_like(value, float("nan")))
    gradient = torch.where(
        contains[:, :, None],
        gradient,
        torch.full_like(gradient, float("nan")),
    )
    if squeeze:
        return value[:, 0], gradient[:, 0], contains[:, 0]
    return value, gradient, contains


def evaluate_exact_spline(
    *,
    field: ExactSplineGpuField,
    points: object,
) -> tuple[object, object, object, object]:
    """Evaluate psi, gradient and Hessian from exact bicubic coefficients."""
    import torch

    local, u, v, contains, squeeze = _exact_spline_cells(field=field, points=points)
    geometry = field.geometry
    derivatives = _evaluate_local_bicubic(local, u, v, derivative_order=2)
    value = derivatives[:, :, 0, 0]
    d_r = derivatives[:, :, 1, 0] / geometry.dr
    d_z = derivatives[:, :, 0, 1] / geometry.dz
    d_rr = derivatives[:, :, 2, 0] / (geometry.dr * geometry.dr)
    d_rz = derivatives[:, :, 1, 1] / (geometry.dr * geometry.dz)
    d_zz = derivatives[:, :, 0, 2] / (geometry.dz * geometry.dz)
    gradient = torch.stack((d_r, d_z), dim=2)
    hessian = torch.stack(
        (
            torch.stack((d_rr, d_rz), dim=2),
            torch.stack((d_rz, d_zz), dim=2),
        ),
        dim=2,
    )
    value = torch.where(contains, value, torch.full_like(value, float("nan")))
    gradient = torch.where(
        contains[:, :, None],
        gradient,
        torch.full_like(gradient, float("nan")),
    )
    hessian = torch.where(
        contains[:, :, None, None],
        hessian,
        torch.full_like(hessian, float("nan")),
    )
    if squeeze:
        return value[:, 0], gradient[:, 0], hessian[:, 0], contains[:, 0]
    return value, gradient, hessian, contains


def _tailored_coordinates_exact_gpu(
    *,
    lower: float,
    upper: float,
    step: float,
    axis_coordinates: object,
    x_coordinates: object,
    x_valid: object,
) -> tuple[object, object]:
    """Torch translation of ``level_set_graph._tailored_coordinates``.

    The static phased coordinates are identical to the CPU implementation.
    Selected X-points remove the interior of one local element and insert its
    two faces; the magnetic axis is inserted as a grid node unless that would
    split a reserved X-point element.
    """
    import torch

    if upper <= lower or step <= 0.0:
        raise ValueError("invalid topology-grid bounds or step")
    axis = torch.as_tensor(axis_coordinates)
    centers = torch.as_tensor(x_coordinates, dtype=axis.dtype, device=axis.device)
    valid_centers = torch.as_tensor(x_valid, dtype=torch.bool, device=axis.device)
    batch_size = int(axis.shape[0])
    x_count = int(centers.shape[1]) if centers.ndim == 2 else 0

    phase = 0.3819660112501051
    base_values = [float(lower), float(upper)]
    coordinate = float(lower) + phase * float(step)
    while coordinate < float(upper):
        base_values.append(float(coordinate))
        coordinate += float(step)
    base = torch.as_tensor(base_values, dtype=axis.dtype, device=axis.device)
    capacity = int(base.numel()) + 2 * x_count + 1
    values = torch.full((batch_size, capacity), float("inf"), dtype=axis.dtype, device=axis.device)
    present = torch.zeros((batch_size, capacity), dtype=torch.bool, device=axis.device)
    values[:, : int(base.numel())] = base[None, :]
    present[:, : int(base.numel())] = True

    if x_count:
        sorted_centers, order = torch.sort(
            torch.where(valid_centers, centers, torch.full_like(centers, float("inf"))),
            dim=1,
        )
        sorted_valid = torch.gather(valid_centers, 1, order)
        for slot in range(x_count):
            center = sorted_centers[:, slot]
            active = sorted_valid[:, slot] & (center > lower) & (center < upper)
            half = 0.5 * float(step)
            left = torch.clamp(center - half, min=float(lower), max=float(upper))
            right = torch.clamp(center + half, min=float(lower), max=float(upper))
            active &= (right - left) >= 0.5 * float(step)
            remove = active[:, None] & present & (values > left[:, None]) & (values < right[:, None])
            present &= ~remove
            insert = int(base.numel()) + 2 * slot
            values[:, insert] = left
            values[:, insert + 1] = right
            present[:, insert] = active
            present[:, insert + 1] = active

    axis_inside = (axis > lower) & (axis < upper)
    if x_count:
        reserved = torch.any(
            valid_centers & (torch.abs(axis[:, None] - centers) < 0.49 * float(step)),
            dim=1,
        )
    else:
        reserved = torch.zeros_like(axis_inside)
    axis_insert = axis_inside & ~reserved
    values[:, -1] = axis
    present[:, -1] = axis_insert

    sortable = torch.where(present, values, torch.full_like(values, float("inf")))
    ordered = torch.sort(sortable, dim=1).values
    finite = torch.isfinite(ordered)
    tolerance = 1.0e-12 * max(abs(float(lower)), abs(float(upper)), 1.0)
    clipped = torch.clamp(ordered, min=float(lower), max=float(upper))
    unique = finite.clone()
    unique[:, 1:] &= torch.abs(clipped[:, 1:] - clipped[:, :-1]) > float(tolerance)
    rank = torch.cumsum(unique.to(torch.int64), dim=1) - 1
    count = torch.sum(unique.to(torch.int64), dim=1)
    compact = torch.full_like(clipped, float("inf"))
    batch = torch.arange(batch_size, device=axis.device)[:, None].expand_as(rank)
    compact[batch[unique], rank[unique]] = clipped[unique]
    return compact, count


def _exact_edge_roots_gpu(
    *,
    field: ExactSplineGpuField,
    starts: object,
    ends: object,
    start_values: object,
    end_values: object,
    edge_valid: object,
    exhaustive: object,
    level: object,
    tolerance: object,
    start_vertex: object,
    end_vertex: object,
    interior_node_base: int,
    root_iterations: int,
    max_roots: int = 4,
) -> tuple[object, object, object, object]:
    """Find exact bicubic edge roots without refining empty edge slots.

    Ordinary topology edges use the already evaluated values at their two
    endpoints.  Only X-halo edges receive the seven additional interior samples
    used by the CPU extractor.  Sign-changing brackets are then compacted per
    lane, so each bisection evaluates the exact spline only at real brackets
    instead of at every padded topology edge.
    """
    import torch

    coefficients = torch.as_tensor(field.coefficients)
    starts_t = torch.as_tensor(
        starts, dtype=coefficients.dtype, device=coefficients.device
    )
    ends_t = torch.as_tensor(
        ends, dtype=coefficients.dtype, device=coefficients.device
    )
    start_values_t = torch.as_tensor(
        start_values, dtype=coefficients.dtype, device=coefficients.device
    )
    end_values_t = torch.as_tensor(
        end_values, dtype=coefficients.dtype, device=coefficients.device
    )
    valid_t = torch.as_tensor(
        edge_valid, dtype=torch.bool, device=coefficients.device
    )
    exhaustive_t = torch.as_tensor(
        exhaustive, dtype=torch.bool, device=coefficients.device
    ) & valid_t
    level_t = torch.as_tensor(
        level, dtype=coefficients.dtype, device=coefficients.device
    ).reshape(-1)
    tolerance_t = torch.as_tensor(
        tolerance, dtype=coefficients.dtype, device=coefficients.device
    ).reshape(-1)
    batch_size, edge_count, _ = starts_t.shape
    if start_values_t.shape != (batch_size, edge_count):
        raise ValueError("start_values must match the topology edge batch")
    if end_values_t.shape != (batch_size, edge_count):
        raise ValueError("end_values must match the topology edge batch")

    vector = ends_t - starts_t
    sample_t = torch.linspace(
        0.0, 1.0, 9, dtype=coefficients.dtype, device=coefficients.device
    )
    residuals = torch.full(
        (batch_size, edge_count, 9),
        float("nan"),
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    residuals[:, :, 0] = start_values_t - level_t[:, None]
    residuals[:, :, 8] = end_values_t - level_t[:, None]
    sample_available = torch.zeros(
        (batch_size, edge_count, 9), dtype=torch.bool, device=coefficients.device
    )
    sample_available[:, :, 0] = valid_t
    sample_available[:, :, 8] = valid_t

    exhaustive_count = torch.sum(exhaustive_t.to(torch.int64), dim=1)
    exhaustive_capacity = int(torch.max(exhaustive_count).item())
    if exhaustive_capacity > 0:
        exhaustive_order = torch.argsort(
            (~exhaustive_t).to(torch.int64), dim=1, stable=True
        )[:, :exhaustive_capacity]
        compact_exhaustive_valid = torch.gather(
            exhaustive_t, 1, exhaustive_order
        )
        batch_index = torch.arange(
            batch_size, device=coefficients.device
        )[:, None]
        compact_starts = starts_t[batch_index, exhaustive_order]
        compact_vectors = vector[batch_index, exhaustive_order]
        internal_t = sample_t[1:8]
        internal_points = (
            compact_starts[:, :, None, :]
            + internal_t[None, None, :, None]
            * compact_vectors[:, :, None, :]
        )
        internal_values, internal_contains = evaluate_exact_spline_value(
            field=field,
            points=internal_points.reshape(
                batch_size, exhaustive_capacity * 7, 2
            ),
        )
        internal_values = internal_values.reshape(
            batch_size, exhaustive_capacity, 7
        ) - level_t[:, None, None]
        internal_contains = internal_contains.reshape(
            batch_size, exhaustive_capacity, 7
        )
        internal_valid = (
            compact_exhaustive_valid[:, :, None]
            & internal_contains
            & torch.isfinite(internal_values)
        )
        scatter_batch = batch_index[:, :, None].expand(
            batch_size, exhaustive_capacity, 7
        )
        scatter_edge = exhaustive_order[:, :, None].expand_as(scatter_batch)
        scatter_slot = torch.arange(
            1, 8, device=coefficients.device
        )[None, None, :].expand_as(scatter_batch)
        residuals[scatter_batch, scatter_edge, scatter_slot] = torch.where(
            internal_valid,
            internal_values,
            torch.full_like(internal_values, float("nan")),
        )
        sample_available[scatter_batch, scatter_edge, scatter_slot] = (
            internal_valid
        )

    sample_root_valid = (
        sample_available
        & torch.isfinite(residuals)
        & (torch.abs(residuals) <= tolerance_t[:, None, None])
    )

    interval_slot = torch.arange(
        8, dtype=torch.int64, device=coefficients.device
    )[None, None, :]
    interval_enabled = exhaustive_t[:, :, None] | (
        (~exhaustive_t)[:, :, None] & (interval_slot == 0)
    )
    left_index = interval_slot.expand(batch_size, edge_count, -1).clone()
    right_index = left_index + 1
    right_index[:, :, 0] = torch.where(
        exhaustive_t,
        right_index[:, :, 0],
        torch.full_like(right_index[:, :, 0], 8),
    )
    left_value = torch.gather(residuals, 2, left_index)
    right_value = torch.gather(residuals, 2, right_index)
    left_available = torch.gather(sample_available, 2, left_index)
    right_available = torch.gather(sample_available, 2, right_index)
    bracket = (
        valid_t[:, :, None]
        & interval_enabled
        & left_available
        & right_available
        & torch.isfinite(left_value)
        & torch.isfinite(right_value)
        & (left_value * right_value < 0.0)
    )

    bracket_count = torch.sum(bracket.to(torch.int64), dim=(1, 2))
    bracket_capacity = int(torch.max(bracket_count).item())
    bracket_t_dense = torch.full(
        (batch_size, edge_count, 8),
        float("nan"),
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    if bracket_capacity > 0:
        flat_bracket = bracket.reshape(batch_size, edge_count * 8)
        bracket_order = torch.argsort(
            (~flat_bracket).to(torch.int64), dim=1, stable=True
        )[:, :bracket_capacity]
        compact_bracket_valid = torch.gather(
            flat_bracket, 1, bracket_order
        )
        compact_edge = bracket_order // 8
        compact_slot = bracket_order % 8
        compact_left_index = torch.gather(
            left_index.reshape(batch_size, edge_count * 8),
            1,
            bracket_order,
        )
        compact_right_index = torch.gather(
            right_index.reshape(batch_size, edge_count * 8),
            1,
            bracket_order,
        )
        compact_left_t = sample_t[compact_left_index]
        compact_right_t = sample_t[compact_right_index]
        compact_f_left = torch.gather(
            left_value.reshape(batch_size, edge_count * 8),
            1,
            bracket_order,
        )
        compact_batch = torch.arange(
            batch_size, device=coefficients.device
        )[:, None]
        compact_starts = starts_t[compact_batch, compact_edge]
        compact_vectors = vector[compact_batch, compact_edge]
        active = compact_bracket_valid.clone()
        xtol = torch.as_tensor(
            1.0e-13, dtype=coefficients.dtype, device=coefficients.device
        )
        rtol = torch.as_tensor(
            4.0 * np.finfo(np.float64).eps,
            dtype=coefficients.dtype,
            device=coefficients.device,
        )
        for _ in range(max(int(root_iterations), 1)):
            middle_t = 0.5 * (compact_left_t + compact_right_t)
            middle_points = (
                compact_starts
                + middle_t[:, :, None] * compact_vectors
            )
            middle_value, middle_contains = evaluate_exact_spline_value(
                field=field,
                points=middle_points,
            )
            middle_value = middle_value - level_t[:, None]
            same_left = (compact_f_left * middle_value) > 0.0
            move_left = active & middle_contains & same_left
            move_right = active & middle_contains & ~same_left
            compact_left_t = torch.where(
                move_left, middle_t, compact_left_t
            )
            compact_f_left = torch.where(
                move_left, middle_value, compact_f_left
            )
            compact_right_t = torch.where(
                move_right, middle_t, compact_right_t
            )
            active &= middle_contains & (
                (compact_right_t - compact_left_t)
                > (xtol + rtol * torch.abs(middle_t))
            )
        compact_root_t = 0.5 * (
            compact_left_t + compact_right_t
        )
        bracket_t_dense[
            compact_batch.expand_as(compact_edge)[compact_bracket_valid],
            compact_edge[compact_bracket_valid],
            compact_slot[compact_bracket_valid],
        ] = compact_root_t[compact_bracket_valid]

    candidate_t = torch.cat(
        (
            sample_t.reshape(1, 1, 9).expand(
                batch_size, edge_count, -1
            ),
            bracket_t_dense,
        ),
        dim=2,
    )
    candidate_valid = torch.cat(
        (sample_root_valid, torch.isfinite(bracket_t_dense)), dim=2
    )
    sortable = torch.where(
        candidate_valid,
        candidate_t,
        torch.full_like(candidate_t, float("inf")),
    )
    sorted_t = torch.sort(sortable, dim=2).values
    finite = torch.isfinite(sorted_t)
    unique = finite.clone()
    unique[:, :, 1:] &= (
        torch.abs(sorted_t[:, :, 1:] - sorted_t[:, :, :-1]) > 1.0e-9
    )
    rank = torch.cumsum(unique.to(torch.int64), dim=2) - 1
    roots_t = torch.full(
        (batch_size, edge_count, max_roots),
        float("nan"),
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    batch = torch.arange(
        batch_size, device=coefficients.device
    )[:, None, None].expand_as(rank)
    edge = torch.arange(
        edge_count, device=coefficients.device
    )[None, :, None].expand_as(rank)
    keep = unique & (rank < max_roots)
    roots_t[batch[keep], edge[keep], rank[keep]] = torch.clamp(
        sorted_t[keep], 0.0, 1.0
    )
    root_valid = torch.isfinite(roots_t)
    root_points = (
        starts_t[:, :, None, :]
        + roots_t[:, :, :, None] * vector[:, :, None, :]
    )

    start_vertex_t = torch.as_tensor(
        start_vertex, dtype=torch.int64, device=coefficients.device
    ).reshape(1, edge_count, 1).expand(batch_size, -1, max_roots)
    end_vertex_t = torch.as_tensor(
        end_vertex, dtype=torch.int64, device=coefficients.device
    ).reshape(1, edge_count, 1).expand(batch_size, -1, max_roots)
    edge_length = torch.linalg.norm(vector, dim=2)
    endpoint_tolerance = 5.0e-13 / torch.clamp(
        edge_length, min=torch.finfo(coefficients.dtype).tiny
    )
    at_start = root_valid & (
        torch.abs(roots_t) <= endpoint_tolerance[:, :, None]
    )
    at_end = root_valid & (
        torch.abs(roots_t - 1.0) <= endpoint_tolerance[:, :, None]
    )
    slot = torch.arange(
        max_roots, device=coefficients.device, dtype=torch.int64
    ).reshape(1, 1, max_roots)
    edge_ids = torch.arange(
        edge_count, device=coefficients.device, dtype=torch.int64
    ).reshape(1, edge_count, 1)
    interior = int(interior_node_base) + edge_ids * max_roots + slot
    node_ids = torch.where(
        at_start,
        start_vertex_t,
        torch.where(at_end, end_vertex_t, interior),
    )
    node_ids = torch.where(
        root_valid, node_ids, torch.full_like(node_ids, -1)
    )
    root_points = torch.where(
        root_valid[:, :, :, None],
        root_points,
        torch.full_like(root_points, float("nan")),
    )
    root_count = torch.sum(root_valid.to(torch.int64), dim=2)
    return root_points, roots_t, node_ids, root_count

def build_level_set_segments_exact_gpu(
    *,
    field: ExactSplineGpuField,
    grid: Grid2D,
    level: object,
    axis_points: object,
    limiter: object,
    x_points: object,
    x_levels: object,
    x_valid: object,
    refinement: int = 2,
) -> tuple[object, object, object, int]:
    """Tensor representation of the CPU ``extract_core_level_set`` graph.

    This function preserves the CPU topology grid, exact edge-root search,
    explicit X-point nodes and four-root cell pairing.  It returns the same
    primitive segment graph in padded tensors; fixed-angle rays are not used to
    define or repair the level set.
    """
    import torch

    coefficients = torch.as_tensor(field.coefficients)
    geometry = field.geometry
    axis = torch.as_tensor(axis_points, dtype=coefficients.dtype, device=coefficients.device)
    x_points_t = torch.as_tensor(x_points, dtype=coefficients.dtype, device=coefficients.device)
    x_levels_t = torch.as_tensor(x_levels, dtype=coefficients.dtype, device=coefficients.device)
    x_valid_t = torch.as_tensor(x_valid, dtype=torch.bool, device=coefficients.device)
    level_t = torch.as_tensor(level, dtype=coefficients.dtype, device=coefficients.device).reshape(-1)
    batch_size = int(coefficients.shape[0])
    x_count = int(x_points_t.shape[1])

    grid_scale = max(abs(float(grid.r.step)), abs(float(grid.z.step)))
    r_lower = float(geometry.topology_r_lower)
    r_upper = float(geometry.topology_r_upper)
    z_lower = float(geometry.topology_z_lower)
    z_upper = float(geometry.topology_z_upper)
    r_step = abs(float(grid.r.step)) / float(max(int(refinement), 1))
    z_step = abs(float(grid.z.step)) / float(max(int(refinement), 1))

    r_coordinates, r_count = _tailored_coordinates_exact_gpu(
        lower=r_lower,
        upper=r_upper,
        step=r_step,
        axis_coordinates=axis[:, 0],
        x_coordinates=x_points_t[:, :, 0],
        x_valid=x_valid_t,
    )
    z_coordinates, z_count = _tailored_coordinates_exact_gpu(
        lower=z_lower,
        upper=z_upper,
        step=z_step,
        axis_coordinates=axis[:, 1],
        x_coordinates=x_points_t[:, :, 1],
        x_valid=x_valid_t,
    )
    max_r = int(r_coordinates.shape[1])
    max_z = int(z_coordinates.shape[1])
    safe_r = torch.where(torch.isfinite(r_coordinates), r_coordinates, torch.full_like(r_coordinates, r_upper))
    safe_z = torch.where(torch.isfinite(z_coordinates), z_coordinates, torch.full_like(z_coordinates, z_upper))
    R = safe_r[:, None, :].expand(batch_size, max_z, max_r)
    Z = safe_z[:, :, None].expand(batch_size, max_z, max_r)
    topology_points = torch.stack((R, Z), dim=3)
    topology_values, topology_contains = evaluate_exact_spline_value(
        field=field,
        points=topology_points.reshape(batch_size, max_z * max_r, 2),
    )
    topology_values = topology_values.reshape(batch_size, max_z, max_r)
    topology_contains = topology_contains.reshape(batch_size, max_z, max_r)
    node_i = torch.arange(max_r, device=coefficients.device)[None, None, :]
    node_j = torch.arange(max_z, device=coefficients.device)[None, :, None]
    topology_node_valid = (
        (node_i < r_count[:, None, None])
        & (node_j < z_count[:, None, None])
        & topology_contains
    )

    cell_r_count = max_r - 1
    cell_z_count = max_z - 1
    cell_i = torch.arange(cell_r_count, device=coefficients.device)[None, None, :]
    cell_j = torch.arange(cell_z_count, device=coefficients.device)[None, :, None]
    cell_valid = (cell_i < (r_count - 1)[:, None, None]) & (cell_j < (z_count - 1)[:, None, None])

    safe_x_r = torch.where(x_valid_t, x_points_t[:, :, 0], torch.full_like(x_points_t[:, :, 0], r_lower))
    safe_x_z = torch.where(x_valid_t, x_points_t[:, :, 1], torch.full_like(x_points_t[:, :, 1], z_lower))
    x_i = torch.searchsorted(safe_r, safe_x_r, right=True) - 1
    x_j = torch.searchsorted(safe_z, safe_x_z, right=True) - 1
    x_i = torch.clamp(x_i, 0, cell_r_count - 1)
    x_j = torch.clamp(x_j, 0, cell_z_count - 1)
    x_valid_t &= (x_i < (r_count - 1)[:, None]) & (x_j < (z_count - 1)[:, None])

    x_halo = torch.zeros((batch_size, cell_z_count, cell_r_count), dtype=torch.bool, device=coefficients.device)
    for slot in range(x_count):
        active = x_valid_t[:, slot]
        x_halo |= active[:, None, None] & (torch.abs(cell_i - x_i[:, slot, None, None]) <= 1) & (
            torch.abs(cell_j - x_j[:, slot, None, None]) <= 1
        )
    x_halo &= cell_valid

    # The CPU edge cache uses whichever adjacent cell is visited first.  These
    # masks reproduce that row-major first-owner rule exactly.
    h_exhaustive = torch.zeros((batch_size, max_z, cell_r_count), dtype=torch.bool, device=coefficients.device)
    h_exhaustive[:, 0, :] = x_halo[:, 0, :]
    h_exhaustive[:, 1:, :] = x_halo
    v_exhaustive = torch.zeros((batch_size, cell_z_count, max_r), dtype=torch.bool, device=coefficients.device)
    v_exhaustive[:, :, 0] = x_halo[:, :, 0]
    v_exhaustive[:, :, 1:] = x_halo

    vertex_count = max_z * max_r
    h_edge_count = max_z * cell_r_count
    v_edge_count = cell_z_count * max_r
    max_roots = 4
    h_interior_base = vertex_count
    base_node_count = vertex_count + (h_edge_count + v_edge_count) * max_roots

    h_starts = topology_points[:, :, :-1, :].reshape(batch_size, h_edge_count, 2)
    h_ends = topology_points[:, :, 1:, :].reshape(batch_size, h_edge_count, 2)
    h_valid = (topology_node_valid[:, :, :-1] & topology_node_valid[:, :, 1:]).reshape(batch_size, h_edge_count)
    h_j = torch.arange(max_z, device=coefficients.device)[:, None].expand(max_z, cell_r_count).reshape(-1)
    h_i = torch.arange(cell_r_count, device=coefficients.device)[None, :].expand(max_z, cell_r_count).reshape(-1)
    h_start_vertex = h_j * max_r + h_i
    h_end_vertex = h_start_vertex + 1

    v_starts = topology_points[:, :-1, :, :].reshape(batch_size, v_edge_count, 2)
    v_ends = topology_points[:, 1:, :, :].reshape(batch_size, v_edge_count, 2)
    v_valid = (topology_node_valid[:, :-1, :] & topology_node_valid[:, 1:, :]).reshape(batch_size, v_edge_count)
    v_j = torch.arange(cell_z_count, device=coefficients.device)[:, None].expand(cell_z_count, max_r).reshape(-1)
    v_i = torch.arange(max_r, device=coefficients.device)[None, :].expand(cell_z_count, max_r).reshape(-1)
    v_start_vertex = v_j * max_r + v_i
    v_end_vertex = (v_j + 1) * max_r + v_i

    flux_scale = torch.maximum(
        torch.amax(torch.abs(topology_values), dim=(1, 2)),
        torch.amax(topology_values, dim=(1, 2)) - torch.amin(topology_values, dim=(1, 2)),
    )
    flux_tolerance = torch.maximum(
        2.0e-12 * flux_scale,
        128.0 * torch.finfo(coefficients.dtype).eps * torch.clamp(torch.abs(level_t), min=1.0),
    )
    # Horizontal and vertical topology edges share one exact root-refinement
    # call.  This preserves the CPU edge ordering because horizontal edges still
    # precede vertical edges, while halving the number of fixed refinement
    # launches in the per-candidate hot path.
    all_starts = torch.cat((h_starts, v_starts), dim=1)
    all_ends = torch.cat((h_ends, v_ends), dim=1)
    all_start_values = torch.cat(
        (
            topology_values[:, :, :-1].reshape(batch_size, h_edge_count),
            topology_values[:, :-1, :].reshape(batch_size, v_edge_count),
        ),
        dim=1,
    )
    all_end_values = torch.cat(
        (
            topology_values[:, :, 1:].reshape(batch_size, h_edge_count),
            topology_values[:, 1:, :].reshape(batch_size, v_edge_count),
        ),
        dim=1,
    )
    all_valid = torch.cat((h_valid, v_valid), dim=1)
    all_exhaustive = torch.cat(
        (
            h_exhaustive.reshape(batch_size, h_edge_count),
            v_exhaustive.reshape(batch_size, v_edge_count),
        ),
        dim=1,
    )
    all_start_vertex = torch.cat((h_start_vertex, v_start_vertex), dim=0)
    all_end_vertex = torch.cat((h_end_vertex, v_end_vertex), dim=0)
    all_points, all_t, all_nodes, _all_count = _exact_edge_roots_gpu(
        field=field,
        starts=all_starts,
        ends=all_ends,
        start_values=all_start_values,
        end_values=all_end_values,
        edge_valid=all_valid,
        exhaustive=all_exhaustive,
        level=level_t,
        tolerance=flux_tolerance,
        start_vertex=all_start_vertex,
        end_vertex=all_end_vertex,
        interior_node_base=h_interior_base,
        root_iterations=max(
            1,
            int(
                np.ceil(
                    np.log2(
                        max(r_step, z_step) / (2.0 * 1.0e-6)
                    )
                )
            ),
        ),
        max_roots=max_roots,
    )
    h_points = all_points[:, :h_edge_count]
    h_t = all_t[:, :h_edge_count]
    h_nodes = all_nodes[:, :h_edge_count]
    v_points = all_points[:, h_edge_count:]
    v_t = all_t[:, h_edge_count:]
    v_nodes = all_nodes[:, h_edge_count:]
    h_points = h_points.reshape(batch_size, max_z, cell_r_count, max_roots, 2)
    h_t = h_t.reshape(batch_size, max_z, cell_r_count, max_roots)
    h_nodes = h_nodes.reshape(batch_size, max_z, cell_r_count, max_roots)
    v_points = v_points.reshape(batch_size, cell_z_count, max_r, max_roots, 2)
    v_t = v_t.reshape(batch_size, cell_z_count, max_r, max_roots)
    v_nodes = v_nodes.reshape(batch_size, cell_z_count, max_r, max_roots)

    bottom_points = h_points[:, :-1, :, :, :]
    bottom_nodes = h_nodes[:, :-1, :, :]
    bottom_pos = h_t[:, :-1, :, :]
    right_points = v_points[:, :, 1:, :, :]
    right_nodes = v_nodes[:, :, 1:, :]
    right_pos = 1.0 + v_t[:, :, 1:, :]
    reverse = torch.arange(max_roots - 1, -1, -1, device=coefficients.device)
    top_points = h_points[:, 1:, :, reverse, :]
    top_nodes = h_nodes[:, 1:, :, reverse]
    top_pos = 2.0 + (1.0 - h_t[:, 1:, :, reverse])
    left_points = v_points[:, :, :-1, reverse, :]
    left_nodes = v_nodes[:, :, :-1, reverse]
    left_pos = 3.0 + (1.0 - v_t[:, :, :-1, reverse])

    perimeter_points = torch.cat((bottom_points, right_points, top_points, left_points), dim=3)
    perimeter_nodes = torch.cat((bottom_nodes, right_nodes, top_nodes, left_nodes), dim=3)
    perimeter_pos = torch.cat((bottom_pos, right_pos, top_pos, left_pos), dim=3)
    perimeter_valid = (perimeter_nodes >= 0) & torch.isfinite(perimeter_pos)
    sort_pos = torch.where(perimeter_valid, perimeter_pos, torch.full_like(perimeter_pos, float("inf")))
    order = torch.argsort(sort_pos, dim=3)
    perimeter_pos = torch.gather(perimeter_pos, 3, order)
    perimeter_nodes = torch.gather(perimeter_nodes, 3, order)
    perimeter_points = torch.gather(
        perimeter_points,
        3,
        order[:, :, :, :, None].expand(-1, -1, -1, -1, 2),
    )
    perimeter_valid = torch.gather(perimeter_valid, 3, order)
    spatial_tolerance = 2.0e-10 * max(grid_scale, 1.0)
    unique = perimeter_valid.clone()
    distance_previous = torch.linalg.norm(perimeter_points[:, :, :, 1:, :] - perimeter_points[:, :, :, :-1, :], dim=4)
    unique[:, :, :, 1:] &= distance_previous > float(spatial_tolerance)
    rank = torch.cumsum(unique.to(torch.int64), dim=3) - 1
    root_capacity = 4 * max_roots
    compact_points = torch.full_like(perimeter_points, float("nan"))
    compact_nodes = torch.full_like(perimeter_nodes, -1)
    compact_pos = torch.full_like(perimeter_pos, float("nan"))
    bidx = torch.arange(batch_size, device=coefficients.device)[:, None, None, None].expand_as(rank)
    jidx = torch.arange(cell_z_count, device=coefficients.device)[None, :, None, None].expand_as(rank)
    iidx = torch.arange(cell_r_count, device=coefficients.device)[None, None, :, None].expand_as(rank)
    compact_points[bidx[unique], jidx[unique], iidx[unique], rank[unique], :] = perimeter_points[unique]
    compact_nodes[bidx[unique], jidx[unique], iidx[unique], rank[unique]] = perimeter_nodes[unique]
    compact_pos[bidx[unique], jidx[unique], iidx[unique], rank[unique]] = perimeter_pos[unique]
    root_count = torch.sum(unique.to(torch.int64), dim=3)
    has_multiple = root_count > 1
    last_index = torch.clamp(root_count - 1, min=0)
    first_point = compact_points[:, :, :, 0, :]
    last_point = torch.gather(
        compact_points,
        3,
        last_index[:, :, :, None, None].expand(-1, -1, -1, 1, 2),
    )[:, :, :, 0, :]
    wrap_duplicate = has_multiple & (torch.linalg.norm(first_point - last_point, dim=3) <= float(spatial_tolerance))
    root_count = root_count - wrap_duplicate.to(torch.int64)
    valid_root_slots = torch.arange(root_capacity, device=coefficients.device)[None, None, None, :] < root_count[:, :, :, None]
    compact_nodes = torch.where(valid_root_slots, compact_nodes, torch.full_like(compact_nodes, -1))
    compact_points = torch.where(valid_root_slots[:, :, :, :, None], compact_points, torch.full_like(compact_points, float("nan")))

    cell_count = cell_z_count * cell_r_count
    x_slot_by_cell = torch.full((batch_size, cell_count), -1, dtype=torch.int64, device=coefficients.device)
    for slot in range(x_count):
        close_level = torch.abs(x_levels_t[:, slot] - level_t) <= 8.0 * flux_tolerance
        active = x_valid_t[:, slot] & close_level
        flat_cell = x_j[:, slot] * cell_r_count + x_i[:, slot]
        safe_cell = torch.clamp(flat_cell, 0, cell_count - 1)
        batch_ids = torch.arange(batch_size, device=coefficients.device)
        x_slot_by_cell[batch_ids[active], safe_cell[active]] = slot
    x_slot_by_cell = x_slot_by_cell.reshape(batch_size, cell_z_count, cell_r_count)
    exact_x_cell = x_slot_by_cell >= 0

    cell_centers = torch.stack(
        (
            0.5 * (safe_r[:, :-1, None] + safe_r[:, 1:, None]).transpose(1, 2).expand(-1, cell_z_count, -1),
            0.5 * (safe_z[:, :-1, None] + safe_z[:, 1:, None]).expand(-1, -1, cell_r_count),
        ),
        dim=3,
    )
    center_values, _contains = evaluate_exact_spline_value(
        field=field,
        points=cell_centers.reshape(batch_size, cell_count, 2),
    )
    center_values = center_values.reshape(batch_size, cell_z_count, cell_r_count)
    corner_values = topology_values[:, :-1, :-1]
    same_sign = (center_values > level_t[:, None, None]) == (corner_values > level_t[:, None, None])

    roots = compact_points
    nodes = compact_nodes
    segment_points = torch.full(
        (batch_size, cell_z_count, cell_r_count, 4, 2, 2),
        float("nan"),
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    segment_nodes = torch.full(
        (batch_size, cell_z_count, cell_r_count, 4, 2),
        -1,
        dtype=torch.int64,
        device=coefficients.device,
    )
    segment_valid = torch.zeros(
        (batch_size, cell_z_count, cell_r_count, 4),
        dtype=torch.bool,
        device=coefficients.device,
    )

    regular_two = cell_valid & ~exact_x_cell & (root_count == 2)
    segment_points[:, :, :, 0, 0, :] = roots[:, :, :, 0, :]
    segment_points[:, :, :, 0, 1, :] = roots[:, :, :, 1, :]
    segment_nodes[:, :, :, 0, 0] = nodes[:, :, :, 0]
    segment_nodes[:, :, :, 0, 1] = nodes[:, :, :, 1]
    segment_valid[:, :, :, 0] |= regular_two

    regular_four = cell_valid & ~exact_x_cell & (root_count == 4)
    pair_a0 = torch.where(same_sign, torch.zeros_like(root_count), torch.ones_like(root_count))
    pair_a1 = torch.where(same_sign, torch.ones_like(root_count), torch.full_like(root_count, 2))
    pair_b0 = torch.where(same_sign, torch.full_like(root_count, 2), torch.full_like(root_count, 3))
    pair_b1 = torch.where(same_sign, torch.full_like(root_count, 3), torch.zeros_like(root_count))
    for segment_slot, first_index, second_index in (
        (0, pair_a0, pair_a1),
        (1, pair_b0, pair_b1),
    ):
        first_point = torch.gather(roots, 3, first_index[:, :, :, None, None].expand(-1, -1, -1, 1, 2))[:, :, :, 0, :]
        second_point = torch.gather(roots, 3, second_index[:, :, :, None, None].expand(-1, -1, -1, 1, 2))[:, :, :, 0, :]
        first_node = torch.gather(nodes, 3, first_index[:, :, :, None])[:, :, :, 0]
        second_node = torch.gather(nodes, 3, second_index[:, :, :, None])[:, :, :, 0]
        segment_points[:, :, :, segment_slot, 0, :] = torch.where(
            regular_four[:, :, :, None], first_point, segment_points[:, :, :, segment_slot, 0, :]
        )
        segment_points[:, :, :, segment_slot, 1, :] = torch.where(
            regular_four[:, :, :, None], second_point, segment_points[:, :, :, segment_slot, 1, :]
        )
        segment_nodes[:, :, :, segment_slot, 0] = torch.where(
            regular_four, first_node, segment_nodes[:, :, :, segment_slot, 0]
        )
        segment_nodes[:, :, :, segment_slot, 1] = torch.where(
            regular_four, second_node, segment_nodes[:, :, :, segment_slot, 1]
        )
        segment_valid[:, :, :, segment_slot] |= regular_four

    safe_x_slot = torch.clamp(x_slot_by_cell, min=0)
    batch_grid = torch.arange(batch_size, device=coefficients.device)[:, None, None]
    x_coordinate = x_points_t[batch_grid, safe_x_slot]
    x_node = base_node_count + safe_x_slot
    x_cell_valid = cell_valid & exact_x_cell & (root_count == 4)
    for root_slot in range(4):
        segment_points[:, :, :, root_slot, 0, :] = torch.where(
            x_cell_valid[:, :, :, None], roots[:, :, :, root_slot, :], segment_points[:, :, :, root_slot, 0, :]
        )
        segment_points[:, :, :, root_slot, 1, :] = torch.where(
            x_cell_valid[:, :, :, None], x_coordinate, segment_points[:, :, :, root_slot, 1, :]
        )
        segment_nodes[:, :, :, root_slot, 0] = torch.where(
            x_cell_valid, nodes[:, :, :, root_slot], segment_nodes[:, :, :, root_slot, 0]
        )
        segment_nodes[:, :, :, root_slot, 1] = torch.where(
            x_cell_valid, x_node, segment_nodes[:, :, :, root_slot, 1]
        )
        segment_valid[:, :, :, root_slot] |= x_cell_valid

    segment_points = segment_points.reshape(batch_size, cell_count * 4, 2, 2)
    segment_nodes = segment_nodes.reshape(batch_size, cell_count * 4, 2)
    segment_valid = segment_valid.reshape(batch_size, cell_count * 4)
    length = torch.linalg.norm(segment_points[:, :, 1, :] - segment_points[:, :, 0, :], dim=2)
    segment_valid &= torch.isfinite(length) & (length > 1.0e-13)
    segment_points = torch.where(segment_valid[:, :, None, None], segment_points, torch.full_like(segment_points, float("nan")))
    segment_nodes = torch.where(segment_valid[:, :, None], segment_nodes, torch.full_like(segment_nodes, -1))
    return segment_points, segment_nodes, segment_valid, int(base_node_count)


def _points_in_or_on_polygon_exact_gpu(points: object, polygon: object, *, tolerance: float) -> object:
    """Torch equivalent of ``boundary_common.points_in_or_on_polygon``."""
    import torch

    pts = torch.as_tensor(points)
    poly = torch.as_tensor(polygon, dtype=pts.dtype, device=pts.device).reshape(-1, 2)
    starts = poly[:-1]
    vectors = poly[1:] - starts
    px = pts[..., 0, None]
    py = pts[..., 1, None]
    x1 = starts[:, 0]
    y1 = starts[:, 1]
    x2 = poly[1:, 0]
    y2 = poly[1:, 1]
    crossing = ((y1 > py) != (y2 > py)) & (
        px < (x2 - x1) * (py - y1) / torch.where(
            torch.abs(y2 - y1) > torch.finfo(pts.dtype).tiny,
            y2 - y1,
            torch.ones_like(y2 - y1),
        ) + x1
    )
    inside = (torch.sum(crossing.to(torch.int64), dim=-1) % 2) == 1
    denom = torch.sum(vectors * vectors, dim=1)
    relative = pts[..., None, :] - starts
    fraction = torch.clamp(
        torch.sum(relative * vectors, dim=-1)
        / torch.where(denom > 0.0, denom, torch.ones_like(denom)),
        0.0,
        1.0,
    )
    closest = starts + fraction[..., None] * vectors
    distance = torch.sqrt(torch.clamp(torch.amin(torch.sum((pts[..., None, :] - closest) ** 2, dim=-1), dim=-1), min=0.0))
    return inside | (distance <= float(tolerance))


def find_critical_points_exact_gpu(
    *,
    psi: object,
    field: ExactSplineGpuField,
    center_hint: tuple[float, float],
    max_candidates: int = 256,
    max_o_points: int = 16,
    max_x_points: int = 32,
) -> tuple[object, object, object, object, object, object, object]:
    """Find the primary O-point and relevant X-point candidates on tensors.

    The physical classification, safeguarded Newton refinement, Hessian test,
    limiter filtering and deterministic ordering follow the CPU contract.  Seed
    suppression is vectorized rather than reproducing the CPU greedy loop, and
    parity is enforced on the accepted critical points and downstream LCFS.
    """
    import torch
    import torch.nn.functional as F

    psi_tensor = torch.as_tensor(psi)
    coefficients = torch.as_tensor(field.coefficients)
    geometry = field.geometry
    if coefficients.dtype != psi_tensor.dtype or coefficients.device != psi_tensor.device:
        raise ValueError("combined exact spline must match psi dtype and device")
    batch_size, nz, nr = psi_tensor.shape
    grid_scale = max(abs(float(geometry.dr)), abs(float(geometry.dz)))
    native = geometry.native_points.reshape(1, nz * nr, 2).expand(batch_size, -1, -1)
    _values, gradients_flat, native_contains = evaluate_exact_spline_value_gradient(
        field=field,
        points=native,
    )
    gradients = gradients_flat.reshape(batch_size, nz, nr, 2)
    grad2 = torch.sum(gradients * gradients, dim=3)
    local_min = grad2 <= -F.max_pool2d((-grad2).unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    interior_node = torch.zeros_like(local_min)
    interior_node[:, 1:-1, 1:-1] = True
    local_valid = local_min & interior_node & native_contains.reshape(batch_size, nz, nr) & torch.isfinite(grad2)

    node_points = native
    node_scores = grad2.reshape(batch_size, -1)
    node_valid = local_valid.reshape(batch_size, -1)

    d_r = gradients[:, :, :, 0]
    d_z = gradients[:, :, :, 1]
    cell_r = torch.stack((d_r[:, :-1, :-1], d_r[:, :-1, 1:], d_r[:, 1:, :-1], d_r[:, 1:, 1:]), dim=3)
    cell_z = torch.stack((d_z[:, :-1, :-1], d_z[:, :-1, 1:], d_z[:, 1:, :-1], d_z[:, 1:, 1:]), dim=3)
    cell_finite = torch.all(torch.isfinite(cell_r), dim=3) & torch.all(torch.isfinite(cell_z), dim=3)
    cell_sign = (
        (torch.amin(cell_r, dim=3) <= 0.0)
        & (torch.amax(cell_r, dim=3) >= 0.0)
        & (torch.amin(cell_z, dim=3) <= 0.0)
        & (torch.amax(cell_z, dim=3) >= 0.0)
    )
    cell_valid = cell_finite & cell_sign
    r_coords = geometry.r0 + torch.arange(nr, dtype=psi_tensor.dtype, device=psi_tensor.device) * geometry.dr
    z_coords = geometry.z0 + torch.arange(nz, dtype=psi_tensor.dtype, device=psi_tensor.device) * geometry.dz
    cell_r_mid = 0.5 * (r_coords[:-1] + r_coords[1:])
    cell_z_mid = 0.5 * (z_coords[:-1] + z_coords[1:])
    Z_mid, R_mid = torch.meshgrid(cell_z_mid, cell_r_mid, indexing="ij")
    cell_points_one = torch.stack((R_mid.reshape(-1), Z_mid.reshape(-1)), dim=1)
    cell_points = cell_points_one.reshape(1, -1, 2).expand(batch_size, -1, -1)
    cell_scores = torch.mean(cell_r * cell_r + cell_z * cell_z, dim=3).reshape(batch_size, -1)
    cell_valid = cell_valid.reshape(batch_size, -1)

    candidate_points = torch.cat((node_points, cell_points), dim=1)
    candidate_scores = torch.cat((node_scores, cell_scores), dim=1)
    candidate_valid = torch.cat((node_valid, cell_valid), dim=1) & torch.isfinite(candidate_scores)
    candidate_scores = torch.where(candidate_valid, candidate_scores, torch.full_like(candidate_scores, float("inf")))

    # First retain the same broad candidate budget as the CPU extractor.  A
    # vectorized spatial non-maximum suppression then keeps the best-scoring
    # seed in every 0.75-grid-scale neighbourhood.  This prevents one critical
    # point from filling the Newton budget with nearby node and cell seeds,
    # while avoiding the CPU implementation's 256 sequential reductions.
    preliminary_capacity = min(
        max(int(max_candidates), 1),
        256,
        int(candidate_points.shape[1]),
    )
    preliminary_score, preliminary_index = torch.topk(
        candidate_scores,
        k=preliminary_capacity,
        dim=1,
        largest=False,
        sorted=True,
    )
    preliminary_valid = torch.isfinite(preliminary_score)
    preliminary_points = torch.gather(
        candidate_points,
        1,
        preliminary_index[:, :, None].expand(-1, -1, 2),
    )
    pair_distance = torch.linalg.norm(
        preliminary_points[:, :, None, :]
        - preliminary_points[:, None, :, :],
        dim=3,
    )
    seed_index = torch.arange(
        preliminary_capacity, device=psi_tensor.device
    )
    earlier = seed_index[None, None, :] < seed_index[None, :, None]
    suppressed = torch.any(
        preliminary_valid[:, None, :]
        & earlier
        & torch.isfinite(pair_distance)
        & (pair_distance <= 0.75 * grid_scale),
        dim=2,
    )
    nms_valid = preliminary_valid & ~suppressed
    nms_score = torch.where(
        nms_valid,
        preliminary_score,
        torch.full_like(preliminary_score, float("inf")),
    )
    capacity = min(
        max(int(max_o_points) + int(max_x_points) + 16, 32),
        preliminary_capacity,
    )
    selected_score, selected_nms_index = torch.topk(
        nms_score,
        k=capacity,
        dim=1,
        largest=False,
        sorted=True,
    )
    selected_valid = torch.isfinite(selected_score)
    selected_points = torch.gather(
        preliminary_points,
        1,
        selected_nms_index[:, :, None].expand(-1, -1, 2),
    )
    selected_points = torch.where(
        selected_valid[:, :, None],
        selected_points,
        torch.full_like(selected_points, float("nan")),
    )

    center = torch.as_tensor(center_hint, dtype=psi_tensor.dtype, device=psi_tensor.device).reshape(1, 1, 2).expand(batch_size, -1, -1)
    center_valid = (
        (center[:, :, 0] >= geometry.r0)
        & (center[:, :, 0] <= geometry.r0 + geometry.dr * float(geometry.nr - 1))
        & (center[:, :, 1] >= geometry.z0)
        & (center[:, :, 1] <= geometry.z0 + geometry.dz * float(geometry.nz - 1))
    )
    seeds = torch.cat((center, selected_points), dim=1)
    seed_valid = torch.cat((center_valid, selected_valid), dim=1)

    points = seeds.clone()
    origin = seeds.clone()
    alive = seed_valid & torch.all(torch.isfinite(points), dim=2)
    flux_span = torch.amax(psi_tensor, dim=(1, 2)) - torch.amin(psi_tensor, dim=(1, 2))
    flux_abs = torch.amax(torch.abs(psi_tensor), dim=(1, 2))
    flux_scale = torch.maximum(torch.maximum(flux_span, flux_abs), torch.full_like(flux_span, 1.0e-12))
    grad_tol = torch.maximum(
        torch.full_like(flux_scale, 1.0e-11),
        1.0e-8 * flux_scale / max(grid_scale, 1.0e-12),
    )
    search_radius = 3.5 * float(np.hypot(float(geometry.dr), float(geometry.dz)))
    alphas = torch.pow(
        torch.full((5,), 0.5, dtype=psi_tensor.dtype, device=psi_tensor.device),
        torch.arange(5, dtype=psi_tensor.dtype, device=psi_tensor.device),
    )

    # Seeds originate in derivative sign-change cells or local |grad psi|
    # minima, so safeguarded Newton converges rapidly.  Sixteen iterations with
    # five vectorized line-search trials preserve the CPU acceptance cases while
    # bounding the hot-path launch count.
    for _iteration in range(16):
        _level_now, gradient, hessian, contains = evaluate_exact_spline(
            field=field,
            points=points,
        )
        grad_norm = torch.linalg.norm(gradient, dim=2)
        finite = alive & contains & torch.isfinite(grad_norm) & torch.all(torch.isfinite(hessian), dim=(2, 3))
        reached = finite & (grad_norm <= grad_tol[:, None])
        active = finite & ~reached
        alive &= finite

        safe_gradient = torch.where(finite[:, :, None], gradient, torch.zeros_like(gradient))
        h00 = hessian[:, :, 0, 0]
        h01 = hessian[:, :, 0, 1]
        h11 = hessian[:, :, 1, 1]
        determinant = h00 * h11 - h01 * h01
        hessian_scale = torch.maximum(
            torch.maximum(torch.abs(h00), torch.abs(h01)),
            torch.abs(h11),
        )
        determinant_ok = torch.abs(determinant) > (1.0e-14 * torch.clamp(hessian_scale * hessian_scale, min=1.0e-30))
        safe_determinant = torch.where(determinant_ok, determinant, torch.ones_like(determinant))
        delta = torch.stack(
            (
                (h11 * safe_gradient[:, :, 0] - h01 * safe_gradient[:, :, 1]) / safe_determinant,
                (-h01 * safe_gradient[:, :, 0] + h00 * safe_gradient[:, :, 1]) / safe_determinant,
            ),
            dim=2,
        )
        delta_valid = determinant_ok & torch.all(torch.isfinite(delta), dim=2)
        active &= delta_valid

        trials = points[:, :, None, :] - alphas[None, None, :, None] * delta[:, :, None, :]
        field_margin = 0.25 * grid_scale
        trial_contains = (
            (trials[:, :, :, 0] >= geometry.r0 + field_margin)
            & (trials[:, :, :, 0] <= geometry.r0 + geometry.dr * float(geometry.nr - 1) - field_margin)
            & (trials[:, :, :, 1] >= geometry.z0 + field_margin)
            & (trials[:, :, :, 1] <= geometry.z0 + geometry.dz * float(geometry.nz - 1) - field_margin)
        )
        trial_contains &= torch.linalg.norm(trials - origin[:, :, None, :], dim=3) <= search_radius
        _trial_level, trial_gradient, trial_spline_contains = evaluate_exact_spline_value_gradient(
            field=field,
            points=trials.reshape(batch_size, -1, 2),
        )
        trial_norm = torch.linalg.norm(trial_gradient.reshape(batch_size, points.shape[1], 5, 2), dim=3)
        trial_spline_contains = trial_spline_contains.reshape(batch_size, points.shape[1], 5)
        acceptable = (
            active[:, :, None]
            & trial_contains
            & trial_spline_contains
            & torch.isfinite(trial_norm)
            & (trial_norm < grad_norm[:, :, None])
        )
        has_acceptable = torch.any(acceptable, dim=2)
        first = torch.argmax(acceptable.to(torch.int64), dim=2)
        chosen = torch.gather(
            trials,
            2,
            first[:, :, None, None].expand(-1, -1, 1, 2),
        )[:, :, 0, :]
        points = torch.where(has_acceptable[:, :, None], chosen, points)
        failed_active = active & ~has_acceptable
        alive &= ~failed_active

    levels, gradient, hessian, contains = evaluate_exact_spline(
        field=field,
        points=points,
    )
    final_norm = torch.linalg.norm(gradient, dim=2)
    refined_valid = alive & contains & torch.isfinite(final_norm) & (final_norm <= 10.0 * grad_tol[:, None])
    eigenvalues = torch.linalg.eigvalsh(hessian)
    eigen_scale = torch.clamp(torch.amax(torch.abs(eigenvalues), dim=2), min=1.0e-30)
    nondegenerate = torch.amin(torch.abs(eigenvalues), dim=2) > 1.0e-8 * eigen_scale
    x_kind = (eigenvalues[:, :, 0] < 0.0) & (eigenvalues[:, :, 1] > 0.0)
    minimum = eigenvalues[:, :, 0] > 0.0
    maximum = eigenvalues[:, :, 1] < 0.0
    o_kind = minimum | maximum
    classified = refined_valid & nondegenerate & (x_kind | o_kind)
    limiter_inside = _points_in_or_on_polygon_exact_gpu(
        points,
        geometry.limiter_vertices,
        tolerance=grid_scale,
    )
    classified &= limiter_inside

    dedup_tolerance = 0.35 * grid_scale
    point_count = int(points.shape[1])
    pair_distance = torch.linalg.norm(
        points[:, :, None, :] - points[:, None, :, :],
        dim=3,
    )
    point_index = torch.arange(point_count, device=psi_tensor.device)
    earlier = point_index[None, None, :] < point_index[None, :, None]
    duplicate = torch.any(
        classified[:, None, :]
        & earlier
        & torch.isfinite(pair_distance)
        & (pair_distance <= dedup_tolerance),
        dim=2,
    )
    keep = classified & ~duplicate

    hint = torch.as_tensor(center_hint, dtype=psi_tensor.dtype, device=psi_tensor.device)
    o_valid = keep & o_kind
    o_distance = torch.linalg.norm(points - hint[None, None, :], dim=2)
    o_order = torch.argsort(torch.where(o_valid, o_distance, torch.full_like(o_distance, float("inf"))), dim=1, stable=True)
    o_capacity = max(int(max_o_points), 1)
    o_order = o_order[:, :o_capacity]
    o_points = torch.gather(points, 1, o_order[:, :, None].expand(-1, -1, 2))
    o_levels = torch.gather(levels, 1, o_order)
    o_valid_sorted = torch.gather(o_valid, 1, o_order)
    axis_points = o_points[:, 0, :]
    axis_levels = o_levels[:, 0]
    axis_valid = o_valid_sorted[:, 0]
    axis_minimum = torch.gather(minimum, 1, o_order)[:, 0]
    axis_kind = torch.where(
        axis_minimum,
        -torch.ones((batch_size,), dtype=torch.int64, device=psi_tensor.device),
        torch.ones((batch_size,), dtype=torch.int64, device=psi_tensor.device),
    )
    axis_points = torch.where(axis_valid[:, None], axis_points, torch.full_like(axis_points, float("nan")))
    axis_levels = torch.where(axis_valid, axis_levels, torch.full_like(axis_levels, float("nan")))
    axis_kind = torch.where(axis_valid, axis_kind, torch.zeros_like(axis_kind))

    x_valid_all = keep & x_kind
    x_capacity = max(int(max_x_points), 0)
    x_overflow = torch.sum(x_valid_all.to(torch.int64), dim=1) > x_capacity
    x_distance = torch.abs(levels - axis_levels[:, None])
    x_order = torch.argsort(
        torch.where(
            x_valid_all,
            x_distance,
            torch.full_like(x_distance, float("inf")),
        ),
        dim=1,
        stable=True,
    )
    if x_capacity:
        x_order = x_order[:, :x_capacity]
        x_points = torch.gather(points, 1, x_order[:, :, None].expand(-1, -1, 2))
        x_levels = torch.gather(levels, 1, x_order)
        x_valid = torch.gather(x_valid_all, 1, x_order)
        x_points = torch.where(x_valid[:, :, None], x_points, torch.full_like(x_points, float("nan")))
        x_levels = torch.where(x_valid, x_levels, torch.full_like(x_levels, float("nan")))
    else:
        x_points = torch.empty((batch_size, 0, 2), dtype=psi_tensor.dtype, device=psi_tensor.device)
        x_levels = torch.empty((batch_size, 0), dtype=psi_tensor.dtype, device=psi_tensor.device)
        x_valid = torch.empty((batch_size, 0), dtype=torch.bool, device=psi_tensor.device)

    # Never return a partial critical-point set.  If a lane exceeds the declared
    # X-point capacity, invalidate the lane so the public boundary result is
    # ``found=False`` instead of silently changing the topology.
    axis_valid &= ~x_overflow
    axis_points = torch.where(
        axis_valid[:, None], axis_points, torch.full_like(axis_points, float("nan"))
    )
    axis_levels = torch.where(
        axis_valid, axis_levels, torch.full_like(axis_levels, float("nan"))
    )
    axis_kind = torch.where(axis_valid, axis_kind, torch.zeros_like(axis_kind))
    if x_capacity:
        x_valid &= ~x_overflow[:, None]
        x_points = torch.where(
            x_valid[:, :, None], x_points, torch.full_like(x_points, float("nan"))
        )
        x_levels = torch.where(
            x_valid, x_levels, torch.full_like(x_levels, float("nan"))
        )

    return axis_points, axis_levels, axis_kind, axis_valid, x_points, x_levels, x_valid


def _bounded_minimize_limiter_chi_exact_gpu(
    *,
    field: ExactSplineGpuField,
    orientation: object,
    axis_level: object,
    lower: object,
    upper: object,
    segment_index: object,
    valid: object,
) -> tuple[object, object, object]:
    """Refine all sampled limiter minima with one batched golden search.

    The previous literal SciPy port executed 63 branch-heavy iterations over
    every padded limiter slot.  Here all valid brackets are refined together.
    Twenty golden updates reduce the initial limiter interval below 2e-6 m on
    the T-15 geometry, while every objective evaluation remains on the exact
    bicubic spline.
    """
    import torch

    coefficients = torch.as_tensor(field.coefficients)
    geometry = field.geometry
    orient = torch.as_tensor(orientation, dtype=coefficients.dtype, device=coefficients.device)
    axis = torch.as_tensor(axis_level, dtype=coefficients.dtype, device=coefficients.device)
    a = torch.as_tensor(lower, dtype=coefficients.dtype, device=coefficients.device).clone()
    b = torch.as_tensor(upper, dtype=coefficients.dtype, device=coefficients.device).clone()
    segment = torch.as_tensor(segment_index, dtype=torch.int64, device=coefficients.device)
    active = torch.as_tensor(valid, dtype=torch.bool, device=coefficients.device)
    safe_segment = torch.clamp(
        segment,
        0,
        int(geometry.limiter_segment_starts.shape[0]) - 1,
    )
    starts = geometry.limiter_segment_starts[safe_segment]
    vectors = geometry.limiter_segment_vectors[safe_segment]

    def evaluate(t_value: object) -> tuple[object, object, object]:
        points = starts + torch.as_tensor(t_value)[:, :, None] * vectors
        levels, contains = evaluate_exact_spline_value(
            field=field,
            points=points,
        )
        chi = orient[:, None] * (levels - axis[:, None])
        finite = contains & torch.isfinite(chi) & torch.isfinite(levels)
        return chi, levels, finite

    golden = float((np.sqrt(5.0) - 1.0) / 2.0)
    c = b - golden * (b - a)
    d = a + golden * (b - a)
    fc, _level_c, valid_c = evaluate(c)
    fd, _level_d, valid_d = evaluate(d)
    running = active & valid_c & valid_d

    for _ in range(20):
        choose_left = running & (fc <= fd)
        new_a = torch.where(choose_left, a, c)
        new_b = torch.where(choose_left, d, b)
        retained_t = torch.where(choose_left, c, d)
        retained_f = torch.where(choose_left, fc, fd)
        new_t = torch.where(
            choose_left,
            new_b - golden * (new_b - new_a),
            new_a + golden * (new_b - new_a),
        )
        new_f, _new_level, new_valid = evaluate(new_t)
        running &= new_valid
        a = torch.where(running, new_a, a)
        b = torch.where(running, new_b, b)
        c = torch.where(running, torch.where(choose_left, new_t, retained_t), c)
        d = torch.where(running, torch.where(choose_left, retained_t, new_t), d)
        fc = torch.where(running, torch.where(choose_left, new_f, retained_f), fc)
        fd = torch.where(running, torch.where(choose_left, retained_f, new_f), fd)

    best_t = torch.where(fc <= fd, c, d)
    levels, contains = evaluate_exact_spline_value(
        field=field,
        points=starts + best_t[:, :, None] * vectors,
    )
    success = active & running & contains & torch.isfinite(best_t) & torch.isfinite(levels)
    return best_t, levels, success


def limiter_flux_candidates_exact_gpu(
    *,
    field: ExactSplineGpuField,
    axis_level: object,
    orientation: object,
    flux_scale: object,
    flux_floor: object,
) -> tuple[object, object, object, object, object, object, object]:
    """Literal tensor port of ``lcfs._limiter_flux_candidates``.

    Returns grouped candidate chi/levels plus every raw contact point and its
    group id so the caller can perform the CPU contact test without collapsing
    a multi-contact wall level to one point.
    """
    import torch

    coefficients = torch.as_tensor(field.coefficients)
    geometry = field.geometry
    batch_size = int(coefficients.shape[0])
    segment_count, sample_capacity = geometry.limiter_sample_valid.shape
    # Static basis values on all limiter samples are prepared once.  Runtime
    # sample flux is therefore one dense contraction instead of a full spline
    # cell lookup over every padded sample.
    levels = torch.as_tensor(field.limiter_segment_values)
    contains = geometry.limiter_sample_valid[None, :, :].expand(batch_size, -1, -1)
    orientation_t = torch.as_tensor(orientation, dtype=coefficients.dtype, device=coefficients.device).reshape(batch_size)
    axis_t = torch.as_tensor(axis_level, dtype=coefficients.dtype, device=coefficients.device).reshape(batch_size)
    chi = orientation_t[:, None, None] * (levels - axis_t[:, None, None])
    sample_valid = geometry.limiter_sample_valid[None, :, :].expand(batch_size, -1, -1) & contains & torch.isfinite(chi)
    sample_count = torch.sum(geometry.limiter_sample_valid.to(torch.int64), dim=1)
    index = torch.arange(sample_capacity, device=coefficients.device)[None, None, :]
    previous_index = torch.clamp(index - 1, min=0)
    next_index = torch.minimum(index + 1, (sample_count - 1)[None, :, None])
    previous = torch.gather(chi, 2, previous_index.expand(batch_size, segment_count, -1))
    following = torch.gather(chi, 2, next_index.expand(batch_size, segment_count, -1))
    first = index == 0
    last = index == (sample_count - 1)[None, :, None]
    interior = ~first & ~last & (index < sample_count[None, :, None])
    candidate = sample_valid & (
        (interior & (chi <= previous) & (chi <= following))
        | (first & (chi <= following))
        | (last & (chi <= previous))
    )

    flat_count = int(segment_count * sample_capacity)
    flat_candidate = candidate.reshape(batch_size, flat_count)
    flat_segment = torch.arange(segment_count, device=coefficients.device, dtype=torch.int64)[:, None].expand(segment_count, sample_capacity).reshape(-1)
    flat_sample = torch.arange(sample_capacity, device=coefficients.device, dtype=torch.int64)[None, :].expand(segment_count, sample_capacity).reshape(-1)
    sample_count_flat = sample_count[flat_segment]
    endpoint = (flat_sample == 0) | (flat_sample == sample_count_flat - 1)
    interior_candidate = flat_candidate & ~endpoint[None, :]
    lower_sample = torch.clamp(flat_sample - 1, min=0)
    upper_sample = torch.minimum(flat_sample + 1, sample_count_flat - 1)
    t_values = geometry.limiter_sample_t[flat_segment, flat_sample][None, :].expand(batch_size, -1).clone()
    lower = geometry.limiter_sample_t[flat_segment, lower_sample][None, :].expand(batch_size, -1)
    upper = geometry.limiter_sample_t[flat_segment, upper_sample][None, :].expand(batch_size, -1)
    segment_matrix = flat_segment[None, :].expand(batch_size, -1)
    interior_capacity = max(
        1,
        int(
            torch.max(
                torch.sum(interior_candidate.to(torch.int64), dim=1)
            ).item()
        ),
    )
    interior_order = torch.argsort(
        (~interior_candidate).to(torch.int64), dim=1, stable=True
    )[:, :interior_capacity]
    compact_interior_valid = torch.gather(
        interior_candidate, 1, interior_order
    )
    compact_lower = torch.gather(lower, 1, interior_order)
    compact_upper = torch.gather(upper, 1, interior_order)
    compact_segment = torch.gather(segment_matrix, 1, interior_order)
    compact_refined_t, _compact_levels, compact_refined_success = (
        _bounded_minimize_limiter_chi_exact_gpu(
            field=field,
            orientation=orientation_t,
            axis_level=axis_t,
            lower=compact_lower,
            upper=compact_upper,
            segment_index=compact_segment,
            valid=compact_interior_valid,
        )
    )
    refined_t = torch.zeros_like(t_values)
    refined_success = torch.zeros_like(interior_candidate)
    refined_t.scatter_(1, interior_order, compact_refined_t)
    refined_success.scatter_(1, interior_order, compact_refined_success)
    t_values = torch.where(
        interior_candidate & refined_success,
        torch.clamp(refined_t, 0.0, 1.0),
        t_values,
    )
    candidate_valid = flat_candidate & (endpoint[None, :] | refined_success)
    starts = geometry.limiter_segment_starts[flat_segment][None, :, :]
    vectors = geometry.limiter_segment_vectors[flat_segment][None, :, :]
    raw_points = starts + t_values[:, :, None] * vectors
    raw_levels, raw_contains = evaluate_exact_spline_value(
        field=field,
        points=raw_points,
    )
    raw_chi = orientation_t[:, None] * (raw_levels - axis_t[:, None])
    candidate_valid &= raw_contains & torch.isfinite(raw_chi) & (raw_chi > torch.as_tensor(flux_floor, dtype=coefficients.dtype, device=coefficients.device)[:, None])
    raw_chi = torch.where(candidate_valid, raw_chi, torch.full_like(raw_chi, float("inf")))
    order = torch.argsort(raw_chi, dim=1, stable=True)
    sorted_chi = torch.gather(raw_chi, 1, order)
    sorted_levels = torch.gather(raw_levels, 1, order)
    sorted_points = torch.gather(raw_points, 1, order[:, :, None].expand(-1, -1, 2))
    sorted_segments = flat_segment[None, :].expand(batch_size, -1).gather(1, order)
    sorted_valid = torch.isfinite(sorted_chi)
    raw_capacity = max(
        1,
        int(
            torch.max(
                torch.sum(sorted_valid.to(torch.int64), dim=1)
            ).item()
        ),
    )
    sorted_chi = sorted_chi[:, :raw_capacity]
    sorted_levels = sorted_levels[:, :raw_capacity]
    sorted_points = sorted_points[:, :raw_capacity]
    sorted_segments = sorted_segments[:, :raw_capacity]
    sorted_valid = sorted_valid[:, :raw_capacity]

    first_chi = sorted_chi[:, 0]
    flux_tolerance = torch.maximum(
        1.0e-10
        * torch.as_tensor(
            flux_scale, dtype=coefficients.dtype, device=coefficients.device
        ),
        1.0e-7
        * torch.maximum(first_chi, torch.full_like(first_chi, 1.0e-12)),
    )

    # The CPU wall grouping compares every value with the first value of the
    # current group, not merely with its immediate predecessor.  The compact
    # raw candidate list is small, so this short ordered loop preserves that
    # exact non-transitive grouping rule without iterating over padded limiter
    # samples.
    group_id = torch.full(
        (batch_size, raw_capacity),
        -1,
        dtype=torch.int64,
        device=coefficients.device,
    )
    current_group = torch.full(
        (batch_size,), -1, dtype=torch.int64, device=coefficients.device
    )
    group_first_chi = torch.zeros(
        (batch_size,), dtype=coefficients.dtype, device=coefficients.device
    )
    for index in range(raw_capacity):
        value = sorted_chi[:, index]
        valid_now = sorted_valid[:, index]
        new_group = valid_now & (
            (current_group < 0)
            | (torch.abs(value - group_first_chi) > flux_tolerance)
        )
        current_group = torch.where(
            new_group, current_group + 1, current_group
        )
        group_first_chi = torch.where(
            new_group, value, group_first_chi
        )
        group_id[:, index] = torch.where(
            valid_now, current_group, torch.full_like(current_group, -1)
        )

    group_capacity = raw_capacity
    safe_group = torch.clamp(group_id, min=0)
    group_count = torch.zeros(
        (batch_size, group_capacity),
        dtype=torch.int64,
        device=coefficients.device,
    )
    group_chi_sum = torch.zeros(
        (batch_size, group_capacity),
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    group_level_sum = torch.zeros_like(group_chi_sum)
    group_count.scatter_add_(1, safe_group, sorted_valid.to(torch.int64))
    group_chi_sum.scatter_add_(
        1,
        safe_group,
        torch.where(sorted_valid, sorted_chi, torch.zeros_like(sorted_chi)),
    )
    group_level_sum.scatter_add_(
        1,
        safe_group,
        torch.where(
            sorted_valid, sorted_levels, torch.zeros_like(sorted_levels)
        ),
    )
    group_valid = group_count > 0
    group_chi = torch.where(
        group_valid,
        group_chi_sum
        / torch.clamp(group_count.to(coefficients.dtype), min=1.0),
        torch.full_like(group_chi_sum, float("inf")),
    )
    group_level = torch.where(
        group_valid,
        group_level_sum
        / torch.clamp(group_count.to(coefficients.dtype), min=1.0),
        torch.full_like(group_level_sum, float("nan")),
    )
    compact_group_capacity = max(
        1,
        int(
            torch.max(
                torch.sum(group_valid.to(torch.int64), dim=1)
            ).item()
        ),
    )
    return (
        group_chi[:, :compact_group_capacity],
        group_level[:, :compact_group_capacity],
        group_valid[:, :compact_group_capacity],
        sorted_points,
        sorted_segments,
        group_id,
        sorted_valid,
    )
