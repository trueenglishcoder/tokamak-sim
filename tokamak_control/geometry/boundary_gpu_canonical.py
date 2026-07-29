"""Correctness-first batched bridge to the canonical equilibrium LCFS graph.

The magnetic field can remain batched on the accelerator, but topology is a
non-differentiable graph operation.  Until the identical FLUSH-style graph is
ported to tensor kernels, this bridge copies one flux map at a time to the CPU,
runs the canonical extractor, and returns tensors on the original device.  It
prevents the GPU dispatcher from using a second, physically different LCFS
algorithm.
"""

from __future__ import annotations

import numpy as np

from tokamak_control.core.grid import Grid2D
from tokamak_control.geometry.lcfs import find_equilibrium_lcfs


_TOPOLOGY_CODE = {
    "limited": 1,
    "single_null": 2,
    "double_null": 3,
    "multi_null": 4,
}


def fixed_angle_boundary_gpu_canonical(
    *,
    field,
    grid: Grid2D,
    center: tuple[float, float],
    angles,
    limiter,
    return_dense_boundary: bool,
):
    """Run the canonical LCFS extractor for every lane of a torch batch."""
    import torch
    from tokamak_control.geometry.boundary_gpu import FixedAngleBoundaryGpuResult

    if field.ndim != 3:
        raise ValueError(f"field must have shape (B, Z, R), got {tuple(field.shape)}")
    batch_size = int(field.shape[0])
    angle_values = np.asarray(angles.detach().cpu(), dtype=np.float64).reshape(-1)
    limiter_values = np.asarray(limiter.detach().cpu(), dtype=np.float64).reshape(-1, 2)
    dtype = field.dtype
    device = field.device

    results: list[object | None] = []
    physical_found: list[bool] = []
    projection_found: list[bool] = []
    for batch_index in range(batch_size):
        try:
            equilibrium = find_equilibrium_lcfs(
                psi=np.asarray(field[batch_index].detach().cpu(), dtype=np.float64),
                grid=grid,
                center_hint=(float(center[0]), float(center[1])),
                limiter_shape=limiter_values,
                fixed_angles=angle_values,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            results.append(None)
            physical_found.append(False)
            projection_found.append(False)
            continue
        results.append(equilibrium)
        physical_found.append(bool(equilibrium.found))
        projection_found.append(bool(equilibrium.found and equilibrium.fixed_angle_projection.valid))

    angle_count = int(angle_values.size)
    found = torch.as_tensor(projection_found, dtype=torch.bool, device=device)
    status_code = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    topology_code = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    level = torch.full((batch_size,), float("nan"), dtype=dtype, device=device)
    points = torch.full((batch_size, angle_count, 2), float("nan"), dtype=dtype, device=device)
    radii = torch.full((batch_size, angle_count), float("nan"), dtype=dtype, device=device)
    intersection_counts = torch.zeros((batch_size, angle_count), dtype=torch.int64, device=device)
    axis_points = torch.full((batch_size, 2), float("nan"), dtype=dtype, device=device)
    x_points = torch.full((batch_size, 8, 2), float("nan"), dtype=dtype, device=device)
    quality = torch.full((batch_size, 6), float("nan"), dtype=dtype, device=device)

    dense_size = 0
    contact_size = 0
    if return_dense_boundary:
        dense_size = max(
            (int(result.core_boundary.shape[0]) for result in results if result is not None),
            default=0,
        )
    contact_size = max(
        (len(result.limiter_contacts) for result in results if result is not None),
        default=0,
    )
    core_boundary = torch.full(
        (batch_size, dense_size, 2),
        float("nan"),
        dtype=dtype,
        device=device,
    )
    core_boundary_count = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    limiter_contacts = torch.full(
        (batch_size, contact_size, 2),
        float("nan"),
        dtype=dtype,
        device=device,
    )
    limiter_contact_count = torch.zeros((batch_size,), dtype=torch.int64, device=device)

    directions = np.column_stack((np.cos(angle_values), np.sin(angle_values)))
    center_array = np.asarray(center, dtype=np.float64).reshape(1, 2)
    for batch_index, result in enumerate(results):
        if result is None:
            continue
        code = int(_TOPOLOGY_CODE[result.topology])
        topology_code[batch_index] = code
        level[batch_index] = float(result.psi_boundary)
        axis_points[batch_index] = torch.as_tensor(result.magnetic_axis.point, dtype=dtype, device=device)

        x_count = min(len(result.x_points), 8)
        if x_count:
            x_values = np.asarray([point.point for point in result.x_points[:x_count]], dtype=np.float64)
            x_points[batch_index, :x_count] = torch.as_tensor(x_values, dtype=dtype, device=device)

        projection = result.fixed_angle_projection
        intersection_counts[batch_index] = torch.as_tensor(
            projection.intersection_counts,
            dtype=torch.int64,
            device=device,
        )
        if projection.valid:
            radii_values = np.asarray(projection.radii, dtype=np.float64)
            point_values = center_array + radii_values[:, None] * directions
            radii[batch_index] = torch.as_tensor(radii_values, dtype=dtype, device=device)
            points[batch_index] = torch.as_tensor(point_values, dtype=dtype, device=device)
            status_code[batch_index] = code + 7

        if return_dense_boundary:
            count = int(result.core_boundary.shape[0])
            core_boundary_count[batch_index] = count
            core_boundary[batch_index, :count] = torch.as_tensor(
                result.core_boundary,
                dtype=dtype,
                device=device,
            )

        contact_count = len(result.limiter_contacts)
        limiter_contact_count[batch_index] = contact_count
        if contact_count:
            contact_values = np.asarray([contact.point for contact in result.limiter_contacts], dtype=np.float64)
            limiter_contacts[batch_index, :contact_count] = torch.as_tensor(
                contact_values,
                dtype=dtype,
                device=device,
            )

        q = result.quality
        quality[batch_index] = torch.as_tensor(
            [
                q.max_flux_residual,
                q.normalized_flux_residual,
                q.closure_error,
                q.minimum_regular_gradient,
                float(q.limiter_violation_count),
                float(q.core_component_size),
            ],
            dtype=dtype,
            device=device,
        )

    # Preserve physical topology diagnostics even when the requested radial
    # projection is not single-valued.  ``found`` remains the fixed-angle
    # contract expected by the existing dispatcher.
    physical = torch.as_tensor(physical_found, dtype=torch.bool, device=device)
    topology_code = torch.where(physical, topology_code, torch.zeros_like(topology_code))
    level = torch.where(physical, level, torch.full_like(level, float("nan")))

    return FixedAngleBoundaryGpuResult(
        found=found,
        status_code=status_code,
        topology_code=topology_code,
        level=level,
        points=points,
        radii=radii,
        intersection_counts=intersection_counts,
        axis_points=axis_points,
        x_points=x_points,
        core_boundary=core_boundary,
        core_boundary_count=core_boundary_count,
        limiter_contacts=limiter_contacts,
        limiter_contact_count=limiter_contact_count,
        quality=quality,
    )
