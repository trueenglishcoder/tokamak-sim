from __future__ import annotations

import inspect

from tokamak_control.core.batched_gpu_simulator import BatchedGpuTokamakSimulator


def test_batched_gpu_result_does_not_use_cpu_contour_extraction() -> None:
    source = inspect.getsource(BatchedGpuTokamakSimulator._result)
    assert "find_plasma_boundary_with_status" not in source
    assert "legacy_radii_at_angles" not in source
    assert "detach().cpu().numpy()" not in source
    assert "fixed_angle_boundary_gpu" in source
