from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tokamak_control.control.hinf_current import HinftyCurrentController
from tokamak_control.control.hinf_joint import HinftyJointController
from tokamak_control.control.lqr_current import LQRCurrentController
from tokamak_control.control.lqr_joint import LQRJointController


class _FakeIpModel:
    def __init__(self) -> None:
        self.pfc = SimpleNamespace(n_coils=2)
        self.sol = SimpleNamespace(n_coils=1)
        self.state = SimpleNamespace(Ip=100.0)
        self.Ip0 = 100.0
        self.t_step = 1.0e-3
        self._b = np.array([0.12, -0.08, 0.05], dtype=float)

    def predict_Ip_decay_baseline_next(self) -> float:
        return 90.0

    def get_ip_B_row(self) -> np.ndarray:
        return self._b.copy()


def _active(action) -> np.ndarray:
    return np.concatenate([np.asarray(action.pfc_derivs, dtype=float), np.asarray(action.sol_derivs, dtype=float)])


def test_hinf_current_large_gamma_matches_lqr_current_next_step_model() -> None:
    model = _FakeIpModel()
    params = dict(q_ip=3.0, r_pfc=1.0e-3, r_sol=2.0e-3, ridge=1.0e-12, u_clip=None)

    lqr = LQRCurrentController(**params)
    hinf = HinftyCurrentController(**params, gamma=1.0e9)

    lqr_u = _active(lqr.compute_control(model=model, Ip_ref=150.0))
    hinf_u = _active(hinf.compute_control(model=model, Ip_ref=150.0))

    assert np.allclose(hinf_u, lqr_u, rtol=1.0e-8, atol=1.0e-8)


def test_hinf_joint_without_boundary_large_gamma_matches_lqr_joint_ip_term() -> None:
    model = _FakeIpModel()
    params = dict(q_error=5.0, q_ip=3.0, r_pfc=1.0e-3, r_sol=2.0e-3, ridge=1.0e-12, u_clip=None)

    lqr = LQRJointController(**params)
    hinf = HinftyJointController(**params, gamma=1.0e9)

    common = dict(
        model=model,
        psi=np.zeros((2, 2), dtype=float),
        boundary_poly=None,
        center=(0.0, 0.0),
        measure_angles=np.zeros((0,), dtype=float),
        ref_radii=np.zeros((0,), dtype=float),
        Ip_ref=150.0,
    )
    lqr_u = _active(lqr.compute_control(**common))
    hinf_u = _active(hinf.compute_control(**common))

    assert np.allclose(hinf_u, lqr_u, rtol=1.0e-8, atol=1.0e-8)
