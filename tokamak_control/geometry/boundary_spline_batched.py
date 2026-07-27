"""Batched periodic cubic spline — векторизованные версии для GPU обучения."""
from __future__ import annotations
import torch


def fit_periodic_spline_batched(radii: torch.Tensor, h: float):
    """
    Фитирует периодический кубический сплайн для батча радиусов.
    
    Args:
        radii: (B, N) — радиусы для B элементов батча, N углов
        h: шаг между углами
    
    Returns:
        coeffs: (B, 4, N) — коэффициенты a, b, c, d
    """
    B, N = radii.shape
    device = radii.device
    dtype = radii.dtype

    # ── RHS: циклические разности ──
    r_next = torch.roll(radii, -1, dims=1)   # (B, N)
    r_prev = torch.roll(radii, 1, dims=1)
    rhs = 3.0 * (r_next - radii) / h - 3.0 * (radii - r_prev) / h  # (B, N)

    # ── Циклическая трёхдиагональная система ──
    lower = torch.full((B, N), h, dtype=dtype, device=device)
    diag  = torch.full((B, N), 4.0 * h, dtype=dtype, device=device)
    upper = torch.full((B, N), h, dtype=dtype, device=device)

    # Sherman-Morrison для цикличности
    gamma = -diag[:, 0]  # (B,)

    diag_mod = diag.clone()
    diag_mod[:, 0] = diag[:, 0] - gamma
    diag_mod[:, N-1] = diag[:, N-1] - upper[:, N-1] * lower[:, 0] / gamma

    lower_mod = lower.clone()
    upper_mod = upper.clone()
    lower_mod[:, 0] = 0.0
    upper_mod[:, N-1] = 0.0

    u = torch.zeros(B, N, dtype=dtype, device=device)
    u[:, 0] = gamma
    u[:, N-1] = upper[:, N-1]

    x = _thomas_batched(lower_mod, diag_mod, upper_mod, rhs)   # (B, N)
    z = _thomas_batched(lower_mod, diag_mod, upper_mod, u)     # (B, N)

    v_factor = x[:, 0] + x[:, N-1] * lower[:, 0] / gamma
    z_factor = z[:, 0] + z[:, N-1] * lower[:, 0] / gamma

    safe = torch.abs(z_factor) > 1e-15
    correction = torch.where(
        safe,
        v_factor / z_factor,
        torch.zeros(B, dtype=dtype, device=device),
    )
    c = x - correction.unsqueeze(1) * z  # (B, N)

    # ── Коэффициенты a, b, d ──
    a = radii  # (B, N)
    r_next = torch.roll(radii, -1, dims=1)
    c_next = torch.roll(c, -1, dims=1)
    b = (r_next - radii) / h - h * (2.0 * c + c_next) / 3.0
    d = (c_next - c) / (3.0 * h)

    return torch.stack([a, b, c, d], dim=1)  # (B, 4, N)


def evaluate_spline_at_angles_batched(coeffs, angles, query_angles):
    """
    Вычисляет значение сплайна в заданных углах для всего батча.
    
    Args:
        coeffs: (B, 4, N) — коэффициенты
        angles: (N,) — углы узлов
        query_angles: (Q,) — углы для вычисления
    
    Returns:
        values: (B, Q)
    """
    B = coeffs.shape[0]
    N = int(angles.numel())
    Q = int(query_angles.numel())
    device = coeffs.device
    dtype = coeffs.dtype

    h = float(angles[1] - angles[0])
    a0 = float(angles[0])
    period = float(angles[-1]) - a0 + h  # [a0, a0 + period)
    
    queries = query_angles.unsqueeze(0).expand(B, Q)  # (B, Q)
    
    # wrap angles to [a0, a0+period)
    shifted = queries - a0
    wrapped = shifted - period * torch.floor(shifted / period)
    queries_wrapped = wrapped + a0  # (B, Q)
    
    # segment index
    seg_idx = ((queries_wrapped - a0) / h).long()
    seg_idx = torch.clamp(seg_idx, 0, N - 1)  # (B, Q)
    
    # dx from knot point (in radians, h-scaled)
    knot_angles = a0 + seg_idx.float() * h  # (B, Q)
    dx = queries_wrapped - knot_angles  # (B, Q)
    
    # gather coefficients
    b_idx = torch.arange(B, device=device)[:, None].expand(B, Q)
    
    a_val = coeffs[b_idx, 0, seg_idx]  # (B, Q)
    b_val = coeffs[b_idx, 1, seg_idx]
    c_val = coeffs[b_idx, 2, seg_idx]
    d_val = coeffs[b_idx, 3, seg_idx]
    
    return a_val + b_val * dx + c_val * dx * dx + d_val * dx * dx * dx


def _thomas_batched(lower, diag, upper, rhs):
    """Алгоритм Томаса для батча трёхдиагональных систем.
    
    Args:
        lower: (B, N), diag: (B, N), upper: (B, N), rhs: (B, N)
    Returns:
        x: (B, N)
    """
    B, N = rhs.shape
    device = rhs.device
    dtype = rhs.dtype
    
    c_prime = torch.zeros(B, N, dtype=dtype, device=device)
    d_prime = torch.zeros(B, N, dtype=dtype, device=device)
    
    # Прямой ход
    c_prime[:, 0] = upper[:, 0] / diag[:, 0]
    d_prime[:, 0] = rhs[:, 0] / diag[:, 0]
    
    for i in range(1, N):
        denom = diag[:, i] - lower[:, i] * c_prime[:, i - 1]
        if i < N - 1:
            c_prime[:, i] = upper[:, i] / denom
        else:
            c_prime[:, i] = torch.zeros(B, dtype=dtype, device=device)
        d_prime[:, i] = (rhs[:, i] - lower[:, i] * d_prime[:, i - 1]) / denom
    
    # Обратный ход
    x = torch.zeros(B, N, dtype=dtype, device=device)
    x[:, N - 1] = d_prime[:, N - 1]
    
    for i in range(N - 2, -1, -1):
        x[:, i] = d_prime[:, i] - c_prime[:, i] * x[:, i + 1]
    
    return x


def compute_spline_width_score_batched(radii, angles):
    """Вычисляет spline_max_width для батча радиусов (векторизованно).
    
    Args:
        radii: (B, N) — радиусы
        angles: (N,) — углы
    
    Returns:
        scores: (B,) — ширина сплайна для каждого элемента батча
    """
    B, N = radii.shape
    device = radii.device
    dtype = radii.dtype
    
    h = float(angles[1] - angles[0])
    
    # Проверяем finiteness
    valid = torch.all(torch.isfinite(radii), dim=1)  # (B,)
    
    scores = torch.full((B,), -float("inf"), dtype=dtype, device=device)
    
    if not valid.any():
        return scores
    
    # Фитит сплайн только для валидных элементов
    valid_radii = radii[valid]  # (V, N)
    
    try:
        coeffs = fit_periodic_spline_batched(valid_radii, h)  # (V, 4, N)
        
        # Вычисляем радиус на углах 0 и π
        query_angles = torch.tensor([0.0, torch.pi], dtype=dtype, device=device)
        values = evaluate_spline_at_angles_batched(coeffs, angles, query_angles)  # (V, 2)
        
        valid_scores = values[:, 0] + values[:, 1]  # (V,)
        scores[valid] = valid_scores
    except Exception:
        pass
    
    return scores
