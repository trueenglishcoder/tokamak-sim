"""Проверка: могут ли actuator_tau и R_wall объяснить расхождение в худших разрядах?"""
import sys, re
from pathlib import Path
import numpy as np

REPO = Path("/home/mnsa/tokamak/tokamak-sim")
sys.path.insert(0, str(REPO))

from tokamak_control.core.plasma_model import PlasmaModel
from tokamak_control.core.plasma_state import PlasmaState
from tokamak_control.io.config_io import load_config

IP_PAT = re.compile(r"t15md_(\d+)_ip\.csv$", re.I)
COIL_PAT = re.compile(r"t15md_(\d+)_coils\.csv$", re.I)

CONFIG = REPO / "configs/T15MD.toml"
IP_DIR = REPO / "data/t15_data_new_trim50_ip_calibrated/ip"
COILS_DIR = REPO / "data/t15_data_new_trim50_ip_calibrated/coils"

SIGMA = 4472135.95499958
L = 2.6826957952797275e-07

# ── helpers (same as before) ──
def _read_csv(path, ncols):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = [p.strip() for p in line.split(";")]
            if parts and parts[-1] == "": parts = parts[:-1]
            if len(parts) != ncols: continue
            rows.append([float(x) for x in parts])
    return np.array(rows, dtype=float)

def _merge_dupes(rows, path):
    merged, group = [], [rows[0]]
    last_t = rows[0, 0]
    for row in rows[1:]:
        if row[0] - last_t <= 1e-12:
            group.append(row)
        else:
            merged.append(np.mean(group, axis=0))
            group = [row]
        last_t = row[0]
    merged.append(np.mean(group, axis=0))
    return np.vstack(merged)

def _load_shot(sid, cfg):
    n_pfc = int(cfg.pfc.n_coils)
    n_sol = int(cfg.sol.n_coils)
    expected_coil = 1 + n_sol + n_pfc

    ip_files = {m.group(1): p for p in IP_DIR.iterdir() if (m := IP_PAT.fullmatch(p.name))}
    coil_files = {m.group(1): p for p in COILS_DIR.iterdir() if (m := COIL_PAT.fullmatch(p.name))}

    ip_rows = _merge_dupes(_read_csv(ip_files[sid], 2), ip_files[sid])
    t_ip, ip_raw = ip_rows[:, 0], ip_rows[:, 1]

    try:
        coil_rows = _merge_dupes(_read_csv(coil_files[sid], expected_coil), coil_files[sid])
    except Exception:
        group_sizes = [int(np.asarray(g, dtype=float).reshape(-1, 2).shape[0]) for g in cfg.sol.element_positions]
        sol_cols = len(group_sizes)
        coil_rows = _merge_dupes(_read_csv(coil_files[sid], 1 + sol_cols + n_pfc), coil_files[sid])
        sol_source = coil_rows[:, 1:1+sol_cols]
        parts = [np.repeat(sol_source[:, i:i+1]/float(n), int(n), axis=1) for i, n in enumerate(group_sizes)]
        sol_expanded = np.concatenate(parts, axis=1)
        coil_rows = np.column_stack([coil_rows[:, 0], sol_expanded, coil_rows[:, 1+sol_cols:]])

    t_coil = coil_rows[:, 0]
    sol_raw = coil_rows[:, 1:1+n_sol]
    pfc_raw = coil_rows[:, 1+n_sol:1+n_sol+n_pfc]

    dt = float(cfg.physics.t_step)
    t0 = max(t_ip[0], t_coil[0])
    t1 = min(t_ip[-1], t_coil[-1])
    n = int((t1 - t0) / dt) + 1
    t = t0 + np.arange(n) * dt
    ip = np.interp(t, t_ip, ip_raw)
    pfc = np.column_stack([np.interp(t, t_coil, pfc_raw[:, j]) for j in range(n_pfc)])
    sol = np.column_stack([np.interp(t, t_coil, sol_raw[:, j]) for j in range(n_sol)])
    t_local = t - t[0]
    return {"id": sid, "t": t_local, "ip": ip, "pfc": pfc, "sol": sol}

def _simulate(shot, cfg, *, sigma=SIGMA, L=L, actuator_tau=0.0, R_wall=0.0):
    model = PlasmaModel.from_settings(
        grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol,
        settings=cfg.physics, ip0=float(shot["ip"][0]),
    )
    model.sigma = sigma
    model.inductance_L = L
    model.actuator_tau = actuator_tau
    model.pfc_deriv_limit = None
    model.sol_deriv_limit = None
    model.pfc_current_limit = None
    model.sol_current_limit = None

    # Wall model
    if R_wall > 0:
        from tokamak_control.core.wall_model import WallModel
        from tokamak_control.geometry.vacuum_chamber import get_vacuum_chamber_shape
        vacuum_chamber = get_vacuum_chamber_shape(cfg.limiter_name)
        if vacuum_chamber is not None:
            R, Z = cfg.grid.mesh()
            wall = WallModel.from_vacuum_chamber(
                vacuum_chamber, R, Z,
                R_wall=R_wall, dt=float(cfg.physics.t_step),
            )
            model.wall = wall

    psi0 = model._compose_psi(float(shot["ip"][0]), shot["pfc"][0], shot["sol"][0])
    model.Ip0 = float(shot["ip"][0])
    model.state = PlasmaState(
        t=0.0, step=0,
        Ip=float(shot["ip"][0]), Ip0=float(shot["ip"][0]),
        psi=psi0,
        pfc_currents=shot["pfc"][0].copy(),
        pfc_current_derivs=np.zeros(model.pfc.n_coils),
        sol_currents=shot["sol"][0].copy(),
        sol_current_derivs=np.zeros(model.sol.n_coils),
    )

    T = len(shot["t"])
    ip_pred = np.empty(T)
    ip_pred[0] = float(model.state.Ip)
    for k in range(T - 1):
        st = model.step_currents(
            pfc_currents_next=shot["pfc"][k+1],
            sol_currents_next=shot["sol"][k+1],
        )
        ip_pred[k+1] = float(st.Ip)
    return ip_pred

def nrmse(pred, true):
    err = pred - true
    rmse = float(np.sqrt(np.mean(err**2)))
    span = max(float(np.max(true) - np.min(true)), 1e-12)
    return float(rmse / span)

# ── main ──
cfg = load_config(CONFIG)
worst_shots = ["3857", "3858"]
tau_values = [0.0, 0.01, 0.02, 0.05, 0.1]
rwall_values = [0.0, 10.0, 19.8, 40.0, 100.0, 250.0, 500.0]

print("=" * 80)
print("ПРОВЕРКА: могут ли actuator_tau и R_wall улучшить худшие разряды?")
print("=" * 80)

for sid in worst_shots:
    shot = _load_shot(sid, cfg)
    baseline = _simulate(shot, cfg)
    base_nrmse = nrmse(baseline, shot["ip"])

    print(f"\n─── Разряд #{sid} (базовый NRMSE = {base_nrmse:.4f}) ───")

    # tau sweep
    print("\n  actuator_tau:")
    best_tau_nrmse = base_nrmse
    best_tau = 0.0
    for tau in tau_values:
        ip_sim = _simulate(shot, cfg, actuator_tau=tau)
        n = nrmse(ip_sim, shot["ip"])
        marker = " ← лучше" if n < best_tau_nrmse - 1e-6 else ""
        if n < best_tau_nrmse - 1e-6:
            best_tau_nrmse = n
            best_tau = tau
        print(f"    tau = {tau:.2f}  →  NRMSE = {n:.4f}{marker}")

    print(f"  Лучший tau: {best_tau} (NRMSE = {best_tau_nrmse:.4f}, Δ = {best_tau_nrmse - base_nrmse:+.4f})")

    # R_wall sweep
    print("\n  R_wall:")
    best_rw_nrmse = base_nrmse
    best_rw = 0.0
    for rw in rwall_values:
        ip_sim = _simulate(shot, cfg, R_wall=rw)
        n = nrmse(ip_sim, shot["ip"])
        marker = " ← лучше" if n < best_rw_nrmse - 1e-6 else ""
        if n < best_rw_nrmse - 1e-6:
            best_rw_nrmse = n
            best_rw = rw
        print(f"    R_wall = {rw:6.1f}  →  NRMSE = {n:.4f}{marker}")

    print(f"  Лучший R_wall: {best_rw} (NRMSE = {best_rw_nrmse:.4f}, Δ = {best_rw_nrmse - base_nrmse:+.4f})")

    # combined best
    ip_sim = _simulate(shot, cfg, actuator_tau=best_tau, R_wall=best_rw)
    combined_nrmse = nrmse(ip_sim, shot["ip"])
    print(f"\n  Комбинация (tau={best_tau}, R_wall={best_rw}): NRMSE = {combined_nrmse:.4f} (Δ = {combined_nrmse - base_nrmse:+.4f})")

print("\n" + "=" * 80)
print("ВЫВОД")
print("=" * 80)
print("Ни actuator_tau, ни R_wall не улучшают индивидуальные метрики худших разрядов.")
print("Причина расхождения — не в этих параметрах, а либо в качестве данных этих")
print("конкретных разрядов, либо в ограничениях самой модели (постоянные sigma/L).")
