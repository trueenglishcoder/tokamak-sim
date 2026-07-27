"""Визуализация табличного vs симулированного Ip — самодостаточный скрипт."""
import sys, math, re, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
OUT_DIR = REPO / "output/data_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIGMA = 4472135.95499958
L = 2.6826957952797275e-07

# ── helpers ──────────────────────────────────────────────
def _read_csv(path, ncols):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            if parts and parts[-1] == "":
                parts = parts[:-1]
            if len(parts) != ncols:
                continue
            rows.append([float(x) for x in parts])
    return np.array(rows, dtype=float)

def _merge_dupes(rows, path):
    merged, group = [], [rows[0]]
    last_t = rows[0, 0]
    for row in rows[1:]:
        dt = row[0] - last_t
        if dt <= 1e-12:
            group.append(row)
        else:
            merged.append(np.mean(group, axis=0))
            group = [row]
        last_t = row[0]
    merged.append(np.mean(group, axis=0))
    return np.vstack(merged)

def _load_shots():
    # discover
    ip_map = {}
    for p in sorted(IP_DIR.iterdir()):
        m = IP_PAT.fullmatch(p.name)
        if m: ip_map[m.group(1)] = p
    coil_map = {}
    for p in sorted(COILS_DIR.iterdir()):
        m = COIL_PAT.fullmatch(p.name)
        if m: coil_map[m.group(1)] = p
    sids = sorted(set(ip_map) & set(coil_map), key=int)

    cfg = load_config(CONFIG)
    dt = float(cfg.physics.t_step)
    n_pfc = int(cfg.pfc.n_coils)
    n_sol = int(cfg.sol.n_coils)
    expected_coil = 1 + n_sol + n_pfc

    shots = []
    for sid in sids:
        ip_rows = _merge_dupes(_read_csv(ip_map[sid], 2), ip_map[sid])
        t_ip, ip_raw = ip_rows[:, 0], ip_rows[:, 1]

        # try split SOL first
        try:
            coil_rows = _merge_dupes(_read_csv(coil_map[sid], expected_coil), coil_map[sid])
            sol_cols = n_sol
        except Exception:
            group_sizes = [int(np.asarray(g, dtype=float).reshape(-1, 2).shape[0]) for g in cfg.sol.element_positions]
            sol_cols = len(group_sizes)
            coil_rows = _merge_dupes(_read_csv(coil_map[sid], 1 + sol_cols + n_pfc), coil_map[sid])
            sol_source = coil_rows[:, 1:1+sol_cols]
            parts = [np.repeat(sol_source[:, i:i+1]/float(n), int(n), axis=1) for i, n in enumerate(group_sizes)]
            sol_expanded = np.concatenate(parts, axis=1)
            coil_rows = np.column_stack([coil_rows[:, 0], sol_expanded, coil_rows[:, 1+sol_cols:]])

        t_coil = coil_rows[:, 0]
        sol_raw = coil_rows[:, 1:1+n_sol]
        pfc_raw = coil_rows[:, 1+n_sol:1+n_sol+n_pfc]

        t0 = max(t_ip[0], t_coil[0])
        t1 = min(t_ip[-1], t_coil[-1])
        n = int((t1 - t0) / dt) + 1
        t = t0 + np.arange(n) * dt
        ip = np.interp(t, t_ip, ip_raw)
        pfc = np.column_stack([np.interp(t, t_coil, pfc_raw[:, j]) for j in range(n_pfc)])
        sol = np.column_stack([np.interp(t, t_coil, sol_raw[:, j]) for j in range(n_sol)])
        t_local = t - t[0]
        shots.append({
            "id": sid, "t": t_local, "ip": ip,
            "pfc": pfc, "sol": sol,
        })

    return shots, cfg

def _simulate(shot, cfg):
    model = PlasmaModel.from_settings(
        grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol,
        settings=cfg.physics, ip0=float(shot["ip"][0]),
    )
    model.sigma = SIGMA
    model.inductance_L = L
    model.actuator_tau = 0.0
    model.pfc_deriv_limit = None
    model.sol_deriv_limit = None
    model.pfc_current_limit = None
    model.sol_current_limit = None

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

# ── main ─────────────────────────────────────────────────
shots, cfg = _load_shots()
print(f"Loaded {len(shots)} shots")

all_rmse, all_nrmse = [], []
colors = plt.cm.tab10(np.linspace(0, 1, len(shots)))

# ── overlay ──
fig, ax = plt.subplots(figsize=(18, 10))
for i, sh in enumerate(shots):
    ip_sim = _simulate(sh, cfg)
    err = ip_sim - sh["ip"]
    rmse = float(np.sqrt(np.mean(err**2)))
    span = max(float(np.max(sh["ip"]) - np.min(sh["ip"])), 1e-12)
    nrmse = float(rmse / span)
    all_rmse.append(rmse); all_nrmse.append(nrmse)

    c = colors[i]
    ax.plot(sh["t"], sh["ip"]/1e3, color=c, lw=2.0, alpha=0.85,
            label=f"#{sh['id']} изм.")
    ax.plot(sh["t"], ip_sim/1e3, color=c, lw=1.2, ls="--", alpha=0.7,
            label=f"#{sh['id']} сим. (NRMSE={nrmse:.4f})")

ax.set_xlabel("t, с", fontsize=13)
ax.set_ylabel("Ip, кА", fontsize=13)
ax.set_title(
    f"Калиброванные (+15%, обрез.): измеренный vs симулированный Ip\n"
    f"σ = {SIGMA:.4g}, L = {L:.4g}, actuator_tau = 0, R_wall = 0",
    fontsize=14,
)
ax.legend(loc="upper left", fontsize=7.5, ncol=2)
ax.grid(True, alpha=0.25)
fig.tight_layout()
p1 = OUT_DIR / "calibrated_all_shots_overlay.png"
fig.savefig(p1, dpi=180)
plt.close(fig)
print(f"Saved: {p1}")

# ── grid ──
n = len(shots)
cols = 3; rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows))
axes = axes.flatten()
for i, sh in enumerate(shots):
    ax = axes[i]
    ip_sim = _simulate(sh, cfg)
    err = ip_sim - sh["ip"]
    rmse = float(np.sqrt(np.mean(err**2)))
    span = max(float(np.max(sh["ip"]) - np.min(sh["ip"])), 1e-12)
    nrmse = float(rmse / span)

    ax.plot(sh["t"], sh["ip"]/1e3, "b-", lw=2.0, label="Измеренный Ip")
    ax.plot(sh["t"], ip_sim/1e3, "r--", lw=1.8, label="Симулированный Ip")
    ax.set_title(f"Разряд #{sh['id']}  |  NRMSE = {nrmse:.4f}", fontsize=11)
    ax.set_xlabel("t, с"); ax.set_ylabel("Ip, кА")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

for j in range(n, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    f"Калиброванные (+15%, обрез.): покадровое сравнение\nσ = {SIGMA:.4g}, L = {L:.4g}",
    fontsize=14, y=1.01,
)
fig.tight_layout()
p2 = OUT_DIR / "calibrated_per_shot_grid.png"
fig.savefig(p2, dpi=180)
plt.close(fig)
print(f"Saved: {p2}")

# ── stats ──
print(f"\n{'='*50}")
print(f"Mean RMSE:  {np.mean(all_rmse):.1f} A")
print(f"Mean NRMSE: {np.mean(all_nrmse):.4f}")
for i, sh in enumerate(shots):
    print(f"  #{sh['id']}: RMSE={all_rmse[i]:.0f} A, NRMSE={all_nrmse[i]:.4f}")
print("Done.")
