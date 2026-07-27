"""Сетка визуализации: влияние actuator_tau на все 9 разрядов."""
import sys, re
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

# ── data loading (same as before) ──
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
    t0 = max(t_ip[0], t_coil[0]); t1 = min(t_ip[-1], t_coil[-1])
    n = int((t1 - t0) / dt) + 1
    t = t0 + np.arange(n) * dt
    ip = np.interp(t, t_ip, ip_raw)
    pfc = np.column_stack([np.interp(t, t_coil, pfc_raw[:, j]) for j in range(n_pfc)])
    sol = np.column_stack([np.interp(t, t_coil, sol_raw[:, j]) for j in range(n_sol)])
    return {"id": sid, "t": t - t[0], "ip": ip, "pfc": pfc, "sol": sol}

def _simulate(shot, cfg, actuator_tau=0.0):
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics, ip0=float(shot["ip"][0]))
    model.sigma = SIGMA; model.inductance_L = L
    model.actuator_tau = actuator_tau
    model.pfc_deriv_limit = None; model.sol_deriv_limit = None
    model.pfc_current_limit = None; model.sol_current_limit = None
    psi0 = model._compose_psi(float(shot["ip"][0]), shot["pfc"][0], shot["sol"][0])
    model.Ip0 = float(shot["ip"][0])
    model.state = PlasmaState(t=0.0, step=0, Ip=float(shot["ip"][0]), Ip0=float(shot["ip"][0]), psi=psi0,
        pfc_currents=shot["pfc"][0].copy(), pfc_current_derivs=np.zeros(model.pfc.n_coils),
        sol_currents=shot["sol"][0].copy(), sol_current_derivs=np.zeros(model.sol.n_coils))
    T = len(shot["t"]); ip_pred = np.empty(T); ip_pred[0] = float(model.state.Ip)
    for k in range(T - 1):
        st = model.step_currents(pfc_currents_next=shot["pfc"][k+1], sol_currents_next=shot["sol"][k+1])
        ip_pred[k+1] = float(st.Ip)
    return ip_pred

def nrmse(pred, true):
    err = pred - true
    rmse = float(np.sqrt(np.mean(err**2)))
    span = max(float(np.max(true) - np.min(true)), 1e-12)
    return float(rmse / span)

# ── main ──
cfg = load_config(CONFIG)
shots_ids = ["3854","3855","3856","3857","3858","3859","3862","3863","3864"]

# Pre-simulate all
results = {}
for sid in shots_ids:
    shot = _load_shot(sid, cfg)
    ip0 = _simulate(shot, cfg, actuator_tau=0.0)
    
    # find best tau
    best_n, best_tau = nrmse(ip0, shot["ip"]), 0.0
    best_ip = ip0
    for t in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]:
        ip_t = _simulate(shot, cfg, actuator_tau=t)
        nt = nrmse(ip_t, shot["ip"])
        if nt < best_n:
            best_n, best_tau, best_ip = nt, t, ip_t
    
    results[sid] = {
        "shot": shot,
        "ip_tau0": ip0,
        "nrmse0": nrmse(ip0, shot["ip"]),
        "ip_best": best_ip,
        "best_tau": best_tau,
        "nrmse_best": best_n,
    }
    print(f"#{sid}: tau=0 → NRMSE={results[sid]['nrmse0']:.4f}, best tau={best_tau} → NRMSE={best_n:.4f}")

# ── plot: 3×3 grid with tau=0 and best tau ──
n = len(shots_ids)
cols = 3; rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(20, 5.5 * rows))
axes = axes.flatten()

for i, sid in enumerate(shots_ids):
    ax = axes[i]
    r = results[sid]
    sh = r["shot"]
    
    # Measured (always the same)
    ax.plot(sh["t"], sh["ip"]/1e3, "k-", lw=2.2, label="Измеренный Ip", zorder=3)
    
    # tau=0
    ax.plot(sh["t"], r["ip_tau0"]/1e3, color="#d62728", lw=1.5, ls="--", alpha=0.9,
            label=f"tau=0 (NRMSE={r['nrmse0']:.4f})")
    
    # Best tau (only if different from 0)
    if r["best_tau"] > 0:
        ax.plot(sh["t"], r["ip_best"]/1e3, color="#2ca02c", lw=1.8, ls="-", alpha=0.9,
                label=f"tau={r['best_tau']:.2f} (NRMSE={r['nrmse_best']:.4f})")
        improved = r['nrmse0'] - r['nrmse_best']
        title = (f"Разряд #{sid}  |  tau=0: {r['nrmse0']:.4f}  →  "
                 f"tau={r['best_tau']:.2f}: {r['nrmse_best']:.4f}  (Δ={improved:+.4f})")
    else:
        title = f"Разряд #{sid}  |  tau=0: NRMSE={r['nrmse0']:.4f} (оптимум)"
    
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("t, с")
    ax.set_ylabel("Ip, кА")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.25)

for j in range(n, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    f"Влияние actuator_tau на точность воспроизведения Ip\n"
    f"Калиброванные данные (+15%, обрез.) | σ={SIGMA:.4g}, L={L:.4g}",
    fontsize=14, y=1.01,
)
fig.tight_layout()
path = OUT_DIR / "calibrated_tau_comparison_grid.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {path}")

# ── second plot: just the 4 shots that benefit from tau, with error bands ──
benefit_shots = [sid for sid in shots_ids if results[sid]["best_tau"] > 0]
print(f"Shots benefiting from tau: {benefit_shots}")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for i, sid in enumerate(benefit_shots):
    ax = axes[i]
    r = results[sid]
    sh = r["shot"]
    
    ax.plot(sh["t"], sh["ip"]/1e3, "k-", lw=2.2, label="Измеренный Ip")
    ax.plot(sh["t"], r["ip_tau0"]/1e3, color="#d62728", lw=1.5, ls="--",
            label=f"tau=0 (NRMSE={r['nrmse0']:.4f})")
    ax.plot(sh["t"], r["ip_best"]/1e3, color="#2ca02c", lw=2.0,
            label=f"tau={r['best_tau']:.2f} (NRMSE={r['nrmse_best']:.4f})")
    
    # Error delta
    err0 = (r["ip_tau0"] - sh["ip"]) / 1e3
    err_best = (r["ip_best"] - sh["ip"]) / 1e3
    ax.fill_between(sh["t"], 0, err0, alpha=0.1, color="#d62728", label="Ошибка tau=0")
    ax.fill_between(sh["t"], 0, err_best, alpha=0.15, color="#2ca02c", label="Ошибка best tau")
    
    improved = r['nrmse0'] - r['nrmse_best']
    pct = 100 * improved / r['nrmse0']
    ax.set_title(f"Разряд #{sid}: улучшение на {pct:.0f}% (ΔNRMSE={improved:+.4f})", fontsize=12)
    ax.set_xlabel("t, с"); ax.set_ylabel("Ip, кА")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

fig.suptitle("Разряды, где actuator_tau снижает ошибку", fontsize=14, y=1.01)
fig.tight_layout()
path2 = OUT_DIR / "calibrated_tau_benefit_detail.png"
fig.savefig(path2, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path2}")

# ── summary ──
print(f"\n{'='*60}")
print("СВОДКА")
print(f"{'='*60}")
mean0 = np.mean([r["nrmse0"] for r in results.values()])
print(f"Средний NRMSE (tau=0): {mean0:.4f}")
for sid in shots_ids:
    r = results[sid]
    flag = " ← улучшается" if r["best_tau"] > 0 else ""
    print(f"  #{sid}: tau=0={r['nrmse0']:.4f}  →  best tau={r['best_tau']:.2f} ({r['nrmse_best']:.4f}){flag}")
print("Done.")
