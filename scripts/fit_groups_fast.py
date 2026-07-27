"""Быстрая раздельная калибровка — грубая сетка."""
import sys, re, math
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

GROUP1 = ["3855", "3856", "3859", "3862", "3863"]
GROUP2 = ["3854", "3857", "3858", "3864"]

# ── data loading (compact) ──
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
        if row[0] - last_t <= 1e-12: group.append(row)
        else: merged.append(np.mean(group, axis=0)); group = [row]
        last_t = row[0]
    merged.append(np.mean(group, axis=0))
    return np.vstack(merged)

def _load_shot(sid, cfg):
    n_pfc = int(cfg.pfc.n_coils); n_sol = int(cfg.sol.n_coils)
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
        coil_rows = np.column_stack([coil_rows[:, 0], np.concatenate(parts, axis=1), coil_rows[:, 1+sol_cols:]])
    t_coil = coil_rows[:, 0]; sol_raw = coil_rows[:, 1:1+n_sol]; pfc_raw = coil_rows[:, 1+n_sol:1+n_sol+n_pfc]
    dt = float(cfg.physics.t_step)
    t0 = max(t_ip[0], t_coil[0]); t1 = min(t_ip[-1], t_coil[-1])
    n = int((t1 - t0) / dt) + 1
    t = t0 + np.arange(n) * dt
    ip = np.interp(t, t_ip, ip_raw)
    pfc = np.column_stack([np.interp(t, t_coil, pfc_raw[:, j]) for j in range(n_pfc)])
    sol = np.column_stack([np.interp(t, t_coil, sol_raw[:, j]) for j in range(n_sol)])
    return {"id": sid, "t": t - t[0], "ip": ip, "pfc": pfc, "sol": sol}

def _simulate(shot, cfg, sigma, L, actuator_tau=0.0):
    model = PlasmaModel.from_settings(grid=cfg.grid, pfc=cfg.pfc, sol=cfg.sol, settings=cfg.physics, ip0=float(shot["ip"][0]))
    model.sigma = sigma; model.inductance_L = L; model.actuator_tau = actuator_tau
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
    err = pred - true; rmse = float(np.sqrt(np.mean(err**2)))
    return float(rmse / max(float(np.max(true) - np.min(true)), 1e-12))

def score_params(shots, cfg, sigma, L, tau=0.0):
    nrmses = [nrmse(_simulate(sh, cfg, sigma=sigma, L=L, actuator_tau=tau), sh["ip"]) for sh in shots]
    return float(np.mean(nrmses))

def grid_search(shots, cfg, sigma_grid, L_grid, tau_values=[0.0]):
    best_score = float("inf"); best_params = None
    total = len(sigma_grid) * len(L_grid) * len(tau_values)
    done = 0
    for sigma in sigma_grid:
        for L in L_grid:
            for tau in tau_values:
                s = score_params(shots, cfg, sigma, L, tau)
                if s < best_score:
                    best_score = s; best_params = (sigma, L, tau)
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"  {done}/{total} best={best_score:.6f} (σ={best_params[0]:.4g}, L={best_params[1]:.4g}, τ={best_params[2]:.3f})")
    return best_params, best_score

# ── main ──
cfg = load_config(CONFIG)
print("Loading shots...")
all_shots = {sid: _load_shot(sid, cfg) for sid in GROUP1 + GROUP2}
shots1 = [all_shots[sid] for sid in GROUP1]
shots2 = [all_shots[sid] for sid in GROUP2]

# Grid: 12 points each (faster than 15)
sigma_grid = np.logspace(math.log10(1e6), math.log10(2e7), 12)
L_grid = np.logspace(math.log10(1e-8), math.log10(1e-6), 12)

# Group 1: tau=0 only (we know tau doesn't help)
print("\n=== GROUP 1: tau-insensitive (5 shots), sigma×L grid (tau=0) ===")
best1, nrmse1 = grid_search(shots1, cfg, sigma_grid, L_grid, [0.0])
print(f"  Best1: σ={best1[0]:.6g}, L={best1[1]:.6g}, τ={best1[2]:.3f}, NRMSE={nrmse1:.6f}")

# Group 2: sigma×L×tau
print("\n=== GROUP 2: tau-sensitive (4 shots), sigma×L×tau grid ===")
tau_grid2 = [0.0, 0.01, 0.02, 0.03, 0.05]
best2, nrmse2 = grid_search(shots2, cfg, sigma_grid, L_grid, tau_grid2)
print(f"  Best2: σ={best2[0]:.6g}, L={best2[1]:.6g}, τ={best2[2]:.3f}, NRMSE={nrmse2:.6f}")

# Global baseline
nrmse_global_1 = score_params(shots1, cfg, 4472135.95499958, 2.6826957952797275e-07, 0.0)
nrmse_global_2 = score_params(shots2, cfg, 4472135.95499958, 2.6826957952797275e-07, 0.0)
print(f"\nGlobal baseline: G1={nrmse_global_1:.6f}, G2={nrmse_global_2:.6f}")

# ── Cross-evaluation ──
g1_with_g2 = score_params(shots1, cfg, best2[0], best2[1], best2[2])
g2_with_g1 = score_params(shots2, cfg, best1[0], best1[1], best1[2])
g1_with_g1 = score_params(shots1, cfg, best1[0], best1[1], best1[2])
g2_with_g2 = score_params(shots2, cfg, best2[0], best2[1], best2[2])

# ── Visualisation ──
print("\nPlotting...")

# Figure 1: all 9 shots grid
fig, axes = plt.subplots(3, 3, figsize=(20, 17))
axes = axes.flatten()
all_sids = GROUP1 + GROUP2

for i, sid in enumerate(all_sids):
    ax = axes[i]; sh = all_shots[sid]
    in_g1 = sid in GROUP1
    bs, bl, bt = best1 if in_g1 else best2
    
    ip_global = _simulate(sh, cfg, sigma=4472135.95499958, L=2.6826957952797275e-07, actuator_tau=0.0)
    n_global = nrmse(ip_global, sh["ip"])
    ip_group = _simulate(sh, cfg, sigma=bs, L=bl, actuator_tau=bt)
    n_group = nrmse(ip_group, sh["ip"])
    
    ax.plot(sh["t"], sh["ip"]/1e3, "k-", lw=2.2, label="Измеренный")
    ax.plot(sh["t"], ip_global/1e3, color="#d62728", lw=1.3, ls="--", alpha=0.85,
            label=f"Общий (NRMSE={n_global:.4f})")
    ax.plot(sh["t"], ip_group/1e3, color="#2ca02c", lw=1.8, alpha=0.9,
            label=f"Групповой (NRMSE={n_group:.4f})")
    
    delta = n_group - n_global
    grp_lbl = "G1" if in_g1 else "G2"
    ax.set_title(f"#{sid} [{grp_lbl}] | общий={n_global:.4f} → групп.={n_group:.4f} (Δ={delta:+.4f})",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("t, с"); ax.set_ylabel("Ip, кА")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(True, alpha=0.25)

fig.suptitle(
    f"Раздельная калибровка: единый фит vs групповые\n"
    f"G1: σ={best1[0]:.4g}, L={best1[1]:.4g}, τ={best1[2]:.3f} | "
    f"G2: σ={best2[0]:.4g}, L={best2[1]:.4g}, τ={best2[2]:.3f}",
    fontsize=12, y=1.01,
)
fig.tight_layout()
p1 = OUT_DIR / "groups_split_fit_grid.png"
fig.savefig(p1, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {p1}")

# Figure 2: per-group overlay
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

for ax, sids, bs, bl, bt, title in [
    (ax1, GROUP1, best1[0], best1[1], best1[2], "Группа 1: tau-нечувствительные"),
    (ax2, GROUP2, best2[0], best2[1], best2[2], "Группа 2: tau-чувствительные"),
]:
    colors = plt.cm.tab10(np.linspace(0, 1, len(sids)))
    nrmses_global = []; nrmses_group = []
    for j, sid in enumerate(sids):
        sh = all_shots[sid]; c = colors[j]
        ip_g = _simulate(sh, cfg, sigma=4472135.95499958, L=2.6826957952797275e-07, actuator_tau=0.0)
        ip_p = _simulate(sh, cfg, sigma=bs, L=bl, actuator_tau=bt)
        ng = nrmse(ip_g, sh["ip"]); np_ = nrmse(ip_p, sh["ip"])
        nrmses_global.append(ng); nrmses_group.append(np_)
        ax.plot(sh["t"], sh["ip"]/1e3, color=c, lw=2.0, alpha=0.8, label=f"#{sid} изм.")
        ax.plot(sh["t"], ip_p/1e3, color=c, lw=1.3, ls="--", alpha=0.7, label=f"#{sid} фит ({np_:.4f})")
    
    ax.set_title(
        f"{title}\nσ={bs:.4g}, L={bl:.4g}, τ={bt:.3f} | "
        f"NRMSE(общ)={np.mean(nrmses_global):.4f} → NRMSE(груп)={np.mean(nrmses_group):.4f}",
        fontsize=11,
    )
    ax.set_xlabel("t, с"); ax.set_ylabel("Ip, кА")
    ax.legend(fontsize=7, loc="upper left", ncol=2); ax.grid(True, alpha=0.25)

fig.suptitle("Раздельная калибровка: групповые фиты", fontsize=14, y=1.01)
fig.tight_layout()
p2 = OUT_DIR / "groups_split_fit_per_group.png"
fig.savefig(p2, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {p2}")

# ── Summary ──
print(f"\n{'='*75}")
print("ИТОГИ")
print(f"{'='*75}")
g0_s, g0_l, g0_t = 4472135.95499958, 2.6826957952797275e-07, 0.0
print(f"{'':<25} | {'σ':<16} | {'L':<16} | {'τ':<8} | {'NRMSE(G1)':<12} | {'NRMSE(G2)':<12}")
print("-" * 85)
print(f"{'Единый фит':<25} | {g0_s:<16.6g} | {g0_l:<16.6g} | {g0_t:<8.3f} | {nrmse_global_1:<12.6f} | {nrmse_global_2:<12.6f}")
print(f"{'Фиты G1':<25} | {best1[0]:<16.6g} | {best1[1]:<16.6g} | {best1[2]:<8.3f} | {g1_with_g1:<12.6f} | {g2_with_g1:<12.6f}")
print(f"{'Фиты G2':<25} | {best2[0]:<16.6g} | {best2[1]:<16.6g} | {best2[2]:<8.3f} | {g1_with_g2:<12.6f} | {g2_with_g2:<12.6f}")

print(f"\nDone. Script: scripts/fit_groups_fast.py")
