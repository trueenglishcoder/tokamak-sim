"""Раздельная калибровка sigma, L для двух групп разрядов + визуализация."""
import sys, re, math, json
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

# ── helpers ──
IP_PAT = re.compile(r"t15md_(\d+)_ip\.csv$", re.I)
COIL_PAT = re.compile(r"t15md_(\d+)_coils\.csv$", re.I)

CONFIG = REPO / "configs/T15MD.toml"
IP_DIR = REPO / "data/t15_data_new_trim50_ip_calibrated/ip"
COILS_DIR = REPO / "data/t15_data_new_trim50_ip_calibrated/coils"
OUT_DIR = REPO / "output/data_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP1 = ["3855", "3856", "3859", "3862", "3863"]  # tau-insensitive, good
GROUP2 = ["3854", "3857", "3858", "3864"]           # tau-sensitive, worse

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
        sol_expanded = np.concatenate(parts, axis=1)
        coil_rows = np.column_stack([coil_rows[:, 0], sol_expanded, coil_rows[:, 1+sol_cols:]])
    t_coil = coil_rows[:, 0]
    sol_raw = coil_rows[:, 1:1+n_sol]; pfc_raw = coil_rows[:, 1+n_sol:1+n_sol+n_pfc]
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

def grid_search(shots, cfg, name, sigma_grid, L_grid, tau_values=[0.0]):
    """Grid search sigma × L × tau for a group of shots. Returns best params and full results."""
    best_score = float("inf")
    best_params = None
    all_results = []
    total = len(sigma_grid) * len(L_grid) * len(tau_values)
    done = 0

    for sigma in sigma_grid:
        for L in L_grid:
            for tau in tau_values:
                nrmses = []
                for sh in shots:
                    ip_sim = _simulate(sh, cfg, sigma=sigma, L=L, actuator_tau=tau)
                    nrmses.append(nrmse(ip_sim, sh["ip"]))
                mean_nrmse = float(np.mean(nrmses))
                all_results.append((sigma, L, tau, mean_nrmse))
                if mean_nrmse < best_score:
                    best_score = mean_nrmse
                    best_params = (sigma, L, tau)
                done += 1
                if done % 50 == 0:
                    print(f"  [{name}] {done}/{total} best={best_score:.6f}")

    all_results.sort(key=lambda x: x[3])
    return best_params, best_score, all_results

# ── main ──
cfg = load_config(CONFIG)
print("Loading shots...")
all_shots = {sid: _load_shot(sid, cfg) for sid in GROUP1 + GROUP2}

# Grids
sigma_grid = np.logspace(math.log10(1e6), math.log10(2e7), 15)
L_grid = np.logspace(math.log10(1e-8), math.log10(1e-6), 15)
tau_grid = [0.0, 0.01, 0.02, 0.03, 0.05]

groups = [
    ("group1_tau_insensitive", GROUP1, "Группа 1: tau-нечувствительные (#3855, #3856, #3859, #3862, #3863)"),
    ("group2_tau_sensitive", GROUP2, "Группа 2: tau-чувствительные (#3854, #3857, #3858, #3864)"),
]

results = {}
for key, sids, desc in groups:
    print(f"\n{'='*60}")
    print(f"Fitting: {desc}")
    print(f"{'='*60}")
    shots = [all_shots[sid] for sid in sids]
    best, best_nrmse, all_res = grid_search(shots, cfg, key, sigma_grid, L_grid, tau_grid)
    results[key] = {
        "sids": sids, "desc": desc,
        "best_sigma": best[0], "best_L": best[1], "best_tau": best[2],
        "best_nrmse": best_nrmse, "all_results": all_res[:10],
    }
    print(f"  Best: sigma={best[0]:.6g}, L={best[1]:.6g}, tau={best[2]:.3f}, NRMSE={best_nrmse:.6f}")
    for i, (s, l, t, n) in enumerate(all_res[:5]):
        print(f"    #{i+1}: sigma={s:.6g}, L={l:.6g}, tau={t:.3f}, NRMSE={n:.6f}")

# ── Visualisation ──

# Figure 1: Both groups with their own best params, all 9 shots
fig, axes = plt.subplots(3, 3, figsize=(20, 17))
axes = axes.flatten()
all_sids = GROUP1 + GROUP2
group_best = {
    **{sid: results["group1_tau_insensitive"] for sid in GROUP1},
    **{sid: results["group2_tau_sensitive"] for sid in GROUP2},
}

for i, sid in enumerate(all_sids):
    ax = axes[i]
    sh = all_shots[sid]
    grp = group_best[sid]
    
    # Global best (single set for all)
    ip_global = _simulate(sh, cfg, sigma=4472135.95499958, L=2.6826957952797275e-07, actuator_tau=0.0)
    n_global = nrmse(ip_global, sh["ip"])
    
    # Group best
    ip_group = _simulate(sh, cfg, sigma=grp["best_sigma"], L=grp["best_L"], actuator_tau=grp["best_tau"])
    n_group = nrmse(ip_group, sh["ip"])
    
    ax.plot(sh["t"], sh["ip"]/1e3, "k-", lw=2.2, label="Измеренный")
    ax.plot(sh["t"], ip_global/1e3, color="#d62728", lw=1.3, ls="--", alpha=0.85,
            label=f"Общий фит (NRMSE={n_global:.4f})")
    ax.plot(sh["t"], ip_group/1e3, color="#2ca02c", lw=1.8, alpha=0.9,
            label=f"Групповой фит (NRMSE={n_group:.4f})")
    
    grp_label = "G1" if sid in GROUP1 else "G2"
    delta = n_group - n_global
    ax.set_title(f"#{sid} [{grp_label}]  |  общий={n_global:.4f} → групповой={n_group:.4f}  (Δ={delta:+.4f})",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("t, с"); ax.set_ylabel("Ip, кА")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(True, alpha=0.25)

fig.suptitle(
    "Сравнение: единый фит (σ=4.47×10⁶, L=2.68×10⁻⁷, tau=0) vs групповые фиты\n"
    f"G1: σ={results['group1_tau_insensitive']['best_sigma']:.4g}, L={results['group1_tau_insensitive']['best_L']:.4g}, tau={results['group1_tau_insensitive']['best_tau']:.3f}\n"
    f"G2: σ={results['group2_tau_sensitive']['best_sigma']:.4g}, L={results['group2_tau_sensitive']['best_L']:.4g}, tau={results['group2_tau_sensitive']['best_tau']:.3f}",
    fontsize=12, y=1.02,
)
fig.tight_layout()
p1 = OUT_DIR / "groups_split_fit_grid.png"
fig.savefig(p1, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {p1}")

# Figure 2: Per-group summary with all group shots overlaid
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

for ax, grp_key, title in [
    (ax1, "group1_tau_insensitive", "Группа 1: tau-нечувствительные"),
    (ax2, "group2_tau_sensitive", "Группа 2: tau-чувствительные"),
]:
    grp = results[grp_key]
    colors = plt.cm.tab10(np.linspace(0, 1, len(grp["sids"])))
    group_nrmses_global = []
    group_nrmses_group = []
    
    for j, sid in enumerate(grp["sids"]):
        sh = all_shots[sid]
        c = colors[j]
        
        ip_global = _simulate(sh, cfg, sigma=4472135.95499958, L=2.6826957952797275e-07, actuator_tau=0.0)
        n_g = nrmse(ip_global, sh["ip"])
        
        ip_grp = _simulate(sh, cfg, sigma=grp["best_sigma"], L=grp["best_L"], actuator_tau=grp["best_tau"])
        n_p = nrmse(ip_grp, sh["ip"])
        
        group_nrmses_global.append(n_g)
        group_nrmses_group.append(n_p)
        
        ax.plot(sh["t"], sh["ip"]/1e3, color=c, lw=2.0, alpha=0.8, label=f"#{sid} изм.")
        ax.plot(sh["t"], ip_grp/1e3, color=c, lw=1.3, ls="--", alpha=0.7, label=f"#{sid} фит ({n_p:.4f})")
    
    mean_global = np.mean(group_nrmses_global)
    mean_group = np.mean(group_nrmses_group)
    
    ax.set_title(
        f"{title}\n"
        f"σ={grp['best_sigma']:.4g}, L={grp['best_L']:.4g}, tau={grp['best_tau']:.3f} | "
        f"NRMSE(общий)={mean_global:.4f} → NRMSE(групповой)={mean_group:.4f}",
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

# ── Summary table ──
print(f"\n{'='*70}")
print("ИТОГИ РАЗДЕЛЬНОЙ КАЛИБРОВКИ")
print(f"{'='*70}")
print(f"{'Параметр':<25} | {'Единый фит':<22} | {'G1 (хорошие)':<22} | {'G2 (плохие)':<22}")
print("-" * 95)
g0_s, g0_l, g0_t = 4472135.95499958, 2.6826957952797275e-07, 0.0
g1 = results["group1_tau_insensitive"]
g2 = results["group2_tau_sensitive"]
print(f"{'sigma':<25} | {g0_s:<22.6g} | {g1['best_sigma']:<22.6g} | {g2['best_sigma']:<22.6g}")
print(f"{'L':<25} | {g0_l:<22.6g} | {g1['best_L']:<22.6g} | {g2['best_L']:<22.6g}")
print(f"{'tau':<25} | {g0_t:<22.3f} | {g1['best_tau']:<22.3f} | {g2['best_tau']:<22.3f}")

# Compute mean NRMSE for each group under each parameter set
def mean_nrmse_for_group(sids, sigma, L, tau):
    nrmses = []
    for sid in sids:
        ip_sim = _simulate(all_shots[sid], cfg, sigma=sigma, L=L, actuator_tau=tau)
        nrmses.append(nrmse(ip_sim, all_shots[sid]["ip"]))
    return float(np.mean(nrmses))

g1_global = mean_nrmse_for_group(GROUP1, g0_s, g0_l, g0_t)
g1_own = mean_nrmse_for_group(GROUP1, g1["best_sigma"], g1["best_L"], g1["best_tau"])
g2_global = mean_nrmse_for_group(GROUP2, g0_s, g0_l, g0_t)
g2_own = mean_nrmse_for_group(GROUP2, g2["best_sigma"], g2["best_L"], g2["best_tau"])

# Cross: G2 with G1 params, G1 with G2 params
g1_with_g2 = mean_nrmse_for_group(GROUP1, g2["best_sigma"], g2["best_L"], g2["best_tau"])
g2_with_g1 = mean_nrmse_for_group(GROUP2, g1["best_sigma"], g1["best_L"], g1["best_tau"])

print(f"\n{'NRMSE для':<25} | {'Единые пар-ры':<22} | {'Пар-ры G1':<22} | {'Пар-ры G2':<22}")
print("-" * 95)
print(f"{'Группа 1 (хорошие)':<25} | {g1_global:<22.6f} | {g1_own:<22.6f} | {g1_with_g2:<22.6f}")
print(f"{'Группа 2 (плохие)':<25} | {g2_global:<22.6f} | {g2_with_g1:<22.6f} | {g2_own:<22.6f}")
print(f"{'Все 9 разрядов':<25} | {np.mean([g1_global,g2_global]):<22.6f} | "
      f"{np.mean([g1_own,g2_with_g1]):<22.6f} | {np.mean([g1_with_g2,g2_own]):<22.6f}")

print(f"\nDone. All outputs in {OUT_DIR}")
