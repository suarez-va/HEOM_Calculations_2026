"""
Collect the per-point result.json files into one table and plot log10(beta*k) vs log10(beta*Delta)
at fixed epsilon, alongside the Zusman and quantum nonadiabatic reference rates.

    python3 2_collect.py                            # config.yaml
    python3 2_collect.py --config config_smoke.yaml  # results of a test scan

Writes delta_grid/results.csv and delta_grid/scan.png. Points that have not finished yet are
reported as missing rather than silently omitted, so a partial scan is obvious.

Two HEOM rates per point: k_fit, the slope of a line fit of K2(t) = -P2eq ln|1 - P2/P2eq| over
[fitmin, fitmax] (config `rate:` block), and k_HEOM, the last point of k(t) (shown faded).
    python3 2_collect.py --fitmin 5 --fitmax 15   # refit from the saved trajectories, no HEOM rerun
    python3 2_collect.py --plot-fits              # also write fits.png: K2(t) + fit for every point
"""

import os
import csv
import argparse
import json

import numpy as np
from scipy.special import iv
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from libra_py import units
from run_heom_point import log10_delta_grid, fit_window, refit_rows, plot_fits

script_dir = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml", help="same config the scan was run with")
ap.add_argument("--fitmin", type=float, default=None, help="K2 fit window start, t/(beta*hbar) (default: config rate.fitmin)")
ap.add_argument("--fitmax", type=float, default=None, help="K2 fit window end, t/(beta*hbar) (default: config rate.fitmax)")
ap.add_argument("--plot-fits", action="store_true", help="also write fits.png (K2(t) and its fit, per point)")
args = ap.parse_args()
cfg_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
grid_dir = os.path.join(script_dir, cfg.get("runtime", {}).get("grid_dir", "delta_grid"))

log10_delta_ar = log10_delta_grid(cfg)
T = cfg["system"]["temperature"]
beta = units.hartree / (units.boltzmann * T)
Omega = cfg["system"]["Omega"] / beta
Gamma = cfg["system"]["Gamma"] / beta
Lambda = cfg["system"]["Lambda"] / beta
epsilon = cfg["system"]["epsilon"] / beta

rows, missing = [], []
for i in range(len(log10_delta_ar)):
    p = os.path.join(grid_dir, f"val{i}", "result.json")
    if not os.path.exists(p):
        missing.append(i)
        continue
    with open(p) as f:
        rows.append(json.load(f))

if missing:
    print(f"WARNING: {len(missing)} point(s) have no result.json yet: {missing}")
    print("         re-run: python3 1_run_delta_scan_local.py")
if not rows:
    raise SystemExit("no results found - run 1_run_delta_scan_local.py first")

rows.sort(key=lambda r: r["log10_Delta"])

# K2 line-fit rate: reuse result.json's fit if it used this window, else refit from the trajectory
fitmin, fitmax = fit_window(cfg)
fitmin = args.fitmin if args.fitmin is not None else fitmin
fitmax = args.fitmax if args.fitmax is not None else fitmax
nofit = refit_rows(rows, grid_dir, fitmin, fitmax)
if nofit:
    print(f"WARNING: no saved trajectory for point(s) {nofit} - no K2 fit for them")

csv_path = os.path.join(grid_dir, "results.csv")
fields = ["idx", "log10_Delta", "beta_Delta", "beta_epsilon", "k_HEOM", "log10_beta_k_HEOM",
          "k_fit", "log10_beta_k_fit", "fit_r", "fit_npts", "fitmin", "fitmax",
          "P2eq", "P2t_final", "nn_tot", "KK", "LL", "t_therm_s", "t_dyn_s"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"wrote {csv_path}  ({len(rows)} points)")

ld = np.array([r["log10_Delta"] for r in rows])
lk = np.array([np.nan if r["log10_beta_k_HEOM"] is None else r["log10_beta_k_HEOM"]
               for r in rows])
lkf = np.array([np.nan if r["log10_beta_k_fit"] is None else r["log10_beta_k_fit"]
                for r in rows])

# --- reference rate theories vs Delta (epsilon fixed), on a fine grid for smooth curves ---
ldmin = ld.min()
ldmax = ld.max()
Dld = max(ldmax - ldmin, 0.1)
ld_fine = np.linspace(ldmin - 0.1*Dld, ldmax + 0.1*Dld, 400)
Delta_fine = 10.0 ** ld_fine / beta
kMT = 2 * np.pi * Delta_fine**2 * np.sqrt(beta / (4 * np.pi * Lambda)) \
      * np.exp(-beta * (Lambda - epsilon)**2 / (4 * Lambda))
kZUS = kMT / (1 + 4 * np.pi * Delta_fine**2 * Gamma / (Lambda * Omega**2))
kNAD = 2 * np.pi / Omega * Delta_fine**2 * np.exp(0.5 * beta * epsilon) \
       * np.exp(-Lambda / (Omega * np.tanh(0.5 * beta * Omega))) \
       * iv(epsilon / Omega, 2 * Lambda * np.exp(-0.5 * beta * Omega) \
       / (Omega * (1 - np.exp(-beta * Omega))))

# --- MASH-RPMD rates ---
#delta = np.array([])

#bo_classical_mk = np.array([])
#bo_classical_sk = np.array([])

#mash_classical_mk = np.array([])
#mash_classical_sk = np.array([])

#mash_quantum_mk = np.array([])
#mash_quantum_sk = np.array([])

plt.rcParams.update({

    'figure.figsize': (6.0, 4.0),
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'lines.linewidth': 2,
    'axes.linewidth': 3,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.direction': 'in',
    'xtick.top': True,
    'ytick.direction': 'in',
    'ytick.right': True,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'legend.frameon': False,
})

fig, (ax1) = plt.subplots(1, 1, figsize=(6, 4))
ax1.set_ylabel('$log(k\\beta\\hbar)$')
ax1.set_xlabel('$\\log_{10}(\\beta\\Delta)$')
ax1.set_xlim([-0.575, 0.575])
ax1.plot(ld_fine, np.log10(kMT * beta), color="darkred", lw=1.5, label="Marcus")
#ax1.plot(ld_fine, np.log10(kZUS * beta), color="darkorange", lw=1.5, label="Zusman")
#ax1.plot(ld_fine, np.log10(kNAD * beta), color="darkgreen", lw=1.5, label="Quantum")
ax1.plot(ld, lkf, color="k", lw=1.5, label=f"HEOM") #($K_2$ fit [{fitmin:g},{fitmax:g}])
#ax1.plot(ld, lk, color="k", lw=1.5, label="HEOM") #(last $k(t)$)
#ax1.errorbar(np.log10(delta), np.log10(bo_classical_mk * beta), bo_classical_sk*beta / (bo_classical_mk * np.log(10)), fmt='o', markersize=5, linewidth=1.5, color='g', label=r'BOMD')
#ax1.errorbar(np.log10(delta), np.log10(mash_classical_mk * beta), mash_classical_sk*beta / (mash_classical_mk * np.log(10)), fmt='o', markersize=5, linewidth=1.5, color='r', label=r'MASH')
#ax1.errorbar(np.log10(delta), np.log10(mash_quantum_mk * beta), mash_quantum_sk*beta / (mash_quantum_mk * np.log(10)), fmt='o', markersize=5, linewidth=1.5, color='b', label=r'MASH-RPMD')
ax1.legend(fontsize=9, loc='best', handletextpad=0.2)
fig.tight_layout()
png = os.path.join(grid_dir, "scan.png")
fig.savefig(png)
print(f"wrote {png}")

print(f"\nK2 fit window: t/(beta*hbar) = [{fitmin:g}, {fitmax:g}]")
print(f"\n{'log10(bD)':>9} {'log10(bk) fit':>14} {'fit r':>7} {'log10(bk) last':>15} "
      f"{'P2t_final':>11} {'P2eq':>9} {'t_dyn':>8}")
for r in rows:
    vf, v = r["log10_beta_k_fit"], r["log10_beta_k_HEOM"]
    print(f"{r['log10_Delta']:>9.2f} {('%.5f' % vf) if vf is not None else 'n/a':>14} "
          f"{('%.4f' % r['fit_r']) if r.get('fit_r') is not None else 'n/a':>7} "
          f"{('%.5f' % v) if v is not None else 'k<=0':>15} "
          f"{r['P2t_final']:>11.6f} {r['P2eq']:>9.6f} {r['t_dyn_s']:>7.0f}s")

if args.plot_fits:
    print(f"wrote {plot_fits(rows, grid_dir, fitmin, fitmax, 'log10_Delta', 'log10(beta*Delta) = {:.2f}')}")
