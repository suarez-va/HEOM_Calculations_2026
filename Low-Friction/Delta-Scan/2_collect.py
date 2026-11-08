"""
Collect the per-point result.json files into one table and plot log10(beta*k) vs log10(beta*Delta)
at fixed epsilon, alongside the Zusman and quantum nonadiabatic reference rates.

    python3 2_collect.py

Writes delta_grid/results.csv and delta_grid/scan.png. Points that have not finished yet are
reported as missing rather than silently omitted, so a partial scan is obvious.
"""

import os
import csv
import json

import numpy as np
from scipy.special import iv
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from libra_py import units
from run_heom_point import log10_delta_grid

script_dir = os.path.dirname(os.path.abspath(__file__))
grid_dir = os.path.join(script_dir, "delta_grid")

with open(os.path.join(script_dir, "config.yaml")) as f:
    cfg = yaml.safe_load(f)

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

csv_path = os.path.join(grid_dir, "results.csv")
fields = ["idx", "log10_Delta", "beta_Delta", "beta_epsilon", "k_HEOM", "log10_beta_k_HEOM",
          "P2eq", "P2t_final", "nn_tot", "KK", "LL", "t_therm_s", "t_dyn_s"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"wrote {csv_path}  ({len(rows)} points)")

ld = np.array([r["log10_Delta"] for r in rows])
lk = np.array([np.nan if r["log10_beta_k_HEOM"] is None else r["log10_beta_k_HEOM"]
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
#ax1.plot(ld_fine, np.log10(kMT * beta), color="darkred", lw=1.5, label="Marcus")
ax1.plot(ld_fine, np.log10(kZUS * beta), color="darkorange", lw=1.5, label="Zusman")
ax1.plot(ld_fine, np.log10(kNAD * beta), color="darkgreen", lw=1.5, label="Quantum")
ax1.plot(ld, lk, "o", color="k", ms=5, label="HEOM")
ax1.legend(fontsize=9, loc='best', handletextpad=0.2)
fig.tight_layout()
png = os.path.join(grid_dir, "scan.png")
fig.savefig(png)
print(f"wrote {png}")

print(f"\n{'log10(bD)':>9} {'log10(beta k)':>14} {'P2t_final':>11} {'P2eq':>9} {'t_dyn':>8}")
for r in rows:
    v = r["log10_beta_k_HEOM"]
    print(f"{r['log10_Delta']:>9.2f} {('%.5f' % v) if v is not None else 'k<=0':>14} "
          f"{r['P2t_final']:>11.6f} {r['P2eq']:>9.6f} {r['t_dyn_s']:>7.0f}s")
