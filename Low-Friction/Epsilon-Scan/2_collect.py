"""
Collect the per-point result.json files into one table and plot log10(beta*k) vs beta*epsilon,
alongside the Marcus and Zusman reference rates.

    python3 2_collect.py                            # config.yaml
    python3 2_collect.py --config config_smoke.yaml  # results of a test scan

Writes eps_grid/results.csv and eps_grid/scan.png. Points that have not finished yet are
reported as missing rather than silently omitted, so a partial scan is obvious.
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
from run_heom_point import epsilon_grid

script_dir = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml", help="same config the scan was run with")
args = ap.parse_args()
cfg_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
grid_dir = os.path.join(script_dir, cfg.get("runtime", {}).get("grid_dir", "eps_grid"))

beta_eps_ar = epsilon_grid(cfg)
T = cfg["system"]["temperature"]
beta = units.hartree / (units.boltzmann * T)
Omega = cfg["system"]["Omega"] / beta
Gamma = cfg["system"]["Gamma"] / beta
Lambda = cfg["system"]["Lambda"] / beta
Delta = 10.0 ** cfg["system"]["log10_Delta"] / beta

rows, missing = [], []
for i, be in enumerate(beta_eps_ar):
    p = os.path.join(grid_dir, f"val{i}", "result.json")
    if not os.path.exists(p):
        missing.append(i)
        continue
    with open(p) as f:
        rows.append(json.load(f))

if missing:
    print(f"WARNING: {len(missing)} point(s) have no result.json yet: {missing}")
    print("         re-run: python3 1_run_eps_scan_local.py")
if not rows:
    raise SystemExit("no results found - run 1_run_eps_scan_local.py first")

rows.sort(key=lambda r: r["beta_epsilon"])

csv_path = os.path.join(grid_dir, "results.csv")
fields = ["idx", "beta_epsilon", "epsilon", "k_HEOM", "log10_beta_k_HEOM",
          "P2eq", "P2t_final", "nn_tot", "KK", "LL", "t_therm_s", "t_dyn_s"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"wrote {csv_path}  ({len(rows)} points)")

be = np.array([r["beta_epsilon"] for r in rows])
lk = np.array([np.nan if r["log10_beta_k_HEOM"] is None else r["log10_beta_k_HEOM"]
               for r in rows])

# --- reference rate theories, on a fine grid for smooth curves ---
bemin = be.min()
bemax = be.max()
Dbe = bemax - bemin
be_fine = np.linspace(bemin - 0.1*Dbe, bemax + 0.1*Dbe, 400)
eps_fine = be_fine / beta
kMT = 2 * np.pi * Delta**2 * np.sqrt(beta / (4 * np.pi * Lambda)) \
      * np.exp(-beta * (Lambda - eps_fine)**2 / (4 * Lambda))
kZUS = kMT / (1 + 4 * np.pi * Delta**2 * Gamma / (Lambda * Omega**2))
kNAD = 2 * np.pi / Omega * Delta**2 * np.exp(0.5 * beta * eps_fine) \
       * np.exp(-Lambda / (Omega * np.tanh(0.5 * beta * Omega))) \
       * iv(eps_fine / Omega, 2 * Lambda * np.exp(-0.5 * beta * Omega) \
       / (Omega * (1 - np.exp(-beta * Omega))))

# --- MASH-RPMD rates ---
epsil = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0])

mash_classical_mk = np.array([3.525e-05, 1.416e-04, 3.778e-04, 6.812e-04, 7.869e-04, 5.940e-04, 3.073e-04, 1.185e-04, 3.346e-05, 7.684e-06, 1.495e-06])
mash_classical_sk = np.array([0.086e-05, 0.022e-04, 0.016e-04, 0.075e-04, 0.088e-04, 0.072e-04, 0.025e-04, 0.010e-04, 0.055e-05, 0.168e-06, 0.070e-06])

mash_quantum_mk = np.array([5.785e-05, 2.442e-04, 6.766e-04, 1.194e-03, 1.451e-03, 1.195e-03, 6.664e-04, 2.511e-04, 6.687e-05, 1.322e-05, 2.001e-06])
mash_quantum_sk = np.array([0.119e-05, 0.019e-04, 0.048e-04, 0.007e-03, 0.008e-03, 0.005e-03, 0.064e-04, 0.030e-04, 0.166e-05, 0.041e-05, 0.111e-06])

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

fig, (ax1) = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
ax1.set_ylabel('$log(k\\beta\hbar)$')
ax1.set_xlabel('$\epsilon/\lambda$')
ax1.set_ylim([-5.25, -2.75])
ax1.set_xlim([-0.05, 2.55])
#ax1.plot(eps_fine / Lambda, np.log10(kMT * beta), color="darkred", lw=1.5, label="Marcus")
ax1.plot(eps_fine / Lambda, np.log10(kZUS * beta), color="darkorange", lw=1.5, label="Zusman")
ax1.plot(eps_fine / Lambda, np.log10(kNAD * beta), color="darkgreen", lw=1.5, label="Quantum")
ax1.plot(be / (beta * Lambda), lk, "o", color="k", ms=5, label="HEOM")
#ax1.plot(be / (beta * Lambda), lk, ls='-', color="k", lw=1.5, label="HEOM")
ax1.errorbar(epsil / Lambda, np.log10(mash_classical_mk), mash_classical_sk / (mash_classical_mk * np.log(10)), fmt='o', markersize=5, linewidth=1.5, color='r', label=r'MASH')
ax1.errorbar(epsil / Lambda, np.log10(mash_quantum_mk), mash_quantum_sk / (mash_quantum_mk * np.log(10)), fmt='o', markersize=5, linewidth=1.5, color='b', label=r'MASH-RPMD')
#ax1.plot(be / (beta * Lambda), lk+0.01, color="k", label="HEOMd")
#ax1.plot(be / (beta * Lambda), lk-0.01, color="k", label="HEOMdd")
ax1.legend(fontsize=9, loc='best', handletextpad=0.2)
fig.tight_layout()
png = os.path.join(grid_dir, "scan.png")
fig.savefig(png)
print(f"wrote {png}")

print(f"\n{'beta*eps':>9} {'log10(beta k)':>14} {'P2t_final':>11} {'P2eq':>9} {'t_dyn':>8}")
for r in rows:
    v = r["log10_beta_k_HEOM"]
    print(f"{r['beta_epsilon']:>9.1f} {('%.5f' % v) if v is not None else 'k<=0':>14} "
          f"{r['P2t_final']:>11.6f} {r['P2eq']:>9.6f} {r['t_dyn_s']:>7.0f}s")

