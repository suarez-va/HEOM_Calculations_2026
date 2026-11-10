"""
Collect the per-point result.json files into one table and plot log10(beta*k) vs log10(beta*Delta)
at fixed epsilon, alongside the Zusman and quantum nonadiabatic reference rates and the classical adiabatic TST rate.

    python3 2_collect.py

Writes delta_grid/results.csv and delta_grid/scan_test.png. Points that have not finished yet are
reported as missing rather than silently omitted, so a partial scan is obvious.
"""

import os
import csv
import argparse
import json

import numpy as np
from scipy.special import iv
from scipy.integrate import quad
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from libra_py import units
from run_heom_point import log10_delta_grid

script_dir = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml", help="same config the scan was run with")
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

# --- classical adiabatic TST on the lower adiabat of the reaction coordinate ---
# The Brownian-oscillator bath is a damped mass-weighted reaction coordinate q (m = 1) of
# frequency Omega. Diabats displaced by +/- q0 with Lambda = 2 Omega^2 q0^2; state 1 (donor,
# H[0,0] = +epsilon/2, initially populated) sits at +q0. Dividing surface = diabatic crossing.
q0 = np.sqrt(Lambda / (2 * Omega**2))
q_dag = epsilon / (2 * Omega**2 * q0)                 # V1(q_dag) = V2(q_dag)

def V_lower(q, Delta):
    V1 = 0.5 * Omega**2 * (q - q0)**2 + 0.5 * epsilon
    V2 = 0.5 * Omega**2 * (q + q0)**2 - 0.5 * epsilon
    return 0.5 * (V1 + V2) - np.sqrt((0.5 * (V1 - V2))**2 + Delta**2)

def k_adiabatic_TST(Delta):
    """k = <|v|>/2 * exp(-beta V-(q_dag)) / int_{donor basin} exp(-beta V-(q)) dq, with the
    basin integral done numerically (energies measured from V-(q_dag) for stability)."""
    V_dag = V_lower(q_dag, Delta)
    q_max = max(q_dag, q0) + 12.0 / np.sqrt(beta * Omega**2)
    pts = [q0] if q_dag < q0 < q_max else None
    Z, _ = quad(lambda q: np.exp(-beta * (V_lower(q, Delta) - V_dag)), q_dag, q_max,
                points=pts, limit=200)
    return np.sqrt(1.0 / (2 * np.pi * beta)) / Z

kTST = np.array([k_adiabatic_TST(D) for D in Delta_fine])

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
ax1.plot(ld_fine, np.log10(kTST * beta), color="purple", lw=1.5, label="Adiabatic TST")
ax1.plot(ld, lk, "o", color="k", ms=5, label="HEOM")
ax1.legend(fontsize=9, loc='best', handletextpad=0.2)
fig.tight_layout()
png = os.path.join(grid_dir, "scan_test.png")
fig.savefig(png)
print(f"wrote {png}")

print(f"\n{'log10(bD)':>9} {'log10(beta k)':>14} {'P2t_final':>11} {'P2eq':>9} {'t_dyn':>8}")
for r in rows:
    v = r["log10_beta_k_HEOM"]
    print(f"{r['log10_Delta']:>9.2f} {('%.5f' % v) if v is not None else 'k<=0':>14} "
          f"{r['P2t_final']:>11.6f} {r['P2eq']:>9.6f} {r['t_dyn_s']:>7.0f}s")
