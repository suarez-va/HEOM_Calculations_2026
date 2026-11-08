"""
Plot the time-dependent HEOM density matrix of a single point produced by 1_run_point.py:

    python3 2_plot.py                 # reads run/dynamics.npz
    python3 2_plot.py --outdir run_b

Panels (x axis = t / (beta*hbar)):
  1. populations   P1(t) = rho_11, P2(t) = rho_22
  2. coherence     Re rho_12(t), Im rho_12(t)
  3. rate          beta*hbar*k(t),  k(t) = (dP2/dt) / (1 - P2/P2eq)

Writes <outdir>/dynamics.png.
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", default="run")
args = ap.parse_args()
outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(script_dir, args.outdir)

npz_path = os.path.join(outdir, "dynamics.npz")
if not os.path.exists(npz_path):
    raise SystemExit(f"{npz_path} not found - run 1_run_point.py first")
d = np.load(npz_path)

beta = float(d["beta"])
t = d["t_beta"]
rho = d["rho"]
kt_b = d["kt"] * beta                 # beta*hbar*k(t)
k_final_b = float(d["k_HEOM"]) * beta
P2eq = float(d["P2eq"])

plt.rcParams.update({
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'lines.linewidth': 2,
    'axes.linewidth': 3,
    'axes.labelsize': 15,
    'axes.titlesize': 13,
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

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
fig.suptitle(f"$\\beta\\epsilon$ = {float(d['beta_epsilon']):g},   "
             f"$\\log_{{10}}(\\beta\\Delta)$ = {float(d['log10_Delta']):g},   "
             f"KK = {int(d['KK'])}, LL = {int(d['LL'])}", fontsize=12)

#ax1.plot(t, rho[:, 0, 0].real, color="tab:blue", label="$P_1(t)$")
ax1.plot(t, rho[:, 1, 1].real, color="tab:red", label="$P_2(t)$")
ax1.set_ylabel("population")
ax1.legend(loc="best")

ax2.plot(t, rho[:, 0, 1].real, color="tab:purple", label="Re $\\rho_{12}(t)$")
ax2.plot(t, rho[:, 0, 1].imag, color="tab:green", label="Im $\\rho_{12}(t)$")
ax2.set_ylabel("coherence")
ax2.legend(loc="best")

ax3.plot(t, kt_b, color="k", lw=1.5, label="$k(t)$")
ax3.axhline(k_final_b, color="gray", ls="--", lw=1, label=f"$k$ final = {k_final_b:.3e}")
ax3.axhline(0.0, color="k", ls="--", lw=1)
# clip the y range to the latter half of the trajectory so the early-time transient
# (and the 1/(1 - P2/P2eq) blow-up near t = 0) does not set the scale
#tail = kt_b[len(kt_b) // 2:]
#tail = tail[np.isfinite(tail)]
#if tail.size:
#    lo, hi = tail.min(), tail.max()
#    pad = 0.5 * max(hi - lo, 0.1 * abs(k_final_b), 1e-12)
#    ax3.set_ylim(lo - pad, hi + pad)
ax3.set_ylim(0.0 - 0.1*k_final_b, 5.1*k_final_b)
ax3.set_ylabel("$\\beta\\hbar\\, k(t)$")
ax3.set_xlabel("$t\\, /\\, \\beta\\hbar$")
ax3.legend(loc="best")

fig.tight_layout()
png = os.path.join(outdir, "dynamics.png")
fig.savefig(png)
print(f"wrote {png}")

print(f"P2(t_final) = {rho[-1, 1, 1].real:.6f}   P2eq = {P2eq:.6f}")
print(f"|rho_12(t_final)| = {abs(rho[-1, 0, 1]):.3e}")
print(f"beta*hbar*k_HEOM = {k_final_b:.6e}"
      + (f"   log10 = {np.log10(k_final_b):.5f}" if k_final_b > 0 else "   (k <= 0)"))
