"""
Like 2_plot.py, but the rate comes from a line fit of K2(t) instead of the last point of k(t).
Same method as MASH-RPMD_Calculations_2026/MASH-RPMD_System-Bath/analyze_rate.py:

    K2(t) = -P2eq * ln| 1 - P2(t)/P2eq |        (= k t + const once the dynamics is rate-like)
    beta*hbar*k = slope of a least-squares line fit of K2 vs t/(beta*hbar) over [fitmin, fitmax]

    python3 2_plot.py                              # reads run/dynamics.npz, fit over [10, 20]
    python3 2_plot.py --outdir run_b --fitmin 5 --fitmax 15

2x2 panels (x axis = t / (beta*hbar)):
  top-left     populations   P2(t) = rho_22
  top-right    coherence     Re rho_12(t), Im rho_12(t)
  bottom-left  rate          beta*hbar*k(t),  k(t) = (dP2/dt) / (1 - P2/P2eq)   (as in 2_plot.py)
  bottom-right K2(t) with its line fit over the shaded window [fitmin, fitmax]

Writes <outdir>/dynamics.png (2_plot.py's dynamics.png is left alone).
"""

import os
import argparse

import numpy as np
from scipy.stats import linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", default="run")
ap.add_argument("--fitmin", type=float, default=10.0, help="fit window start, in t/(beta*hbar)")
ap.add_argument("--fitmax", type=float, default=20.0, help="fit window end, in t/(beta*hbar)")
args = ap.parse_args()
outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(script_dir, args.outdir)

npz_path = os.path.join(outdir, "dynamics.npz")
if not os.path.exists(npz_path):
    raise SystemExit(f"{npz_path} not found - run 1_run_point.py first")
d = np.load(npz_path)


# ---- the two helpers below are copied from MASH-RPMD_System-Bath/analyze_rate.py
# ---- (k12_from_p2, line_fit), minus the statistical-error parts HEOM does not need
def k2_from_p2(P2, P2eq):
    """K2 = -P2eq*ln|1 - P2/P2eq|."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -P2eq * np.log(np.abs(1.0 - P2 / P2eq))


def line_fit(time, K, fitmin, fitmax):
    """Least-squares m*t+b of K over [fitmin, fitmax], finite points only. NaNs if too few."""
    mask = (time >= fitmin) & (time <= fitmax) & np.isfinite(K)
    if int(mask.sum()) < 2:
        return float("nan"), float("nan"), float("nan"), int(mask.sum())
    fit = linregress(time[mask], K[mask])
    return float(fit.slope), float(fit.intercept), float(fit.rvalue), int(mask.sum())


beta = float(d["beta"])
t = d["t_beta"]
rho = d["rho"]
kt_b = d["kt"] * beta                          # beta*hbar*k(t)
k_final_b = float(d["k_HEOM"]) * beta          # 2_plot.py's last-point value, for comparison
P2eq = float(d["P2eq"])
P2 = rho[:, 1, 1].real

fmin, fmax = args.fitmin, args.fitmax
K2 = k2_from_p2(P2, P2eq)
m, b, r, npts = line_fit(t, K2, fmin, fmax)    # m = d K2 / d(t/beta) = beta*hbar*k

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

fig, ((ax1, ax2), (ax4, ax3)) = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
fig.suptitle(f"$\\beta\\epsilon$ = {float(d['beta_epsilon']):g},   "
             f"$\\log_{{10}}(\\beta\\Delta)$ = {float(d['log10_Delta']):g},   "
             f"KK = {int(d['KK'])}, LL = {int(d['LL'])}", fontsize=12)

ax1.plot(t, P2, color="tab:red", label="$P_2(t)$")
ax1.set_ylabel("population")
ax1.legend(loc="best")

ax2.plot(t, rho[:, 0, 1].real, color="tab:purple", label="Re $\\rho_{12}(t)$")
ax2.plot(t, rho[:, 0, 1].imag, color="tab:green", label="Im $\\rho_{12}(t)$")
ax2.set_ylabel("coherence")
ax2.legend(loc="best")

# bottom-left: the original k(t) panel of 2_plot.py
ax4.plot(t, kt_b, color="k", lw=1.5, label="$k(t)$")
ax4.axhline(k_final_b, color="gray", ls="--", lw=1, label=f"$k$ final = {k_final_b:.3e}")
ax4.axhline(0.0, color="k", ls="--", lw=1)
if k_final_b > 0:
    ax4.set_ylim(0.0 - 0.1*k_final_b, 5.1*k_final_b)
else:
    # 2_plot.py's limits assume k_final > 0; otherwise scale to the back half of k(t)
    tail = kt_b[len(kt_b) // 2:]
    tail = tail[np.isfinite(tail)]
    if tail.size:
        lo, hi = np.percentile(tail, [2.0, 98.0])
        pad = 0.2 * max(hi - lo, 1e-12)
        ax4.set_ylim(lo - pad, hi + pad)
ax4.set_ylabel("$\\beta\\hbar\\, k(t)$")
ax4.set_xlabel("$t\\, /\\, \\beta\\hbar$")
ax4.legend(loc="best")

# bottom-right: K2(t) and its line fit
ax3.axvspan(fmin, fmax, color="0.92", zorder=0)
ax3.axhline(0.0, color="0.7", lw=0.8, ls="--")
ax3.plot(t, K2, color="C0", lw=1.6, zorder=4,
         label=r"$K_2(t)=-P_2^{\mathrm{eq}}\ln|1-P_2/P_2^{\mathrm{eq}}|$")
if np.isfinite(m):
    tf = np.array([fmin, min(fmax, t[-1])])
    ax3.plot(tf, m * tf + b, color="k", lw=2.0, ls="--", zorder=6,
             label=f"fit [{fmin:g},{fmax:g}]:  m={m:.4g}, b={b:.4g}, r={r:.4f}")

    def _log(x):
        return f"{np.log10(x):.4f}" if x > 0 else "n/a"
    ax3.text(0.97, 0.05,
             f"$\\beta\\hbar k$ (fit) = {m:.4e}   (log$_{{10}}$ {_log(m)})\n"
             f"$\\beta\\hbar k_\\mathrm{{HEOM}}$ (last pt) = {k_final_b:.4e}   "
             f"(log$_{{10}}$ {_log(k_final_b)})",
             transform=ax3.transAxes, fontsize=8, va="bottom", ha="right")
ax3.set_ylabel("$K_2(t)$")
ax3.set_xlabel("$t\\, /\\, \\beta\\hbar$")
ax3.legend(loc="upper left", fontsize=8)

fig.tight_layout()
png = os.path.join(outdir, "dynamics.png")
fig.savefig(png)
print(f"wrote {png}")

print(f"P2(t_final) = {P2[-1]:.6f}   P2eq = {P2eq:.6f}")
if np.isfinite(m):
    print(f"K2 fit over t/beta = [{fmin:g}, {fmax:g}] ({npts} points): "
          f"slope = {m:.6e}, intercept = {b:.6e}, r = {r:.6f}")
    print(f"beta*hbar*k (fit)       = {m:.6e}" + (f"   log10 = {np.log10(m):.5f}" if m > 0 else "   (k <= 0)"))
else:
    print(f"WARNING: only {npts} finite point(s) in [{fmin:g}, {fmax:g}] - "
          f"the trajectory ends at t/beta = {t[-1]:g}; adjust --fitmin/--fitmax")
print(f"beta*hbar*k_HEOM (last) = {k_final_b:.6e}"
      + (f"   log10 = {np.log10(k_final_b):.5f}" if k_final_b > 0 else "   (k <= 0)"))
