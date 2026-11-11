"""
Run ONE point of the Delta scan: a HEOM thermalization leg followed by a dynamics leg, and
extract the rate constant two ways: k_HEOM (last point of k(t)) and k_fit (slope of
a line fit of K2(t) = -P2eq ln|1 - P2/P2eq|, see k2_fit).

This is the single-point worker invoked by 1_run_delta_scan_local.py. It is also runnable by
hand for debugging a single value:

    python3 run_heom_point.py --idx 6

Everything physical comes from config.yaml. The only thing that varies across the scan is
log10(beta*Delta); epsilon is fixed by config.yaml. Each point chdir's into its own
delta_grid/val<idx>/ directory, which is what
keeps the concurrent runs from clobbering each other - run_dynamics writes its output into
relative paths ("heom_therm/", "heom_dyn/") that would otherwise collide.
"""

# glibc's dynamic mmap threshold otherwise makes RSS ratchet upward for the whole run at large
# hierarchies. Must happen before the big allocations.
import ctypes
_libc = ctypes.CDLL("libc.so.6")
_libc.mallopt(-3, 131072)   # M_MMAP_THRESHOLD - also disables the dynamic adjustment
_libc.mallopt(-1, 131072)   # M_TRIM_THRESHOLD
_libc.mallopt(-8, 2)        # M_ARENA_MAX - many processes per node, so keep arenas minimal

# Threads per point: --num-threads if given, else heom.num_threads from the config. Has to be
# known before numpy/scipy/numba are imported, hence the early pre-parse of the command line.
import os
import sys
import json
import time
import argparse

import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))

# Only when run as a script: the driver/collector also import this module for its grid function.
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, required=True, help="index into the scan grid")
    ap.add_argument("--config", default=None)
    ap.add_argument("--num-threads", type=int, default=None,
                    help="cores for this point (default: heom.num_threads from the config)")
    args = ap.parse_args()

    cfg_path = args.config or os.path.join(script_dir, "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    num_threads = int(args.num_threads if args.num_threads is not None else cfg["heom"]["num_threads"])
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = str(num_threads)

import numpy as np
import h5py

if sys.platform == "cygwin":
    from cyglibra_core import *
else:
    from liblibra_core import *

from libra_py import units
import libra_py.dynamics.heom.compute as compute


def log10_delta_grid(cfg):
    """The log10(beta*Delta) values being scanned (evenly spaced, endpoints inclusive). Single
    source of truth, imported by the driver and the collector too, so the grid can never
    disagree between them. Rounded so values come out as e.g. -1.4 rather than -1.4000000000000001."""
    g = cfg["grids"]["log10_Delta"]
    return np.round(np.linspace(g["start"], g["stop"], int(g["num"])), 10)


# ---------------- K2 line-fit rate ----------------
# Same method as Single-Point/2_plot.py (taken from MASH-RPMD_System-Bath/analyze_rate.py):
#     K2(t) = -P2eq ln|1 - P2(t)/P2eq|,   beta*hbar*k = slope of K2 vs t/(beta*hbar) over [fitmin, fitmax]
# These helpers are also imported by the collectors, which refit from each point's dynamics.npz.

DEFAULT_FIT = (10.0, 20.0)


def fit_window(cfg):
    """(fitmin, fitmax) in units of t/(beta*hbar), from the config's `rate:` block."""
    r = cfg.get("rate") or {}
    return float(r.get("fitmin", DEFAULT_FIT[0])), float(r.get("fitmax", DEFAULT_FIT[1]))


def k2_from_p2(P2, P2eq):
    """K2 = -P2eq*ln|1 - P2/P2eq|."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -P2eq * np.log(np.abs(1.0 - P2 / P2eq))


def line_fit(time, K, fitmin, fitmax):
    """Least-squares m*t+b of K over [fitmin, fitmax], finite points only. NaNs if too few."""
    from scipy.stats import linregress
    mask = (time >= fitmin) & (time <= fitmax) & np.isfinite(K)
    if int(mask.sum()) < 2:
        return float("nan"), float("nan"), float("nan"), int(mask.sum())
    fit = linregress(time[mask], K[mask])
    return float(fit.slope), float(fit.intercept), float(fit.rvalue), int(mask.sum())


def k2_fit(t_beta, P2, P2eq, beta, fitmin, fitmax):
    """Fit K2 over the window; returns the result.json fields of the fitted rate."""
    m, b, r, n = line_fit(t_beta, k2_from_p2(P2, P2eq), fitmin, fitmax)
    ok = np.isfinite(m) and m > 0
    return {"k_fit": (m / beta) if np.isfinite(m) else None,
            "log10_beta_k_fit": float(np.log10(m)) if ok else None,
            "fit_slope": m if np.isfinite(m) else None,
            "fit_intercept": b if np.isfinite(b) else None,
            "fit_r": r if np.isfinite(r) else None,
            "fit_npts": n, "fitmin": fitmin, "fitmax": fitmax}


def load_p2(valdir):
    """(t_beta, P2, P2eq, beta) for one finished point: from dynamics.npz, or for points run
    before it was saved, from heom_dyn/mem_data.hdf + result.json. None if neither exists."""
    npz = os.path.join(valdir, "dynamics.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        return d["t_beta"], d["rho"][:, 1, 1].real, float(d["P2eq"]), float(d["beta"])
    hdf = os.path.join(valdir, "heom_dyn", "mem_data.hdf")
    res = os.path.join(valdir, "result.json")
    if os.path.exists(hdf) and os.path.exists(res):
        with open(res) as f:
            r = json.load(f)
        with h5py.File(hdf, "r") as f:
            t = np.array(f["time/data"][:]).ravel()
            P2 = np.array(f["denmat/data"][:, 1, 1]).real
        return t / r["beta"], P2, float(r["P2eq"]), float(r["beta"])
    return None


def refit_rows(rows, grid_dir, fitmin, fitmax):
    """Make every row's fit fields correspond to [fitmin, fitmax]: rows whose result.json was
    fitted over that window are kept, the rest are refitted from their saved trajectory.
    Returns the indices that could not be fitted (no trajectory on disk)."""
    nofit = []
    for row in rows:
        if row.get("fitmin") == fitmin and row.get("fitmax") == fitmax and "k_fit" in row:
            continue
        data = load_p2(os.path.join(grid_dir, f"val{row['idx']}"))
        if data is None:
            nofit.append(row["idx"])
            row.update({"k_fit": None, "log10_beta_k_fit": None, "fit_r": None,
                        "fit_npts": 0, "fitmin": fitmin, "fitmax": fitmax})
            continue
        t_beta, P2, P2eq, beta = data
        row.update(k2_fit(t_beta, P2, P2eq, beta, fitmin, fitmax))
    return nofit


def plot_fits(rows, grid_dir, fitmin, fitmax, label_key, label_fmt, png_name="fits.png"):
    """Small multiples: K2(t), the shaded fit window and the fit line, one panel per point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows)
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, row in zip(axes.ravel(), rows):
        data = load_p2(os.path.join(grid_dir, f"val{row['idx']}"))
        ax.set_title(label_fmt.format(row[label_key]), fontsize=9)
        ax.tick_params(labelsize=7)
        if data is None:
            ax.text(0.5, 0.5, "no trajectory", ha="center", va="center", transform=ax.transAxes)
            continue
        t_beta, P2, P2eq, beta = data
        K2 = k2_from_p2(P2, P2eq)
        ax.axvspan(fitmin, fitmax, color="0.92", zorder=0)
        ax.plot(t_beta, K2, color="C0", lw=1.0)
        if row.get("fit_slope") is not None:
            tf = np.array([fitmin, min(fitmax, t_beta[-1])])
            ax.plot(tf, row["fit_slope"] * tf + row["fit_intercept"], "k--", lw=1.2)
            ax.text(0.03, 0.97, f"m={row['fit_slope']:.3e}\nr={row['fit_r']:.3f}",
                    transform=ax.transAxes, fontsize=7, va="top")
        fin = K2[np.isfinite(K2)]
        if fin.size:                         # ln spikes once P2 ~ P2eq: clip the view
            lo, hi = np.percentile(fin, [0.5, 99.5])
            pad = 0.1 * max(hi - lo, 1e-12)
            ax.set_ylim(min(lo, 0.0) - pad, hi + pad)
    fig.supxlabel("$t\\, /\\, \\beta\\hbar$")
    fig.supylabel("$K_2(t)=-P_2^{eq}\\ln|1-P_2/P_2^{eq}|$")
    fig.tight_layout()
    png = os.path.join(grid_dir, png_name)
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return png


def main():
    log10_delta_ar = log10_delta_grid(cfg)
    log10_delta = float(log10_delta_ar[args.idx])

    # ---------------- physical parameters ----------------
    T = cfg["system"]["temperature"]
    beta = units.hartree / (units.boltzmann * T)
    Omega = cfg["system"]["Omega"] / beta
    Gamma = cfg["system"]["Gamma"] / beta
    Lambda = cfg["system"]["Lambda"] / beta
    beta_eps = float(cfg["system"]["epsilon"])
    epsilon = beta_eps / beta
    beta_Delta = 10.0 ** log10_delta
    Delta = beta_Delta / beta                      # <-- the scanned quantity

    P1eq = np.exp(-0.5 * beta * epsilon) / (np.exp(-0.5 * beta * epsilon) + np.exp(0.5 * beta * epsilon))
    P2eq = np.exp(0.5 * beta * epsilon) / (np.exp(-0.5 * beta * epsilon) + np.exp(0.5 * beta * epsilon))

    grid_dir = os.path.join(script_dir, cfg.get("runtime", {}).get("grid_dir", "delta_grid"))
    outdir = os.path.join(grid_dir, f"val{args.idx}")
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)

    print(f"idx={args.idx}  log10(beta*Delta)={log10_delta}  Delta={Delta}  "
          f"beta*epsilon={beta_eps}", flush=True)

    # Thermalization Hamiltonian - diagonal, no electronic coupling
    Ham0 = CMATRIX(2, 2)
    Ham0.set(0, 0, 0.5 * epsilon); Ham0.set(0, 1, 0.0)
    Ham0.set(1, 0, 0.0);           Ham0.set(1, 1, -0.5 * epsilon)

    # Full Hamiltonian
    Ham = CMATRIX(2, 2)
    Ham.set(0, 0, 0.5 * epsilon); Ham.set(0, 1, Delta)
    Ham.set(1, 0, Delta);         Ham.set(1, 1, -0.5 * epsilon)

    h = cfg["heom"]
    params = {
        "KK": h["KK"], "LL": h["LL"],
        "spectral_density": h["spectral_density"],
        "gamma": Gamma, "eta": Lambda, "omega": Omega, "temperature": T,
        "el_phon_couplings": initialize_el_phonon_couplings(2),

        "dt": cfg["therm"]["dt"] * beta,
        "nsteps": cfg["therm"]["nsteps"],
        "nprint": cfg["therm"]["nprint"],
        "verbosity": -1, "progress_frequency": 0.2,

        "truncation_scheme": h["truncation_scheme"], "do_scale": h["do_scale"],
        "adm_tolerance": h["adm_tolerance"], "adm_deriv_tolerance": h["adm_deriv_tolerance"],
        "filter_after_steps": h["filter_after_steps"], "do_zeroing": h["do_zeroing"],
        "num_threads": num_threads,
        "backend": h.get("backend", "cpp"),

        "prefix": "heom_therm",
        "hdf5_output_level": 0, "txt_output_level": 0, "mem_output_level": 4,
        "properties_to_save": ["timestep", "time", "denmat", "auxdenmat"],
        "use_compression": 0, "compression_level": [0, 0, 0],
    }

    slurm = os.environ.get("SLURM_JOB_ID")
    print(f"threads = {num_threads}   cores in affinity mask = {len(os.sched_getaffinity(0))}"
          + (f"   SLURM job {slurm} task {os.environ.get('SLURM_ARRAY_TASK_ID')} "
             f"on {os.uname().nodename}" if slurm else ""), flush=True)

    nn_tot = compute_nn_tot(2 * (h["KK"] + (2 if h["spectral_density"] == 1 else 1)), h["LL"])
    print(f"nn_tot = {nn_tot}", flush=True)

    # ---------------- leg 1: thermalization ----------------
    t0 = time.time()
    rho0 = CMATRIX(2, 2); rho0.set(0, 0, 1.0 + 0.0j)
    compute.run_dynamics(params, Ham0, rho0)
    t_therm = time.time() - t0

    # ---------------- leg 2: dynamics ----------------
    params.update({"dt": cfg["dyn"]["dt"] * beta,
                   "nsteps": cfg["dyn"]["nsteps"],
                   "nprint": cfg["dyn"]["nprint"]})
    params.update({"prefix": "heom_dyn", "mem_output_level": 3,
                   "properties_to_save": ["timestep", "time", "denmat"]})

    with h5py.File("heom_therm/mem_data.hdf", "r") as f:
        rho = np.array(f["denmat/data"][-1, :, :])
        adm = np.array(f["auxdenmat/data"][-1, :, :, :])

    if params["backend"] == "numba":
        # the numba backend takes the numpy arrays directly
        rho_init, adm_init = rho, adm
    else:
        rho_init = CMATRIX(2, 2)
        rho_init.set(0, 0, rho[0, 0]); rho_init.set(0, 1, rho[0, 1])
        rho_init.set(1, 0, rho[1, 0]); rho_init.set(1, 1, rho[1, 1])

        adm_init = CMATRIXList()
        for n in range(adm.shape[0]):
            M = CMATRIX(2, 2)
            M.set(0, 0, adm[n, 0, 0]); M.set(0, 1, adm[n, 0, 1])
            M.set(1, 0, adm[n, 1, 0]); M.set(1, 1, adm[n, 1, 1])
            adm_init.append(M)

    t0 = time.time()
    compute.run_dynamics(params, Ham, rho_init, adm_init)
    t_dyn = time.time() - t0

    # ---------------- extract the rate ----------------
    with h5py.File("heom_dyn/mem_data.hdf", "r") as f:
        t_ar = np.array(f["time/data"][:])
        rhot = np.array(f["denmat/data"][:, :, :])

    P2t = rhot[:, 1, 1].real
    kt = np.gradient(P2t, t_ar) / (1 - P2t / P2eq)
    k_HEOM = float(kt[-1])

    log10_k = float(np.log10(k_HEOM * beta)) if k_HEOM > 0 else None

    # ---------------- K2 line-fit rate (see k2_fit) ----------------
    fitmin, fitmax = fit_window(cfg)
    t_ar = np.ravel(t_ar)
    fit = k2_fit(t_ar / beta, P2t, P2eq, beta, fitmin, fitmax)
    if fit["fit_npts"] < 2:
        print(f"WARNING: K2 fit window [{fitmin:g}, {fitmax:g}] has {fit['fit_npts']} point(s); "
              f"the dynamics ends at t/beta = {t_ar[-1] / beta:g}", flush=True)
    np.savez("dynamics.npz", t=t_ar, t_beta=t_ar / beta, rho=rhot, kt=kt,
             P2eq=P2eq, beta=beta)

    result = {
        "idx": args.idx,
        "log10_Delta": log10_delta,
        "beta_Delta": beta_Delta,
        "Delta": Delta,
        "beta_epsilon": beta_eps,
        "epsilon": epsilon,
        "k_HEOM": k_HEOM,
        "log10_beta_k_HEOM": log10_k,
        "P1eq": float(P1eq), "P2eq": float(P2eq),
        "P2t_final": float(P2t[-1]),
        "beta": beta, "nn_tot": int(nn_tot),
        "KK": h["KK"], "LL": h["LL"], "spectral_density": h["spectral_density"],
        "backend": params["backend"], "num_threads": num_threads,
        **fit,
        "t_therm_s": t_therm, "t_dyn_s": t_dyn,
    }
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"k_HEOM = {k_HEOM:.6e}   log10(beta*k_HEOM) = {log10_k}   (last point of k(t))", flush=True)
    print(f"k_fit  = {fit['k_fit']}   log10(beta*k_fit)  = {fit['log10_beta_k_fit']}   "
          f"(K2 fit over [{fitmin:g}, {fitmax:g}], r = {fit['fit_r']})", flush=True)
    print(f"timing: therm {t_therm:.0f}s, dyn {t_dyn:.0f}s", flush=True)


if __name__ == "__main__":
    main()
