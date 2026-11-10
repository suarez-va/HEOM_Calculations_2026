"""
Run ONE point of the epsilon scan: a HEOM thermalization leg followed by a dynamics leg, and
extract the rate constant k_HEOM.

This is the single-point worker invoked by 1_run_eps_scan_local.py. It is also runnable by
hand for debugging a single value:

    python3 run_heom_point.py --idx 6

Everything physical comes from config.yaml. The only thing that varies across the scan is
beta*epsilon. Each point chdir's into its own eps_grid/val<idx>/ directory, which is what
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


def epsilon_grid(cfg):
    """The beta*epsilon values being scanned. Single source of truth, imported by the driver
    and the collector too, so the grid can never disagree between them."""
    g = cfg["grids"]["epsilon"]
    return np.arange(g["start"], g["stop"], g["step"], dtype=float)


def main():
    beta_eps_ar = epsilon_grid(cfg)
    beta_eps = float(beta_eps_ar[args.idx])

    # ---------------- physical parameters ----------------
    T = cfg["system"]["temperature"]
    beta = units.hartree / (units.boltzmann * T)
    Omega = cfg["system"]["Omega"] / beta
    Gamma = cfg["system"]["Gamma"] / beta
    Lambda = cfg["system"]["Lambda"] / beta
    Delta = 10.0 ** cfg["system"]["log10_Delta"] / beta
    epsilon = beta_eps / beta                      # <-- the scanned quantity

    P1eq = np.exp(-0.5 * beta * epsilon) / (np.exp(-0.5 * beta * epsilon) + np.exp(0.5 * beta * epsilon))
    P2eq = np.exp(0.5 * beta * epsilon) / (np.exp(-0.5 * beta * epsilon) + np.exp(0.5 * beta * epsilon))

    grid_dir = os.path.join(script_dir, cfg.get("runtime", {}).get("grid_dir", "eps_grid"))
    outdir = os.path.join(grid_dir, f"val{args.idx}")
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)

    print(f"idx={args.idx}  beta*epsilon={beta_eps}  epsilon={epsilon}", flush=True)

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

    result = {
        "idx": args.idx,
        "beta_epsilon": beta_eps,
        "epsilon": epsilon,
        "k_HEOM": k_HEOM,
        "log10_beta_k_HEOM": log10_k,
        "P1eq": float(P1eq), "P2eq": float(P2eq),
        "P2t_final": float(P2t[-1]),
        "beta": beta, "nn_tot": int(nn_tot),
        "KK": h["KK"], "LL": h["LL"], "spectral_density": h["spectral_density"],
        "backend": params["backend"], "num_threads": num_threads,
        "t_therm_s": t_therm, "t_dyn_s": t_dyn,
    }
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"k_HEOM = {k_HEOM:.6e}   log10(beta*k_HEOM) = {log10_k}", flush=True)
    print(f"timing: therm {t_therm:.0f}s, dyn {t_dyn:.0f}s", flush=True)


if __name__ == "__main__":
    main()
