"""
Run ONE HEOM point: a thermalization leg (Delta = 0) followed by a dynamics leg (full
Hamiltonian), and save the full time-dependent density matrix rho(t) of the dynamics leg.

    python3 1_run_point.py                                  # config.yaml -> run/
    python3 1_run_point.py --config my.yaml --outdir run_b  # several test points side by side

Physics and Libra usage are identical to Low-Friction/Delta-Scan/run_heom_point.py; the only
difference is that epsilon and Delta both come straight from config.yaml (no grid) and that
the whole trajectory is kept, not just the final rate. Plot with:  python3 2_plot.py
"""

# glibc's dynamic mmap threshold otherwise makes RSS ratchet upward for the whole run at large
# hierarchies. Must happen before the big allocations.
import ctypes
_libc = ctypes.CDLL("libc.so.6")
_libc.mallopt(-3, 131072)   # M_MMAP_THRESHOLD - also disables the dynamic adjustment
_libc.mallopt(-1, 131072)   # M_TRIM_THRESHOLD

import os
import sys
import json
import time
import shutil
import argparse

import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--outdir", default="run", help="output directory (relative to this script)")
args = ap.parse_args()

cfg_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

# Thread count from config. Must precede numpy/scipy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(cfg["heom"]["num_threads"])

import numpy as np
import h5py

if sys.platform == "cygwin":
    from cyglibra_core import *
else:
    from liblibra_core import *

from libra_py import units
import libra_py.dynamics.heom.compute as compute


def main():
    # ---------------- physical parameters ----------------
    T = cfg["system"]["temperature"]
    beta = units.hartree / (units.boltzmann * T)
    Omega = cfg["system"]["Omega"] / beta
    Gamma = cfg["system"]["Gamma"] / beta
    Lambda = cfg["system"]["Lambda"] / beta
    beta_eps = float(cfg["system"]["epsilon"])
    epsilon = beta_eps / beta
    log10_delta = float(cfg["system"]["log10_Delta"])
    beta_Delta = 10.0 ** log10_delta
    Delta = beta_Delta / beta

    P1eq = np.exp(-0.5 * beta * epsilon) / (np.exp(-0.5 * beta * epsilon) + np.exp(0.5 * beta * epsilon))
    P2eq = np.exp(0.5 * beta * epsilon) / (np.exp(-0.5 * beta * epsilon) + np.exp(0.5 * beta * epsilon))

    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(script_dir, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(outdir, "config_used.yaml"))
    os.chdir(outdir)

    print(f"beta*epsilon={beta_eps}  log10(beta*Delta)={log10_delta}  Delta={Delta}", flush=True)
    print(f"output -> {outdir}", flush=True)

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
        "num_threads": h["num_threads"],
        "backend": h.get("backend", "cpp"),

        "prefix": "heom_therm",
        "hdf5_output_level": 0, "txt_output_level": 0, "mem_output_level": 4,
        "properties_to_save": ["timestep", "time", "denmat", "auxdenmat"],
        "use_compression": 0, "compression_level": [0, 0, 0],
    }

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

    # ---------------- collect rho(t) and the rate ----------------
    with h5py.File("heom_dyn/mem_data.hdf", "r") as f:
        t_ar = np.array(f["time/data"][:]).ravel()
        rhot = np.array(f["denmat/data"][:, :, :])

    P2t = rhot[:, 1, 1].real
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.gradient(P2t, t_ar) / (1 - P2t / P2eq)
    k_HEOM = float(kt[-1])
    log10_k = float(np.log10(k_HEOM * beta)) if k_HEOM > 0 else None

    np.savez("dynamics.npz",
             t=t_ar, t_beta=t_ar / beta, rho=rhot, kt=kt,
             beta=beta, epsilon=epsilon, beta_epsilon=beta_eps,
             Delta=Delta, log10_Delta=log10_delta,
             P1eq=P1eq, P2eq=P2eq, k_HEOM=k_HEOM,
             nn_tot=int(nn_tot), KK=h["KK"], LL=h["LL"])

    result = {
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
        "backend": params["backend"],
        "t_therm_s": t_therm, "t_dyn_s": t_dyn,
    }
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"k_HEOM = {k_HEOM:.6e}   log10(beta*k_HEOM) = {log10_k}", flush=True)
    print(f"timing: therm {t_therm:.0f}s, dyn {t_dyn:.0f}s", flush=True)
    print("next: python3 2_plot.py" + ("" if args.outdir == "run" else f" --outdir {args.outdir}"))


if __name__ == "__main__":
    main()
