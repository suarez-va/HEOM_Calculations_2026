"""
Driver: run the whole log10(beta*Delta) scan locally. Each point gets heom.num_threads cores
(pinned with taskset), and runtime.ncores // heom.num_threads points run at once.

    python3 1_run_delta_scan_local.py                # run whatever is not yet done
    python3 1_run_delta_scan_local.py --force        # re-run everything from scratch
    python3 1_run_delta_scan_local.py --ncores 12    # total cores to use (override config.yaml)
    python3 1_run_delta_scan_local.py --threads 4    # cores per point (override heom.num_threads)
    python3 1_run_delta_scan_local.py --only 0 1 2   # just these indices
    python3 1_run_delta_scan_local.py --config config_smoke.yaml   # cheap test, own grid_dir

Already-completed points (those with a result.json) are skipped unless --force, so the scan is
restartable - if it is interrupted, or a few points fail, just run it again.
For an HPC cluster use 1_submit_delta_scan_slurm.py instead (one SLURM array task per point).
"""

import os
import sys
import argparse

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_grid import run_grid_local, prepare_grid
from run_heom_point import log10_delta_grid

script_dir = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--ncores", type=int, default=None, help="total cores (override runtime.ncores)")
ap.add_argument("--threads", type=int, default=None,
                help="cores per point (override heom.num_threads)")
ap.add_argument("--force", action="store_true", help="re-run points that already have results")
ap.add_argument("--only", type=int, nargs="+", default=None, help="run only these indices")
ap.add_argument("--config", default="config.yaml",
                help="config file to use (e.g. config_smoke.yaml for a fast dry run)")
args = ap.parse_args()

cfg_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

log10_delta_ar = log10_delta_grid(cfg)
ncores = args.ncores if args.ncores is not None else cfg["runtime"]["ncores"]
threads = args.threads if args.threads is not None else cfg["heom"]["num_threads"]

grid_dir = os.path.join(script_dir, cfg["runtime"].get("grid_dir", "delta_grid"))
todo = prepare_grid(grid_dir, log10_delta_ar,
                    header="log10(beta*Delta), one line per val<idx> directory",
                    only=args.only, force=args.force)
if not todo:
    print("nothing to do - every requested point already has a result.json "
          "(use --force to recompute)")
    sys.exit(0)

npoints = len(log10_delta_ar)
args_list = [["--idx", str(i), "--config", cfg_path, "--num-threads", str(threads)]
             for i in range(npoints)]
cwd_list = [script_dir] * npoints                  # the point script chdir's itself
log_list = [os.path.join(grid_dir, f"val{i}", "run.log") for i in range(npoints)]

print(f"log10(beta*Delta) grid ({npoints} points): {log10_delta_ar}")
print(f"output -> {grid_dir}")
print(f"running indices: {todo}")

codes = run_grid_local(os.path.join(script_dir, "run_heom_point.py"),
                       args_list, ncores=ncores, threads_per_job=threads,
                       cwd_list=cwd_list, log_list=log_list, indices=todo)

print("\nnext: python3 2_collect.py" + ("" if args.config == "config.yaml" else f" --config {args.config}"))
sys.exit(1 if any(rc != 0 for rc in codes.values()) else 0)
