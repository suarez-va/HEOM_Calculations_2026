"""
Driver: run the whole beta*epsilon scan locally, one single-threaded point per core.

    python3 1_run_eps_scan_local.py                # run whatever is not yet done
    python3 1_run_eps_scan_local.py --force        # re-run everything from scratch
    python3 1_run_eps_scan_local.py --ncores 12    # override config.yaml
    python3 1_run_eps_scan_local.py --only 0 1 2   # just these indices

Already-completed points (those with a result.json) are skipped unless --force, so the scan is
restartable - if it is interrupted, or a few points fail, just run it again.
"""

import os
import sys
import argparse

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_grid import run_grid_local
from run_heom_point import epsilon_grid

script_dir = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--ncores", type=int, default=None, help="override runtime.ncores")
ap.add_argument("--force", action="store_true", help="re-run points that already have results")
ap.add_argument("--only", type=int, nargs="+", default=None, help="run only these indices")
ap.add_argument("--config", default="config.yaml",
                help="config file to use (e.g. config_smoke.yaml for a fast dry run)")
args = ap.parse_args()

cfg_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

beta_eps_ar = epsilon_grid(cfg)
ncores = args.ncores if args.ncores is not None else cfg["runtime"]["ncores"]

grid_dir = os.path.join(script_dir, "eps_grid")
os.makedirs(grid_dir, exist_ok=True)
np.savetxt(os.path.join(grid_dir, "vals.txt"), beta_eps_ar,
           header="beta*epsilon, one line per val<idx> directory")

npoints = len(beta_eps_ar)
args_list = [["--idx", str(i), "--config", cfg_path] for i in range(npoints)]
cwd_list = [script_dir] * npoints                  # the point script chdir's itself
log_list = [os.path.join(grid_dir, f"val{i}", "run.log") for i in range(npoints)]
for i in range(npoints):
    os.makedirs(os.path.join(grid_dir, f"val{i}"), exist_ok=True)

# decide what to run
candidates = args.only if args.only is not None else list(range(npoints))
if args.force:
    todo = candidates
else:
    todo = [i for i in candidates
            if not os.path.exists(os.path.join(grid_dir, f"val{i}", "result.json"))]
    skipped = sorted(set(candidates) - set(todo))
    if skipped:
        print(f"skipping {len(skipped)} already-complete point(s): {skipped}")

if not todo:
    print("nothing to do - every requested point already has a result.json "
          "(use --force to recompute)")
    sys.exit(0)

print(f"beta*epsilon grid ({npoints} points): {beta_eps_ar}")
print(f"running indices: {todo}")

codes = run_grid_local(os.path.join(script_dir, "run_heom_point.py"),
                       args_list, ncores=ncores,
                       cwd_list=cwd_list, log_list=log_list, indices=todo)

print("\nnext: python3 2_collect.py")
sys.exit(1 if any(rc != 0 for rc in codes.values()) else 0)
