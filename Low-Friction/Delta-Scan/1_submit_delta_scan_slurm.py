"""
Driver: submit the log10(beta*Delta) scan to SLURM as ONE job array, with one array task per
grid point. Each task asks for heom.num_threads cores (--cpus-per-task) and runs
run_heom_point.py --idx $SLURM_ARRAY_TASK_ID. Scheduler settings are in the `slurm:` block
of the config.

    python3 1_submit_delta_scan_slurm.py                # submit every point without a result.json
    python3 1_submit_delta_scan_slurm.py --dry-run      # only write the sbatch script, print it
    python3 1_submit_delta_scan_slurm.py --force        # resubmit every point
    python3 1_submit_delta_scan_slurm.py --only 3 7     # just these indices
    python3 1_submit_delta_scan_slurm.py --threads 8    # cores per point (override heom.num_threads)
    python3 1_submit_delta_scan_slurm.py --config config_smoke.yaml

The array indices ARE the grid indices, so finished points are left out of the array. When some
tasks fail or time out, run this again to resubmit only the missing ones. Per-point logs go to
<grid_dir>/val<idx>/slurm-<jobid>_<idx>.log. When everything is done: python3 2_collect.py
"""

import os
import sys
import shutil
import argparse
import subprocess

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_grid import prepare_grid, write_sbatch_array
from run_heom_point import log10_delta_grid

script_dir = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=None,
                help="cores per point (override heom.num_threads)")
ap.add_argument("--force", action="store_true", help="resubmit points that already have results")
ap.add_argument("--only", type=int, nargs="+", default=None, help="submit only these indices")
ap.add_argument("--dry-run", action="store_true", help="write the sbatch script but do not submit")
ap.add_argument("--config", default="config.yaml")
args = ap.parse_args()

cfg_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
cfg_path = os.path.abspath(cfg_path)
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

log10_delta_ar = log10_delta_grid(cfg)
threads = args.threads if args.threads is not None else cfg["heom"]["num_threads"]

grid_dir = os.path.join(script_dir, cfg["runtime"].get("grid_dir", "delta_grid"))
todo = prepare_grid(grid_dir, log10_delta_ar,
                    header="log10(beta*Delta), one line per val<idx> directory",
                    only=args.only, force=args.force)
if not todo:
    print("nothing to submit - every requested point already has a result.json "
          "(use --force to recompute)")
    sys.exit(0)

sbatch_path = os.path.join(grid_dir, "submit_array.sbatch")
spec = write_sbatch_array(sbatch_path, os.path.join(script_dir, "run_heom_point.py"), cfg_path,
                          grid_dir, todo, threads, cfg.get("slurm"), cwd=script_dir)

print(f"log10(beta*Delta) grid ({len(log10_delta_ar)} points), submitting {len(todo)}: {todo}")
print(f"array = {spec}   cores per point = {threads}")
print(f"wrote {sbatch_path}")

if args.dry_run:
    print("\n--dry-run: not submitted. Script:\n")
    print(open(sbatch_path).read())
    sys.exit(0)

if shutil.which("sbatch") is None:
    sys.exit("sbatch not found on this machine - submit it yourself with:  sbatch " + sbatch_path)

out = subprocess.run(["sbatch", sbatch_path], capture_output=True, text=True)
print(out.stdout.strip())
if out.returncode != 0:
    sys.exit(f"sbatch failed (exit {out.returncode}):\n{out.stderr.strip()}")
print(f"monitor with:  squeue -u $USER      logs in {grid_dir}/val<idx>/slurm-*.log")
print("when finished: python3 2_collect.py" + ("" if args.config == "config.yaml" else f" --config {args.config}"))
