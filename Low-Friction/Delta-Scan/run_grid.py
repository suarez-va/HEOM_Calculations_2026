"""
Grid helpers shared by the local driver (1_run_*_local.py) and the SLURM driver
(1_submit_*_slurm.py):

  - prepare_grid      : create the grid directory + val<idx>/ subdirs, write vals.txt and decide
                        which indices still need running (skip finished ones unless --force)
  - run_grid_local    : execute independent jobs locally, each pinned with taskset to its own
                        block of `threads_per_job` cores
  - array_spec        : compress a list of indices into a SLURM --array spec, e.g. "0-5,8,10-12"
  - write_sbatch_array: write the sbatch job-array script (one array task per grid point)

Adapted from EDMD/src/workflow/run_grid.py, kept self-contained here so this scan directory
has no cross-repo import. Differences from that version:

  - each job's stdout/stderr is captured to its own log file, so a failure is diagnosable
    after the fact instead of being interleaved on the terminal
  - returns the per-job exit codes and prints a failure summary
  - jobs are dispatched by index, so the caller can pre-filter to only the incomplete ones
  - a job may use several cores (threads_per_job), pinned to a contiguous core range
"""

import os
import sys
import time
import shlex
import subprocess

import numpy as np


def prepare_grid(grid_dir, vals, header, only=None, force=False):
    """
    Make grid_dir/ and grid_dir/val<idx>/ for every point, write grid_dir/vals.txt and return the
    list of indices to run: `only` (or all points), minus those that already have a result.json
    unless force=True.
    """
    os.makedirs(grid_dir, exist_ok=True)
    np.savetxt(os.path.join(grid_dir, "vals.txt"), vals, header=header)
    npoints = len(vals)
    for i in range(npoints):
        os.makedirs(os.path.join(grid_dir, f"val{i}"), exist_ok=True)

    candidates = list(only) if only is not None else list(range(npoints))
    bad = [i for i in candidates if not 0 <= i < npoints]
    if bad:
        raise SystemExit(f"--only indices out of range 0..{npoints-1}: {bad}")
    if force:
        return candidates

    todo = [i for i in candidates
            if not os.path.exists(os.path.join(grid_dir, f"val{i}", "result.json"))]
    skipped = sorted(set(candidates) - set(todo))
    if skipped:
        print(f"skipping {len(skipped)} already-complete point(s): {skipped}")
    return todo


def run_grid_local(script, args_list, ncores=8, cwd_list=None, log_list=None, indices=None,
                   threads_per_job=1):
    """
    script          : absolute path to the Python script to run
    args_list       : list of CLI argument lists, one per job, e.g. [['--idx','0'], ...]
    ncores          : total number of cores to use
    cwd_list        : optional per-job working directory
    log_list        : optional per-job path for combined stdout/stderr
    indices         : optional subset of job indices to actually run (default: all)
    threads_per_job : cores given to each job; ncores // threads_per_job jobs run at once, and
                      slot k is pinned to cores [k*t, (k+1)*t - 1]

    Returns {job_idx: returncode}.
    """
    t = int(threads_per_job)
    nslots = ncores // t
    if nslots < 1:
        raise ValueError(f"ncores={ncores} is smaller than threads_per_job={t}")
    cores = [f"{k*t}-{k*t+t-1}" if t > 1 else str(k) for k in range(nslots)]

    queue = list(indices) if indices is not None else list(range(len(args_list)))
    total = len(queue)
    slots = [None] * nslots            # slots[k] = (Popen, job_idx, filehandle) or None
    codes = {}
    done = 0
    t0 = time.time()

    print(f"dispatching {total} job(s): {nslots} at a time x {t} core(s) each "
          f"(cores 0-{nslots*t-1})", flush=True)

    while queue or any(s is not None for s in slots):

        for k in range(nslots):
            if slots[k] is not None:
                proc, job_idx, fh = slots[k]
                if proc.poll() is not None:
                    if fh is not None:
                        fh.close()
                    codes[job_idx] = proc.returncode
                    done += 1
                    status = "done" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                    print(f"[slot {k} cores {cores[k]}] job {job_idx} {status}   "
                          f"({done}/{total}, {time.time()-t0:.0f}s elapsed)", flush=True)
                    slots[k] = None

        for k in range(nslots):
            if slots[k] is None and queue:
                job_idx = queue.pop(0)
                cwd = cwd_list[job_idx] if cwd_list is not None else None
                fh = open(log_list[job_idx], "w") if log_list is not None else None
                cmd = ["taskset", "-c", cores[k], sys.executable, script] + args_list[job_idx]
                proc = subprocess.Popen(cmd, cwd=cwd,
                                        stdout=fh if fh is not None else None,
                                        stderr=subprocess.STDOUT if fh is not None else None)
                slots[k] = (proc, job_idx, fh)
                print(f"[slot {k} cores {cores[k]}] job {job_idx} started: "
                      f"{os.path.basename(script)} {' '.join(args_list[job_idx])}", flush=True)

        time.sleep(0.1)

    failed = sorted(i for i, rc in codes.items() if rc != 0)
    print(f"\nall jobs finished in {time.time()-t0:.0f}s: "
          f"{len(codes)-len(failed)} ok, {len(failed)} failed", flush=True)
    if failed:
        print(f"failed job indices: {failed}")
        if log_list is not None:
            print(f"see e.g. {log_list[failed[0]]}")
    return codes


def array_spec(indices):
    """[0,1,2,3,5,7,8] -> '0-3,5,7-8' (SLURM --array syntax)."""
    idx = sorted(set(int(i) for i in indices))
    parts, start = [], None
    for j, i in enumerate(idx):
        if start is None:
            start = i
        if j == len(idx) - 1 or idx[j + 1] != i + 1:
            parts.append(str(start) if start == i else f"{start}-{i}")
            start = None
    return ",".join(parts)


def write_sbatch_array(path, script, cfg_path, grid_dir, indices, threads, slurm_cfg, cwd):
    """
    Write an sbatch job-array script: one array task per grid index, each with `threads` cores,
    running  <python> <script> --idx $SLURM_ARRAY_TASK_ID --config <cfg_path> --num-threads N.
    The val<idx>/ directories must already exist (SLURM does not create the --output dir).
    """
    s = slurm_cfg or {}
    spec = array_spec(indices)
    if s.get("max_concurrent"):
        spec += f"%{int(s['max_concurrent'])}"

    lines = ["#!/bin/bash",
             f"#SBATCH --job-name={s.get('job_name', 'heom_scan')}",
             f"#SBATCH --array={spec}",
             "#SBATCH --nodes=1",
             "#SBATCH --ntasks=1",
             f"#SBATCH --cpus-per-task={threads}",
             f"#SBATCH --output={grid_dir}/val%a/slurm-%A_%a.log"]
    for key in ("time", "mem", "partition", "account", "qos"):
        if s.get(key) is not None:
            lines.append(f"#SBATCH --{key}={s[key]}")
    for extra in s.get("extra_sbatch") or []:
        extra = str(extra).strip()
        lines.append(extra if extra.startswith("#SBATCH") else f"#SBATCH {extra}")

    python = s.get("python", "python3")
    lines += ["",
              "set -eo pipefail",
              "",
              "# ---- environment setup (slurm.setup in the config) ----",
              (s.get("setup") or "").rstrip(),
              "",
              "# one point per array task; it gets exactly the cores SLURM allocated",
              f"NT=${{SLURM_CPUS_PER_TASK:-{threads}}}",
              "for v in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS "
              "NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS NUMBA_NUM_THREADS; do export $v=$NT; done",
              "",
              "echo \"job ${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID} on $(hostname), "
              "$NT thread(s), started $(date)\"",
              f"cd {shlex.quote(cwd)}",
              f"{python} {shlex.quote(script)} --idx \"$SLURM_ARRAY_TASK_ID\" --config {shlex.quote(cfg_path)} "
              f"--num-threads \"$NT\"",
              "echo \"finished $(date)\"",
              ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return spec
