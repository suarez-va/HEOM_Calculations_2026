"""
Minimal local job runner: execute a list of independent single-process jobs across `ncores`
cores, each pinned with taskset.

Adapted from EDMD/src/workflow/run_grid.py, kept self-contained here so this scan directory
has no cross-repo import. Differences from that version:

  - each job's stdout/stderr is captured to its own log file, so a failure is diagnosable
    after the fact instead of being interleaved on the terminal
  - returns the per-job exit codes and prints a failure summary
  - jobs are dispatched by index, so the caller can pre-filter to only the incomplete ones
"""

import os
import sys
import time
import subprocess


def run_grid_local(script, args_list, ncores=8, cwd_list=None, log_list=None, indices=None):
    """
    script    : absolute path to the Python script to run
    args_list : list of CLI argument lists, one per job, e.g. [['--idx','0'], ...]
    ncores    : how many jobs to keep running at once (each pinned to one core)
    cwd_list  : optional per-job working directory
    log_list  : optional per-job path for combined stdout/stderr
    indices   : optional subset of job indices to actually run (default: all)

    Returns {job_idx: returncode}.
    """
    queue = list(indices) if indices is not None else list(range(len(args_list)))
    total = len(queue)
    slots = [None] * ncores            # slots[core] = (Popen, job_idx, filehandle) or None
    codes = {}
    done = 0
    t0 = time.time()

    print(f"dispatching {total} job(s) over {ncores} core(s)", flush=True)

    while queue or any(s is not None for s in slots):

        for core in range(ncores):
            if slots[core] is not None:
                proc, job_idx, fh = slots[core]
                if proc.poll() is not None:
                    if fh is not None:
                        fh.close()
                    codes[job_idx] = proc.returncode
                    done += 1
                    status = "done" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                    print(f"[core {core}] job {job_idx} {status}   "
                          f"({done}/{total}, {time.time()-t0:.0f}s elapsed)", flush=True)
                    slots[core] = None

        for core in range(ncores):
            if slots[core] is None and queue:
                job_idx = queue.pop(0)
                cwd = cwd_list[job_idx] if cwd_list is not None else None
                fh = open(log_list[job_idx], "w") if log_list is not None else None
                cmd = ["taskset", "-c", str(core), sys.executable, script] + args_list[job_idx]
                proc = subprocess.Popen(cmd, cwd=cwd,
                                        stdout=fh if fh is not None else None,
                                        stderr=subprocess.STDOUT if fh is not None else None)
                slots[core] = (proc, job_idx, fh)
                print(f"[core {core}] job {job_idx} started: "
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
