"""GPU allocation and job execution.

Two levels of parallelism:
  * if there are at least as many jobs (ligand x leg x repeat) as GPUs, every job gets one
    GPU and jobs queue up;
  * if GPUs are left over, complex legs (the expensive ones) spread their replicas over
    several GPUs.
Each job runs in its own process and starts one worker process per GPU it was given.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import subprocess
import time

from cufep.workflow import LegJob, run_leg_safe

MAX_GPUS_PER_JOB = 8


def detect_gpus(max_utilization: int = 50) -> list[int]:
    """Indices of visible NVIDIA GPUs that are not busy computing right now.

    Memory held by idle processes does not count as busy; utilization does.
    """
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception:  # noqa: BLE001
        return []
    gpus, busy = [], []
    for line in out.strip().splitlines():
        try:
            index, util = (int(x) for x in line.split(","))
        except ValueError:
            continue
        (busy if util > max_utilization else gpus).append(index)
    return gpus or busy            # if everything is busy, share rather than refuse


def assign_gpu_counts(jobs: list[LegJob], n_gpus: int) -> dict[int, int]:
    """Number of GPUs for each job (keyed by position in ``jobs``)."""
    counts = {i: 1 for i in range(len(jobs))}
    n_gpus = max(1, n_gpus)
    complex_jobs = [i for i, j in enumerate(jobs) if j.leg == "complex"]
    others = len(jobs) - len(complex_jobs)
    if complex_jobs and len(jobs) < n_gpus:
        per = max(1, min(MAX_GPUS_PER_JOB, (n_gpus - others) // len(complex_jobs)))
        for i in complex_jobs:
            counts[i] = per
    return counts


def _entry(job: LegJob) -> None:
    run_leg_safe(job)


def _read_result(job: LegJob) -> dict:
    for name in ("result.json", "error.json"):
        f = job.leg_dir / name
        if f.exists():
            return json.loads(f.read_text())
    return {"ligand": job.ligand, "leg": job.leg, "repeat": job.repeat,
            "error": "the job process ended without writing a result (killed?)"}


def run_jobs(jobs: list[LegJob], gpus: list, log=print, poll_seconds: float = 2.0) -> list[dict]:
    """Run all jobs on the given devices and return their results in order."""
    if not jobs:
        return []
    if not gpus or gpus == [None]:
        results = []                                   # no GPU: one job at a time, in this process
        for job in jobs:
            job.devices, job.inline = [None], True
            (job.leg_dir / "error.json").unlink(missing_ok=True)
            results.append(run_leg_safe(job))
        return results
    counts = assign_gpu_counts(jobs, len(gpus))
    order = sorted(range(len(jobs)), key=lambda i: (jobs[i].leg != "complex", i))
    free = list(gpus)
    running: dict[int, tuple] = {}
    ctx = mp.get_context("spawn")
    t0 = time.time()
    while order or running:
        started = True
        while started and order:
            started = False
            for i in list(order):
                need = min(counts[i], len(gpus))
                if len(free) >= need:
                    devs, free = free[:need], free[need:]
                    job = jobs[i]
                    job.devices, job.inline = devs, False
                    (job.leg_dir / "error.json").unlink(missing_ok=True) if job.leg_dir.exists() else None
                    proc = ctx.Process(target=_entry, args=(job,))
                    proc.start()
                    running[i] = (proc, devs)
                    order.remove(i)
                    log(f"started {job.tag} on GPU {','.join(str(d) for d in devs)} "
                        f"({len(running)} running, {len(order)} waiting)")
                    started = True
                    break
        for i in list(running):
            proc, devs = running[i]
            if not proc.is_alive():
                proc.join()
                free.extend(devs)
                del running[i]
                res = _read_result(jobs[i])
                state = "FAILED: " + res["error"] if "error" in res else "finished"
                log(f"{jobs[i].tag} {state} ({(time.time() - t0) / 60:.1f} min since start)")
        time.sleep(poll_seconds)
    return [_read_result(job) for job in jobs]
