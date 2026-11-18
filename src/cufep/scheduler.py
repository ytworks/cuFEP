"""GPU allocation and job execution.

Units of work
    A *job* is one leg of one repeat of one ligand. Jobs are independent of each other and
    each runs in its own process. Inside a job the *replicas* (one per lambda window) are
    independent within an iteration, so they can be advanced on different GPUs and, under
    NVIDIA MPS, several at a time on the same GPU. A *worker* is a process with one OpenMM
    Context on one GPU; it advances the replicas assigned to it one after another.

Capacity
    The *oversubscription factor* of a job is the number of its workers that one GPU runs at
    the same time (1 without MPS). A worker of a job with factor f takes 1/f of a GPU.

Allocation (recomputed whenever a job starts, changes phase or ends)
    * A job that is in a part that cannot be spread out (plain MD before the alchemical
      run, restraint search, lambda schedule) holds one worker.
    * The rest of the capacity is divided among the jobs that are sampling, in proportion
      to the work they still have to do, and never more than one worker per replica.
    * A job learns about a change at its next iteration boundary and starts or stops
      workers there. Replica coordinates live in shared memory owned by the job, so no data
      move when workers change.

If there are more jobs than ``max_running`` they wait in a queue; freed capacity goes to
waiting jobs first.
"""
from __future__ import annotations

import multiprocessing as mp
import subprocess
import threading
import time
from dataclasses import dataclass, field

from cufep.workflow import JobControl, LegJob, read_result, run_leg_safe


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


def oversubscription_for(n_atoms: int, setting="auto", mps: bool = True) -> int:
    """Workers of a system of ``n_atoms`` that share one GPU at the same time.

    ``setting`` is ``resources.oversubscription_factor``. The automatic choice follows
    measurements of the total throughput on one GPU for systems of different sizes: the
    smaller the system, the more of them a GPU can run at once.
    """
    if not mps:
        return 1
    if setting != "auto":
        return max(1, int(setting))
    if n_atoms <= 10_000:
        return 16
    if n_atoms <= 60_000:
        return 8
    if n_atoms <= 200_000:
        return 4
    return 2


@dataclass
class Demand:
    """What one running job needs at the moment."""

    job: int
    phase: str = "serial"                # serial | sampling
    factor: int = 1
    max_workers: int = 1                 # sampling: the number of replicas
    work: float = 1.0                    # remaining work in arbitrary units; only ratios matter
    devices: list[int] = field(default_factory=list)     # GPU of each current worker


def allocate(gpus: list[int], demands: list[Demand]) -> dict[int, list[int]]:
    """GPU of every worker of every job, as ``{job: [gpu, gpu, ...]}``."""
    if not gpus:
        return {d.job: [] for d in demands}
    free = {g: 1.0 for g in gpus}
    out: dict[int, list[int]] = {}

    def emptiest(cost: float) -> int:
        fits = [g for g in gpus if free[g] + 1e-9 >= cost]
        return max(fits or gpus, key=lambda g: (round(free[g], 9), -gpus.index(g)))

    # 1. one worker for every job in a part that cannot be spread out; it stays where it is
    for d in (x for x in demands if x.phase != "sampling"):
        cost = 1.0 / d.factor
        g = d.devices[0] if d.devices and d.devices[0] in free else emptiest(cost)
        free[g] -= cost
        out[d.job] = [g]

    # 2. the rest in proportion to the remaining work, at most one worker per replica
    sampling = [x for x in demands if x.phase == "sampling"]
    share = {d.job: 0.0 for d in sampling}
    open_jobs, capacity = list(sampling), max(0.0, sum(max(0.0, v) for v in free.values()))
    while open_jobs:
        total = sum(max(d.work, 1e-12) for d in open_jobs)
        capped = [d for d in open_jobs if d.max_workers / d.factor <= capacity * max(d.work, 1e-12) / total + 1e-12]
        if not capped:
            for d in open_jobs:
                share[d.job] = capacity * max(d.work, 1e-12) / total
            break
        for d in capped:
            share[d.job] = d.max_workers / d.factor
            capacity -= share[d.job]
            open_jobs.remove(d)
    count = {d.job: int(max(1, min(d.max_workers, int(share[d.job] * d.factor + 1e-9)))) for d in sampling}

    # 3. place the workers: keep workers where they are if there is room, then the emptiest GPU
    for d in sorted(sampling, key=lambda x: -count[x.job] / x.factor):
        cost, placed = 1.0 / d.factor, []
        for g in d.devices:
            if len(placed) < count[d.job] and g in free and free[g] + 1e-9 >= cost \
                    and placed.count(g) < -(-count[d.job] // len(gpus)):
                placed.append(g); free[g] -= cost
        while len(placed) < count[d.job]:
            g = emptiest(cost)
            placed.append(g); free[g] -= cost
        out[d.job] = sorted(placed)

    # 4. hand out what rounding left over, one worker at a time, to the job with most work per worker
    spare = True
    while spare:
        spare = False
        for d in sorted(sampling, key=lambda x: -x.work / len(out[x.job])):
            cost = 1.0 / d.factor
            fits = [g for g in gpus if free[g] + 1e-9 >= cost]
            if len(out[d.job]) < d.max_workers and fits:
                g = max(fits, key=lambda g_: (round(free[g_], 9), -gpus.index(g_)))
                out[d.job] = sorted(out[d.job] + [g]); free[g] -= cost
                spare = True
                break
    return out


class GpuMonitor:
    """Samples the utilization of the GPUs of a run in the background."""

    def __init__(self, gpus: list[int], interval: float = 5.0) -> None:
        self.gpus, self.interval = list(gpus), interval
        self.samples: list[tuple[float, dict[int, float]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "GpuMonitor":
        if self.gpus:
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=20, check=True).stdout
                row = {}
                for line in out.strip().splitlines():
                    i, util = (int(x) for x in line.split(","))
                    if i in self.gpus:
                        row[i] = float(util)
                self.samples.append((time.time(), row))
            except Exception:  # noqa: BLE001 - monitoring must never disturb the run
                pass

    def stop(self) -> None:
        self._stop.set()

    def mean(self) -> dict[int, float]:
        return self._mean([r for _, r in self.samples])

    def mean_over(self, intervals: list[tuple[float, float]]) -> dict[int, float]:
        """Mean utilization over the samples that fall into any of the (start, end) intervals."""
        return self._mean([r for t, r in self.samples if any(a <= t <= b for a, b in intervals)])

    def _mean(self, rows: list[dict[int, float]]) -> dict[int, float | None]:
        return {g: (sum(r.get(g, 0.0) for r in rows) / len(rows) if rows else None) for g in self.gpus}


def _entry(job: LegJob, conn) -> None:
    job.control = JobControl(conn)
    run_leg_safe(job)


def run_jobs(jobs: list[LegJob], gpus: list, log=print, oversubscription="auto", mps: bool = False,
             max_running: int | None = None, poll_seconds: float = 0.2) -> list[dict]:
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
    gpus = list(gpus)
    max_running = max_running or max(2, len(gpus))
    for job in jobs:
        job.oversubscription = oversubscription_for(job.n_atoms, oversubscription, mps)
    waiting = list(range(len(jobs)))
    running: dict[int, dict] = {}
    ctx = mp.get_context("spawn")
    t0 = time.time()

    def demands() -> list[Demand]:
        return [Demand(job=i, phase=r["phase"], factor=jobs[i].oversubscription, max_workers=r["max_workers"],
                       work=r["work"], devices=list(r["devices"])) for i, r in running.items()]

    def rebalance(reason: str) -> None:
        plan = allocate(gpus, demands())
        for i, devices in plan.items():
            r = running[i]
            if r["phase"] == "sampling" and (devices != r["devices"] or r["waiting_for_devices"]):
                try:
                    r["conn"].send({"devices": devices})
                except (BrokenPipeError, OSError):
                    continue
                if devices != r["devices"] or r["waiting_for_devices"]:
                    log(f"{jobs[i].tag}: {len(devices)} worker(s) on GPU " + _summary(devices) + f" ({reason})")
                r["waiting_for_devices"] = False
            r["devices"] = devices

    while waiting or running:
        # start waiting jobs
        while waiting and len(running) < max_running:
            i = waiting.pop(0)
            job = jobs[i]
            running[i] = {"phase": "serial", "max_workers": 1, "work": 1.0, "devices": [], "waiting_for_devices": False}
            device = allocate(gpus, demands())[i][0]
            running[i]["devices"] = [device]
            job.devices, job.inline = [device], False
            if job.leg_dir.exists():
                (job.leg_dir / "error.json").unlink(missing_ok=True)
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_entry, args=(job, child))
            proc.start()
            child.close()
            running[i].update(proc=proc, conn=parent)
            log(f"started {job.tag} on GPU {device} ({len(running)} running, {len(waiting)} waiting)")
            rebalance(f"{job.tag} started")
        # messages from the jobs
        for i in list(running):
            r = running[i]
            try:
                while r["conn"].poll():
                    msg = r["conn"].recv()
                    if msg.get("event") == "sampling":
                        r.update(phase="sampling", max_workers=int(msg["replicas"]), work=float(msg["work"]),
                                 waiting_for_devices=True)
                        rebalance(f"{jobs[i].tag} starts sampling {msg['replicas']} lambda windows")
                    elif msg.get("event") == "progress":
                        r["work"] = float(msg["work"])
                    elif msg.get("event") == "serial":
                        r.update(phase="serial", max_workers=1, devices=r["devices"][:1])
                        rebalance(f"{jobs[i].tag} finished sampling")
            except (EOFError, OSError):
                pass
        # finished jobs
        for i in list(running):
            proc = running[i]["proc"]
            if not proc.is_alive():
                proc.join()
                running[i]["conn"].close()
                del running[i]
                res = read_result(jobs[i])
                state = "FAILED: " + res["error"] if "error" in res else "finished"
                log(f"{jobs[i].tag} {state} ({(time.time() - t0) / 60:.1f} min since start)")
                if running:
                    rebalance(f"{jobs[i].tag} ended")
        time.sleep(poll_seconds)
    return [read_result(job) for job in jobs]


def _summary(devices: list[int]) -> str:
    return ", ".join(f"{g}x{devices.count(g)}" if devices.count(g) > 1 else str(g) for g in sorted(set(devices)))
