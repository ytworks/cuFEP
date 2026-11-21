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
    * A job whose workers cannot move asks for them and keeps them until it asks again or is
      serial again: the windows of the lambda precalculation, and the trajectories from which
      a restraint is chosen (``restraints.selection: interactions``), one worker each. When
      it is serial again it goes on with the first GPU of what it was granted.
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
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field

from cufep.workflow import JobControl, LegJob, begin_leg, read_result, record_ended_job, run_leg_safe


@dataclass(frozen=True)
class VisibleGpu:
    """A GPU this process can use: its CUDA device number (what OpenMM's DeviceIndex and
    ``resources.gpus`` mean) and its number and UUID in nvidia-smi, which counts every GPU of
    the machine."""

    logical: int
    physical: int
    uuid: str
    utilization: float = 0.0


def query_gpus() -> list[tuple[int, str, float]]:
    """(nvidia-smi index, UUID, utilization in %) of every GPU of the machine; [] without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        try:
            rows.append((int(parts[0]), parts[1], float(parts[2])))
        except (ValueError, IndexError):
            continue
    return rows


def use_pci_bus_order(environ=None) -> None:
    """Make CUDA number the GPUs in the order of their PCI bus, as nvidia-smi does. CUDA's own
    default (FASTEST_FIRST) leaves the order unspecified, so a CUDA device number could not be
    matched with the GPU nvidia-smi reports. Must be set before a process makes a CUDA context;
    the processes of a run inherit it. Another order set by the user is refused."""
    environ = os.environ if environ is None else environ
    order = environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if order != "PCI_BUS_ID":
        raise ValueError(f"CUDA_DEVICE_ORDER={order}: cuFEP needs PCI_BUS_ID (or no setting) to tell which GPU a CUDA "
                         "device number is")


def visible_gpus(rows=None, environ=None) -> list[VisibleGpu]:
    """The GPUs this process can use, in the order of their CUDA device numbers.

    ``CUDA_VISIBLE_DEVICES`` is read as CUDA reads it: a comma-separated list of nvidia-smi
    numbers or (prefixes of) UUIDs; CUDA numbers the listed GPUs 0, 1, ... in that order and
    stops at the first entry that is not a GPU (``-1`` or an empty value hide all). Review of
    PR 1: with ``CUDA_VISIBLE_DEVICES=2`` on a machine of four GPUs only device 0 exists for
    CUDA, and it is the GPU nvidia-smi calls 2. MIG instances are not supported.
    """
    environ = os.environ if environ is None else environ
    rows = query_gpus() if rows is None else list(rows)
    by_index = {index: (index, uuid, util) for index, uuid, util in rows}
    mask = environ.get("CUDA_VISIBLE_DEVICES")
    if mask is None:
        chosen = [by_index[i] for i in sorted(by_index)]
    else:
        chosen = []
        for entry in (e.strip() for e in mask.split(",")):
            if entry.upper().startswith("MIG-"):
                raise ValueError(f"CUDA_VISIBLE_DEVICES={mask}: MIG instances are not supported by cuFEP")
            if entry.isdigit() and int(entry) in by_index:
                row = by_index[int(entry)]
            elif entry.upper().startswith("GPU-"):
                matches = [r for r in rows if r[1].upper().startswith(entry.upper())]
                if len(matches) != 1:
                    break
                row = matches[0]
            else:
                break                                   # CUDA stops at the first entry that is no GPU
            if row in chosen:
                break
            chosen.append(row)
    return [VisibleGpu(logical=k, physical=index, uuid=uuid, utilization=util)
            for k, (index, uuid, util) in enumerate(chosen)]


def detect_gpus(max_utilization: int = 50) -> list[int]:
    """CUDA device numbers of the visible GPUs that are not busy computing right now.

    Memory held by idle processes does not count as busy; utilization does.
    """
    use_pci_bus_order()
    visible = visible_gpus()
    free = [g.logical for g in visible if g.utilization <= max_utilization]
    return free or [g.logical for g in visible]        # if everything is busy, share rather than refuse


def check_gpus(gpus: list[int]) -> None:
    """ValueError if ``resources.gpus`` names a CUDA device number that this process does not have."""
    use_pci_bus_order()
    visible = visible_gpus()
    if not visible:
        return                                          # nvidia-smi cannot tell; CUDA will
    wrong = [g for g in gpus if g is not None and not 0 <= int(g) < len(visible)]
    if wrong:
        mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        raise ValueError(f"resources.gpus: {wrong} are no CUDA devices of this process, which has "
                         f"{len(visible)} (numbered from 0{'' if mask is None else f'; CUDA_VISIBLE_DEVICES={mask}'})")


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
    #: serial: one worker, where it runs; sampling: workers that can be moved between iterations;
    #: pinned: workers that cannot be moved (the precalculation of adaptive lambda scheduling)
    phase: str = "serial"                # serial | sampling | pinned
    factor: int = 1
    max_workers: int = 1                 # sampling: the number of replicas
    work: float = 1.0                    # remaining work in arbitrary units; only ratios matter
    devices: list[int] = field(default_factory=list)     # GPU of each current worker


def allocate(gpus: list[int], demands: list[Demand], strict: bool = False) -> dict[int, list[int]] | None:
    """GPU of every worker of every job, as ``{job: [gpu, gpu, ...]}``.

    A worker of a job with oversubscription factor ``f`` takes 1/f of a GPU. Every job gets one
    worker first: a job in a part that cannot be spread out (plain MD, analysis) keeps the GPU it
    runs on, a sampling job keeps one of its GPUs if there is room. What is left is shared among
    the sampling jobs in proportion to their remaining work, at most one worker per replica, and
    a worker is placed only where it fits. With ``strict`` the result is None if the first
    workers do not fit; the scheduler starts a job only when they do, so that no GPU ever holds
    more than it can take (without MPS: one worker per GPU; review of PR 1). Without
    ``strict`` a first worker that does not fit goes to the emptiest GPU, which only a state the
    scheduler never makes can need.
    """
    if not gpus:
        return {d.job: [] for d in demands}
    free = {g: 1.0 for g in gpus}
    out: dict[int, list[int]] = {}

    def fits(g: int, cost: float) -> bool:
        return free[g] + 1e-9 >= cost

    def emptiest(cost: float, over: bool = False) -> int | None:
        candidates = [g for g in gpus if fits(g, cost)] or (gpus if over else [])
        return max(candidates, key=lambda g: (round(free[g], 9), -gpus.index(g))) if candidates else None

    # 1. one worker for every job: plain MD stays where it runs (it cannot move), then the jobs that
    #    can go anywhere, the larger workers first
    def pinned(d: Demand) -> bool:
        return d.phase != "sampling" and bool(d.devices) and d.devices[0] in free

    first = sorted(demands, key=lambda d: (not pinned(d), d.phase == "sampling", -1.0 / d.factor, d.job))
    for d in [x for x in demands if x.phase == "pinned"]:
        # every worker of a job that cannot move its workers stays where it is
        cost = 1.0 / d.factor
        for g in d.devices:
            if g in free:
                if strict and not fits(g, cost):
                    return None
                free[g] -= cost
        out[d.job] = list(d.devices)
    for d in [x for x in first if x.phase != "pinned"]:
        cost = 1.0 / d.factor
        if pinned(d):
            g = d.devices[0]                                   # it is running there
            if strict and not fits(g, cost):
                return None
        else:
            kept = [g for g in d.devices if g in free and fits(g, cost)]
            g = kept[0] if kept else emptiest(cost, over=not strict)
            if g is None:
                return None
        free[g] -= cost
        out[d.job] = [g]

    # 2. the rest of the capacity in proportion to the remaining work, at most one worker per replica
    sampling = [x for x in demands if x.phase == "sampling"]
    share = {d.job: 0.0 for d in sampling}
    open_jobs, capacity = list(sampling), max(0.0, sum(max(0.0, v) for v in free.values()))
    while open_jobs:
        total = sum(max(d.work, 1e-12) for d in open_jobs)
        capped = [d for d in open_jobs
                  if (d.max_workers - 1) / d.factor <= capacity * max(d.work, 1e-12) / total + 1e-12]
        if not capped:
            for d in open_jobs:
                share[d.job] = capacity * max(d.work, 1e-12) / total
            break
        for d in capped:
            share[d.job] = (d.max_workers - 1) / d.factor
            capacity -= share[d.job]
            open_jobs.remove(d)
    count = {d.job: int(min(d.max_workers, 1 + int(share[d.job] * d.factor + 1e-9))) for d in sampling}

    # 3. place them: keep workers where they are if there is room, then the emptiest GPU that has room
    for d in sorted(sampling, key=lambda x: -count[x.job] / x.factor):
        cost, placed = 1.0 / d.factor, list(out[d.job])
        spread = -(-count[d.job] // len(gpus))
        rest = list(d.devices)
        if placed[0] in rest:
            rest.remove(placed[0])
        for g in rest:
            if len(placed) < count[d.job] and g in free and fits(g, cost) and placed.count(g) < spread:
                placed.append(g)
                free[g] -= cost
        while len(placed) < count[d.job]:
            g = emptiest(cost)
            if g is None:
                break
            placed.append(g)
            free[g] -= cost
        out[d.job] = sorted(placed)

    # 4. hand out what rounding left over, one worker at a time, to the job with most work per worker
    spare = True
    while spare:
        spare = False
        for d in sorted(sampling, key=lambda x: -x.work / len(out[x.job])):
            cost = 1.0 / d.factor
            g = emptiest(cost)
            if len(out[d.job]) < d.max_workers and g is not None:
                out[d.job] = sorted(out[d.job] + [g])
                free[g] -= cost
                spare = True
                break
    return out


class GpuMonitor:
    """Samples the utilization of the GPUs of a run in the background."""

    def __init__(self, gpus: list[int], interval: float = 5.0) -> None:
        self.gpus, self.interval = list(gpus), interval
        # nvidia-smi's number of every CUDA device of the run, taken now: once cuFEP's MPS daemon
        # runs, the processes of the run no longer carry CUDA_VISIBLE_DEVICES (cufep.mps)
        self.physical = {g.logical: g.physical for g in visible_gpus()}
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
                # by CUDA device number, the numbers of the run, not by nvidia-smi's own
                load = {index: util for index, _uuid, util in query_gpus()}
                row = {g: load[self.physical[g]] for g in self.gpus if self.physical.get(g) in load}
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


#: what a job process runs (tests put a stand-in here; it must be importable by the job process)
JOB_FUNCTION = run_leg_safe


def _entry(job: LegJob, conn, scheduler_pid: int | None = None, function=None) -> None:
    import signal

    from cufep.processes import die_with_parent, hold_job_lock

    # a process group of its own: the scheduler stops the job and every worker it started together,
    # and an interrupt at the terminal reaches the scheduler, which stops the jobs in order
    os.setpgrp()

    def stop(signum, frame):                    # noqa: ARG001
        raise SystemExit(128 + signum)          # leaves every `with` and `finally`: workers stop, files close

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # a scheduler that ends without stopping its jobs (SIGKILL, a crash) stops them all the same,
    # and while this job lives no new run uses its leg (cufep.processes)
    if scheduler_pid is not None:
        die_with_parent(scheduler_pid)
    lock = hold_job_lock(job.leg_dir)           # noqa: F841  held until this process ends
    job.control = JobControl(conn)
    (function or run_leg_safe)(job)


#: how long a job is given to stop its workers and close its files after SIGTERM before it is killed
STOP_GRACE_SECONDS = 120.0


def stop_jobs(running: dict[int, dict], jobs: list[LegJob], log=print, grace: float | None = None) -> None:
    """Stop every started job process and everything it started, and wait until they have ended.

    Each job gets SIGTERM (it stops its workers and closes its files), then SIGKILL for its whole
    process group if it has not ended within ``grace`` seconds. What a job left is recorded in its
    leg's files (:func:`record_ended_job`). Called when the scheduler itself fails or is
    interrupted: the output directory's lock must not be released while a job can still write
    there (review of PR 1).
    """
    import signal

    grace = STOP_GRACE_SECONDS if grace is None else float(grace)
    # a second interrupt (or a hangup, a SIGTERM) while the jobs stop must not end this function: the
    # lock of the output directory would go with it while jobs still write (review of PR 1)
    told = []

    def wait_please(signum, frame):            # noqa: ARG001
        if not told:
            told.append(signum)
            log(f"stopping the jobs; this ends by itself within {grace:.0f} s")

    previous = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, wait_please)
    try:
        _stop_started(running, jobs, log, grace)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _stop_started(running: dict[int, dict], jobs: list[LegJob], log, grace: float) -> None:
    import signal

    started = {i: r for i, r in running.items() if r.get("proc") is not None and r["proc"].pid is not None}
    for _i, r in started.items():
        if r["proc"].is_alive():
            try:
                r["proc"].terminate()
            except Exception:  # noqa: BLE001
                pass
    deadline = time.time() + grace
    for i, r in started.items():
        proc = r["proc"]
        proc.join(max(0.0, deadline - time.time()))
        if proc.is_alive():
            log(f"{jobs[i].tag} did not stop within {grace:.0f} s; killing it and its workers")
            try:
                os.killpg(proc.pid, signal.SIGKILL)     # its group: the job and every worker it started
            except (ProcessLookupError, PermissionError):
                proc.kill()
        proc.join()
        try:
            r["conn"].close()
        except Exception:  # noqa: BLE001
            pass
        try:
            record_ended_job(jobs[i], proc.exitcode)
        except Exception as exc:  # noqa: BLE001
            log(f"{jobs[i].tag}: what it left could not be recorded: {exc}")
        log(f"{jobs[i].tag} stopped (exit code {proc.exitcode})")


def run_jobs(jobs: list[LegJob], gpus: list, log=print, oversubscription="auto", mps: bool = False,
             max_running: int | None = None, poll_seconds: float = 0.2) -> list[dict]:
    """Run all jobs on the given devices and return their results in order."""
    if not jobs:
        return []
    if not gpus or gpus == [None]:
        results = []                                   # no GPU: one job at a time, in this process
        for job in jobs:
            job.devices, job.inline = [None], True
            results.append(run_leg_safe(job))
        return results
    gpus = list(gpus)
    max_running = max_running or max(2, len(gpus))
    return _run_started_jobs(jobs, gpus, log, oversubscription, mps, max_running, poll_seconds)


def _run_started_jobs(jobs: list[LegJob], gpus: list, log, oversubscription, mps: bool, max_running: int,
                      poll_seconds: float) -> list[dict]:
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

    def first_worker_fits(i: int) -> bool:
        """Whether job ``i`` can start now: its first worker and the first worker of every
        running job fit on the GPUs (without MPS: one per GPU)."""
        candidate = Demand(job=i, phase="serial", factor=jobs[i].oversubscription)
        return allocate(gpus, demands() + [candidate], strict=True) is not None

    def interrupted(signum, frame):            # noqa: ARG001
        raise SystemExit(128 + signum)

    previous = {}
    if threading.current_thread() is threading.main_thread():
        # a SIGTERM (kill, GNU timeout) or a hangup ends the run as Ctrl-C does: the jobs are stopped
        # before the lock of the output directory goes (review of PR 1)
        for sig in (signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, interrupted)
    try:
        while waiting or running:
            # start waiting jobs, as long as their first worker fits
            while waiting and len(running) < max_running and (not running or first_worker_fits(waiting[0])):
                i = waiting.pop(0)
                job = jobs[i]
                plan = allocate(gpus, demands() + [Demand(job=i, phase="serial", factor=job.oversubscription)],
                                strict=True) or allocate(gpus, demands() + [Demand(job=i, factor=job.oversubscription)])
                device = plan[i][0]
                running[i] = {"phase": "serial", "max_workers": 1, "work": 1.0, "devices": [device],
                              "waiting_for_devices": False, "proc": None, "conn": None}
                job.devices, job.inline = [device], False
                # set the leg's earlier result aside here, before the process starts: if the process
                # dies at once, the leg must not be read as finished
                begin_leg(job.leg_dir, job)
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=_entry, args=(job, child, os.getpid(), JOB_FUNCTION))
                running[i].update(proc=proc, conn=parent)       # known before it starts: stopped on any failure
                proc.start()
                child.close()
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
                            rebalance(f"{jobs[i].tag} starts "
                                      + (msg.get("what") or f"sampling {msg['replicas']} lambda windows"))
                            if msg.get("pinned"):
                                # it cannot move them: they stay its own until it asks again (review of PR 1)
                                r["phase"] = "pinned"
                        elif msg.get("event") == "progress":
                            r["work"] = float(msg["work"])
                        elif msg.get("event") == "serial":
                            r.update(phase="serial", max_workers=1, devices=r["devices"][:1])
                            rebalance(f"{jobs[i].tag} goes on with one worker")
                except (EOFError, OSError):
                    pass
            # finished jobs
            for i in list(running):
                proc = running[i]["proc"]
                if not proc.is_alive():
                    proc.join()
                    running[i]["conn"].close()
                    del running[i]
                    # a process that was killed wrote nothing: record that in the leg's files
                    res = record_ended_job(jobs[i], proc.exitcode)
                    state = "FAILED: " + res["error"] if "error" in res else "finished"
                    log(f"{jobs[i].tag} {state} ({(time.time() - t0) / 60:.1f} min since start)")
                    if running:
                        rebalance(f"{jobs[i].tag} ended")
            time.sleep(poll_seconds)
    except BaseException:
        stop_jobs(running, jobs, log)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return [read_result(job) for job in jobs]


def _summary(devices: list[int]) -> str:
    return ", ".join(f"{g}x{devices.count(g)}" if devices.count(g) > 1 else str(g) for g in sorted(set(devices)))
