"""The scheduler of the GPUs: which devices a run has, how many jobs share them, and that a run that
fails or is interrupted stops every process it started before it gives up its output directory
(review of PR 1 on a3462a5)."""
import json
import os
import random
import signal
import time
from pathlib import Path

import pytest

from cufep import scheduler
from cufep.scheduler import Demand, VisibleGpu, allocate, check_gpus, detect_gpus, use_pci_bus_order, visible_gpus
from cufep.workflow import LegJob

FOUR = [(0, "GPU-aaaa-0", 0.0), (1, "GPU-bbbb-1", 0.0), (2, "GPU-cccc-2", 0.0), (3, "GPU-dddd-3", 0.0)]


# ---- which GPUs a process has -----------------------------------------------------------------------------

def test_cuda_visible_devices_is_read_as_cuda_reads_it():
    """Review of PR 1 (P1): with CUDA_VISIBLE_DEVICES=2 on four GPUs only CUDA device 0 exists, and it is the GPU
    nvidia-smi calls 2; nvidia-smi's numbers were used as CUDA's."""
    assert visible_gpus(FOUR, {}) == [VisibleGpu(k, k, FOUR[k][1]) for k in range(4)]
    assert [(g.logical, g.physical) for g in visible_gpus(FOUR, {"CUDA_VISIBLE_DEVICES": "2"})] == [(0, 2)]
    assert [(g.logical, g.physical) for g in visible_gpus(FOUR, {"CUDA_VISIBLE_DEVICES": "3,1"})] == [(0, 3), (1, 1)]
    assert [(g.logical, g.physical) for g in visible_gpus(FOUR, {"CUDA_VISIBLE_DEVICES": "GPU-cccc"})] == [(0, 2)]
    assert (
        visible_gpus(FOUR, {"CUDA_VISIBLE_DEVICES": "-1"}) == []
        and visible_gpus(FOUR, {"CUDA_VISIBLE_DEVICES": ""}) == []
    )
    # CUDA stops at the first entry that is not a GPU, and at a repeated one
    assert [g.physical for g in visible_gpus(FOUR, {"CUDA_VISIBLE_DEVICES": "1,7,3"})] == [1]
    assert [g.physical for g in visible_gpus(FOUR, {"CUDA_VISIBLE_DEVICES": "1,1,3"})] == [1]
    with pytest.raises(ValueError, match="MIG"):
        visible_gpus(FOUR, {"CUDA_VISIBLE_DEVICES": "MIG-1234"})


def test_detected_gpus_are_cuda_device_numbers_and_their_load_is_their_own(monkeypatch):
    rows = [(0, "GPU-a", 95.0), (1, "GPU-b", 0.0), (2, "GPU-c", 97.0), (3, "GPU-d", 0.0)]
    monkeypatch.setattr(scheduler, "query_gpus", lambda: rows)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert detect_gpus() == [0]                       # the only device, though busy: shared rather than refused
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,0")
    assert detect_gpus() == [1]                       # nvidia-smi's 3 is CUDA's 1, the one that is free
    with pytest.raises(ValueError, match=r"resources.gpus: \[3\] are no CUDA devices of this process, which has 3"):
        check_gpus([0, 3])
    check_gpus([0, 1, 2])
    monitor = scheduler.GpuMonitor([0, 1], interval=0.01).start()
    time.sleep(0.2)
    monitor.stop()
    assert monitor.samples and all(row == {0: 97.0, 1: 0.0} for _, row in monitor.samples)


def test_cuda_numbers_the_gpus_as_nvidia_smi_does():
    env = {}
    use_pci_bus_order(env)
    assert env == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    with pytest.raises(ValueError, match="CUDA_DEVICE_ORDER=FASTEST_FIRST"):
        use_pci_bus_order({"CUDA_DEVICE_ORDER": "FASTEST_FIRST"})


# ---- how many jobs share the GPUs --------------------------------------------------------------------------

def _loads(plan, demands, gpus):
    load = {g: 0.0 for g in gpus}
    for d in demands:
        for g in plan[d.job]:
            load[g] += 1.0 / d.factor
    return load


def test_without_mps_one_gpu_takes_one_job_at_a_time():
    """Review of PR 1 (P2): on one GPU without MPS the complex and the solvent leg were started together and both
    put a Context on it."""
    running = [Demand(job=0, phase="serial", factor=1, devices=[0])]
    assert allocate([0], running + [Demand(job=1, factor=1)], strict=True) is None
    sampling = [Demand(job=0, phase="sampling", factor=1, max_workers=14, work=1.0, devices=[0])]
    assert allocate([0], sampling + [Demand(job=1, factor=1)], strict=True) is None
    assert allocate([0, 1], running + [Demand(job=1, factor=1)], strict=True) == {0: [0], 1: [1]}


@pytest.mark.parametrize("seed", range(200))
def test_no_gpu_is_ever_given_more_than_it_takes(seed):
    """Jobs start only when the strict allocation has room for their first worker; whatever they then do (plain MD,
    sampling with any number of replicas, any remaining work), no GPU carries more than its capacity."""
    rng = random.Random(seed)
    gpus = list(range(rng.randint(1, 4)))
    mps = rng.random() < 0.7
    running: list[Demand] = []
    for job in range(rng.randint(1, 10)):
        factor = rng.choice([2, 4, 8, 16]) if mps else 1
        candidate = Demand(job=job, phase="serial", factor=factor)
        plan = allocate(gpus, running + [candidate], strict=True)
        if plan is None:
            continue                                  # the scheduler lets it wait
        candidate.devices = plan[job]
        running.append(candidate)
        plan = allocate(gpus, running)                # the scheduler rebalances as soon as a job has started
        assert max(_loads(plan, running, gpus).values()) <= 1.0 + 1e-9, (seed, plan)
        for d in running:
            d.devices = plan[d.job]
        for _ in range(3):                            # jobs move between plain MD and sampling, work goes down
            for d in running:
                if rng.random() < 0.5:
                    d.phase = "sampling"
                    d.max_workers = rng.randint(2, 40)
                elif d.phase == "sampling":
                    d.phase, d.max_workers, d.devices = "serial", 1, d.devices[:1]
                d.work = rng.uniform(0.001, 5.0)
            plan = allocate(gpus, running)
            loads = _loads(plan, running, gpus)
            assert max(loads.values()) <= 1.0 + 1e-9, (seed, plan)
            assert all(1 <= len(plan[d.job]) <= d.max_workers for d in running)
            for d in running:
                d.devices = plan[d.job]


# ---- a run that fails stops everything it started --------------------------------------------------------

import subprocess  # noqa: E402
import sys  # noqa: E402

import job_doubles  # noqa: E402

TESTS = Path(__file__).parent


def _jobs(tmp_path, names):
    return [LegJob(ligand=name, leg="solvent", repeat=0, lig_dir=str(tmp_path / name), devices=[], config={},
                   production_ns=1.0) for name in names]


def _gone(pid: int) -> bool:
    """The process has ended: it does not exist, or it is a zombie that nobody has reaped yet (a job whose
    scheduler was killed belongs to pid 1, which in a test container does not reap)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] == "Z"
    except OSError:
        return True


def _wait_for(condition, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize("name", ["polite", "stubborn"])
def test_a_scheduler_that_fails_stops_its_jobs_before_it_returns(tmp_path, monkeypatch, name):
    """Review of PR 1 (P1): an exception in the scheduler right after a job had started (here: the log that
    reports the start fails) released the lock of the output directory while the job went on writing there."""
    monkeypatch.setattr(scheduler, "JOB_FUNCTION", job_doubles.heartbeat_job)
    monkeypatch.setattr(scheduler, "STOP_GRACE_SECONDS", 2.0)
    jobs = _jobs(tmp_path, [name])

    def log(message):
        if message.startswith("started"):
            _wait_for(lambda: (jobs[0].leg_dir / "heartbeat").exists())
            raise OSError("the log cannot be written")

    with pytest.raises(OSError, match="the log cannot be written"):
        scheduler.run_jobs(jobs, [0], log=log, mps=False)
    leg = jobs[0].leg_dir
    assert _gone(int((leg / "pid").read_text()))       # gone, not only asked to go
    beat = (leg / "heartbeat").read_text()
    time.sleep(0.5)
    assert (leg / "heartbeat").read_text() == beat     # nothing writes there any more
    assert "the job process ended without a result" in json.loads((leg / "error.json").read_text())["error"]


def test_jobs_wait_for_room_on_a_single_gpu_without_mps(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "JOB_FUNCTION", job_doubles.heartbeat_job)
    jobs = _jobs(tmp_path, ["short-0.6", "short-0.7"])
    scheduler.run_jobs(jobs, [0], log=lambda m: None, mps=False)
    spans = [(float((j.leg_dir / "began").read_text()), float((j.leg_dir / "ended").read_text())) for j in jobs]
    assert spans[1][0] >= spans[0][1]                  # the second began after the first had ended


def _start_run(out, names):
    """A run of heartbeat jobs in a process of its own, leader of its own process group, as at a terminal."""
    code = (f"import sys; sys.path.insert(0, {str(TESTS)!r}); import job_doubles; "
            f"job_doubles.scheduler_main({str(out)!r}, {list(names)!r})")
    proc = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    legs = [out / name / "repeat_0" / "solvent" for name in names]
    assert _wait_for(lambda: all((leg / "heartbeat").exists() for leg in legs)), "the jobs did not start"
    return proc, legs


def _lock_is_free(out) -> bool:
    import fcntl

    with open(out / ".cufep.lock", "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        fcntl.flock(fh, fcntl.LOCK_UN)
        return True


@pytest.mark.parametrize("how", ["SIGTERM to the run", "SIGHUP to its process group"])
def test_a_run_ended_by_a_signal_stops_its_jobs_before_its_lock_goes(tmp_path, how):
    """Review of PR 1 (P1): the jobs run in process groups of their own; a SIGTERM (kill, GNU timeout) or a hangup
    ended the run without stopping them, its lock went, and the jobs went on writing (heartbeat 11 -> 17 lines in
    3 s)."""

    out = tmp_path / "out"
    proc, legs = _start_run(out, ["slow-1", "polite"])
    if how.startswith("SIGTERM"):
        proc.send_signal(signal.SIGTERM)
    else:
        os.killpg(proc.pid, signal.SIGHUP)
    assert proc.wait(timeout=60) != 0
    for leg in legs:
        assert _gone(int((leg / "pid").read_text()))
        beat = (leg / "heartbeat").read_text()
        time.sleep(0.3)
        assert (leg / "heartbeat").read_text() == beat
    assert (legs[0] / "stopped").exists()               # the slow one was let finish its stop
    lines = (out / "scheduler.log").read_text().splitlines()
    assert lines[-1] == "released" and sum(" stopped (exit code" in line for line in lines) == 2
    assert _lock_is_free(out)


@pytest.mark.skipif(sys.platform != "linux", reason="the signal at the end of the parent (PR_SET_PDEATHSIG) is Linux's")
def test_a_run_that_is_killed_leaves_no_job_behind_and_no_second_run_meets_one(tmp_path):
    """SIGKILL cannot be caught: the jobs get a signal from the kernel when their scheduler ends and stop in
    order, and while one still stops, a new run in the same directory is refused."""

    from cufep.run import lock_output_directory
    from cufep.workflow import WorkflowError

    out = tmp_path / "out"
    proc, legs = _start_run(out, ["slow-2"])
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=30)
    assert _wait_for(lambda: (legs[0] / "heartbeat").read_text().startswith("stopping"), timeout=10)
    with pytest.raises(WorkflowError, match=r"a job of an earlier run is still running in slow-2/repeat_0/solvent"):
        lock_output_directory(out)
    assert _wait_for(lambda: _gone(int((legs[0] / "pid").read_text())), timeout=30)
    assert (legs[0] / "stopped").exists()
    lock_output_directory(out).close()                  # once it has ended, the directory is free


def test_a_second_interrupt_does_not_release_the_lock_while_jobs_stop(tmp_path):
    """Review of PR 1 (P2): a second Ctrl-C while the jobs stopped left stop_jobs, and the lock went while a job
    was still stopping."""

    out = tmp_path / "out"
    proc, legs = _start_run(out, ["slow-3"])
    proc.send_signal(signal.SIGINT)
    time.sleep(0.5)
    proc.send_signal(signal.SIGINT)
    time.sleep(1.0)
    assert proc.poll() is None and not _lock_is_free(out)     # still stopping its job, still holding the lock
    proc.wait(timeout=60)
    assert (legs[0] / "stopped").exists() and _gone(int((legs[0] / "pid").read_text()))
    assert _lock_is_free(out)


def test_the_clients_of_cufeps_mps_daemon_do_not_carry_the_device_mask(monkeypatch):
    """Found by the check chain of plan revision 13 (B1): with CUDA_VISIBLE_DEVICES=2 the run chose CUDA device 0,
    GPU 2, and every job failed with CUDA_ERROR_NO_DEVICE: the clients of an MPS server number its GPUs from 0 and
    look for a client's mask among them. While cuFEP's daemon runs, its processes drop the mask; the monitor keeps
    the map from CUDA device numbers to nvidia-smi's that it took before."""
    import subprocess as sp

    from cufep import mps

    rows = [(0, "GPU-a", 0.0), (1, "GPU-b", 0.0), (2, "GPU-c", 77.0), (3, "GPU-d", 0.0)]
    monkeypatch.setattr(scheduler, "query_gpus", lambda: rows)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(mps.shutil, "which", lambda name: "/usr/bin/" + name)
    seen = []
    monkeypatch.setattr(
        mps.subprocess,
        "run",
        lambda args, **kw: (
            seen.append(kw.get("env", {}).get("CUDA_VISIBLE_DEVICES")) or sp.CompletedProcess(args, 0, "", "")
        ),
    )
    monitor = scheduler.GpuMonitor([0], interval=0.01)
    daemon = mps.start(lambda m: None)
    try:
        assert seen == ["2"]                                 # the server sees GPU 2 alone
        assert "CUDA_VISIBLE_DEVICES" not in os.environ      # its clients find it as their device 0
        monitor.start()
        time.sleep(0.2)
        monitor.stop()
        assert monitor.samples and all(row == {0: 77.0} for _, row in monitor.samples)
    finally:
        daemon.stop()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


# ---- review of PR 1 on 64751ca: the GPUs and processes of the precalculation, and of the CPU ---------------

def test_the_precalculation_keeps_the_gpus_it_runs_on():
    """P2: the precalculation keeps its workers where they started, but the scheduler moved them on paper when a
    short job on GPU 1 ended, and started a new job on GPU 0, where the precalculation still ran."""
    precalculation = Demand(job=0, phase="pinned", factor=1, max_workers=1, devices=[0])
    plan = allocate([0, 1], [precalculation])
    assert plan[0] == [0]
    plan = allocate([0, 1], [precalculation, Demand(job=1, factor=1)], strict=True)
    assert plan == {0: [0], 1: [1]}
    assert allocate([0, 1], [precalculation, Demand(job=1, factor=1, devices=[1]),
                             Demand(job=2, factor=1)], strict=True) is None      # no room: the third waits


def _pool_of(seconds, n_workers=1):
    import multiprocessing as mp

    from cufep.sampling.hrex import WindowPool, _Process

    ctx = mp.get_context("spawn")
    workers = []
    for k in range(n_workers):
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=job_doubles.window_worker, args=(child, seconds))
        proc.start()
        child.close()
        made = _Process.__new__(_Process)
        made.parent, made.proc, made.device, made.worker_id, made.start = parent, proc, 0, k, 0
        workers.append(made)
    pool = WindowPool.__new__(WindowPool)
    pool.workers, pool.log = workers, (lambda message: None)
    return pool


def test_the_precalculation_limit_is_one_of_a_window_not_of_the_batch():
    """P2: one deadline for the whole batch stopped few workers that ran many windows, each in good time."""
    from cufep.sampling.hrex import WorkerError

    pool = _pool_of(0.3)
    try:
        results = pool.run([{"window": k} for k in range(5)], window_timeout=1.0)     # 1.5 s in all, 0.3 s each
        assert [r["window"] for r in results] == list(range(5))
    finally:
        for w in pool.workers:
            w.stop()
    slow = _pool_of(3.0)
    try:
        with pytest.raises(WorkerError, match="did not finish within 1 s of its start"):
            slow.run([{"window": 0}], window_timeout=1.0)
    finally:
        for w in slow.workers:
            w.proc.kill()


def test_arrays_whose_owner_ended_are_removed(tmp_path, monkeypatch):
    """P2: a killed run left its replica arrays (288 MB for 30 replicas of 100,000 atoms) in /dev/shm."""
    from cufep.sampling.worker import ARRAYS_PREFIX, OWNER_LOCK, SharedArrays, remove_orphaned_arrays

    monkeypatch.setenv("CUFEP_SHM_DIR", str(tmp_path))
    living = SharedArrays(3, 10, 3)
    orphan = tmp_path / f"{ARRAYS_PREFIX}dead"
    orphan.mkdir()
    (orphan / OWNER_LOCK).write_text("")               # nobody holds it: its owner has ended
    (orphan / "pos.f8").write_bytes(b"\0" * 1000)
    earlier = tmp_path / f"{ARRAYS_PREFIX}of_an_earlier_version"
    earlier.mkdir()
    assert remove_orphaned_arrays(str(tmp_path)) == [str(orphan)]
    assert Path(living.directory).exists() and earlier.exists() and not orphan.exists()
    second = SharedArrays(3, 10, 3)                     # a new owner sweeps too, and leaves the living one
    assert Path(living.directory).exists()
    living.close()
    second.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == [earlier.name]


def test_arrays_are_not_removed_while_their_owner_makes_them(tmp_path, monkeypatch):
    """Review of 037bdf1 (P2): between the creation of owner.lock and the lock, another job's removal of orphaned
    arrays found the lock free and removed the directory being made. With real files and flock, the removal runs just
    before the lock is taken: nothing is removed while the owner holds the lock, its directory once it is released."""
    import os

    from cufep.sampling import worker

    swept = []

    def sweep():
        swept.extend(worker.remove_orphaned_arrays(str(tmp_path)))

    directory, fd = worker.owned_directory(str(tmp_path), _before_lock=sweep)
    assert swept == [] and Path(directory).is_dir() and Path(directory).name.startswith(worker.ARRAYS_PREFIX)
    assert worker.remove_orphaned_arrays(str(tmp_path)) == []            # owned: 0 removed
    os.close(fd)
    assert worker.remove_orphaned_arrays(str(tmp_path)) == [directory]   # released: 1 removed
    # the order before the fix: the lock file under the public name, the lock taken after it; in between it went
    made = tmp_path / f"{worker.ARRAYS_PREFIX}being_made"
    made.mkdir()
    (made / worker.OWNER_LOCK).write_text("")
    assert worker.remove_orphaned_arrays(str(tmp_path)) == [str(made)]
    # and the arrays themselves, made with the removal just before their lock
    original = worker.owned_directory
    monkeypatch.setattr(worker, "owned_directory", lambda base: original(base, _before_lock=sweep))
    monkeypatch.setenv("CUFEP_SHM_DIR", str(tmp_path))
    arrays = worker.SharedArrays(2, 5, 2)
    arrays.pos[:] = 1.0
    assert swept == [] and Path(arrays.directory).is_dir() and float(arrays.pos.sum()) == 2 * 2 * 5 * 3
    assert worker.remove_orphaned_arrays(str(tmp_path)) == []
    arrays.close()
    assert list(tmp_path.iterdir()) == []                                 # no staging directory is left either


def test_a_cpu_context_has_the_threads_of_the_configuration(monkeypatch):
    """P2: the inline workers of CPU runs took OPENMM_CPU_THREADS (3) instead of resources.cpu_threads_per_worker
    (1). OpenCL got no precision at all."""
    import openmm as mm

    from cufep.backends import get_backend

    monkeypatch.setenv("OPENMM_CPU_THREADS", "3")
    backend = get_backend("openmm", platform="CPU", cpu_threads=1)
    system = mm.System()
    system.addParticle(1.0)
    context = backend.make_context(system, mm.VerletIntegrator(0.001))
    assert context.getPlatform().getPropertyValue(context, "Threads") == "1"
    assert get_backend("openmm", platform="OpenCL", precision="double").platform_properties(0) == {
        "Precision": "double", "DeviceIndex": "0"}
