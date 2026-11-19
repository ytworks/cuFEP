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
    assert visible_gpus(FOUR, {"CUDA_VISIBLE_DEVICES": "-1"}) == [] and visible_gpus(FOUR, {"CUDA_VISIBLE_DEVICES": ""}) == []
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

def child_job(job, conn):
    """Stands in for a job process: writes its pid and a heartbeat into the leg directory until it is stopped.
    A job named 'stubborn' ignores SIGTERM; one named 'short-<s>' ends after s seconds with a result."""
    os.setpgrp()
    leg = Path(job.leg_dir)
    leg.mkdir(parents=True, exist_ok=True)
    if job.ligand == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(SystemExit(143)))
    (leg / "pid").write_text(str(os.getpid()))
    t0 = time.time()
    (leg / "began").write_text(repr(t0))
    while True:
        (leg / "heartbeat").write_text(repr(time.time()))
        if job.ligand.startswith("short-") and time.time() - t0 > float(job.ligand.split("-")[1]):
            (leg / "ended").write_text(repr(time.time()))
            (leg / "result.json").write_text(json.dumps({"ligand": job.ligand, "leg": job.leg, "repeat": 0,
                                                         "analysis": {}}))
            (leg / ".started").unlink(missing_ok=True)
            return
        time.sleep(0.05)


def _jobs(tmp_path, names):
    return [LegJob(ligand=name, leg="solvent", repeat=0, lig_dir=str(tmp_path / name), devices=[], config={},
                   production_ns=1.0) for name in names]


@pytest.mark.parametrize("name", ["polite", "stubborn"])
def test_a_scheduler_that_fails_stops_its_jobs_before_it_returns(tmp_path, monkeypatch, name):
    """Review of PR 1 (P1): an exception in the scheduler right after a job had started (here: the log that
    reports the start fails) released the lock of the output directory while the job went on writing there."""
    monkeypatch.setattr(scheduler, "_entry", child_job)
    monkeypatch.setattr(scheduler, "STOP_GRACE_SECONDS", 2.0)
    jobs = _jobs(tmp_path, [name])

    def log(message):
        if message.startswith("started"):
            time.sleep(1.0)                            # the job is running
            raise OSError("the log cannot be written")

    with pytest.raises(OSError, match="the log cannot be written"):
        scheduler.run_jobs(jobs, [0], log=log, mps=False)
    leg = jobs[0].leg_dir
    pid = int((leg / "pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)                                # gone, not only asked to go
    beat = (leg / "heartbeat").read_text()
    time.sleep(0.5)
    assert (leg / "heartbeat").read_text() == beat     # nothing writes there any more
    assert "the job process ended without a result" in json.loads((leg / "error.json").read_text())["error"]


def test_jobs_wait_for_room_on_a_single_gpu_without_mps(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "_entry", child_job)
    jobs = _jobs(tmp_path, ["short-0.6", "short-0.7"])
    scheduler.run_jobs(jobs, [0], log=lambda m: None, mps=False)
    spans = [(float((j.leg_dir / "began").read_text()), float((j.leg_dir / "ended").read_text())) for j in jobs]
    assert spans[1][0] >= spans[0][1]                  # the second began after the first had ended
