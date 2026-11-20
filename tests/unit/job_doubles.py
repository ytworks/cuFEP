"""Stand-ins for the work of a job process and for a scheduler process, for the tests of how a run stops.

They run in processes of their own (spawned jobs, a scheduler started with subprocess), so they live in a
module those processes can import.
"""
import json
import os
import sys
import time
from pathlib import Path


def heartbeat_job(job):
    """Writes its pid and a heartbeat into the leg directory until it is stopped. Named 'short-<s>': ends after s
    seconds with a result. Named 'slow-<s>': needs s seconds to stop after SIGTERM (its workers, its files).
    Named 'stubborn': ignores SIGTERM."""
    import signal

    leg = Path(job.leg_dir)
    leg.mkdir(parents=True, exist_ok=True)
    if job.ligand == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (leg / "pid").write_text(str(os.getpid()))
    t0 = time.time()
    (leg / "began").write_text(repr(t0))
    try:
        while True:
            (leg / "heartbeat").write_text(repr(time.time()))
            if job.ligand.startswith("short-") and time.time() - t0 > float(job.ligand.split("-")[1]):
                (leg / "ended").write_text(repr(time.time()))
                (leg / "result.json").write_text(json.dumps({"ligand": job.ligand, "leg": job.leg, "repeat": 0,
                                                             "analysis": {}}))
                (leg / "started.json").unlink(missing_ok=True)
                return {}
            time.sleep(0.05)
    finally:
        if job.ligand.startswith("slow-"):
            end = time.time() + float(job.ligand.split("-")[1])
            while time.time() < end:
                (leg / "heartbeat").write_text("stopping " + repr(time.time()))
                time.sleep(0.05)
            (leg / "stopped").write_text(repr(time.time()))


def scheduler_main(out_dir: str, names: list[str]) -> None:
    """A run as `cufep run` makes it: the lock of the output directory, then the scheduler with jobs that run
    heartbeat_job."""
    from cufep import scheduler
    from cufep.run import lock_output_directory
    from cufep.workflow import LegJob

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scheduler.JOB_FUNCTION = heartbeat_job
    scheduler.STOP_GRACE_SECONDS = 20.0
    jobs = [LegJob(ligand=name, leg="solvent", repeat=0, lig_dir=str(out / name), devices=[], config={},
                   production_ns=1.0) for name in names]
    lock = lock_output_directory(out)
    log = open(out / "scheduler.log", "a", buffering=1)
    try:
        # one device per job (the jobs use none): without MPS a GPU takes one job at a time
        scheduler.run_jobs(jobs, list(range(len(jobs))), log=lambda m: log.write(m + "\n"), mps=False)
    finally:
        log.write("released\n")
        lock.close()


if __name__ == "__main__":
    scheduler_main(sys.argv[1], sys.argv[2:])


def window_worker(conn, seconds: float) -> None:
    """Stands in for a worker of the lambda precalculation: every window takes ``seconds``."""
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message[0] == "stop":
            return
        time.sleep(seconds)
        conn.send(("done", {"window": message[1].get("window")}))
