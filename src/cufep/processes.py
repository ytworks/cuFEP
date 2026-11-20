"""The processes of a run end with the process that started them, and a new run cannot start while
a job of an earlier one is still writing.

A run consists of the scheduler (the `cufep` process, which holds the lock of the output
directory), one process per job (leg), each in a process group of its own, and the worker
processes of a job. If the scheduler ends without stopping its jobs (SIGKILL, a crash), its lock
is released while the jobs may still write into the output directory (review of PR 1). Two things
close that:

* every job and every worker asks the kernel for a signal when the process that started it ends
  (Linux ``PR_SET_PDEATHSIG``): a job then stops in order (its SIGTERM handler), a worker ends;
* every job holds a lock of its own leg (``.job.lock`` in the leg directory) for as long as it
  lives; a new run takes the lock of the output directory and then refuses while any leg lock is
  still held (:func:`legs_still_running`).
"""
from __future__ import annotations

import os
import signal
from pathlib import Path

#: the lock a job process holds on its leg directory for as long as it runs
JOB_LOCK = ".job.lock"


def die_with_parent(parent_pid: int, sig: int = signal.SIGTERM) -> None:
    """Ask the kernel to send ``sig`` to this process when the process that started it ends; if it
    has already ended (this process has another parent now), send it at once. Linux only; elsewhere
    nothing happens. ``parent_pid`` is the pid of the starting process, passed by it: a process
    whose parent is pid 1 is not orphaned when the parent is a container's first process."""
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        if libc.prctl(PR_SET_PDEATHSIG, int(sig), 0, 0, 0) != 0:
            return
    except Exception:  # noqa: BLE001  not Linux
        return
    if os.getppid() != int(parent_pid):
        os.kill(os.getpid(), sig)


def hold_job_lock(leg_dir: Path | str):
    """Take the lock of a leg directory for the life of this job process and return the handle
    (keep it). Released by the kernel when the process ends, however it ends."""
    import fcntl

    fd = os.open(Path(leg_dir) / JOB_LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def legs_still_running(out_dir: Path | str) -> list[str]:
    """Leg directories under ``out_dir`` whose job lock is held by a living process: a job of an
    earlier run that has not ended yet."""
    import fcntl

    held = []
    for lock in sorted(Path(out_dir).glob(f"*/repeat_*/*/{JOB_LOCK}")):
        try:
            fd = os.open(lock, os.O_RDWR)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            held.append(str(lock.parent.relative_to(out_dir)))
        finally:
            os.close(fd)
    return held
