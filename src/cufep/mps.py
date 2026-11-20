"""A private NVIDIA MPS daemon, so that several worker processes can share a GPU concurrently.

Without MPS, processes on the same GPU take turns. Small systems leave most of a modern GPU
idle, so running several replicas of a leg on one GPU at the same time raises the total
throughput (SOMD2 calls this the oversubscription factor).

cuFEP starts its own daemon with its own pipe directory and never attaches to a daemon that
was started by somebody else. If the daemon cannot be started, cuFEP runs one worker per
GPU, as without MPS.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

CONTROL = "nvidia-cuda-mps-control"


class MpsDaemon:
    """Handle of a running daemon; ``stop()`` shuts it down."""

    def __init__(self, directory: str, previous: dict[str, str | None]) -> None:
        self.directory = directory
        self._previous = previous
        self.running = True

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        try:
            subprocess.run([CONTROL], input="quit\n", text=True, capture_output=True, timeout=60, env=os.environ.copy())
        except Exception:  # noqa: BLE001 - nothing more can be done at shutdown
            pass
        for key, value in self._previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self.directory, ignore_errors=True)


def start(log=print) -> MpsDaemon | None:
    """Start a daemon for this process tree and point the environment at it.

    Must be called before any process of the run creates a CUDA context. Returns ``None``
    (and logs why) if MPS is not available. The daemon serves the GPUs this process can see;
    device indices stay the same.
    """
    if shutil.which(CONTROL) is None:
        log("NVIDIA MPS is not available (nvidia-cuda-mps-control not found): one worker per GPU")
        return None
    directory = tempfile.mkdtemp(prefix="cufep-mps-")
    keys = ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY")
    previous = {k: os.environ.get(k) for k in keys}
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = os.path.join(directory, "pipe")
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = os.path.join(directory, "log")
    for k in keys:
        os.makedirs(os.environ[k], exist_ok=True)
    env = os.environ.copy()
    try:
        done = subprocess.run([CONTROL, "-d"], capture_output=True, text=True, timeout=60, env=env)
        ok = done.returncode == 0
        why = (done.stderr or done.stdout).strip()
    except Exception as exc:  # noqa: BLE001
        ok, why = False, str(exc)
    if not ok:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(directory, ignore_errors=True)
        log(f"NVIDIA MPS could not be started ({why[:200]}): one worker per GPU")
        return None
    # The server sees the GPUs of CUDA_VISIBLE_DEVICES; its clients number those GPUs 0, 1, ... in
    # that order, and a client that keeps the mask looks for its GPUs among the server's and finds
    # none (measured: server and client with CUDA_VISIBLE_DEVICES=5, CUDA_ERROR_NO_DEVICE; client
    # without the mask, the GPU 5). The processes of the run therefore drop the mask while the daemon
    # runs; their device numbers stay the CUDA device numbers of the run.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        previous["CUDA_VISIBLE_DEVICES"] = os.environ.pop("CUDA_VISIBLE_DEVICES")
    return MpsDaemon(directory, previous)


def thread_percentage(factor: int) -> int | None:
    """CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for a worker when ``factor`` workers share a GPU.

    NVIDIA recommends 200 / (number of processes) for OpenMM; the value is kept within 10-100.
    """
    if factor <= 1:
        return None
    return int(max(10, min(100, round(200.0 / factor))))
