"""Hamiltonian replica exchange controller.

One controller per leg. It owns the shared replica state, starts one worker process per
GPU, and treats every iteration as a transaction: workers read buffer ``cur`` and write
buffer ``1 - cur``; only when all of them report does the controller flip the buffer,
mix the states, append the energies and (periodically) write a checkpoint.

Exchanges permute thermodynamic-state indices among replicas, so no coordinates move
between GPUs. The random numbers for mixing depend only on (seed, iteration), never on
how replicas are spread over workers.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass

import numpy as np

from cufep.sampling.storage import LegStorage
from cufep.sampling.worker import LegSpec, ReplicaWorker, SharedArrays, worker_main


class SamplerError(RuntimeError):
    """A worker failed or hung."""


@dataclass
class IterationStats:
    iteration: int
    seconds: float
    swap_fraction: float


def mix_replicas(u: np.ndarray, state_of_replica: np.ndarray, rng: np.random.Generator,
                 n_attempts: int | None = None) -> tuple[np.ndarray, int, int]:
    """Gibbs-style mixing: many Metropolis swaps between random replica pairs.

    ``u[r, k]`` is the reduced potential of replica ``r``'s configuration in state ``k``.
    Returns the new assignment and (accepted, attempted) counts.
    """
    k = len(state_of_replica)
    s = np.array(state_of_replica, dtype=int)
    if k < 2:
        return s, 0, 0
    n = min(k ** 3, 50000) if n_attempts is None else int(n_attempts)
    pairs = rng.integers(0, k, size=(n, 2))
    log_r = np.log(rng.random(n))
    accepted = attempted = 0
    for (i, j), lr in zip(pairs, log_r):
        if i == j:
            continue
        attempted += 1
        si, sj = s[i], s[j]
        log_p = -(u[i, sj] + u[j, si] - u[i, si] - u[j, sj])
        if log_p >= 0.0 or lr < log_p:
            s[i], s[j] = sj, si
            accepted += 1
    return s, accepted, attempted


class _Process:
    """A worker in its own process."""

    def __init__(self, ctx, spec: LegSpec, device, shm_names: dict, worker_id: int) -> None:
        self.parent, child = ctx.Pipe()
        self.proc = ctx.Process(target=worker_main, args=(child, spec, device, shm_names, worker_id), daemon=True)
        self.proc.start()
        child.close()
        self.device, self.worker_id = device, worker_id

    def wait_ready(self, timeout: float) -> str:
        if not self.parent.poll(timeout):
            raise SamplerError(f"worker {self.worker_id} (device {self.device}) did not start in {timeout:.0f} s")
        msg = self.parent.recv()
        if msg[0] != "ready":
            raise SamplerError(f"worker {self.worker_id} failed to start: {msg[1]}")
        return msg[1]

    def submit(self, *args) -> None:
        self.parent.send(("run", *args))

    def result(self, timeout: float) -> dict:
        if not self.parent.poll(timeout):
            raise SamplerError(f"worker {self.worker_id} (device {self.device}) gave no answer in {timeout:.0f} s")
        try:
            msg = self.parent.recv()
        except EOFError as exc:
            raise SamplerError(f"worker {self.worker_id} (device {self.device}) died") from exc
        if msg[0] != "done":
            raise SamplerError(f"worker {self.worker_id} (device {self.device}): {msg[1]}")
        return msg[1]

    def stop(self) -> None:
        try:
            self.parent.send(("stop",))
        except Exception:  # noqa: BLE001
            pass
        self.proc.join(15)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(5)
        self.parent.close()


class _Inline:
    """A worker in the controller's own process (tests, CPU platforms)."""

    def __init__(self, spec: LegSpec, device, arrays: SharedArrays, worker_id: int) -> None:
        self.worker = ReplicaWorker(spec, device, arrays, worker_id)
        self.device, self.worker_id = device, worker_id
        self._pending = None

    def wait_ready(self, timeout: float) -> str:
        return self.worker.describe()

    def submit(self, *args) -> None:
        self._pending = args

    def result(self, timeout: float) -> dict:
        args, self._pending = self._pending, None
        return self.worker.run(*args)

    def stop(self) -> None:
        self.worker.close()


class ReplicaExchangeSampler:
    """Replica exchange over the states of ``spec`` (one replica per state)."""

    def __init__(self, spec: LegSpec, storage_dir, devices: list[int | None] | None = None,
                 inline: bool = False, exchange: bool = True, log=print) -> None:
        self.spec = spec
        self.k = spec.n_states
        self.exchange = exchange
        self.log = log
        self.devices = list(devices) if devices else [None]
        self.inline = inline
        self.cur = 0
        self.state_of_replica = np.arange(self.k)
        self.iteration = 0
        self.storage = LegStorage(storage_dir, self.k, self.k, len(spec.frame_atoms))
        self.arrays = SharedArrays(self.k, spec.n_atoms, self.k, fallback_dir=str(self.storage.dir))
        self.storage.set_meta(states=spec.states, temperature_K=spec.temperature_K,
                              backend=spec.backend, leg=spec.name,
                              iteration_ps=spec.steps_per_iteration * spec.timestep_fs * 1e-3,
                              frame_ligand=list(spec.frame_ligand), frame_ca=list(spec.frame_ca))
        if spec.lrc is not None:
            from cufep.alchemy.lrc import TailEnergy
            tail = TailEnergy.from_dict(spec.lrc)
            self.storage.set_meta(lrc_s6=tail._s6, lrc_s12=tail._s12, lrc_cutoff_nm=tail.cutoff_nm)  # noqa: SLF001
        self.workers: list = []
        self._iter_seconds: float | None = None
        self.accepted = self.attempted = 0

    # -- life cycle -------------------------------------------------------------------
    def start(self) -> None:
        if self.workers:
            return
        n = min(len(self.devices), self.k)
        if self.inline:
            self.workers = [_Inline(self.spec, self.devices[w], self.arrays, w) for w in range(n)]
        else:
            ctx = mp.get_context("spawn")
            self.workers = [_Process(ctx, self.spec, self.devices[w], self.arrays.names, w) for w in range(n)]
        for w in self.workers:
            self.log(f"[{self.spec.name}] worker {w.worker_id}: {w.wait_ready(600.0)}")
        if not self.arrays.location.startswith("/dev/shm"):
            self.log(f"[{self.spec.name}] note: replica arrays are on disk ({self.arrays.location}) because "
                     "/dev/shm is too small; start the container with --shm-size=16g or --ipc=host to avoid this")

    def close(self) -> None:
        for w in self.workers:
            try:
                w.stop()
            except Exception:  # noqa: BLE001
                pass
        self.workers = []
        self.storage.close()
        self.arrays.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- state ------------------------------------------------------------------------
    def initialize(self, positions: np.ndarray, boxes: np.ndarray, velocities: np.ndarray | None = None) -> None:
        """Set replica coordinates: one configuration for all, or one per replica."""
        positions, boxes = np.asarray(positions, dtype=float), np.asarray(boxes, dtype=float)
        if positions.ndim == 2:
            positions = np.broadcast_to(positions, (self.k, *positions.shape))
        if boxes.ndim == 2:
            boxes = np.broadcast_to(boxes, (self.k, 3, 3))
        self.arrays.pos[self.cur] = positions
        self.arrays.box[self.cur] = boxes
        if velocities is None:
            self.arrays.vel[self.cur] = np.nan
        else:
            v = np.asarray(velocities, dtype=float)
            self.arrays.vel[self.cur] = np.broadcast_to(v, (self.k, *v.shape)) if v.ndim == 2 else v

    def restore(self) -> bool:
        """Resume from the checkpoint in the storage directory, if there is one."""
        ckpt = self.storage.read_checkpoint()
        if ckpt is None:
            return False
        if ckpt["positions"].shape != self.arrays.pos[0].shape:
            raise SamplerError("the checkpoint does not match this leg (different size or number of states)")
        self.arrays.pos[self.cur] = ckpt["positions"]
        self.arrays.vel[self.cur] = ckpt["velocities"]
        self.arrays.box[self.cur] = ckpt["boxes"]
        self.state_of_replica = ckpt["state_of_replica"].astype(int)
        self.iteration = ckpt["iteration"]
        self.storage.truncate(self.iteration)
        return True

    def checkpoint(self) -> None:
        self.storage.write_checkpoint(self.arrays.pos[self.cur], self.arrays.vel[self.cur], self.arrays.box[self.cur],
                                      self.state_of_replica, self.iteration)

    def current_configurations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Copies of (positions, boxes, state_of_replica)."""
        return self.arrays.pos[self.cur].copy(), self.arrays.box[self.cur].copy(), self.state_of_replica.copy()

    # -- iterations ---------------------------------------------------------------------
    def _timeout(self) -> float:
        return 900.0 if self._iter_seconds is None else max(300.0, 30.0 * self._iter_seconds)

    def _advance(self, record: bool, steps: int | None) -> list[dict]:
        n = len(self.workers)
        for w, worker in enumerate(self.workers):
            assignments = [(r, int(self.state_of_replica[r])) for r in range(self.k) if r % n == w]
            worker.submit(self.cur, assignments, self.iteration, record, steps)
        timeout = self._timeout()
        return [worker.result(timeout) for worker in self.workers]

    def run(self, n_iterations: int, record_stride: int = 4, checkpoint_stride: int = 100,
            store: bool = True, steps: int | None = None, progress_every: int = 0) -> list[IterationStats]:
        """Run ``n_iterations`` more iterations."""
        self.start()
        stats: list[IterationStats] = []
        for _ in range(int(n_iterations)):
            t0 = time.perf_counter()
            record = store and record_stride > 0 and self.iteration % record_stride == 0
            results = self._advance(record, steps)
            # ---- commit ---------------------------------------------------------------
            self.cur = 1 - self.cur
            u = self.arrays.u.copy()
            sampled_states = self.state_of_replica.copy()
            if store:
                self.storage.append(u, sampled_states, self.arrays.vol.copy())
                for res in results:
                    if res.get("recorded"):
                        self.storage.append_record(self.iteration, res["recorded"])
            frac = 0.0
            if self.exchange:
                rng = np.random.default_rng([int(self.spec.seed) & 0x7FFFFFFF, self.iteration, 0x5EED])
                new, acc, att = mix_replicas(u, self.state_of_replica, rng)
                frac = float(np.mean(new != self.state_of_replica))
                self.state_of_replica = new
                self.accepted += acc
                self.attempted += att
            self.iteration += 1
            if store and checkpoint_stride > 0 and self.iteration % checkpoint_stride == 0:
                self.checkpoint()
            dt = time.perf_counter() - t0
            self._iter_seconds = dt if self._iter_seconds is None else 0.9 * self._iter_seconds + 0.1 * dt
            stats.append(IterationStats(self.iteration, dt, frac))
            if progress_every and self.iteration % progress_every == 0:
                self.log(f"[{self.spec.name}] iteration {self.iteration}: {self._iter_seconds:.2f} s/iteration, "
                         f"{100 * frac:.0f}% of replicas changed state")
        if store:
            self.checkpoint()
        return stats
