"""Hamiltonian replica exchange controller.

One controller per leg. It owns the shared replica state, starts one worker process per
entry of its device list (a GPU listed n times gets n workers, which run concurrently under
NVIDIA MPS), and treats every iteration as a transaction: workers read buffer ``cur`` and write
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
    #: time the slowest worker needed, and the sum over workers (seconds)
    slowest_worker: float = 0.0
    all_workers: float = 0.0


def mix_replicas(u: np.ndarray, state_of_replica: np.ndarray, rng: np.random.Generator,
                 n_attempts: int | None = None) -> tuple[np.ndarray, int, int]:
    """Gibbs-style mixing: many Metropolis swaps between random replica pairs.

    ``u[r, k]`` is the reduced potential of replica ``r``'s configuration in state ``k``.
    Returns the new assignment and (accepted, attempted) counts.
    """
    k = len(state_of_replica)
    if k < 2:
        return np.array(state_of_replica, dtype=int), 0, 0
    n = min(k ** 3, 50000) if n_attempts is None else int(n_attempts)
    pairs = rng.integers(0, k, size=(n, 2))
    log_r = np.log(rng.random(n))
    # Plain Python lists: this loop runs every iteration in the controller, and indexing NumPy
    # scalars made it the largest serial cost of an iteration (40 ms for 28 replicas).
    rows = np.asarray(u, dtype=float).tolist()
    s = [int(x) for x in state_of_replica]
    accepted = attempted = 0
    for i, j, lr in zip(pairs[:, 0].tolist(), pairs[:, 1].tolist(), log_r.tolist()):
        if i == j:
            continue
        attempted += 1
        si, sj = s[i], s[j]
        ui, uj = rows[i], rows[j]
        log_p = -(ui[sj] + uj[si] - ui[si] - uj[sj])
        if log_p >= 0.0 or lr < log_p:
            s[i], s[j] = sj, si
            accepted += 1
    return np.array(s, dtype=int), accepted, attempted


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
        try:
            self.parent.send(("run", *args))
            self._dead = False
        except (BrokenPipeError, OSError):
            self._dead = True                 # reported by result(), which the controller handles

    def result(self, timeout: float) -> dict:
        if getattr(self, "_dead", False):
            raise SamplerError(f"worker {self.worker_id} (device {self.device}) is not running")
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


class WindowPool:
    """Worker processes that each run single lambda windows (the adaptive precalculation).

    One worker per entry of ``devices``; a GPU listed n times gets n workers, which run at the
    same time under NVIDIA MPS. ``run`` hands the tasks to the workers as they become free.
    """

    def __init__(self, spec: LegSpec, devices: list, log=print) -> None:
        ctx = mp.get_context("spawn")
        self.spec, self.log = spec, log
        self.workers = [_Process(ctx, spec, d, None, w) for w, d in enumerate(devices)]
        for w in self.workers:
            w.wait_ready(600.0)

    def run(self, tasks: list[dict], timeout: float = 3600.0) -> list[dict]:
        results: list = [None] * len(tasks)
        pending = list(range(len(tasks)))
        busy: dict[int, int] = {}                          # worker position -> task index
        deadline = time.time() + timeout
        while pending or busy:
            for pos, worker in enumerate(self.workers):
                if pos not in busy and pending:
                    i = pending.pop(0)
                    worker.parent.send(("window", tasks[i]))
                    busy[pos] = i
            for pos in list(busy):
                worker = self.workers[pos]
                try:
                    if not worker.parent.poll(0.02):
                        continue
                    msg = worker.parent.recv()
                except (EOFError, OSError) as exc:
                    raise SamplerError(f"a worker of the lambda precalculation died (device {worker.device})") from exc
                if msg[0] != "done":
                    raise SamplerError(f"lambda precalculation on device {worker.device}: {msg[1]}")
                results[busy.pop(pos)] = msg[1]
            if time.time() > deadline:
                raise SamplerError("the lambda precalculation did not finish in time")
        return results

    def close(self) -> None:
        for w in self.workers:
            try:
                w.stop()
            except Exception:  # noqa: BLE001
                pass
        self.workers = []

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


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
        try:
            # before anything is written: samples are continued only under the conditions they
            # were made under
            self.conditions_of_storage = self.storage.check_conditions(spec.conditions())
        except Exception:
            self.storage.close()
            raise
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
        self._next_worker_id = 0
        self._iter_seconds: float | None = None
        #: seconds the workers were busy, summed, and (iteration, devices) whenever the workers changed
        self.busy_seconds = 0.0
        self.worker_seconds_available = 0.0
        self.device_history: list[tuple[int, list]] = []
        self.accepted = self.attempted = 0
        self.restarts = 0

    # -- life cycle -------------------------------------------------------------------
    def start(self) -> None:
        if self.workers:
            return
        n = min(len(self.devices), self.k)
        self.devices = self.devices[:n]
        if self.inline:
            self.workers = [_Inline(self.spec, self.devices[w], self.arrays, w) for w in range(n)]
        else:
            ctx = mp.get_context("spawn")
            self.workers = [_Process(ctx, self.spec, self.devices[w], self.arrays.names, w) for w in range(n)]
        self._next_worker_id = n
        for w in self.workers:
            described = w.wait_ready(600.0)
        self.log(f"[{self.spec.name}] {n} worker(s) on {self._device_summary()}: {described.split(' (index')[0]}")
        self.device_history.append((self.iteration, list(self.devices)))
        if not self.arrays.location.startswith("/dev/shm"):
            self.log(f"[{self.spec.name}] note: replica arrays are on disk ({self.arrays.location}) because "
                     "/dev/shm is too small; start the container with --shm-size=16g or --ipc=host to avoid this")

    def _device_summary(self) -> str:
        devs = [d for d in self.devices if d is not None]
        if not devs:
            return "the CPU"
        return "GPU " + ", ".join(f"{g}x{devs.count(g)}" if devs.count(g) > 1 else str(g) for g in sorted(set(devs)))

    def set_devices(self, devices: list) -> bool:
        """Change the workers between two iterations: one worker per entry of ``devices``.

        Workers that are still wanted keep running; the others are stopped and new ones are
        started. Replica coordinates are in shared memory, so nothing has to be moved.
        Returns whether anything changed.
        """
        target = list(devices)[: self.k]
        if not target or self.inline:
            return False
        if not self.workers:
            changed = target != self.devices
            self.devices = target
            return changed
        if sorted(target) == sorted(self.devices):
            return False
        wanted: dict = {}
        for d in target:
            wanted[d] = wanted.get(d, 0) + 1
        keep, stop = [], []
        for w in self.workers:
            if wanted.get(w.device, 0) > 0:
                wanted[w.device] -= 1
                keep.append(w)
            else:
                stop.append(w)
        for w in stop:
            try:
                w.stop()
            except Exception:  # noqa: BLE001
                pass
        ctx = mp.get_context("spawn")
        new = []
        for d, n in wanted.items():
            for _ in range(n):
                new.append(_Process(ctx, self.spec, d, self.arrays.names, self._next_worker_id))
                self._next_worker_id += 1
        for w in new:
            w.wait_ready(600.0)
        self.workers = keep + new
        self.devices = [w.device for w in self.workers]
        self.log(f"[{self.spec.name}] iteration {self.iteration}: now {len(self.workers)} worker(s) on "
                 f"{self._device_summary()}")
        self.device_history.append((self.iteration, list(self.devices)))
        return True

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
        """Start the run: set replica coordinates, one configuration for all or one per replica.

        The storage is emptied: samples of a run that was interrupted before its first
        checkpoint would otherwise stay in front of the samples of this one. A checkpoint of
        the start is written, so that such an interruption resumes from here.
        """
        self.iteration = 0
        self.state_of_replica = np.arange(self.k)
        self.storage.truncate(0)
        self.storage.set_meta(n_equilibration_iterations=0)
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
        self.checkpoint()

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

    def _assignments(self, w: int) -> list[tuple[int, int]]:
        n = len(self.workers)
        return [(r, int(self.state_of_replica[r])) for r in range(self.k) if r % n == w]

    def _restart_worker(self, w: int) -> None:
        old = self.workers[w]
        try:
            old.stop()
        except Exception:  # noqa: BLE001
            pass
        ctx = mp.get_context("spawn")
        self.workers[w] = _Process(ctx, self.spec, old.device, self.arrays.names, old.worker_id)
        self.log(f"[{self.spec.name}] worker {old.worker_id} restarted: {self.workers[w].wait_ready(600.0)}")

    def _advance(self, record: bool, steps: int | None) -> list[dict]:
        """One iteration on every worker.

        Workers read buffer ``cur`` and write the other one, so a worker that dies or hangs
        leaves the iteration's input intact. Its replicas are then run again once on a fresh
        worker process. A second failure from the same configurations is not a hardware
        hiccup, so it is raised (the leg is then repeated on the fallback backend).
        """
        for w, worker in enumerate(self.workers):
            worker.submit(self.cur, self._assignments(w), self.iteration, record, steps)
        timeout = self._timeout()
        results: list[dict | None] = [None] * len(self.workers)
        failed: list[tuple[int, SamplerError]] = []
        for w, worker in enumerate(self.workers):
            try:
                results[w] = worker.result(timeout)
            except SamplerError as exc:
                failed.append((w, exc))
        for w, exc in failed:
            if self.inline:
                raise exc
            self.log(f"[{self.spec.name}] iteration {self.iteration}: {str(exc).splitlines()[0][:200]}; "
                     f"restarting worker {w} and repeating its replicas")
            self.restarts += 1
            self._restart_worker(w)
            self.workers[w].submit(self.cur, self._assignments(w), self.iteration, record, steps)
            try:
                results[w] = self.workers[w].result(timeout)
            except SamplerError as again:
                raise SamplerError(f"worker {w} failed twice from the same configurations at iteration "
                                   f"{self.iteration}: {again}") from again
        return results

    def run(self, n_iterations: int, record_stride: int = 4, checkpoint_stride: int = 100,
            store: bool = True, steps: int | None = None, progress_every: int = 0,
            on_iteration=None) -> list[IterationStats]:
        """Run ``n_iterations`` more iterations.

        ``on_iteration(sampler, stats)`` is called after every iteration; it may call
        :meth:`set_devices`.
        """
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
            worker_seconds = [float(res.get("seconds", 0.0)) for res in results if res]
            stats.append(IterationStats(self.iteration, dt, frac, max(worker_seconds, default=0.0),
                                        float(sum(worker_seconds))))
            self.busy_seconds += float(sum(worker_seconds))
            self.worker_seconds_available += dt * len(self.workers)
            if on_iteration is not None:
                on_iteration(self, stats[-1])
            if progress_every and self.iteration % progress_every == 0:
                self.log(f"[{self.spec.name}] iteration {self.iteration}: {self._iter_seconds:.2f} s/iteration, "
                         f"{100 * frac:.0f}% of replicas changed state")
        if store:
            self.checkpoint()
        return stats
