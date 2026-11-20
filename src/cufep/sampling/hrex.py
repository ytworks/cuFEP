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
import os
import time
from dataclasses import dataclass

import numpy as np

from cufep.sampling.storage import LegStorage, StorageError
from cufep.sampling.worker import LegSpec, ReplicaWorker, SharedArrays, worker_main


class SamplerError(RuntimeError):
    """The sampler cannot go on: a worker failed (:class:`WorkerError`), or the sampler was used
    in a way it does not allow, or a checkpoint does not belong to the leg. Failures to store
    the samples are :class:`cufep.sampling.storage.StorageError`, which is not a SamplerError."""


class WorkerError(SamplerError):
    """A worker process failed, hung or did not start: the only kind of failure that says
    something about the backend (and so the only one after which a leg may be repeated on
    another backend)."""


@dataclass
class IterationStats:
    iteration: int
    seconds: float
    swap_fraction: float
    #: time the slowest worker needed, and the sum over workers (seconds)
    slowest_worker: float = 0.0
    all_workers: float = 0.0


def mix_replicas(
    u: np.ndarray,
    state_of_replica: np.ndarray,
    rng: np.random.Generator,
    n_attempts: int | None = None,
    neighbour_counts: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    """Gibbs-style mixing: many Metropolis swaps between random replica pairs.

    ``u[r, k]`` is the reduced potential of replica ``r``'s configuration in state ``k``.
    Returns the new assignment and (accepted, attempted) counts. ``neighbour_counts`` (states - 1
    rows of [proposed, accepted]) counts the proposals between the replicas in neighbouring states
    k and k + 1, row k, and how many of them were accepted: the acceptance of exchange between
    neighbouring windows (review of PR 1: what was reported as such was the fraction of iterations
    after which a replica had moved to a neighbouring state).
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
        neighbours = neighbour_counts is not None and abs(si - sj) == 1
        if neighbours:
            neighbour_counts[min(si, sj), 0] += 1
        if log_p >= 0.0 or lr < log_p:
            s[i], s[j] = sj, si
            accepted += 1
            if neighbours:
                neighbour_counts[min(si, sj), 1] += 1
    return np.array(s, dtype=int), accepted, attempted


class _Process:
    """A worker in its own process."""

    def __init__(self, ctx, spec: LegSpec, device, shm_names: dict, worker_id: int, start: int = 0) -> None:
        self.parent, child = ctx.Pipe()
        self.proc = ctx.Process(target=worker_main, args=(child, spec, device, shm_names, worker_id, start,
                                                          os.getpid()), daemon=True)
        self.proc.start()
        child.close()
        self.device, self.worker_id, self.start = device, worker_id, start

    def wait_ready(self, timeout: float) -> str:
        try:
            if not self.parent.poll(timeout):
                raise WorkerError(f"worker {self.worker_id} (device {self.device}) did not start in {timeout:.0f} s")
            msg = self.parent.recv()
        except (EOFError, OSError) as exc:            # the process ended before it said it was ready
            raise WorkerError(f"worker {self.worker_id} (device {self.device}) ended before it was ready") from exc
        if msg[0] != "ready":
            raise WorkerError(f"worker {self.worker_id} failed to start: {msg[1]}")
        return msg[1]

    def submit(self, *args) -> None:
        try:
            self.parent.send(("run", *args))
            self._dead = False
        except (BrokenPipeError, OSError):
            self._dead = True                 # reported by result(), which the controller handles

    def result(self, timeout: float) -> dict:
        if getattr(self, "_dead", False):
            raise WorkerError(f"worker {self.worker_id} (device {self.device}) is not running")
        try:
            if not self.parent.poll(timeout):
                raise WorkerError(f"worker {self.worker_id} (device {self.device}) gave no answer in {timeout:.0f} s")
            msg = self.parent.recv()
        except (EOFError, OSError) as exc:
            raise WorkerError(f"worker {self.worker_id} (device {self.device}) died") from exc
        if msg[0] != "done":
            raise WorkerError(f"worker {self.worker_id} (device {self.device}): {msg[1]}")
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


def start_workers(factories, timeout: float = 600.0) -> tuple[list, list[str]]:
    """Create workers (each callable of ``factories`` makes one) and wait until every one is ready.

    If one cannot be made or does not become ready, every worker that was made is stopped (its
    process joined, or killed) before the error is raised: a failed start leaves no process that
    holds a GPU while the leg goes on, or is repeated on another backend (review of PR 1).
    """
    workers: list = []
    try:
        for make in factories:
            workers.append(make())
        described = [w.wait_ready(timeout) for w in workers]
    except BaseException:
        for w in workers:
            try:
                w.stop()
            except Exception:  # noqa: BLE001  stopping as many as can be stopped
                pass
        raise
    return workers, described


class WindowPool:
    """Worker processes that each run single lambda windows (the adaptive precalculation).

    One worker per entry of ``devices``; a GPU listed n times gets n workers, which run at the
    same time under NVIDIA MPS. ``run`` hands the tasks to the workers as they become free.
    """

    def __init__(self, spec: LegSpec, devices: list, log=print) -> None:
        ctx = mp.get_context("spawn")
        self.spec, self.log = spec, log
        self.workers, _described = start_workers(
            [lambda d=d, w=w: _Process(ctx, spec, d, None, w) for w, d in enumerate(devices)])

    def run(self, tasks: list[dict], window_timeout: float = 3600.0) -> list[dict]:
        """Run every task (one window each) on the workers as they become free.

        A window that runs longer than ``window_timeout`` seconds from its own start, or a worker
        process that has ended, is a WorkerError. The limit is one of a window, not of the batch:
        few workers that run many windows one after another, each in good time, are not stopped
        after an hour in total (review of PR 1).
        """
        results: list = [None] * len(tasks)
        pending = list(range(len(tasks)))
        busy: dict[int, int] = {}                          # worker position -> task index
        started: dict[int, float] = {}                     # worker position -> start of its window
        while pending or busy:
            for pos, worker in enumerate(self.workers):
                if pos not in busy and pending:
                    i = pending.pop(0)
                    try:
                        worker.parent.send(("window", tasks[i]))
                    except (EOFError, OSError) as exc:     # the process has ended
                        raise WorkerError(f"a worker of the lambda precalculation ended (device {worker.device})") \
                            from exc
                    busy[pos], started[pos] = i, time.time()
            for pos in list(busy):
                worker = self.workers[pos]
                try:
                    if not worker.parent.poll(0.02):
                        if not worker.proc.is_alive():
                            raise WorkerError(f"a worker of the lambda precalculation died (device {worker.device})")
                        if time.time() - started[pos] > window_timeout:
                            raise WorkerError(
                                f"a window of the lambda precalculation on device {worker.device} did not "
                                f"finish within {window_timeout:.0f} s of its start"
                            )
                        continue
                    msg = worker.parent.recv()
                except (EOFError, OSError) as exc:
                    raise WorkerError(f"a worker of the lambda precalculation died (device {worker.device})") from exc
                if msg[0] != "done":
                    raise WorkerError(f"lambda precalculation on device {worker.device}: {msg[1]}")
                results[busy.pop(pos)] = msg[1]
                started.pop(pos, None)
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

    def __init__(self, spec: LegSpec, device, arrays: SharedArrays, worker_id: int, start: int = 0) -> None:
        self.worker = ReplicaWorker(spec, device, arrays, worker_id, start)
        self.device, self.worker_id, self.start = device, worker_id, start
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
                 inline: bool = False, exchange: bool | None = None, log=print) -> None:
        # the method of exchange is decided here, once: ``exchange`` (if given) replaces the one of
        # ``spec``, and the same value runs the leg, is stored with its samples and is compared when
        # the leg is continued (review of PR 1: exchange=False ran independent windows whose samples
        # were recorded as replica exchange, from the default of the spec)
        if exchange is not None:
            from dataclasses import replace

            spec = replace(spec, exchange="hrex" if exchange else "none")
        if spec.exchange not in ("hrex", "none"):
            raise ValueError(f"exchange {spec.exchange!r} of the leg: 'hrex' or 'none'")
        self.spec = spec
        self.k = spec.n_states
        self.exchange = spec.exchange == "hrex"
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
        #: the storage holds samples of an earlier version whose conditions are not recorded: they
        #: are not continued, and nothing is written to their file
        self.unrecorded = self.conditions_of_storage == "unrecorded"
        if not self.unrecorded:
            self._write_meta()
        self.workers: list = []
        self._next_worker_id = 0
        #: how often this leg has been started: 0 for a new run, one more with every resume from a
        #: checkpoint (stored in the checkpoint). Part of the seed of every worker, see stream_seed
        self.start_number = 0
        #: the iteration of the last checkpoint that was written (None before the first)
        self.last_checkpoint: int | None = None
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
            factories = [lambda w=w: _Inline(self.spec, self.devices[w], self.arrays, w, self.start_number)
                         for w in range(n)]
        else:
            ctx = mp.get_context("spawn")
            factories = [lambda w=w: _Process(ctx, self.spec, self.devices[w], self.arrays.names, w, self.start_number)
                         for w in range(n)]
        self.workers, ready = start_workers(factories)
        described = ready[-1]
        self._next_worker_id = n
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
        ids = []
        for d, n in wanted.items():
            for _ in range(n):
                ids.append((d, self._next_worker_id))
                self._next_worker_id += 1
        # the stopped ones are gone, whatever happens next: the workers and the devices they name agree
        # before anything can fail (review of PR 1: after a failed start the devices still named the
        # stopped workers, and asking for them again changed nothing)
        self.workers = keep
        self.devices = [w.device for w in keep]
        new, _ready = start_workers(
            [lambda d=d, i=i: _Process(ctx, self.spec, d, self.arrays.names, i, self.start_number) for d, i in ids]
        )
        self.workers = keep + new
        self.devices = [w.device for w in self.workers]
        self.log(f"[{self.spec.name}] iteration {self.iteration}: now {len(self.workers)} worker(s) on "
                 f"{self._device_summary()}")
        self.device_history.append((self.iteration, list(self.devices)))
        return True

    def close(self, failing: bool = False) -> None:
        """Stop the workers, store the samples that are still in memory and free the arrays.

        If those samples cannot be stored (they are the iterations after the last checkpoint),
        that is logged, and raised unless ``failing`` says that the sampler is being closed
        because of another error, which then stays the one that is raised.
        """
        for w in self.workers:
            try:
                w.stop()
            except Exception:  # noqa: BLE001
                pass
        self.workers = []
        try:
            self.storage.close()
        except StorageError as exc:
            self.log(f"[{self.spec.name}] {exc}")
            if not failing:
                raise
        finally:
            self.arrays.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(failing=exc_type is not None)

    def _write_meta(self) -> None:
        spec = self.spec
        self.storage.set_meta(states=spec.states, temperature_K=spec.temperature_K,
                              backend=spec.backend, leg=spec.name,
                              iteration_ps=spec.steps_per_iteration * spec.timestep_fs * 1e-3,
                              frame_ligand=list(spec.frame_ligand), frame_ca=list(spec.frame_ca))
        if spec.lrc is not None:
            from cufep.alchemy.lrc import TailEnergy
            tail = TailEnergy.from_dict(spec.lrc)
            self.storage.set_meta(lrc_s6=tail._s6, lrc_s12=tail._s12, lrc_cutoff_nm=tail.cutoff_nm)  # noqa: SLF001

    # -- state ------------------------------------------------------------------------
    def initialize(self, positions: np.ndarray, boxes: np.ndarray, velocities: np.ndarray | None = None) -> None:
        """Start the run: set replica coordinates, one configuration for all or one per replica.

        The storage is emptied: samples of a run that was interrupted before its first
        checkpoint would otherwise stay in front of the samples of this one. A checkpoint of
        the start is written, so that such an interruption resumes from here.
        """
        if self.unrecorded:
            self.storage.refuse_unrecorded()
        self._refuse_with_workers("initialize")
        self.iteration = 0
        self.start_number = 0
        self.state_of_replica = np.arange(self.k)
        self.storage.truncate(0)
        self.storage.clear_plan()
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
        self.checkpoint(must=True)

    def restore(self) -> bool:
        """Resume from the checkpoint in the storage directory, if there is one."""
        if self.unrecorded:
            self.storage.refuse_unrecorded()
        self._refuse_with_workers("restore")
        ckpt = self.storage.read_checkpoint()
        if ckpt is None:
            return False
        if ckpt["positions"].shape != self.arrays.pos[0].shape:
            raise SamplerError("the checkpoint does not match this leg (different size or number of states)")
        self.arrays.pos[self.cur] = ckpt["positions"]
        self.arrays.vel[self.cur] = ckpt["velocities"]
        self.arrays.box[self.cur] = ckpt["boxes"]
        # the rows the checkpoint names: iterations that were run without being stored have none
        # (a checkpoint of an earlier version does not say, and then every iteration had a row)
        extra = ckpt.get("extra") or {}
        if "rows" not in extra:
            # a checkpoint of an earlier version does not say how many of its iterations were stored: one
            # with iterations that were run without being stored (a warm-up) cannot be told from one
            # without, whatever its records show (review of PR 1), so it is not continued from
            from cufep.workflow import old_checkpoint_refusal

            raise StorageError(f"the leg in {self.storage.dir}: {old_checkpoint_refusal(self.storage.dir)}")
        rows = int(extra["rows"])
        if self.storage.n_iterations < rows:
            raise StorageError(
                f"{self.storage.dir} holds {self.storage.n_iterations} iterations, fewer than the "
                f"{rows} that its checkpoint names: the samples were changed after the checkpoint was written. "
                "Nothing was changed; the samples can be analysed (cufep analyze), and --reschedule starts the leg anew"
            )
        self.state_of_replica = ckpt["state_of_replica"].astype(int)
        self.iteration = ckpt["iteration"]
        self.last_checkpoint = int(ckpt["iteration"])
        # one more start of this leg: its workers get random streams that no earlier start used
        self.start_number = int((ckpt.get("extra") or {}).get("start", 0)) + 1
        self.storage.truncate(rows)
        return True

    def _refuse_with_workers(self, what: str) -> None:
        if self.workers:
            raise SamplerError(f"{what}() after the workers of this sampler were started: their random streams "
                               "belong to the start they were created for; create a new sampler")

    def checkpoint(self, must: bool = False) -> bool:
        """Store the samples that are in memory and the state to restart from
        (:meth:`LegStorage.write_checkpoint`: a new part of the samples, then the checkpoint).

        A reader of the leg (an analysis of the running leg) never holds a file that is written
        here, so nothing waits for it. A write that fails raises StorageError; the earlier
        checkpoint and the samples up to it are kept, and running the leg again resumes from there
        on the same backend. ``must`` (the start and the end of a run) is kept for the callers.
        """
        self.storage.write_checkpoint(
            self.arrays.pos[self.cur], self.arrays.vel[self.cur], self.arrays.box[self.cur], self.state_of_replica,
            self.iteration, extra={"start": self.start_number, "rows": self.storage.n_iterations})
        self.last_checkpoint = int(self.iteration)
        return True

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
        # a new number, and with it a new random stream: with the old one the worker would draw the
        # thermal noise of its first iterations again
        self.workers[w] = _Process(ctx, self.spec, old.device, self.arrays.names, self._next_worker_id,
                                   self.start_number)
        self._next_worker_id += 1
        self.log(f"[{self.spec.name}] worker {old.worker_id} restarted as worker {self.workers[w].worker_id}: "
                 f"{self.workers[w].wait_ready(600.0)}")

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
        failed: list[tuple[int, WorkerError]] = []
        for w, worker in enumerate(self.workers):
            try:
                results[w] = worker.result(timeout)
            except WorkerError as exc:
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
            except WorkerError as again:
                raise WorkerError(f"worker {w} failed twice from the same configurations at iteration "
                                  f"{self.iteration}: {again}") from again
        return results

    def run(self, n_iterations: int, record_stride: int = 4, checkpoint_stride: int = 100,
            store: bool = True, steps: int | None = None, progress_every: int = 0,
            on_iteration=None) -> list[IterationStats]:
        """Run ``n_iterations`` more iterations.

        ``on_iteration(sampler, stats)`` is called after every iteration; it may call
        :meth:`set_devices`.
        """
        if self.unrecorded and store:
            self.storage.refuse_unrecorded()
        if steps is not None:                        # in this process: never a WorkerError
            self.spec.check_segment(int(steps), f"the MD of one replica in one iteration ({int(steps)} steps asked)")
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
            # the exchange proposed on the configurations of this iteration, counted with its row
            swaps = np.zeros((max(0, len(sampled_states) - 1), 2), dtype="i8")
            frac = 0.0
            new = sampled_states
            if self.exchange:
                rng = np.random.default_rng([int(self.spec.seed) & 0x7FFFFFFF, self.iteration, 0x5EED])
                new, acc, att = mix_replicas(u, self.state_of_replica, rng, neighbour_counts=swaps)
                frac = float(np.mean(new != self.state_of_replica))
                self.accepted += acc
                self.attempted += att
            if store:
                # records carry the row of their iteration: iterations that are not stored (a warm-up
                # with store=False) have no row
                row = self.storage.append(u, sampled_states, self.arrays.vol.copy(), swaps=swaps)
                for res in results:
                    if res.get("recorded"):
                        self.storage.append_record(row, res["recorded"])
            self.state_of_replica = new
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
            self.checkpoint(must=True)
        return stats
