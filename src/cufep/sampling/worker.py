"""Replica workers: processes with one OpenMM Context each, one or several per GPU.

A worker time-slices the replicas assigned to it through its single Context. Replica
coordinates, velocities and boxes live in shared memory owned by the controller and are
double-buffered: a worker reads buffer ``cur`` and writes buffer ``1 - cur``, and the
controller flips ``cur`` only when every worker has reported. A crashed worker therefore
never leaves a half-updated iteration behind.

Only small messages cross the pipes; reduced potentials go to shared memory too.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from cufep.sampling.states import AlchemicalState


@dataclass
class LegSpec:
    """Everything a worker needs to build its Context (picklable)."""

    name: str
    system_xml: str                      # stock alchemical System, restraint included, no barostat
    states: list[dict]
    n_atoms: int
    backend: str = "cupotential"
    precision: str = "mixed"
    platform: str = "CUDA"
    integrator: str = "nv"               # cuPotential backend only: nv | stock
    temperature_K: float = 298.15
    pressure_bar: float | None = 1.0     # None -> constant volume
    barostat: str = "isotropic"          # isotropic | membrane | none
    barostat_frequency: int = 25
    #: hrex = Hamiltonian replica exchange between the windows, none = independent windows
    exchange: str = "hrex"
    friction_per_ps: float = 1.0
    timestep_fs: float = 4.0
    steps_per_iteration: int = 625
    seed: int = 0
    cpu_threads: int = 2
    box_safety: float | None = None
    #: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE of the worker processes when several share a GPU under MPS
    mps_thread_percentage: int | None = None
    #: long-range dispersion bookkeeping for the coupled state (see alchemy/lrc.py)
    lrc: dict | None = None
    #: the settings of the configuration that the samples depend on beyond what is listed here
    #: (soft-core, REST2, tail), in the words of the configuration: recorded with the conditions,
    #: so that a later run can be compared with them before any System is built
    settings: dict = field(default_factory=dict)
    #: atoms whose positions are saved for structural analysis (coupled-state replica)
    frame_atoms: list[int] = field(default_factory=list)
    #: positions *within frame_atoms* of the ligand heavy atoms and the protein C-alpha atoms
    frame_ligand: list[int] = field(default_factory=list)
    frame_ca: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.check_segment(self.steps_per_iteration, "the MD of one replica in one iteration")

    @property
    def has_barostat(self) -> bool:
        return self.pressure_bar is not None and self.barostat != "none"

    def check_segment(self, steps: int, what: str = "a stretch of MD of one configuration") -> None:
        """ValueError if a configuration that is advanced by ``steps`` MD steps in a Context of
        this leg, which other configurations share, would not get a move of the barostat
        (:func:`cufep.steps.barostat_segment_problem`). Checked in the process that plans the
        steps, before a worker is asked for them."""
        from cufep.steps import barostat_segment_problem

        if self.has_barostat:
            problem = barostat_segment_problem(steps, self.barostat_frequency, what)
            if problem:
                raise ValueError(problem)

    @property
    def n_states(self) -> int:
        return len(self.states)

    def alchemical_states(self) -> list[AlchemicalState]:
        return [AlchemicalState(**s) for s in self.states]

    def conditions(self) -> dict:
        """What the samples of this leg are samples of.

        A leg can be continued only under the conditions its samples were made under: the
        lambda states, temperature, pressure and barostat, the dynamics, the length of an
        iteration, the backend, and the System, which holds the force field, the restraint and
        the REST2 region and is represented by a checksum of its XML.
        """
        import hashlib

        return {"states": [dict(s) for s in self.states], "temperature_K": float(self.temperature_K),
                "pressure_bar": None if self.pressure_bar is None else float(self.pressure_bar),
                "barostat": self.barostat, "barostat_frequency": int(self.barostat_frequency),
                "exchange": str(self.exchange),
                "friction_per_ps": float(self.friction_per_ps), "timestep_fs": float(self.timestep_fs),
                "steps_per_iteration": int(self.steps_per_iteration), "backend": self.backend,
                "precision": self.precision, "integrator": self.integrator, "n_atoms": int(self.n_atoms),
                "system_sha256": hashlib.sha256(self.system_xml.encode()).hexdigest(),
                "lrc": None if self.lrc is None else tail_summary(self.lrc),
                **({"settings": dict(self.settings)} if self.settings else {})}


def _shared_directory(required_bytes: int, fallback: str | None = None) -> str:
    """Directory for the shared replica arrays.

    ``/dev/shm`` is used when it has room (no disk traffic). Docker gives containers only
    64 MB there unless started with ``--shm-size`` or ``--ipc=host``; writing past that
    limit kills the process with SIGBUS, so fall back to an ordinary directory instead.
    """
    override = os.environ.get("CUFEP_SHM_DIR")
    if override:
        return override
    try:
        st = os.statvfs("/dev/shm")
        if st.f_bavail * st.f_frsize > 1.25 * required_bytes + (64 << 20):
            return "/dev/shm"
    except OSError:
        pass
    return fallback or tempfile.gettempdir()


#: the directories of the shared replica arrays, and the lock their owner holds while it lives
ARRAYS_PREFIX = "cufep-replicas-"
OWNER_LOCK = "owner.lock"


def remove_orphaned_arrays(base: str) -> list[str]:
    """Remove the replica arrays under ``base`` whose owner has ended without removing them (killed,
    crashed); returns them. An owner holds the lock ``owner.lock`` of its directory while it lives;
    the lock works across processes and containers on one machine, which a process number would not.
    Directories without the lock (of earlier versions) are left alone. Review of PR 1: every killed
    run left its arrays, 288 MB for 30 replicas of 100,000 atoms, in /dev/shm or its fallback."""
    import fcntl

    removed = []
    try:
        names = os.listdir(base)
    except OSError:
        return removed
    for name in names:
        if not name.startswith(ARRAYS_PREFIX):
            continue
        directory = os.path.join(base, name)
        try:
            fd = os.open(os.path.join(directory, OWNER_LOCK), os.O_RDWR)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            continue                                     # its owner lives
        finally:
            os.close(fd)
        shutil.rmtree(directory, ignore_errors=True)
        removed.append(directory)
    return removed


#: the name a directory of replica arrays has until its owner holds its lock: never removed by
#: remove_orphaned_arrays, which looks at ARRAYS_PREFIX only
STAGING_PREFIX = "cufep-staging-"


def owned_directory(base: str, _before_lock=None) -> tuple[str, int]:
    """A new directory for replica arrays under ``base`` and the descriptor of its lock, held.

    The directory is made under a name that :func:`remove_orphaned_arrays` does not look at, its
    lock is taken there, and only then is it renamed to its public name in the same directory: a
    directory under the public name always has a lock that is held while its owner lives (review
    of PR 1: between the creation of the lock file and the lock, another job's removal of
    orphaned arrays found the lock free and removed the directory being made). A directory that
    already has the public name makes the rename fail (it holds a lock file, so it is not empty)
    and a new name is tried. An owner killed in between leaves an empty staging directory.
    ``_before_lock`` is called just before the lock is taken (tests).
    """
    import fcntl

    while True:
        staging = tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=base)
        fd = os.open(os.path.join(staging, OWNER_LOCK), os.O_RDWR | os.O_CREAT, 0o644)
        if _before_lock is not None:
            _before_lock()
        fcntl.flock(fd, fcntl.LOCK_EX)
        public = os.path.join(base, ARRAYS_PREFIX + os.path.basename(staging)[len(STAGING_PREFIX):])
        try:
            os.rename(staging, public)
        except OSError:
            os.close(fd)
            shutil.rmtree(staging, ignore_errors=True)
            continue
        return public, fd


class SharedArrays:
    """Replica state and energies in memory-mapped files shared with the workers."""

    FIELDS = ("pos", "vel", "box", "u", "vol")

    def __init__(self, n_replicas: int, n_atoms: int, n_states: int, names: dict | None = None,
                 fallback_dir: str | None = None) -> None:
        self.shapes = {"pos": (2, n_replicas, n_atoms, 3), "vel": (2, n_replicas, n_atoms, 3),
                       "box": (2, n_replicas, 3, 3), "u": (n_replicas, n_states), "vol": (n_replicas,)}
        self.owner = names is None
        self._owner_lock = None
        if self.owner:
            total = sum(int(np.prod(s)) * 8 for s in self.shapes.values())
            base = _shared_directory(total, fallback_dir)
            os.makedirs(base, exist_ok=True)
            remove_orphaned_arrays(base)
            self.directory, self._owner_lock = owned_directory(base)
            self._paths = {k: os.path.join(self.directory, f"{k}.f8") for k in self.FIELDS}
            mode = "w+"
        else:
            self.directory = None
            self._paths = dict(names)
            mode = "r+"
        self._maps = {k: np.memmap(self._paths[k], dtype=np.float64, mode=mode, shape=self.shapes[k])
                      for k in self.FIELDS}
        self.pos, self.vel, self.box, self.u, self.vol = (self._maps[k] for k in self.FIELDS)
        if self.owner:
            self.vel[:] = np.nan          # NaN velocities = "draw from Maxwell-Boltzmann"

    @property
    def names(self) -> dict:
        return dict(self._paths)

    @property
    def location(self) -> str:
        return os.path.dirname(self._paths["pos"])

    def close(self) -> None:
        self.pos = self.vel = self.box = self.u = self.vol = None
        self._maps = {}
        if self.owner and self.directory:
            shutil.rmtree(self.directory, ignore_errors=True)
            self.directory = None
        if self._owner_lock is not None:
            os.close(self._owner_lock)
            self._owner_lock = None


def tail_summary(lrc: dict) -> dict:
    """The record of a tail correction in the conditions of the samples: its summary
    (:meth:`cufep.alchemy.lrc.TailEnergy.summary`), also for a record that is one already."""
    if "sha256" in lrc and "env" not in lrc:
        return dict(lrc)
    from cufep.alchemy.lrc import TailEnergy

    return TailEnergy.from_dict(lrc).summary()


#: what a random stream of a worker is for
INTEGRATOR_STREAM, BAROSTAT_STREAM = 0, 1


def stream_seed(seed: int, start: int, worker_id: int, purpose: int = INTEGRATOR_STREAM) -> int:
    """The seed of one random stream of a leg, in 1 ... 2**31 - 1.

    A stream belongs to one worker of one start of the leg: ``start`` counts how often the leg
    has been started (0 for a new run, one more each time it is resumed or continued from a
    checkpoint), ``worker_id`` is never used twice within a start, ``purpose`` separates the
    integrator from the barostat. A worker that is created again therefore never draws the
    numbers an earlier one drew: a seed made of the leg's seed and the worker's number alone
    repeated the thermal noise of a leg from its beginning whenever the leg was resumed. Zero is
    avoided because the cuPotential plugin reads it as "a new seed every time".
    """
    entropy = [int(seed) & 0x7FFFFFFF, int(start), int(worker_id), int(purpose)]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0] & 0x7FFFFFFF) or 1


class ReplicaWorker:
    """Owns one Context and advances replicas on request."""

    def __init__(self, spec: LegSpec, device: int | None, arrays: SharedArrays, worker_id: int = 0,
                 start: int = 0) -> None:
        import openmm as mm
        import openmm.unit as u

        from cufep.backends import get_backend
        from cufep.sampling.states import StateEnergyEvaluator

        self.spec = spec
        self.arrays = arrays
        self.worker_id = worker_id
        self._mm, self._u = mm, u
        kwargs = dict(precision=spec.precision, platform=spec.platform, cpu_threads=spec.cpu_threads)
        if spec.backend in ("cupotential", "cupo"):
            kwargs.update(integrator=spec.integrator, box_safety=spec.box_safety)
        self.backend = get_backend(spec.backend, **kwargs)
        system = self.backend.prepare_system(mm.XmlSerializer.deserialize(spec.system_xml))
        temperature = spec.temperature_K * u.kelvin
        self.start = int(start)
        seed = stream_seed(spec.seed, start, worker_id, INTEGRATOR_STREAM)
        barostat_seed = stream_seed(spec.seed, start, worker_id, BAROSTAT_STREAM)
        if spec.pressure_bar is not None and spec.barostat != "none":
            pressure = spec.pressure_bar * u.bar
            if spec.barostat == "membrane":
                system.addForce(self.backend.make_membrane_barostat(pressure, temperature,
                                                                    spec.barostat_frequency, seed=barostat_seed))
            else:
                system.addForce(self.backend.make_isotropic_barostat(pressure, temperature,
                                                                     spec.barostat_frequency, seed=barostat_seed))
        self.integrator = self.backend.make_integrator(temperature, spec.friction_per_ps / u.picosecond,
                                                       spec.timestep_fs * u.femtoseconds, seed=seed)
        self.context = self.backend.make_context(system, self.integrator, device)
        self.states = spec.alchemical_states()
        pressure_for_u = spec.pressure_bar if spec.barostat != "none" else None
        self.evaluator = StateEnergyEvaluator(self.context, self.states, spec.temperature_K, pressure_for_u)
        self.temperature = temperature
        self.lrc = None
        if spec.lrc is not None:
            from cufep.alchemy.lrc import TailEnergy
            self.lrc = TailEnergy.from_dict(spec.lrc)

    def describe(self) -> str:
        platform = self.context.getPlatform()
        names = platform.getPropertyNames()
        device = platform.getPropertyValue(self.context, "DeviceName") if "DeviceName" in names else platform.getName()
        index = platform.getPropertyValue(self.context, "DeviceIndex") if "DeviceIndex" in names else "-"
        return f"{self.spec.backend}/{platform.getName()} on {device} (index {index})"

    def run(self, cur: int, assignments: list[tuple[int, int]], iteration: int, record: bool,
            steps: int | None = None) -> dict:
        """Advance replicas ``[(replica, state_index), ...]`` by one iteration."""
        from cufep.sampling.states import set_state

        u = self._u
        t_start = time.perf_counter()
        arrays, nxt = self.arrays, 1 - cur
        n_steps = self.spec.steps_per_iteration if steps is None else int(steps)
        self.spec.check_segment(n_steps)            # the sampler checks it first; this is the last line
        out: dict = {"recorded": None}
        for replica, state_index in assignments:
            box = arrays.box[cur, replica]
            self.context.setPeriodicBoxVectors(*[self._mm.Vec3(*row) for row in box])
            self.context.setPositions(arrays.pos[cur, replica])
            vel = arrays.vel[cur, replica]
            if np.isnan(vel[0, 0]):
                self.context.setVelocitiesToTemperature(
                    self.temperature, (int(self.spec.seed) + 7919 * replica + 104729 * iteration) % 2147483647)
            else:
                self.context.setVelocities(vel)
            state = self.states[state_index]
            set_state(self.context, state)
            if n_steps > 0:
                self.integrator.step(n_steps)
            st = self.context.getState(getPositions=True, getVelocities=True)
            pos = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)
            new_box = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer)
            reduced, volume = self.evaluator.reduced_potentials()
            if not (np.isfinite(reduced[state_index]) and np.all(np.isfinite(pos[:1]))):
                raise FloatingPointError(f"replica {replica} (state {state_index}) became non-finite "
                                         f"at iteration {iteration}")
            arrays.pos[nxt, replica] = pos
            arrays.vel[nxt, replica] = st.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond)
            arrays.box[nxt, replica] = new_box
            arrays.u[replica] = reduced
            arrays.vol[replica] = volume
            if record and state_index == 0:
                rec = {"replica": replica}
                if self.lrc is not None:
                    rec["lrc_du"] = float(self.lrc.energy(np.asarray(pos), np.asarray(new_box)))
                if self.spec.frame_atoms:
                    rec["frame"] = np.asarray(pos[self.spec.frame_atoms], dtype=np.float32)
                    rec["box"] = np.asarray(new_box, dtype=np.float32)
                out["recorded"] = rec
        out["seconds"] = time.perf_counter() - t_start
        out["n_replicas"] = len(assignments)
        return out

    def close(self) -> None:
        del self.context
        del self.integrator


def worker_environment(cpu_threads: int, mps_thread_percentage: int | None = None,
                       parent_pid: int | None = None) -> None:
    """What every process that a job starts for MD on a GPU sets before it makes a Context: it ends
    with the job that started it (cufep.processes), its threads are those of
    ``resources.cpu_threads_per_worker``, and it uses cuFEP's own MPS daemon if the run started one
    (cufep.mps), never somebody else's. One function for the workers of the replica exchange and
    of the lambda precalculation and for the MD from which a restraint is chosen."""
    if parent_pid is not None:
        from cufep.processes import die_with_parent

        die_with_parent(parent_pid)
    os.environ["OPENMM_CPU_THREADS"] = str(cpu_threads)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(cpu_threads)
    os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", "/tmp/cufep-no-mps")
    if mps_thread_percentage:
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(int(mps_thread_percentage))


def worker_main(conn, spec: LegSpec, device: int | None, shm_names: dict, worker_id: int, start: int = 0,
                parent_pid: int | None = None) -> None:
    """Entry point of a worker process. It ends with the job that started it (cufep.processes)."""
    worker_environment(spec.cpu_threads, spec.mps_thread_percentage, parent_pid)
    arrays = None
    worker = None
    try:
        if shm_names:                                    # none for workers that only run single windows
            arrays = SharedArrays(len(spec.states), spec.n_atoms, len(spec.states), names=shm_names)
        worker = ReplicaWorker(spec, device, arrays, worker_id, start)
        conn.send(("ready", worker.describe()))
        while True:
            msg = conn.recv()
            if msg[0] == "stop":
                break
            if msg[0] == "run":
                _cmd, cur, assignments, iteration, record, steps = msg
                result = worker.run(cur, assignments, iteration, record, steps)
                conn.send(("done", result))
            elif msg[0] == "window":                     # one window of the adaptive precalculation
                from cufep.alchemy.schedule import run_window
                conn.send(("done", run_window(worker.context, worker.integrator, msg[1])))
            else:
                conn.send(("error", f"unknown command {msg[0]!r}"))
    except EOFError:
        pass
    except Exception as exc:  # noqa: BLE001 - report every failure to the controller
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"))
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            if worker is not None:
                worker.close()
        except Exception:  # noqa: BLE001
            pass
        if arrays is not None:
            arrays.close()
        conn.close()
