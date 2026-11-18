"""Replica workers: one process per GPU, one OpenMM Context per worker.

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
    friction_per_ps: float = 1.0
    timestep_fs: float = 4.0
    steps_per_iteration: int = 625
    seed: int = 0
    cpu_threads: int = 2
    box_safety: float | None = None
    #: long-range dispersion bookkeeping for the coupled state (see alchemy/lrc.py)
    lrc: dict | None = None
    #: atoms whose positions are saved for structural analysis (coupled-state replica)
    frame_atoms: list[int] = field(default_factory=list)
    #: positions *within frame_atoms* of the ligand heavy atoms and the protein C-alpha atoms
    frame_ligand: list[int] = field(default_factory=list)
    frame_ca: list[int] = field(default_factory=list)

    @property
    def n_states(self) -> int:
        return len(self.states)

    def alchemical_states(self) -> list[AlchemicalState]:
        return [AlchemicalState(**s) for s in self.states]


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


class SharedArrays:
    """Replica state and energies in memory-mapped files shared with the workers."""

    FIELDS = ("pos", "vel", "box", "u", "vol")

    def __init__(self, n_replicas: int, n_atoms: int, n_states: int, names: dict | None = None,
                 fallback_dir: str | None = None) -> None:
        self.shapes = {"pos": (2, n_replicas, n_atoms, 3), "vel": (2, n_replicas, n_atoms, 3),
                       "box": (2, n_replicas, 3, 3), "u": (n_replicas, n_states), "vol": (n_replicas,)}
        self.owner = names is None
        if self.owner:
            total = sum(int(np.prod(s)) * 8 for s in self.shapes.values())
            base = _shared_directory(total, fallback_dir)
            os.makedirs(base, exist_ok=True)
            self.directory = tempfile.mkdtemp(prefix="cufep-replicas-", dir=base)
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


class ReplicaWorker:
    """Owns one Context and advances replicas on request."""

    def __init__(self, spec: LegSpec, device: int | None, arrays: SharedArrays, worker_id: int = 0) -> None:
        import openmm as mm
        import openmm.unit as u

        from cufep.backends import get_backend
        from cufep.sampling.states import StateEnergyEvaluator

        self.spec = spec
        self.arrays = arrays
        self.worker_id = worker_id
        self._mm, self._u = mm, u
        kwargs = dict(precision=spec.precision, platform=spec.platform)
        if spec.backend in ("cupotential", "cupo"):
            kwargs.update(integrator=spec.integrator, box_safety=spec.box_safety)
        self.backend = get_backend(spec.backend, **kwargs)
        system = self.backend.prepare_system(mm.XmlSerializer.deserialize(spec.system_xml))
        temperature = spec.temperature_K * u.kelvin
        seed = int(spec.seed) + 1000003 * (worker_id + 1)
        if spec.pressure_bar is not None and spec.barostat != "none":
            pressure = spec.pressure_bar * u.bar
            if spec.barostat == "membrane":
                system.addForce(self.backend.make_membrane_barostat(pressure, temperature,
                                                                    spec.barostat_frequency, seed=seed + 1))
            else:
                system.addForce(self.backend.make_isotropic_barostat(pressure, temperature,
                                                                     spec.barostat_frequency, seed=seed + 1))
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


def worker_main(conn, spec: LegSpec, device: int | None, shm_names: dict, worker_id: int) -> None:
    """Entry point of a worker process."""
    os.environ["OPENMM_CPU_THREADS"] = str(spec.cpu_threads)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(spec.cpu_threads)
    # Never attach to someone else's MPS daemon.
    os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", "/tmp/cufep-no-mps")
    arrays = None
    worker = None
    try:
        arrays = SharedArrays(len(spec.states), spec.n_atoms, len(spec.states), names=shm_names)
        worker = ReplicaWorker(spec, device, arrays, worker_id)
        conn.send(("ready", worker.describe()))
        while True:
            msg = conn.recv()
            if msg[0] == "stop":
                break
            if msg[0] == "run":
                _cmd, cur, assignments, iteration, record, steps = msg
                result = worker.run(cur, assignments, iteration, record, steps)
                conn.send(("done", result))
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
