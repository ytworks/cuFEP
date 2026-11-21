"""The unrestrained MD of the complex from which the Boresch restraint is chosen.

``restraints.selection: interactions`` chooses the anchors of the restraint from the interactions
between protein and ligand in plain MD of the bound complex, as Ding et al. do
(arXiv:2603.22274 v3, sec. 2.4: four NPT trajectories at the target temperature, 3 ns each, a
frame every 10 ps, the first 1 ns of each not analysed). This module runs those trajectories.

* Every trajectory starts from the same configuration, the end of the pre-equilibration
  (positions, velocities, box), and differs from the others by its random numbers only: the
  thermostat and the barostat of trajectory k take the streams of the leg's seed and k
  (:func:`trajectory_seeds`), whatever GPU or process runs it and whenever.
* It is a plain MD of the leg like the unrestrained part of the pre-equilibration and goes
  through the same functions (:func:`cufep.sampling.equilibrate.unrestrained_md`,
  :func:`~cufep.sampling.equilibrate.with_time_step_recovery`): the backend, precision and
  fallback of the configuration, the repeat on stock OpenMM and at 2 fs, each said in the log and
  in the notes of the trajectory.
* A trajectory is one file, published in one step when it is complete
  (``restraint_md/trajectory_<k>.npz``): the frames of the solute atoms, their boxes and times,
  the backend that ran it, its notes and the record of what it was made from. A run that is
  interrupted keeps the trajectories that are complete and runs the others from their start.
* With several GPUs granted, the trajectories run at the same time, one process each
  (:func:`run_trajectories`); otherwise one after the other in the process of the job.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from cufep.files import publish_npz
from cufep.sampling.equilibrate import EquilibrationSettings, unrestrained_md, with_time_step_recovery
from cufep.sampling.worker import BAROSTAT_STREAM, INTEGRATOR_STREAM, stream_seed, worker_environment

#: the directory of the trajectories in the directory of the leg
DIRECTORY = "restraint_md"
#: what the leg's seed is raised by for these trajectories, so that they share no random stream
#: with the pre-equilibration, the lambda precalculation or the replica exchange of the leg
SEED_OFFSET = 31


class TrajectoryError(RuntimeError):
    """A trajectory for the restraint could not be made."""


def trajectory_file(leg_dir: Path | str, index: int) -> Path:
    return Path(leg_dir) / DIRECTORY / f"trajectory_{int(index)}.npz"


def trajectory_seeds(seed: int, index: int) -> tuple[int, int]:
    """The seeds of the integrator and of the barostat of trajectory ``index`` of a leg with
    ``seed``: never zero, which the cuPotential plugin reads as "a new seed every time"."""
    base = int(seed) + SEED_OFFSET
    return stream_seed(base, 0, index, INTEGRATOR_STREAM), stream_seed(base, 0, index, BAROSTAT_STREAM)


def start_digest(positions, velocities, box) -> str:
    """What identifies the configuration the trajectories start from: a trajectory of another
    start (a pre-equilibration that was made again) is not one of this leg."""
    digest = hashlib.sha256()
    for array in (positions, velocities, box):
        digest.update(np.ascontiguousarray(np.asarray(array, dtype=np.float64)).tobytes())
    return digest.hexdigest()


def frame_times_ps(n_frames: int, length_ns: float, frame_interval_ps: float, timestep_fs: float) -> np.ndarray:
    """The time of every frame of an MD of ``length_ns`` (ps from its start): one after every frame
    interval and one at the end, in whole MD steps (:func:`cufep.steps.md_steps`)."""
    from cufep.steps import md_steps

    steps, per_frame = md_steps(length_ns * 1000.0, timestep_fs), md_steps(frame_interval_ps, timestep_fs)
    return np.array([min((i + 1) * per_frame, steps) * float(timestep_fs) / 1000.0 for i in range(int(n_frames))])


@dataclass
class TrajectoryTask:
    """One trajectory to run: everything its process needs."""

    index: int
    path: str
    system_xml: str
    positions: np.ndarray
    velocities: np.ndarray
    box: np.ndarray
    frame_atoms: list[int]
    settings: EquilibrationSettings        # production_ns is the length of the trajectory
    timestep_fs: float                     # the time step to begin with
    made_with: dict                        # the record of what the trajectory is made from
    cpu_threads: int = 2
    mps_thread_percentage: int | None = None
    notes: list[str] = field(default_factory=list)       # what happened to it before it ran (a repeat)

    @property
    def what(self) -> str:
        return f"MD {self.index} for the restraint"


def run_trajectory(task: TrajectoryTask, device, log=print) -> dict:
    """Run one trajectory and publish its file; returns what the log of the leg says about it."""
    import openmm as mm

    system = mm.XmlSerializer.deserialize(task.system_xml)
    seed, barostat_seed = trajectory_seeds(task.settings.seed, task.index)
    start = (np.asarray(task.positions), np.asarray(task.velocities), np.asarray(task.box))
    result = with_time_step_recovery(
        lambda dt: unrestrained_md(system, start, task.frame_atoms, task.settings, dt, device, log, what=task.what,
                                   seed=seed, barostat_seed=barostat_seed),
        task.timestep_fs, log, what=task.what)
    notes = [*task.notes, *result.notes]
    times = frame_times_ps(result.frames.shape[0], task.settings.production_ns, task.settings.frame_interval_ps,
                           result.timestep_fs)
    publish_npz(Path(task.path), frames=result.frames, frame_boxes=result.frame_boxes,
                frame_atoms=np.array(result.frame_atoms), time_ps=times, timestep_fs=result.timestep_fs,
                simulated_ns=result.simulated_ns, backend=result.backend, notes=json.dumps(notes),
                made_with=json.dumps(task.made_with))
    return {"index": int(task.index), "backend": result.backend, "timestep_fs": float(result.timestep_fs),
            "frames": int(result.frames.shape[0]), "notes": notes}


def _trajectory_process(conn, task: TrajectoryTask, device, parent_pid: int) -> None:
    """Entry point of the process of one trajectory."""
    worker_environment(task.cpu_threads, task.mps_thread_percentage, parent_pid)
    try:
        summary = run_trajectory(task, device, lambda line: conn.send(("log", line)))
        conn.send(("done", summary))
    except Exception as exc:  # noqa: BLE001 - every failure is reported to the job
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"))
        except Exception:  # noqa: BLE001
            pass
    finally:
        conn.close()


def published_summary(task: TrajectoryTask) -> dict | None:
    """What :func:`run_trajectory` returns for a trajectory whose file is there, made by this task; None
    without such a file. The file is published in one step when the trajectory is complete, so a process
    that ends between that step and its last word has done its work (review of eb6f89f: the trajectory was
    run again on stock OpenMM over the file, or the stage ended with ``backend.fallback: none``)."""
    path = Path(task.path)
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            if json.loads(str(data["made_with"])) != json.loads(json.dumps(task.made_with)):
                return None
            return {"index": int(task.index), "backend": str(data["backend"]),
                    "timestep_fs": float(data["timestep_fs"]), "frames": int(data["frames"].shape[0]),
                    "notes": list(json.loads(str(data["notes"])))}
    except (OSError, ValueError, KeyError):
        return None


def _on_stock_openmm(task: TrajectoryTask) -> TrajectoryTask | None:
    """The task of a trajectory whose process died, to be run once more on stock OpenMM; None if the
    configuration allows no fallback or stock OpenMM was running it already."""
    if task.settings.md_backend == "openmm" or task.settings.fallback == "none":
        return None
    note = f"the process of {task.what} ended without a result on {task.settings.md_backend}; it was repeated on " \
           "stock OpenMM"
    return replace(task, settings=replace(task.settings, md_backend="openmm"), notes=[*task.notes, note])


def run_trajectories(tasks: list[TrajectoryTask], devices: list, log=print, inline: bool = False,
                     poll_seconds: float = 0.05) -> list[dict]:
    """Run every task and return their summaries in the order of the tasks.

    With ``inline`` or fewer than two entries in ``devices`` the trajectories run one after the
    other in this process, on the first device. Otherwise there is one process per entry of
    ``devices`` at a time (a GPU listed n times runs n at once, under NVIDIA MPS), each running one
    trajectory, until all are done. A process that ends without a result is a failure of the
    backend: its trajectory is run once more on stock OpenMM if the configuration has that
    fallback and the note of the trajectory says so; otherwise, and for an error that the process
    reports, the others are stopped and :class:`TrajectoryError` is raised. There is no limit on
    the time of a trajectory: its length is a setting.
    """
    for task in tasks:
        Path(task.path).parent.mkdir(parents=True, exist_ok=True)
    if inline or len(devices) < 2 or len(tasks) < 2:
        device = devices[0] if devices else None
        return [run_trajectory(task, device, log) for task in tasks]
    def last_word(pipe, wait: float):
        """What a process said of how its trajectory ended, if it has; its lines for the log are passed on."""
        try:
            while pipe.poll(wait):
                message = pipe.recv()
                if message[0] != "log":
                    return message
                log(message[1])
                wait = 0.0
        except (EOFError, OSError):
            return ("died",)
        return None

    ctx = mp.get_context("spawn")
    pending = list(tasks)
    running: dict[int, tuple] = {}                         # position in devices -> (process, pipe, task)
    done: dict[int, dict] = {}
    try:
        while pending or running:
            for slot, device in enumerate(devices):
                if slot not in running and pending:
                    task = pending.pop(0)
                    parent, child = ctx.Pipe()
                    process = ctx.Process(target=_trajectory_process, args=(child, task, device, os.getpid()),
                                          daemon=True)
                    process.start()
                    child.close()
                    running[slot] = (process, parent, task)
            for slot in list(running):
                process, pipe, task = running[slot]
                ended = last_word(pipe, poll_seconds)
                if ended is None and not process.is_alive():
                    # it is gone; what it sent between the look at the pipe and this one is still to be read
                    ended = last_word(pipe, 0.0) or ("died",)
                if ended is None:
                    continue
                process.join(15)
                if process.is_alive():
                    # it has said how it ended and does not end: it does not keep its GPU
                    process.kill()
                    process.join(5)
                pipe.close()
                del running[slot]
                if ended[0] == "done":
                    done[task.index] = ended[1]
                elif ended[0] == "error":
                    raise TrajectoryError(f"{task.what} failed (device {devices[slot]}): {ended[1]}")
                elif published_summary(task) is not None:
                    done[task.index] = published_summary(task)
                else:
                    again = _on_stock_openmm(task)
                    if again is None:
                        raise TrajectoryError(f"the process of {task.what} ended without a result (device "
                                              f"{devices[slot]})")
                    log(f"the process of {task.what} ended without a result; repeating the trajectory on stock OpenMM")
                    pending.insert(0, again)
    finally:
        for process, pipe, _task in running.values():
            if process.is_alive():
                process.kill()
            process.join(5)
            pipe.close()
    return [done[task.index] for task in tasks]


@dataclass
class Trajectories:
    """The trajectories of a leg as the analysis reads them."""

    frames: np.ndarray           # all frames x frame atoms x 3 (nm), one trajectory after the other
    boxes: np.ndarray            # all frames x 3 x 3
    frame_atoms: np.ndarray
    trajectory: np.ndarray       # the trajectory of every frame
    time_ps: np.ndarray          # the time of every frame from the start of its trajectory
    backends: list[str]          # the backend that ran each trajectory
    timesteps_fs: list[float]
    notes: list[str]             # of all trajectories, each naming its own

    def after(self, discard_ns: float) -> "Trajectories":
        """The frames whose time is later than ``discard_ns`` ("the first 1 ns of each trajectory is
        discarded"): by the time that was integrated, not by a count of frames, which depends on
        the time step."""
        keep = self.time_ps > float(discard_ns) * 1000.0 + 1e-9
        return replace(self, frames=self.frames[keep], boxes=self.boxes[keep], trajectory=self.trajectory[keep],
                       time_ps=self.time_ps[keep])


def read_trajectories(leg_dir: Path | str, n: int) -> Trajectories:
    """The ``n`` trajectories of a leg, in their order."""
    frames, boxes, which, times, backends, timesteps, notes = [], [], [], [], [], [], []
    frame_atoms = None
    for k in range(int(n)):
        with np.load(trajectory_file(leg_dir, k)) as data:
            atoms = np.asarray(data["frame_atoms"], dtype=int)
            if frame_atoms is not None and not np.array_equal(atoms, frame_atoms):
                raise TrajectoryError(
                    f"{trajectory_file(leg_dir, k)} holds other atoms than the trajectories before it")
            frame_atoms = atoms
            frames.append(np.asarray(data["frames"]))
            boxes.append(np.asarray(data["frame_boxes"]))
            times.append(np.asarray(data["time_ps"], dtype=float))
            which.append(np.full(frames[-1].shape[0], k, dtype=int))
            backends.append(str(data["backend"]))
            timesteps.append(float(data["timestep_fs"]))
            # a note that does not name its trajectory (the lowered time step) is given its name
            notes += [note if "for the restraint" in note else f"MD {k} for the restraint: {note}"
                      for note in json.loads(str(data["notes"]))]
    return Trajectories(np.concatenate(frames), np.concatenate(boxes), frame_atoms, np.concatenate(which),
                        np.concatenate(times), backends, timesteps, notes)
