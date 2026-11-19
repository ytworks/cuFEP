"""Storage for one leg: energies in HDF5, restart state in an atomically written NPZ."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import h5py
import numpy as np


class ConditionsError(Exception):
    """Stored samples were made under other conditions than those of the run that opens them."""


def _same(a, b) -> bool:
    """Equality of stored and current conditions, with a tolerance for numbers."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return a == b
    try:
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)), abs(float(b)))
    except (TypeError, ValueError):
        return a == b


#: Seconds to wait for a samples file that another process holds: a reader for the writer that is
#: storing a checkpoint; the writer for a reader at an ordinary checkpoint, after which that
#: checkpoint is postponed; and the writer where the samples have to go to the disk.
READ_WAIT_S, CHECKPOINT_WAIT_S, WRITE_WAIT_S = 60.0, 20.0, 600.0


def _held_by_another_process(exc: OSError) -> bool:
    return isinstance(exc, BlockingIOError) or "unable to lock file" in str(exc)


def open_samples(path: str | Path, mode: str = "r", wait_s: float = READ_WAIT_S):
    """``h5py.File(path, mode)``; while another process holds the file, wait for it, up to ``wait_s``.

    HDF5 locks a file that is open: a reader against writers, a writer against everybody, and
    an open that meets the lock fails at once. The sampler of a leg has the file open only while
    it stores a checkpoint (:class:`LegStorage`), and a reader only while it loads the samples,
    so a short wait is enough on either side. The error of HDF5 is raised when the wait is over.
    """
    deadline = time.monotonic() + max(0.0, float(wait_s))
    pause = 0.02
    while True:
        try:
            return h5py.File(path, mode)
        except OSError as exc:
            if not _held_by_another_process(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(pause)
            pause = min(0.5, 2.0 * pause)


def stored_conditions(directory: str | Path) -> tuple[int, dict | None]:
    """(number of iterations, conditions) of the samples in ``directory``, read without changing
    anything; (0, None) if there are no samples, (n, None) if their conditions are not recorded."""
    path = Path(directory) / "samples.h5"
    if not path.exists():
        return 0, None
    with open_samples(path) as h5:
        n = int(h5["u"].shape[0])
        if n == 0 or "conditions" not in h5.attrs:
            return n, None
        return n, json.loads(h5.attrs["conditions"])


def conditions_differences(stored: dict, current: dict) -> list[str]:
    """What differs between the conditions stored with samples and those of a run, in words.

    Everything the samples record is compared; what they do not record (a key of a later
    version) is not.
    """
    differing = []
    for key, old in stored.items():
        new = current.get(key)
        if key == "lrc" and isinstance(old, dict) and "env" in old:
            # samples of a version that stored all parameters of the tail: compared as their summary
            from cufep.sampling.worker import tail_summary
            old = tail_summary(old)
        if _same(old, new):
            continue
        if key == "states":
            differing.append("the lambda states")
        elif key == "system_sha256":
            differing.append("the System (force field, soft-core, restraint or REST2 region)")
        elif key == "lrc" and isinstance(old, dict) and isinstance(new, dict):
            radii = [f"{name}: {old.get(name)} -> {new.get(name)}" for name in ("cutoff_nm", "long_cutoff_nm")
                     if not _same(old.get(name), new.get(name))]
            differing += [f"the tail correction ({'; '.join(radii)})"] if radii else \
                ["the Lennard-Jones parameters of the ligand's dispersion tail"]
        elif key == "lrc":
            differing.append("the tail correction (recorded with the samples or not)")
        elif key == "settings" and isinstance(old, dict) and isinstance(new, dict):
            differing += [f"{name}: {old.get(name)} -> {new.get(name)}"
                          for name in sorted(set(old) | set(new)) if not _same(old.get(name), new.get(name))]
        else:
            differing.append(f"{key}: {old} -> {new}")
    return differing


def other_conditions(n_iterations: int, directory, differing: list[str]) -> "ConditionsError":
    """The refusal to continue samples under other conditions than they were made under."""
    return ConditionsError(
        f"the {n_iterations} iterations of samples in {directory} were made under other conditions than "
        f"this run asks for ({'; '.join(differing)}). They are left as they are. A calculation under other "
        "conditions needs a new output.dir; only the lambda windows of a leg can be determined again in "
        "place, with --reschedule.")


def stored_backend(directory: str | Path) -> str | None:
    """Backend with which the samples in ``directory`` were made, or None if there are none."""
    path = Path(directory) / "samples.h5"
    if not path.exists():
        return None
    with open_samples(path) as h5:
        if h5["u"].shape[0] == 0 or "backend" not in h5.attrs:
            return None
        return str(h5.attrs["backend"])


class LegStorage:
    """The samples of one leg and the state to restart it from.

    ``samples.h5``
        u        (n_iter, n_replicas, n_states)  reduced potentials of each replica at every state
        states   (n_iter, n_replicas)            state each replica was simulated in
        volume   (n_iter, n_replicas)            box volume, nm^3
        lrc_iter, lrc_du                         coupled-state tail energies (kJ/mol)
        frame_iter, frames, frame_box            coupled-state coordinates of selected atoms
    ``checkpoint.npz``
        positions, velocities and boxes of every replica, the state permutation and the
        number of committed iterations.

    The rows of the iterations are kept in memory and go to ``samples.h5`` when a checkpoint
    is written, and when the storage is closed; in between the file is closed. Another process
    can therefore read the samples of a leg that is being sampled (``cufep analyze``), and what
    it reads is one state of the run: all datasets to the iteration of the last checkpoint. A
    run that is resumed drops the rows after its checkpoint anyway, so a run that is killed
    loses nothing it would have kept. (HDF5's single-writer-multiple-reader mode was not taken:
    a writer that is killed leaves such a file unreadable until it is repaired, attributes
    cannot be added in that mode, and files of earlier versions could not use it.)
    """

    def __init__(self, directory: str | Path, n_replicas: int, n_states: int, n_frame_atoms: int = 0) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.h5_path = self.dir / "samples.h5"
        self.ckpt_path = self.dir / "checkpoint.npz"
        self.n_replicas, self.n_states, self.n_frame_atoms = n_replicas, n_states, n_frame_atoms
        #: iterations that are not in the file yet: (u, states, volume) each, and the records
        #: (iteration, tail energy and frame of the coupled state) that belong to them
        self._rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._records: list[tuple[int, dict]] = []
        if not self.h5_path.exists():
            with self._file_for_writing() as h5:
                k, s = n_replicas, n_states
                h5.create_dataset("u", shape=(0, k, s), maxshape=(None, k, s), dtype="f8", chunks=(16, k, s))
                h5.create_dataset("states", shape=(0, k), maxshape=(None, k), dtype="i4", chunks=(256, k))
                h5.create_dataset("volume", shape=(0, k), maxshape=(None, k), dtype="f8", chunks=(256, k))
                h5.create_dataset("lrc_iter", shape=(0,), maxshape=(None,), dtype="i8", chunks=(1024,))
                h5.create_dataset("lrc_du", shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
                h5.create_dataset("frame_iter", shape=(0,), maxshape=(None,), dtype="i8", chunks=(256,))
                if n_frame_atoms:
                    h5.create_dataset("frames", shape=(0, n_frame_atoms, 3), maxshape=(None, n_frame_atoms, 3),
                                      dtype="f4", chunks=(8, n_frame_atoms, 3), compression="gzip",
                                      compression_opts=1)
                    h5.create_dataset("frame_box", shape=(0, 3, 3), maxshape=(None, 3, 3), dtype="f4")
                h5.attrs["n_equilibration_iterations"] = 0
            self._stored = 0                               # iterations in the file
        else:
            with open_samples(self.h5_path) as h5:
                if h5["u"].shape[1:] != (n_replicas, n_states):
                    raise ValueError(f"{self.h5_path} was written for a different number of replicas/states")
                self._stored = int(h5["u"].shape[0])

    def _file_for_writing(self, wait_s: float = WRITE_WAIT_S):
        return open_samples(self.h5_path, "a", wait_s)

    def attributes(self) -> dict:
        """The attributes of the samples file as Python values."""
        with open_samples(self.h5_path) as h5:
            return decoded_attributes(h5)

    # -- conditions ---------------------------------------------------------------
    def check_conditions(self, conditions: dict) -> str:
        """Record the conditions of the samples, or refuse to continue them under other ones.

        Nothing is written if they differ: the samples and their metadata stay as they are.
        Returns "new", "same" or "unrecorded".

        "unrecorded": the file holds samples of a version that did not record their conditions.
        Nothing on the disk tells what they were made with (the soft-core, the pressure and the
        friction were stored nowhere, and that version rewrote temperature, backend and states
        every time it opened the file), so they cannot be checked and nothing is written to the
        file. Such samples are read and analysed; they are not continued (:meth:`refuse_unrecorded`).
        """
        if self.n_iterations == 0:
            with self._file_for_writing() as h5:
                h5.attrs["conditions"] = json.dumps(conditions)
            return "new"
        with open_samples(self.h5_path) as h5:
            stored = h5.attrs["conditions"] if "conditions" in h5.attrs else None
        if stored is None:
            return "unrecorded"
        differing = conditions_differences(json.loads(stored), conditions)
        if differing:
            raise other_conditions(self.n_iterations, self.dir, differing)
        return "same"

    def refuse_unrecorded(self) -> None:
        """Raise: samples whose conditions were not recorded are not continued."""
        raise ConditionsError(
            f"the {self.n_iterations} iterations of samples in {self.dir} were written by a version of cuFEP that "
            "did not record the conditions they were made under, so nothing shows that this run would continue "
            "them under the same ones. They are left as they are and can be analysed (cufep analyze). To sample "
            "this leg further, start it anew with --reschedule: the samples it has are kept in previous_<n>/.")

    # -- metadata -----------------------------------------------------------------
    def set_meta(self, **kwargs) -> None:
        with self._file_for_writing() as h5:
            for key, value in kwargs.items():
                h5.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value

    def set_plan(self, n_equilibration: int, n_total: int, analysis_record: dict | None = None) -> None:
        """Record, before the run, how many of its iterations are equilibration and how many it has,
        and the settings its result will be analysed with (``analysis_record``).

        It goes to the disk at once: an analysis of a run that was interrupted must not count
        equilibration as production, and must judge the leg by the levels of its run.
        """
        with self._file_for_writing() as h5:
            h5.attrs["n_equilibration_iterations"] = int(n_equilibration)
            h5.attrs["n_planned_iterations"] = int(n_total)
            if analysis_record is not None:
                h5.attrs["analysis_record"] = json.dumps(analysis_record)

    def clear_plan(self) -> None:
        """A run that starts anew has no plan until :meth:`set_plan` gives it one."""
        with self._file_for_writing() as h5:
            h5.attrs["n_equilibration_iterations"] = 0
            if "n_planned_iterations" in h5.attrs:
                del h5.attrs["n_planned_iterations"]

    @property
    def n_iterations(self) -> int:
        """Iterations this storage holds: those in the file and those still in memory."""
        return self._stored + len(self._rows)

    # -- appending ------------------------------------------------------------------
    def append(self, u: np.ndarray, states: np.ndarray, volume: np.ndarray) -> int:
        n = self.n_iterations
        self._rows.append((np.array(u, dtype="f8"), np.array(states, dtype="i4"), np.array(volume, dtype="f8")))
        return n

    def append_record(self, iteration: int, record: dict) -> None:
        kept = {}
        if "lrc_du" in record:
            kept["lrc_du"] = float(record["lrc_du"])
        if "frame" in record and self.n_frame_atoms:
            kept["frame"] = np.array(record["frame"], dtype="f4")
            kept["box"] = np.array(record["box"], dtype="f4")
        if kept:
            self._records.append((int(iteration), kept))

    def flush(self, wait_s: float = WRITE_WAIT_S) -> bool:
        """Write the iterations that are in memory to the file.

        False, with nothing written and everything still in memory, if another process held the
        file for all of ``wait_s``.
        """
        if not self._rows and not self._records:
            return True
        try:
            h5 = self._file_for_writing(wait_s)
        except OSError as exc:
            if _held_by_another_process(exc):
                return False
            raise
        with h5:
            if self._rows:
                n, m = int(h5["u"].shape[0]), len(self._rows)
                for column, name in enumerate(("u", "states", "volume")):
                    h5[name].resize(n + m, axis=0)
                    h5[name][n:n + m] = np.stack([row[column] for row in self._rows])
            tail = [(it, record["lrc_du"]) for it, record in self._records if "lrc_du" in record]
            if tail:
                n = int(h5["lrc_du"].shape[0])
                for name, values in (("lrc_iter", [it for it, _ in tail]), ("lrc_du", [du for _, du in tail])):
                    h5[name].resize(n + len(tail), axis=0)
                    h5[name][n:n + len(tail)] = values
            frames = [(it, record) for it, record in self._records if "frame" in record]
            if frames and "frames" in h5:
                n = int(h5["frames"].shape[0])
                for name, values in (("frame_iter", [it for it, _ in frames]),
                                     ("frames", np.stack([record["frame"] for _, record in frames])),
                                     ("frame_box", np.stack([record["box"] for _, record in frames]))):
                    h5[name].resize(n + len(frames), axis=0)
                    h5[name][n:n + len(frames)] = values
            h5.flush()
        self._stored += len(self._rows)
        self._rows.clear()
        self._records.clear()
        return True

    def truncate(self, n_iterations: int) -> None:
        """Drop everything after ``n_iterations`` committed iterations, in memory and in the file."""
        n_iterations = int(n_iterations)
        del self._rows[max(0, n_iterations - self._stored):]
        self._records = [(it, record) for it, record in self._records if it < n_iterations]
        with open_samples(self.h5_path) as h5:
            beyond = h5["u"].shape[0] > n_iterations or any(
                h5[name].shape[0] and int(h5[name][-1]) >= n_iterations for name in ("lrc_iter", "frame_iter"))
        if not beyond:
            return                                         # nothing to cut: the file is not opened for writing
        with self._file_for_writing() as h5:
            for name in ("u", "states", "volume"):
                if h5[name].shape[0] > n_iterations:
                    h5[name].resize(n_iterations, axis=0)
            for it_name, data in (("lrc_iter", ("lrc_du",)), ("frame_iter", ("frames", "frame_box"))):
                its = h5[it_name][:]
                keep = int(np.sum(its < n_iterations))
                if keep < len(its):
                    h5[it_name].resize(keep, axis=0)
                    for d in data:
                        if d in h5:
                            h5[d].resize(keep, axis=0)
            h5.flush()
        self._stored = min(self._stored, n_iterations)

    # -- checkpoint -------------------------------------------------------------------
    def write_checkpoint(self, positions, velocities, boxes, state_of_replica, iteration: int,
                         extra: dict | None = None, wait_s: float = WRITE_WAIT_S) -> bool:
        """Store the iterations that are in memory, then the state to restart from.

        In that order: a checkpoint never names an iteration whose samples are not in the file.
        False, with neither written, if another process held the samples file for all of
        ``wait_s``; the iterations stay in memory and the earlier checkpoint stays valid.
        """
        if not self.flush(wait_s):
            return False
        tmp = self.ckpt_path.with_suffix(".tmp.npz")
        np.savez(tmp, positions=positions, velocities=velocities, boxes=boxes,
                 state_of_replica=np.asarray(state_of_replica), iteration=int(iteration),
                 extra=json.dumps(extra or {}))
        os.replace(tmp, self.ckpt_path)
        return True

    def read_checkpoint(self) -> dict | None:
        if not self.ckpt_path.exists():
            return None
        with np.load(self.ckpt_path, allow_pickle=False) as data:
            out = {k: data[k] for k in ("positions", "velocities", "boxes", "state_of_replica")}
            out["iteration"] = int(data["iteration"])
            out["extra"] = json.loads(str(data["extra"]))
        return out

    def close(self) -> None:
        """Store what is in memory. (Iterations after the last checkpoint are dropped by a run that
        resumes; a leg that is only read afterwards has them.)"""
        try:
            self.flush()
        except Exception:  # noqa: BLE001
            pass


def equilibration_boundary(attrs: dict, leg_dir: str | Path | None = None) -> int | None:
    """The number of equilibration iterations of the samples with these attributes, or None if
    the file does not tell.

    The run records its plan before its first iteration (``n_planned_iterations``). Earlier
    versions wrote the boundary only when a run had finished; until then it read 0. For such a
    file the number is taken as written if it is above 0 or if the leg has a result, and is
    unknown otherwise.
    """
    n = int(attrs.get("n_equilibration_iterations", 0))
    if "n_planned_iterations" in attrs or n > 0:
        return n
    if leg_dir is not None and any((Path(leg_dir) / name).exists() for name in ("result.json", "result.previous.json")):
        return n
    return None


def decoded_attributes(h5) -> dict:
    """The attributes of a samples file as Python values (text that is JSON is decoded)."""
    out = {}
    for key, value in h5.attrs.items():
        if isinstance(value, (bytes, str)):
            text = value.decode() if isinstance(value, bytes) else value
            try:
                out[key] = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                out[key] = text
        else:
            out[key] = value.item() if hasattr(value, "item") else value
    return out


def stored_attributes(directory: str | Path) -> tuple[int, dict]:
    """(number of iterations, attributes) of the samples in ``directory``, read without changing
    anything; (0, {}) if there are no samples."""
    path = Path(directory) / "samples.h5"
    if not path.exists():
        return 0, {}
    with open_samples(path) as h5:
        return int(h5["u"].shape[0]), decoded_attributes(h5)


def read_samples(directory: str | Path) -> dict:
    """Load everything analysis needs from a leg directory.

    Also while the leg is being sampled: the file then holds the iterations up to the last
    checkpoint (:class:`LegStorage`). Should a file hold its datasets to different lengths (a
    process killed in the middle of storing a checkpoint), the iterations that all of them hold
    are returned.
    """
    path = Path(directory) / "samples.h5"
    with open_samples(path) as h5:
        out = {name: h5[name][:] for name in ("u", "states", "volume", "lrc_iter", "lrc_du", "frame_iter")}
        for name in ("frames", "frame_box"):
            if name in h5:
                out[name] = h5[name][:]
        out["attrs"] = decoded_attributes(h5)
    n = min(len(out[name]) for name in ("u", "states", "volume"))
    for name in ("u", "states", "volume"):
        out[name] = out[name][:n]
    for iterations, data in (("lrc_iter", ("lrc_du",)), ("frame_iter", ("frames", "frame_box"))):
        present = [name for name in data if name in out]
        keep = min([len(out[iterations])] + [len(out[name]) for name in present])
        keep = int(np.sum(out[iterations][:keep] < n))
        for name in (iterations, *present):
            out[name] = out[name][:keep]
    return out
