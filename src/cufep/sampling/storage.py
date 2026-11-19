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


class StorageError(RuntimeError):
    """The samples of a leg could not be stored or read as they must be. Not a failure of the
    backend: the samples and the checkpoint that were stored stay as they are."""


class StorageBusyError(StorageError):
    """Another process held the samples file for longer than the run waits."""


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


#: the datasets of the samples file: one row per iteration, and the records of some iterations
#: (the tail energy and the frame of the coupled state), which carry their iteration
ROW_DATASETS = ("u", "states", "volume")
TAIL_DATASETS = ("lrc_iter", "lrc_du")
FRAME_DATASETS = ("frame_iter", "frames", "frame_box")
#: the attribute that says how much of every dataset is stored (:func:`committed_counts`)
COMMITTED = "committed"


def committed_counts(h5) -> tuple[int, int, int]:
    """(iterations, tail records, frame records) that an open samples file holds as stored.

    :class:`LegStorage` writes the datasets first and then this count, in one attribute, so the
    file holds one state of the run up to the count, whatever happened to a write after it; every
    reader takes this many and no more. A file of an earlier version has no such attribute: it
    holds the iterations that all three row datasets hold, and the records of those iterations
    that all datasets of the records hold. A count that a dataset is shorter than is a damaged
    file and raises StorageError.
    """
    lengths = {name: int(h5[name].shape[0]) for name in ROW_DATASETS + TAIL_DATASETS + FRAME_DATASETS if name in h5}
    if COMMITTED in h5.attrs:
        raw = h5.attrs[COMMITTED]
        counts = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        rows, tail, frames = int(counts["iterations"]), int(counts["tail_records"]), int(counts["frame_records"])
        short = [name for names, need in ((ROW_DATASETS, rows), (TAIL_DATASETS, tail), (FRAME_DATASETS, frames))
                 for name in names if name in lengths and lengths[name] < need]
        if short:
            raise StorageError(f"{h5.filename}: the datasets {', '.join(short)} are shorter than the {rows} iterations, "
                               f"{tail} tail records and {frames} frame records the file says it holds; it is damaged")
        return rows, tail, frames
    rows = min(lengths[name] for name in ROW_DATASETS)

    def records(names) -> int:
        present = [name for name in names if name in lengths]
        keep = min(lengths[name] for name in present)
        return int(np.sum(h5[names[0]][:keep] < rows))

    return rows, records(TAIL_DATASETS), records(FRAME_DATASETS)


def stored_conditions(directory: str | Path) -> tuple[int, dict | None]:
    """(number of iterations, conditions) of the samples in ``directory``, read without changing
    anything; (0, None) if there are no samples, (n, None) if their conditions are not recorded."""
    path = Path(directory) / "samples.h5"
    if not path.exists():
        return 0, None
    with open_samples(path) as h5:
        n = committed_counts(h5)[0]
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
        if committed_counts(h5)[0] == 0 or "backend" not in h5.attrs:
            return None
        return str(h5.attrs["backend"])


def _write_at(dataset, offset: int, values) -> None:
    """Make ``dataset`` hold ``values`` from row ``offset`` on and nothing after them."""
    dataset.resize(offset + len(values), axis=0)
    if len(values):
        dataset[offset:offset + len(values)] = np.stack([np.asarray(v) for v in values])


def _publish(h5, rows: int, tail: int, frames: int) -> None:
    """Say how much of every dataset is stored (:func:`committed_counts`): one attribute, written
    after the data it counts."""
    h5.attrs[COMMITTED] = json.dumps({"iterations": int(rows), "tail_records": int(tail), "frame_records": int(frames)})


class LegStorage:
    """The samples of one leg and the state to restart it from.

    ``samples.h5``
        u        (n_iter, n_replicas, n_states)  reduced potentials of each replica at every state
        states   (n_iter, n_replicas)            state each replica was simulated in
        volume   (n_iter, n_replicas)            box volume, nm^3
        lrc_iter, lrc_du                         coupled-state tail energies (kJ/mol)
        frame_iter, frames, frame_box            coupled-state coordinates of selected atoms
        committed (attribute)                    how many rows and records are stored
    ``checkpoint.npz``
        positions, velocities and boxes of every replica, the state permutation and the
        number of committed iterations.

    The rows of the iterations are kept in memory and go to ``samples.h5`` when a checkpoint
    is written, and when the storage is closed; in between the file is closed. Another process
    can therefore read the samples of a leg that is being sampled (``cufep analyze``).

    A write is a transaction. Every dataset is written from the stored count on (overwriting
    whatever a failed write left there), and only then is the new count published
    (:func:`committed_counts`), which every reader goes by. A write that fails at any point
    leaves the file at its earlier state for every reader, keeps the rows in memory, and the
    next write starts again from the same place: nothing is stored twice and nothing that was
    not written is read. Truncation publishes the smaller count first and shortens the
    datasets after it. (HDF5's single-writer-multiple-reader mode was not taken: a writer that
    is killed leaves such a file unreadable until it is repaired, attributes cannot be added in
    that mode, and files of earlier versions could not use it.)
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
                _publish(h5, 0, 0, 0)
        with open_samples(self.h5_path) as h5:
            if h5["u"].shape[1:] != (n_replicas, n_states):
                raise ValueError(f"{self.h5_path} was written for a different number of replicas/states")
            #: what the file holds as stored: iterations, tail records, frame records
            self._stored, self._stored_tail, self._stored_frames = committed_counts(h5)
            self._has_frames = "frames" in h5

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
        """Iterations this storage holds: those stored in the file and those still in memory."""
        return self._stored + len(self._rows)

    @property
    def n_stored(self) -> int:
        """Iterations stored in the file (:func:`committed_counts`)."""
        return self._stored

    # -- appending ------------------------------------------------------------------
    def append(self, u: np.ndarray, states: np.ndarray, volume: np.ndarray) -> int:
        n = self.n_iterations
        self._rows.append((np.array(u, dtype="f8"), np.array(states, dtype="i4"), np.array(volume, dtype="f8")))
        return n

    def append_record(self, iteration: int, record: dict) -> None:
        kept = {}
        if "lrc_du" in record:
            kept["lrc_du"] = float(record["lrc_du"])
        if "frame" in record and self.n_frame_atoms and self._has_frames:
            kept["frame"] = np.array(record["frame"], dtype="f4")
            kept["box"] = np.array(record["box"], dtype="f4")
        if kept:
            self._records.append((int(iteration), kept))

    def flush(self, wait_s: float = WRITE_WAIT_S) -> bool:
        """Store the iterations that are in memory: a transaction (see the class).

        False, with nothing written and everything still in memory, if another process held the
        file for all of ``wait_s``. Any other failure is raised with the file at its earlier
        state and everything still in memory; calling it again repeats the same write.
        """
        if not self._rows and not self._records:
            return True
        try:
            h5 = self._file_for_writing(wait_s)
        except OSError as exc:
            if _held_by_another_process(exc):
                return False
            raise
        tail = [(it, record["lrc_du"]) for it, record in self._records if "lrc_du" in record]
        frames = [(it, record) for it, record in self._records if "frame" in record]
        rows = self._stored + len(self._rows)
        n_tail, n_frames = self._stored_tail + len(tail), self._stored_frames + len(frames)
        with h5:
            for column, name in enumerate(ROW_DATASETS):
                _write_at(h5[name], self._stored, [row[column] for row in self._rows])
            _write_at(h5["lrc_iter"], self._stored_tail, [it for it, _ in tail])
            _write_at(h5["lrc_du"], self._stored_tail, [du for _, du in tail])
            _write_at(h5["frame_iter"], self._stored_frames, [it for it, _ in frames])
            if self._has_frames:
                _write_at(h5["frames"], self._stored_frames, [record["frame"] for _, record in frames])
                _write_at(h5["frame_box"], self._stored_frames, [record["box"] for _, record in frames])
            _publish(h5, rows, n_tail, n_frames)
            h5.flush()
        self._stored, self._stored_tail, self._stored_frames = rows, n_tail, n_frames
        self._rows.clear()
        self._records.clear()
        return True

    def truncate(self, n_iterations: int) -> None:
        """Drop everything after ``n_iterations`` iterations, in memory and in the file: the
        smaller count is published first, the datasets are shortened after it."""
        n_iterations = int(n_iterations)
        del self._rows[max(0, n_iterations - self._stored):]
        self._records = [(it, record) for it, record in self._records if it < n_iterations]
        with open_samples(self.h5_path) as h5:
            tail_its = h5["lrc_iter"][:self._stored_tail]
            frame_its = h5["frame_iter"][:self._stored_frames]
            lengths = {name: int(h5[name].shape[0]) for name in ROW_DATASETS + TAIL_DATASETS + FRAME_DATASETS
                       if name in h5}
            published = COMMITTED in h5.attrs
        rows = min(self._stored, n_iterations)
        tail, frames = int(np.sum(tail_its < n_iterations)), int(np.sum(frame_its < n_iterations))
        wanted = {**{name: rows for name in ROW_DATASETS}, **{name: tail for name in TAIL_DATASETS},
                  **{name: frames for name in FRAME_DATASETS}}
        if published and (rows, tail, frames) == (self._stored, self._stored_tail, self._stored_frames) and all(
                lengths[name] == wanted[name] for name in lengths):
            return                                         # nothing to cut: the file is not opened for writing
        with self._file_for_writing() as h5:
            _publish(h5, rows, tail, frames)
            h5.flush()
            self._stored, self._stored_tail, self._stored_frames = rows, tail, frames
            for name in lengths:
                if h5[name].shape[0] != wanted[name]:
                    h5[name].resize(wanted[name], axis=0)
            h5.flush()

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

    def close(self, wait_s: float = CHECKPOINT_WAIT_S) -> None:
        """Store what is in memory.

        These are the iterations after the last checkpoint (a run that resumes drops them; a
        leg that is only read afterwards has them). If they cannot be stored, StorageError says
        so (StorageBusyError if another process held the file for ``wait_s``); the file keeps
        what it held, and the rows stay in memory, so that the caller can try again.
        """
        if not self._rows and not self._records:
            return
        pending, stored = len(self._rows), self._stored
        try:
            written = self.flush(wait_s)
        except Exception as exc:  # noqa: BLE001  said, and raised as what it means for the samples
            raise StorageError(f"the {pending} iteration(s) after iteration {stored} could not be stored in "
                               f"{self.h5_path} ({type(exc).__name__}: {exc}); the file holds the {stored} "
                               "before them") from exc
        if not written:
            raise StorageBusyError(f"the {pending} iteration(s) after iteration {stored} could not be stored: another "
                                   f"process held {self.h5_path} open for {wait_s:g} s; the file holds the {stored} "
                                   "before them")


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
        elif isinstance(value, np.ndarray) and value.size != 1:
            out[key] = value.tolist()
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
        return committed_counts(h5)[0], decoded_attributes(h5)


def stored_tail_iterations(directory: str | Path) -> np.ndarray:
    """The iterations at which the samples in ``directory`` hold a tail energy (stored ones only)."""
    path = Path(directory) / "samples.h5"
    if not path.exists():
        return np.zeros(0, dtype=int)
    with open_samples(path) as h5:
        return np.asarray(h5["lrc_iter"][:committed_counts(h5)[1]], dtype=int)


def read_samples(directory: str | Path) -> dict:
    """Load everything analysis needs from a leg directory: what the file holds as stored
    (:func:`committed_counts`), also while the leg is being sampled (then up to its last
    checkpoint) and after a write that failed half-way."""
    path = Path(directory) / "samples.h5"
    with open_samples(path) as h5:
        rows, tail, frames = committed_counts(h5)
        out = {name: h5[name][:rows] for name in ROW_DATASETS}
        out.update({name: h5[name][:tail] for name in TAIL_DATASETS})
        out.update({name: h5[name][:frames] for name in FRAME_DATASETS if name in h5})
        out["attrs"] = decoded_attributes(h5)
    return out
