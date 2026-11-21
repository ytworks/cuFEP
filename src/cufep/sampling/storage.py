"""Storage for one leg: energies in HDF5 parts that are written once and named by an atomically
replaced manifest, restart state in an atomically written NPZ."""
from __future__ import annotations

import json
import os
import time
from numbers import Integral
from pathlib import Path
from typing import NamedTuple

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
    if isinstance(a, Integral) and isinstance(b, Integral):
        # a seed, a number of frames or of windows: no tolerance (review of eb6f89f: the seeds 1000000000
        # and 1000000001 were the same seed, and trajectories of both were used together)
        return int(a) == int(b)
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
#: rows that only samples of later versions have: proposals and acceptances of exchange between the
#: replicas in neighbouring states, per iteration ((states - 1) x [proposed, accepted]); samples in
#: which any part lacks them have none (no part is refused for it)
OPTIONAL_ROW_DATASETS = ("swaps",)
TAIL_DATASETS = ("lrc_iter", "lrc_du")
FRAME_DATASETS = ("frame_iter", "frames", "frame_box")
#: the dataset in which samples of the previous version say how much of every dataset is stored
#: (:func:`committed_counts`); this version writes it once, with zeros, in the empty samples.h5 of
#: a new leg, so that an earlier version reads such a leg as holding no samples
COUNTS = "committed_counts"
#: the attribute in which one earlier version kept them (h5py replaces an attribute by deleting it
#: and creating it anew, so a failure in between lost them; review of PR 1)
COMMITTED = "committed"


def committed_counts(h5) -> tuple[int, int, int]:
    """(iterations, tail records, frame records) that a samples file of an earlier version holds
    as stored: the samples of a leg without a manifest (:func:`stored_layout`).

    The previous version wrote the datasets first and then these counts, so the file holds one
    state of the run up to them; every reader takes this many and no more. A file of a version
    before it has no counts: it holds the iterations that all three row datasets hold, and the
    records of those iterations that all datasets of the records hold. A count that a dataset is
    shorter than is a damaged file and raises StorageError.
    """
    lengths = {name: int(h5[name].shape[0]) for name in ROW_DATASETS + TAIL_DATASETS + FRAME_DATASETS if name in h5}
    stored = None
    if COUNTS in h5:
        stored = [int(x) for x in h5[COUNTS][:3]]
    elif COMMITTED in h5.attrs:
        raw = h5.attrs[COMMITTED]
        counts = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        stored = [int(counts["iterations"]), int(counts["tail_records"]), int(counts["frame_records"])]
    if stored is not None:
        rows, tail, frames = stored
        short = [name for names, need in ((ROW_DATASETS, rows), (TAIL_DATASETS, tail), (FRAME_DATASETS, frames))
                 for name in names if name in lengths and lengths[name] < need]
        if short:
            raise StorageError(
                f"{h5.filename}: the datasets {', '.join(short)} are shorter than the {rows} iterations, "
                f"{tail} tail records and {frames} frame records the file says it holds; it is damaged"
            )
        return rows, tail, frames
    rows = min(lengths[name] for name in ROW_DATASETS)

    def records(names) -> int:
        present = [name for name in names if name in lengths]
        keep = min(lengths[name] for name in present)
        return int(np.sum(h5[names[0]][:keep] < rows))

    return rows, records(TAIL_DATASETS), records(FRAME_DATASETS)


#: conditions that versions before they were recorded always had: their value for samples that do
#: not record them
CONDITIONS_OF_EARLIER_VERSIONS = {"exchange": "hrex"}


def conditions_differences(stored: dict, current: dict) -> list[str]:
    """What differs between the conditions stored with samples and those of a run, in words.

    Everything the samples record is compared; what they do not record (a key of a later
    version) is not.
    """
    differing = []
    stored = dict(stored)
    for key, earlier in CONDITIONS_OF_EARLIER_VERSIONS.items():
        # what every version before the key was recorded did: samples of hrex were stored without it,
        # and a run with exchange: none must not take them as its own (review of PR 1)
        if key in current and key not in stored:
            stored[key] = earlier
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


# ---------------------------------------------------------------------------------- the layout
#: the file that names the stored parts of the samples of a leg, with their counts, and holds the
#: attributes of the samples
MANIFEST = "manifest.json"
#: the parts that each write adds: written once under another name, made durable, renamed, and
#: never opened for writing again
SEGMENT_PATTERN = "segment_{:06d}.h5"
#: the suffix of a file while it is written; such a file is never read
WRITING = ".writing"


# one way to make a file or a directory durable, for the samples and for every other file (cufep.files)
from cufep.files import _NO_DIRECTORY_SYNC  # noqa: E402,F401
from cufep.files import fsync_path as _fsync_path  # noqa: E402


def _replace(source: Path, target: Path) -> None:
    """Put ``source`` in the place of ``target`` in one step: the rename every write of the storage
    ends with."""
    os.replace(source, target)


def _write_manifest(directory: Path, manifest: dict) -> None:
    """Write the manifest of a leg in one step: the whole file under another name, made durable,
    then renamed over the old one. A reader sees the old manifest or the new one, never a part."""
    from cufep import jsonio

    path = Path(directory) / MANIFEST
    part = path.with_name(MANIFEST + WRITING)
    with open(part, "w") as fh:
        fh.write(jsonio.dumps(manifest))
        fh.flush()
        os.fsync(fh.fileno())
    _replace(part, path)
    _fsync_path(Path(directory))


def read_manifest(directory: str | Path) -> dict | None:
    """The manifest of the samples in ``directory``, or None for samples of an earlier version
    (one samples.h5 that holds everything)."""
    path = Path(directory) / MANIFEST
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text())
        [(str(p["file"]), int(p["rows"]), int(p["tail"]), int(p["frames"])) for p in manifest["parts"]]
    except (ValueError, KeyError, TypeError) as exc:
        raise StorageError(f"{path} cannot be read ({type(exc).__name__}: {exc}); it is damaged") from exc
    return manifest


def stored_layout(directory: str | Path) -> tuple[list[tuple[Path, int, int, int]], dict]:
    """The stored parts of the samples in ``directory`` and their attributes, as every reader takes
    them: [(file, iterations, tail records, frame records), ...] in order, and the attributes.
    ([], {}) if there are no samples. Samples of an earlier version are one part, samples.h5,
    counted by :func:`committed_counts`, with the attributes of that file."""
    d = Path(directory)
    manifest = read_manifest(d)
    if manifest is None:
        parts = sorted(p.name for p in d.glob("segment_*.h5"))
        if parts:
            raise StorageError(
                f"{d} holds parts of samples ({', '.join(parts[:3])}{' ...' if len(parts) > 3 else ''}) but no "
                f"{MANIFEST} that names them; the leg is damaged and nothing was changed. If its {MANIFEST} was "
                "moved away, put it back; if the run of an earlier version stopped at its first write, these parts "
                "were never stored and can be moved away")
        path = d / "samples.h5"
        if not path.exists():
            return [], {}
        with open_samples(path) as h5:
            counts, attrs = committed_counts(h5), decoded_attributes(h5)
        return [(path, *counts)], attrs
    parts = [(d / str(p["file"]), int(p["rows"]), int(p["tail"]), int(p["frames"])) for p in manifest["parts"]]
    missing = [path.name for path, *_counts in parts if not path.exists()]
    if missing:
        raise StorageError(f"{d / MANIFEST} names parts of the samples that are not there ({', '.join(missing)}); "
                           "the leg is damaged")
    return parts, dict(manifest.get("attrs") or {})


def _manifest_text(directory: str | Path) -> str | None:
    try:
        return (Path(directory) / MANIFEST).read_text()
    except OSError:
        return None


def _read(directory: str | Path, names: tuple[str, ...]) -> dict:
    """The datasets ``names`` of the samples in ``directory``: each part to its count, concatenated.
    A dataset that a part holds fewer of than its count, or that a part with records lacks, is a
    damaged leg (StorageError); one that no part has is left out.

    A writer that opens the leg removes the parts that its manifest no longer names; a reader that
    read the manifest before and finds a part gone reads the new manifest and starts again."""
    for attempt in range(3):
        before = _manifest_text(directory)
        try:
            return _read_once(directory, names)
        except (StorageError, OSError):
            if attempt == 2 or _manifest_text(directory) == before:
                raise
    raise AssertionError("unreachable")


def _read_once(directory: str | Path, names: tuple[str, ...]) -> dict:
    parts, attrs = stored_layout(directory)
    if not parts:
        raise StorageError(f"{directory}: no samples")
    count_of = {**{name: 1 for name in ROW_DATASETS + OPTIONAL_ROW_DATASETS}, **{name: 2 for name in TAIL_DATASETS},
                **{name: 3 for name in FRAME_DATASETS}}
    pieces: dict[str, list] = {name: [] for name in names}
    lacking: dict[str, int] = {name: 0 for name in names}
    for part in parts:
        path = part[0]
        with open_samples(path) as h5:
            for name in names:
                need = part[count_of[name]]
                if name not in h5:
                    lacking[name] += need
                    continue
                if h5[name].shape[0] < need:
                    raise StorageError(f"{path}: {name} holds {h5[name].shape[0]} entries, fewer than the {need} "
                                       "the samples say it holds; the leg is damaged")
                pieces[name].append(h5[name][:need])
    out = {}
    for name in names:
        if name in OPTIONAL_ROW_DATASETS and lacking[name]:
            continue                                    # some parts are of an earlier version: none
        if pieces[name] and lacking[name]:
            raise StorageError(f"{directory}: {lacking[name]} entries of {name} are in parts that do not have it; "
                               "the leg is damaged")
        if pieces[name]:
            out[name] = np.concatenate(pieces[name])
    out["attrs"] = attrs
    return out


def stored_conditions(directory: str | Path) -> tuple[int, dict | None]:
    """(number of iterations, conditions) of the samples in ``directory``, read without changing
    anything; (0, None) if there are no samples, (n, None) if their conditions are not recorded."""
    parts, attrs = stored_layout(directory)
    n = sum(rows for _path, rows, _tail, _frames in parts)
    conditions = attrs.get("conditions")
    return (n, conditions) if n and isinstance(conditions, dict) else (n, None)


def stored_backend(directory: str | Path) -> str | None:
    """Backend with which the samples in ``directory`` were made, or None if there are none."""
    parts, attrs = stored_layout(directory)
    if not sum(rows for _path, rows, _tail, _frames in parts) or "backend" not in attrs:
        return None
    return str(attrs["backend"])


def stored_attributes(directory: str | Path) -> tuple[int, dict]:
    """(number of iterations, attributes) of the samples in ``directory``, read without changing
    anything; (0, {}) if there are no samples."""
    parts, attrs = stored_layout(directory)
    return sum(rows for _path, rows, _tail, _frames in parts), attrs


def stored_tail_iterations(directory: str | Path) -> np.ndarray:
    """The iterations at which the samples in ``directory`` hold a tail energy (stored ones only)."""
    if not stored_layout(directory)[0]:
        return np.zeros(0, dtype=int)
    return np.asarray(_read(directory, ("lrc_iter",)).get("lrc_iter", np.zeros(0)), dtype=int)


def read_samples(directory: str | Path) -> dict:
    """Load everything analysis needs from a leg directory: the stored parts, each to its count
    (:func:`stored_layout`), also while the leg is being sampled (then up to its last checkpoint)
    and after a write that failed or a writer that was killed at any point."""
    return _read(directory, ROW_DATASETS + OPTIONAL_ROW_DATASETS + TAIL_DATASETS + FRAME_DATASETS)


class _Memory(NamedTuple):
    """What a LegStorage holds in memory: the rows of the iterations not stored yet, their records,
    and the part a write was storing them to with how many of each it took (None if no write is
    under way). Replaced as one object, so that an interrupt (Ctrl-C) leaves either the state before
    or the state after, never a part of each."""
    rows: list
    records: list
    pending: tuple | None


def _segment_number(path: Path) -> int | None:
    name = path.name.split(".")[0]
    if not name.startswith("segment_"):
        return None
    try:
        return int(name[len("segment_"):])
    except ValueError:
        return None


class LegStorage:
    """The samples of one leg and the state to restart it from.

    ``samples.h5`` and ``segment_<n>.h5``, the parts of the samples
        u        (n_iter, n_replicas, n_states)  reduced potentials of each replica at every state
        states   (n_iter, n_replicas)            state each replica was simulated in
        volume   (n_iter, n_replicas)            box volume, nm^3
        lrc_iter, lrc_du                         coupled-state tail energies (kJ/mol), by row
        frame_iter, frames, frame_box            coupled-state coordinates of selected atoms, by row
    ``manifest.json``
        the parts that are stored, in order, each with its numbers of iterations, tail records and
        frame records, and the attributes of the samples (conditions, states, plan ...)
    ``checkpoint.npz``
        positions, velocities and boxes of every replica, the state permutation, the iteration,
        the number of stored rows and the number of starts of the leg.

    Stored data is never changed (review of PR 1: HDF5 rewrites and moves a compressed chunk of
    frames when rows are added to it, so the frames of the last checkpoint were lost when the
    writer was killed in the middle of the next write, whatever counts said). The rows of the
    iterations are kept in memory; at a checkpoint they go to a new part, ``segment_<n>.h5``,
    written whole under another name, made durable and renamed, and only then is the manifest,
    which names it, replaced in one step. Truncation replaces the manifest alone; parts it drops
    stay on the disk and their names are not used again. Whenever the writer is killed, the
    manifest names complete parts only, and a reader sees one state of the run. A reader never
    meets the writer in a file, since no part is opened for writing once it exists: a leg can be
    analysed while it is sampled, and nobody waits for anybody.

    ``samples.h5`` is the first part: empty for a leg of this version, all the samples of a leg
    of an earlier version, which is never written to again. Such a leg gets its manifest at its
    first write, with the counts (:func:`committed_counts`) and attributes of that file.
    """

    def __init__(self, directory: str | Path, n_replicas: int, n_states: int, n_frame_atoms: int = 0) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.h5_path = self.dir / "samples.h5"
        self.ckpt_path = self.dir / "checkpoint.npz"
        self.n_replicas, self.n_states, self.n_frame_atoms = n_replicas, n_states, n_frame_atoms
        #: iterations that are not stored yet: (u, states, volume) each, the records (row, tail energy
        #: and frame of the coupled state) that belong to them, and the write under way (_Memory)
        self._mem = _Memory([], [], None)
        if not self.h5_path.exists():
            # an empty samples.h5 is made only for a leg that has nothing else: a leg whose samples.h5 is
            # gone but that has a manifest or parts is damaged, and nothing is made or removed (review of PR 1)
            if (self.dir / MANIFEST).exists() or any(self.dir.glob("segment_*.h5")):
                raise StorageError(f"{self.dir} has a {MANIFEST} or parts of samples but no samples.h5; the leg is "
                                   "damaged and nothing was changed. Put samples.h5 back, or start the leg anew with "
                                   "--reschedule")
            part = self.h5_path.with_name(self.h5_path.name + WRITING)
            with h5py.File(part, "w") as h5:
                k, s = n_replicas, n_states
                h5.create_dataset("u", shape=(0, k, s), maxshape=(None, k, s), dtype="f8", chunks=(16, k, s))
                h5.create_dataset("states", shape=(0, k), maxshape=(None, k), dtype="i4", chunks=(256, k))
                h5.create_dataset("volume", shape=(0, k), maxshape=(None, k), dtype="f8", chunks=(256, k))
                h5.create_dataset("lrc_iter", shape=(0,), maxshape=(None,), dtype="i8", chunks=(1024,))
                h5.create_dataset("lrc_du", shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
                h5.create_dataset("frame_iter", shape=(0,), maxshape=(None,), dtype="i8", chunks=(256,))
                if n_frame_atoms:
                    h5.create_dataset("frames", shape=(0, n_frame_atoms, 3), maxshape=(None, n_frame_atoms, 3),
                                      dtype="f4", chunks=(8, n_frame_atoms, 3))
                    h5.create_dataset("frame_box", shape=(0, 3, 3), maxshape=(None, 3, 3), dtype="f4")
                h5.attrs["n_equilibration_iterations"] = 0
                h5.create_dataset(COUNTS, data=np.zeros(3, dtype="i8"))
            _fsync_path(part)
            _replace(part, self.h5_path)
            _write_manifest(
                self.dir,
                {
                    "format": 2,
                    "parts": [{"file": "samples.h5", "rows": 0, "tail": 0, "frames": 0}],
                    "attrs": {"n_equilibration_iterations": 0},
                },
            )
        with open_samples(self.h5_path) as h5:
            if h5["u"].shape[1:] != (n_replicas, n_states):
                raise ValueError(f"{self.h5_path} was written for a different number of replicas/states")
            self._has_frames = "frames" in h5
        self._load()                     # refuses parts that no manifest names (stored_layout)
        self._remove_unused()

    def _load(self) -> None:
        """What is stored: the parts and their counts (:func:`stored_layout`)."""
        self._parts = stored_layout(self.dir)[0]
        self._stored = sum(rows for _p, rows, _t, _f in self._parts)

    def _manifest(self) -> dict:
        """The manifest as it is now; for samples of an earlier version, the one they get at their
        first write: samples.h5 as their only part, with its counts and attributes."""
        manifest = read_manifest(self.dir)
        if manifest is None:
            from cufep import jsonio

            path, rows, tail, frames = self._parts[0]
            manifest = {"format": 2, "parts": [{"file": path.name, "rows": rows, "tail": tail, "frames": frames}],
                        "attrs": jsonio.clean(stored_layout(self.dir)[1])}
        return manifest

    def _commit(self, manifest: dict, what: str) -> None:
        """Replace the manifest by ``manifest``; StorageError, with the old one in place, if that fails."""
        try:
            _write_manifest(self.dir, manifest)
        except Exception as exc:  # noqa: BLE001  raised as what it means for the samples
            self._load()
            raise StorageError(f"{what} could not be stored in {self.dir} ({type(exc).__name__}: {exc}); the "
                               "samples are as they were") from exc
        self._load()

    def _set_attrs(self, changes: dict, remove: tuple = ()) -> None:
        from cufep import jsonio

        manifest = self._manifest()
        attrs = dict(manifest.get("attrs") or {})
        attrs.update(jsonio.clean(changes))
        for key in remove:
            attrs.pop(key, None)
        manifest["attrs"] = attrs
        self._commit(manifest, f"the attributes {', '.join(sorted(set(changes) | set(remove)))}")

    def attributes(self) -> dict:
        """The attributes of the samples as Python values."""
        return stored_layout(self.dir)[1]

    # -- conditions ---------------------------------------------------------------
    def check_conditions(self, conditions: dict) -> str:
        """Record the conditions of the samples, or refuse to continue them under other ones.

        Nothing is written if they differ: the samples and their metadata stay as they are.
        Returns "new", "same" or "unrecorded".

        "unrecorded": the leg holds samples of a version that did not record their conditions.
        Nothing on the disk tells what they were made with (the soft-core, the pressure and the
        friction were stored nowhere, and that version rewrote temperature, backend and states
        every time it opened the file), so they cannot be checked and nothing is written. Such
        samples are read and analysed; they are not continued (:meth:`refuse_unrecorded`).
        """
        if self.n_iterations == 0:
            self._set_attrs({"conditions": conditions})
            return "new"
        stored = self.attributes().get("conditions")
        if not isinstance(stored, dict):
            return "unrecorded"
        differing = conditions_differences(stored, conditions)
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
        self._set_attrs(kwargs)

    def set_plan(self, n_equilibration: int, n_total: int, analysis_record: dict | None = None) -> None:
        """Record, before the run, how many of its iterations are equilibration and how many it has,
        and the settings its result will be analysed with (``analysis_record``).

        It goes to the disk at once: an analysis of a run that was interrupted must not count
        equilibration as production, and must judge the leg by the levels of its run.
        """
        changes = {"n_equilibration_iterations": int(n_equilibration), "n_planned_iterations": int(n_total)}
        if analysis_record is not None:
            changes["analysis_record"] = analysis_record
        self._set_attrs(changes)

    def clear_plan(self) -> None:
        """A run that starts anew has no plan until :meth:`set_plan` gives it one."""
        self._set_attrs({"n_equilibration_iterations": 0}, remove=("n_planned_iterations",))

    @property
    def _rows(self) -> list:
        return self._mem.rows

    @property
    def _records(self) -> list:
        return self._mem.records

    def _settle(self) -> None:
        """If a write was under way and its part is named by the manifest (an interrupt came after it
        was stored), drop from memory what that part holds; then no write is under way."""
        mem = self._mem
        if mem.pending is None:
            return
        target, rows, records = mem.pending
        self._load()
        if any(path == target for path, *_counts in self._parts):
            self._mem = _Memory(mem.rows[rows:], mem.records[records:], None)
        else:
            self._mem = _Memory(mem.rows, mem.records, None)

    @property
    def n_iterations(self) -> int:
        """Iterations this storage holds: those stored and those still in memory."""
        self._settle()
        return self._stored + len(self._rows)

    @property
    def n_stored(self) -> int:
        """Iterations stored (named by the manifest)."""
        return self._stored

    # -- appending ------------------------------------------------------------------
    def append(self, u: np.ndarray, states: np.ndarray, volume: np.ndarray, swaps: np.ndarray | None = None) -> int:
        n = self.n_iterations
        self._mem.rows.append((np.array(u, dtype="f8"), np.array(states, dtype="i4"), np.array(volume, dtype="f8"),
                               None if swaps is None else np.array(swaps, dtype="i8")))
        return n

    def append_record(self, iteration: int, record: dict) -> None:
        kept = {}
        if "lrc_du" in record:
            kept["lrc_du"] = float(record["lrc_du"])
        if "frame" in record and self.n_frame_atoms and self._has_frames:
            kept["frame"] = np.array(record["frame"], dtype="f4")
            kept["box"] = np.array(record["box"], dtype="f4")
        if kept:
            self._mem.records.append((int(iteration), kept))

    def _next_segment(self) -> Path:
        """A name no manifest has named: one more than the number of parts made (which the manifest
        keeps, so that a part dropped by a truncation and then removed does not give its name again)
        and than any number in the directory (a part left by a write that failed)."""
        manifest = read_manifest(self.dir) or {}
        numbers = [n for n in (_segment_number(p) for p in self.dir.glob("segment_*")) if n is not None]
        return self.dir / SEGMENT_PATTERN.format(max(numbers + [int(manifest.get("segments_made", 0))]) + 1)

    def _remove_unused(self) -> None:
        """Remove what the manifest does not name: parts that a truncation dropped and what writes that
        failed or were killed left (review of PR 1: they stayed on the disk). Only the writer, when it
        opens the leg, does this; a reader that meets a part gone reads the manifest again (_read)."""
        manifest = read_manifest(self.dir)
        named = set() if manifest is None else {str(part["file"]) for part in manifest["parts"]}
        # without a manifest there are no parts (stored_layout refuses them): only what writes left
        leftovers = list(self.dir.glob("segment_*" + WRITING)) if manifest is None else list(self.dir.glob("segment_*"))
        for path in [*leftovers, self.dir / (MANIFEST + WRITING), self.dir / ("samples.h5" + WRITING)]:
            if path.name not in named and path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

    def _write_segment(self, target: Path, tail: list, frames: list) -> None:
        """Write the iterations in memory and ``tail`` and ``frames`` records to ``target`` in one step."""
        part = target.with_name(target.name + WRITING)
        k, s, m = self.n_replicas, self.n_states, len(self._rows)
        with h5py.File(part, "w") as h5:
            h5.create_dataset("u", data=np.stack([row[0] for row in self._rows]) if m else np.zeros((0, k, s)))
            h5.create_dataset("states", data=np.stack([row[1] for row in self._rows]) if m else
                              np.zeros((0, k), dtype="i4"))
            h5.create_dataset("volume", data=np.stack([row[2] for row in self._rows]) if m else np.zeros((0, k)))
            if all(len(row) > 3 and row[3] is not None for row in self._rows):
                h5.create_dataset("swaps", data=np.stack([row[3] for row in self._rows]) if m else
                                  np.zeros((0, max(0, s - 1), 2), dtype="i8"))
            h5.create_dataset("lrc_iter", data=np.array([it for it, _ in tail], dtype="i8"))
            h5.create_dataset("lrc_du", data=np.array([du for _, du in tail], dtype="f8"))
            h5.create_dataset("frame_iter", data=np.array([it for it, _ in frames], dtype="i8"))
            if self._has_frames:
                if frames:
                    h5.create_dataset("frames", data=np.stack([r["frame"] for _, r in frames]), compression="gzip",
                                      compression_opts=1)
                    h5.create_dataset("frame_box", data=np.stack([r["box"] for _, r in frames]))
                else:
                    h5.create_dataset("frames", data=np.zeros((0, self.n_frame_atoms, 3), dtype="f4"))
                    h5.create_dataset("frame_box", data=np.zeros((0, 3, 3), dtype="f4"))
            h5.attrs["first_row"] = int(self._stored)
        _fsync_path(part)
        _replace(part, target)
        _fsync_path(self.dir)

    def flush(self, wait_s: float = WRITE_WAIT_S) -> bool:
        """Store the iterations that are in memory as a new part (see the class); returns True.

        A failure raises StorageError with what is stored unchanged and everything still in
        memory; calling it again writes them again, as a part of another name. ``wait_s`` is not
        used: no file is shared with a reader, so nothing has to be waited for.
        """
        self._settle()                   # what an interrupted write stored is not stored again
        if not self._rows and not self._records:
            return True
        tail = [(it, record["lrc_du"]) for it, record in self._records if "lrc_du" in record]
        frames = [(it, record) for it, record in self._records if "frame" in record]
        target = self._next_segment()
        mem = self._mem
        self._mem = _Memory(mem.rows, mem.records, (target, len(mem.rows), len(mem.records)))
        try:
            if read_manifest(self.dir) is None:
                # samples of an earlier version get their manifest before their first part, so that no part
                # exists that no manifest names (stored_layout refuses such parts as damage)
                _write_manifest(self.dir, self._manifest())
            self._write_segment(target, tail, frames)
            manifest = self._manifest()
            manifest["parts"] = list(manifest["parts"]) + [
                {"file": target.name, "rows": len(self._rows), "tail": len(tail), "frames": len(frames)}]
            manifest["segments_made"] = max(int(manifest.get("segments_made", 0)), _segment_number(target) or 0)
            _write_manifest(self.dir, manifest)             # from here on the part is stored
        except BaseException as exc:  # noqa: BLE001  an error or an interrupt (Ctrl-C), at any point
            # what is stored is what the manifest names: if it names the new part, the iterations are
            # stored and leave the memory, whatever ended the write (review of PR 1: an interrupt after
            # the manifest was replaced left them in memory, and the close stored them twice)
            self._settle()
            if any(path == target for path, *_counts in self._parts) or not isinstance(exc, Exception):
                if isinstance(exc, Exception):
                    return True
                raise
            raise StorageError(f"the {len(self._rows)} iteration(s) after iteration {self._stored} could not be "
                               f"stored in {self.dir} ({type(exc).__name__}: {exc}); the {self._stored} before "
                               "them are stored, and these are kept in memory") from exc
        self._settle()                   # the part is named: what it holds leaves the memory
        return True

    def truncate(self, n_iterations: int) -> None:
        """Drop everything after ``n_iterations`` iterations, in memory and in what is stored: the
        manifest is replaced by one that names that many rows and the records of those rows; no
        part is changed."""
        n_iterations = int(n_iterations)
        self._settle()
        mem = self._mem
        self._mem = _Memory(mem.rows[:max(0, n_iterations - self._stored)],
                            [(it, record) for it, record in mem.records if it < n_iterations], None)
        self._load()                                  # the parts and the manifest read together, now
        if n_iterations >= self._stored:
            return
        manifest, kept, start = self._manifest(), [], 0
        for entry, (path, rows, tail, frames) in zip(manifest["parts"], self._parts):
            if start >= n_iterations:
                break
            if start + rows > n_iterations:
                with open_samples(path) as h5:
                    tail = int(np.sum(h5["lrc_iter"][:tail] < n_iterations))
                    frames = int(np.sum(h5["frame_iter"][:frames] < n_iterations))
                rows = n_iterations - start
            kept.append(dict(entry, rows=rows, tail=tail, frames=frames))
            start += rows
        if not kept:            # nothing is left: the first part stays, empty, so that readers find 0 rows
            kept = [dict(manifest["parts"][0], rows=0, tail=0, frames=0)]
        manifest["parts"] = kept
        self._commit(manifest, f"the cut to {n_iterations} iterations")

    # -- checkpoint -------------------------------------------------------------------
    def write_checkpoint(self, positions, velocities, boxes, state_of_replica, iteration: int,
                         extra: dict | None = None, wait_s: float = WRITE_WAIT_S) -> bool:
        """Store the iterations that are in memory, then the state to restart from; returns True.

        In that order: a checkpoint never names an iteration whose samples are not stored. A
        failure raises (StorageError) with the earlier checkpoint and samples as they were."""
        self.flush(wait_s)
        tmp = self.ckpt_path.with_suffix(".tmp.npz")
        try:
            np.savez(tmp, positions=positions, velocities=velocities, boxes=boxes,
                     state_of_replica=np.asarray(state_of_replica), iteration=int(iteration),
                     extra=json.dumps(extra or {}))
            _fsync_path(tmp)
        except Exception as exc:  # noqa: BLE001  the checkpoint before it stays
            try:
                tmp.unlink()
            except OSError:
                pass
            raise StorageError(f"the checkpoint at iteration {iteration} could not be written in {self.dir} "
                               f"({type(exc).__name__}: {exc}); the one before it is kept, with the samples it "
                               "names") from exc
        _replace(tmp, self.ckpt_path)
        try:
            _fsync_path(self.dir)
        except OSError as exc:
            raise StorageError(f"the checkpoint at iteration {iteration} was written in {self.dir}, but the directory "
                               f"could not be made durable ({type(exc).__name__}: {exc})") from exc
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
        so; what was stored is unchanged, and the rows stay in memory, so that the caller can try
        again.
        """
        self.flush(wait_s)
