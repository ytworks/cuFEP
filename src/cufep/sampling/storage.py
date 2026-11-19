"""Storage for one leg: energies in HDF5, restart state in an atomically written NPZ."""
from __future__ import annotations

import json
import os
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


def stored_conditions(directory: str | Path) -> tuple[int, dict | None]:
    """(number of iterations, conditions) of the samples in ``directory``, read without changing
    anything; (0, None) if there are no samples, (n, None) if their conditions are not recorded."""
    path = Path(directory) / "samples.h5"
    if not path.exists():
        return 0, None
    with h5py.File(path, "r") as h5:
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
    with h5py.File(path, "r") as h5:
        if h5["u"].shape[0] == 0 or "backend" not in h5.attrs:
            return None
        return str(h5.attrs["backend"])


class LegStorage:
    """Append-only sample storage with truncation for restarts.

    ``samples.h5``
        u        (n_iter, n_replicas, n_states)  reduced potentials of each replica at every state
        states   (n_iter, n_replicas)            state each replica was simulated in
        volume   (n_iter, n_replicas)            box volume, nm^3
        lrc_iter, lrc_du                         coupled-state tail energies (kJ/mol)
        frame_iter, frames, frame_box            coupled-state coordinates of selected atoms
    ``checkpoint.npz``
        positions, velocities and boxes of every replica, the state permutation and the
        number of committed iterations.
    """

    def __init__(self, directory: str | Path, n_replicas: int, n_states: int, n_frame_atoms: int = 0) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.h5_path = self.dir / "samples.h5"
        self.ckpt_path = self.dir / "checkpoint.npz"
        self.n_replicas, self.n_states, self.n_frame_atoms = n_replicas, n_states, n_frame_atoms
        new = not self.h5_path.exists()
        self.h5 = h5py.File(self.h5_path, "a")
        if new:
            k, s = n_replicas, n_states
            self.h5.create_dataset("u", shape=(0, k, s), maxshape=(None, k, s), dtype="f8", chunks=(16, k, s))
            self.h5.create_dataset("states", shape=(0, k), maxshape=(None, k), dtype="i4", chunks=(256, k))
            self.h5.create_dataset("volume", shape=(0, k), maxshape=(None, k), dtype="f8", chunks=(256, k))
            self.h5.create_dataset("lrc_iter", shape=(0,), maxshape=(None,), dtype="i8", chunks=(1024,))
            self.h5.create_dataset("lrc_du", shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
            self.h5.create_dataset("frame_iter", shape=(0,), maxshape=(None,), dtype="i8", chunks=(256,))
            if n_frame_atoms:
                self.h5.create_dataset("frames", shape=(0, n_frame_atoms, 3), maxshape=(None, n_frame_atoms, 3),
                                       dtype="f4", chunks=(8, n_frame_atoms, 3), compression="gzip",
                                       compression_opts=1)
                self.h5.create_dataset("frame_box", shape=(0, 3, 3), maxshape=(None, 3, 3), dtype="f4")
            self.h5.attrs["n_equilibration_iterations"] = 0
        elif self.h5["u"].shape[1:] != (n_replicas, n_states):
            raise ValueError(f"{self.h5_path} was written for a different number of replicas/states")

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
            self.h5.attrs["conditions"] = json.dumps(conditions)
            return "new"
        if "conditions" not in self.h5.attrs:
            return "unrecorded"
        differing = conditions_differences(json.loads(self.h5.attrs["conditions"]), conditions)
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
        for key, value in kwargs.items():
            self.h5.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value

    def set_plan(self, n_equilibration: int, n_total: int, analysis_record: dict | None = None) -> None:
        """Record, before the run, how many of its iterations are equilibration and how many it has,
        and the settings its result will be analysed with (``analysis_record``).

        It goes to the disk at once: an analysis of a run that was interrupted must not count
        equilibration as production, and must judge the leg by the levels of its run.
        """
        self.h5.attrs["n_equilibration_iterations"] = int(n_equilibration)
        self.h5.attrs["n_planned_iterations"] = int(n_total)
        if analysis_record is not None:
            self.h5.attrs["analysis_record"] = json.dumps(analysis_record)
        self.h5.flush()

    def clear_plan(self) -> None:
        """A run that starts anew has no plan until :meth:`set_plan` gives it one."""
        self.h5.attrs["n_equilibration_iterations"] = 0
        if "n_planned_iterations" in self.h5.attrs:
            del self.h5.attrs["n_planned_iterations"]

    @property
    def n_iterations(self) -> int:
        return int(self.h5["u"].shape[0])

    # -- appending ------------------------------------------------------------------
    def append(self, u: np.ndarray, states: np.ndarray, volume: np.ndarray) -> int:
        n = self.n_iterations
        for name, value in (("u", u), ("states", states), ("volume", volume)):
            ds = self.h5[name]
            ds.resize(n + 1, axis=0)
            ds[n] = value
        return n

    def append_record(self, iteration: int, record: dict) -> None:
        if "lrc_du" in record:
            n = self.h5["lrc_du"].shape[0]
            self.h5["lrc_iter"].resize(n + 1, axis=0); self.h5["lrc_iter"][n] = iteration
            self.h5["lrc_du"].resize(n + 1, axis=0); self.h5["lrc_du"][n] = record["lrc_du"]
        if "frame" in record and self.n_frame_atoms:
            n = self.h5["frames"].shape[0]
            self.h5["frame_iter"].resize(n + 1, axis=0); self.h5["frame_iter"][n] = iteration
            self.h5["frames"].resize(n + 1, axis=0); self.h5["frames"][n] = record["frame"]
            self.h5["frame_box"].resize(n + 1, axis=0); self.h5["frame_box"][n] = record["box"]

    def truncate(self, n_iterations: int) -> None:
        """Drop everything after ``n_iterations`` committed iterations."""
        for name in ("u", "states", "volume"):
            if self.h5[name].shape[0] > n_iterations:
                self.h5[name].resize(n_iterations, axis=0)
        for it_name, data in (("lrc_iter", ("lrc_du",)), ("frame_iter", ("frames", "frame_box"))):
            its = self.h5[it_name][:]
            keep = int(np.sum(its < n_iterations))
            if keep < len(its):
                self.h5[it_name].resize(keep, axis=0)
                for d in data:
                    if d in self.h5:
                        self.h5[d].resize(keep, axis=0)
        self.h5.flush()

    # -- checkpoint -------------------------------------------------------------------
    def write_checkpoint(self, positions, velocities, boxes, state_of_replica, iteration: int,
                         extra: dict | None = None) -> None:
        self.h5.flush()
        tmp = self.ckpt_path.with_suffix(".tmp.npz")
        np.savez(tmp, positions=positions, velocities=velocities, boxes=boxes,
                 state_of_replica=np.asarray(state_of_replica), iteration=int(iteration),
                 extra=json.dumps(extra or {}))
        os.replace(tmp, self.ckpt_path)

    def read_checkpoint(self) -> dict | None:
        if not self.ckpt_path.exists():
            return None
        with np.load(self.ckpt_path, allow_pickle=False) as data:
            out = {k: data[k] for k in ("positions", "velocities", "boxes", "state_of_replica")}
            out["iteration"] = int(data["iteration"])
            out["extra"] = json.loads(str(data["extra"]))
        return out

    def close(self) -> None:
        try:
            self.h5.flush()
            self.h5.close()
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
    with h5py.File(path, "r") as h5:
        return int(h5["u"].shape[0]), decoded_attributes(h5)


def read_samples(directory: str | Path) -> dict:
    """Load everything analysis needs from a leg directory."""
    path = Path(directory) / "samples.h5"
    with h5py.File(path, "r") as h5:
        out = {name: h5[name][:] for name in ("u", "states", "volume", "lrc_iter", "lrc_du", "frame_iter")}
        for name in ("frames", "frame_box"):
            if name in h5:
                out[name] = h5[name][:]
        out["attrs"] = decoded_attributes(h5)
    return out
