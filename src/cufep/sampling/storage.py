"""Storage for one leg: energies in HDF5, restart state in an atomically written NPZ."""
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np


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

    # -- metadata -----------------------------------------------------------------
    def set_meta(self, **kwargs) -> None:
        for key, value in kwargs.items():
            self.h5.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value

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


def read_samples(directory: str | Path) -> dict:
    """Load everything analysis needs from a leg directory."""
    path = Path(directory) / "samples.h5"
    with h5py.File(path, "r") as h5:
        out = {name: h5[name][:] for name in ("u", "states", "volume", "lrc_iter", "lrc_du", "frame_iter")}
        for name in ("frames", "frame_box"):
            if name in h5:
                out[name] = h5[name][:]
        out["attrs"] = {}
        for key, value in h5.attrs.items():
            if isinstance(value, (bytes, str)):
                text = value.decode() if isinstance(value, bytes) else value
                try:
                    out["attrs"][key] = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    out["attrs"][key] = text
            else:
                out["attrs"][key] = value.item() if hasattr(value, "item") else value
    return out
