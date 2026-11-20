"""The frames of the coupled window of a leg, from every stored part of its samples, in order
(:func:`cufep.sampling.storage.stored_layout`: samples.h5, and the segments that later versions add).

A run of an early version that was stopped while it wrote its frames left a chunk that cannot be
read; the frames are then those before it, as these scripts always took them."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def stored_frames(hrex_dir: str | Path, step: int = 25) -> np.ndarray:
    from cufep.sampling.storage import open_samples, stored_layout

    pieces = []
    for path, _rows, _tail, frames in stored_layout(hrex_dir)[0]:
        with open_samples(path) as h:
            for start in range(0, frames, step):
                try:
                    pieces.append(h["frames"][start:min(start + step, frames)])
                except OSError:                      # the frames after it would not follow on in time
                    return np.concatenate(pieces) if pieces else np.zeros((0, 0, 3), dtype="f4")
    return np.concatenate(pieces) if pieces else np.zeros((0, 0, 3), dtype="f4")
