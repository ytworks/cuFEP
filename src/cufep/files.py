"""Files that are written in one step.

cuFEP uses again what an earlier run left: the prepared systems, the pre-equilibration, the
restraint, the lambda schedule and the start configurations of a leg, the cached partial charges
of a ligand, and the results. A file that is used because it exists must never be seen half
written: a run that is stopped while it writes one would otherwise leave a file that the next
run takes as done and then cannot read (review of PR 1: ``start_configs.npz`` was written under
its final name, and a run stopped in the middle could no longer be resumed).

:func:`publish` writes the whole file under another name next to it, makes it durable, renames
it over the final name in one step and makes the directory durable; a reader sees the old file
or the new one, never a part. Every file cuFEP writes goes through it, except the samples of a
leg, which :mod:`cufep.sampling.storage` writes as a transaction of its own (with the same
:func:`fsync_path`), and the lock file of an output directory. A test scans the source for any
other writer.
"""
from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Callable

#: what a system says when it cannot make a directory durable: passed over for directories only
_NO_DIRECTORY_SYNC = {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EBADF, errno.EISDIR, errno.EACCES}

#: the suffix of a file while it is being written
WRITING = ".writing"


def fsync_path(path: Path | str) -> None:
    """Make what was written to ``path`` (a file, or a directory and its entries) durable.

    An error of a file is raised (review of PR 1: an EIO of the fsync of a part was passed over, and
    the checkpoint that named it replaced the one before). A directory that the system cannot sync
    (EINVAL, ENOTSUP ...) is passed over; any other error of a directory is raised too."""
    path = Path(path)
    directory = path.is_dir()
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        if directory and exc.errno in _NO_DIRECTORY_SYNC:
            return
        raise
    try:
        os.fsync(fd)
    except OSError as exc:
        if not (directory and exc.errno in _NO_DIRECTORY_SYNC):
            raise
    finally:
        os.close(fd)


def publish(path: Path | str, write: Callable[[Path], None]) -> Path:
    """Write ``path`` in one step: ``write(tmp)`` writes the whole file to ``tmp``, a name next to
    ``path`` of this call alone; then it is made durable and renamed over ``path``. If anything fails
    or the process is interrupted before the rename, ``path`` is as it was and ``tmp`` is removed.
    Two writers of one file each publish a whole file of their own; the last rename wins (review of
    PR 1: with one temporary name for all, a writer published the half file of the other)."""
    import uuid

    path = Path(path)
    # a name of this call alone, not created here: the writer creates the file with the usual
    # permissions of the process (mkstemp would make every published file readable by its owner only)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}{WRITING}")
    try:
        write(tmp)
        fsync_path(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    fsync_path(path.parent)
    return path


def publish_text(path: Path | str, text: str) -> Path:
    """:func:`publish` a text file."""
    return publish(path, lambda tmp: tmp.write_text(text))


def publish_npz(path: Path | str, **arrays) -> Path:
    """:func:`publish` a NumPy ``.npz`` archive (written to an open file, so that NumPy does not
    append ``.npz`` to the temporary name)."""
    import numpy as np

    def write(tmp: Path) -> None:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)

    return publish(path, write)


def publish_with(path: Path | str, write_to_handle: Callable, mode: str = "w", **open_kwargs) -> Path:
    """:func:`publish` a file that a function writes to an open handle (csv, PDB writers)."""

    def write(tmp: Path) -> None:
        with open(tmp, mode, **open_kwargs) as fh:
            write_to_handle(fh)

    return publish(path, write)
