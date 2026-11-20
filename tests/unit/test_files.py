"""Files that are used again because they exist are never seen half written (review of PR 1 on a3462a5:
start_configs.npz was written under its final name, and a run stopped while writing it could not be resumed)."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from cufep import files
from cufep.files import WRITING, publish, publish_npz, publish_text

#: the only places that may write a file other than through cufep.files: the samples of a leg, which are a
#: transaction of their own; the lock of an output directory; the input of an external program in a
#: temporary directory; the log of a run, to which lines are appended
ALLOWED = {("sampling/storage.py", None), ("run.py", ".cufep.lock"), ("prep/orient.py", 'open(src, "w")'),
           ("files.py", None), ("workflow.py", 'open(self.path, "a")')}    # the last: cufep.log, lines appended

WRITERS = [r"\.write_text\(", r"\.write_bytes\(", r"np\.savez(_compressed)?\(\s*(?!fh\b)", r"\bsavefig\(\s*(?!fh\b)",
           r"shutil\.copy2?\((?![^)]*\btmp\))", r"\bopen\([^)]*,\s*[\"'][wa]b?[\"']"]


def test_every_file_cufep_writes_goes_through_publish(repo):
    offenders = []
    for path in sorted((repo / "src" / "cufep").rglob("*.py")):
        rel = str(path.relative_to(repo / "src" / "cufep"))
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for pattern in WRITERS:
                if re.search(pattern, line):
                    allowed = any(rel == f and (marker is None or marker in line) for f, marker in ALLOWED)
                    if not allowed:
                        offenders.append(f"{rel}:{number}: {line.strip()}")
    assert offenders == []


class Interrupted(BaseException):
    pass


@pytest.mark.parametrize("failure", ["in the writer", "in the fsync of the file", "in the rename", "interrupt"])
def test_a_publish_that_fails_leaves_the_file_as_it_was(tmp_path, monkeypatch, failure):
    target = tmp_path / "start_configs.npz"
    publish_npz(target, positions=np.zeros((2, 3)))
    before = target.read_bytes()

    def half(tmp: Path) -> None:
        tmp.write_bytes(b"PK\x03\x04 half an archive")
        if failure == "in the writer":
            raise OSError("no space left on device")
        if failure == "interrupt":
            raise Interrupted()

    if failure == "in the fsync of the file":
        monkeypatch.setattr(files, "fsync_path", lambda p: (_ for _ in ()).throw(OSError(5, "EIO")))
    if failure == "in the rename":
        monkeypatch.setattr(files.os, "replace", lambda a, b: (_ for _ in ()).throw(OSError(28, "ENOSPC")))
    with pytest.raises((OSError, Interrupted)):
        publish(target, half)
    assert target.read_bytes() == before
    assert not (tmp_path / ("start_configs.npz" + WRITING)).exists()
    assert np.load(target)["positions"].shape == (2, 3)


def test_the_reviewers_start_configurations_stopped_while_written(tmp_path, monkeypatch):
    """The run of a leg makes its start configurations and is stopped while it writes them: the next run finds
    no start_configs.npz (it makes them again) instead of one it cannot read."""
    start = tmp_path / "start_configs.npz"
    real = np.savez

    def stopped(fh, **arrays):
        real(fh, **arrays)
        fh.truncate(100)                               # what the disk holds when the process is killed
        raise Interrupted()

    monkeypatch.setattr(np, "savez", stopped)
    with pytest.raises(Interrupted):
        publish_npz(start, positions=np.zeros((4, 100, 3)), boxes=np.zeros((4, 3, 3)), velocities=np.zeros((4, 100, 3)))
    assert not start.exists() and list(tmp_path.iterdir()) == []


def test_written_files_are_complete(tmp_path):
    publish_text(tmp_path / "a.json", json.dumps({"x": 1}))
    assert json.loads((tmp_path / "a.json").read_text()) == {"x": 1}
    publish_npz(tmp_path / "b.npz", y=np.arange(3))
    assert np.load(tmp_path / "b.npz")["y"].tolist() == [0, 1, 2]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json", "b.npz"]


def test_two_writers_of_one_file_each_publish_a_whole_file(tmp_path):
    """Review of PR 1 (P2): two writers of one file shared the temporary name; the first published the other's half
    file and the second failed with FileNotFoundError."""
    import threading

    target = tmp_path / "benchmark_report.md"
    barrier = threading.Barrier(2)
    errors = []

    def writer(text):
        def write(tmp):
            with open(tmp, "w") as fh:
                fh.write(text[: len(text) // 2])
                barrier.wait(timeout=10)                   # both are half way
                fh.write(text[len(text) // 2:])
        try:
            publish(target, write)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(c * 1000,)) for c in "AB"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert target.read_text() in ("A" * 1000, "B" * 1000)
    assert [p.name for p in tmp_path.iterdir()] == ["benchmark_report.md"]
    # the usual permissions of the process, not those of a private temporary file
    import os

    mask = os.umask(0)
    os.umask(mask)
    assert target.stat().st_mode & 0o777 == 0o666 & ~mask
