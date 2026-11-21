#!/usr/bin/env python3
"""Check where the names of commercial FEP products and of their vendors appear.

The project's rule (.claude/rules/fep-invariants.md): no text says that a decision of cuFEP
follows, copies or was made by looking at a commercial FEP product. Such a name appears only
where accuracy is compared (the benchmark report and the accuracy pages of the wiki against the
published FEP+ predictions) and where a source is cited (the benchmark set). This check holds the
names to the files that do that. Run by the `harness` job of CI. Standard library only.
"""
from __future__ import annotations

import re
import subprocess
import sys
import unicodedata
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: names used neither to compare accuracy nor to cite a source here: nowhere but in this check
#: and its test
NOWHERE = re.compile(r"Flare|Cresset|Desmond", re.IGNORECASE)
#: names used to compare accuracy and to cite the benchmark set, allowed in these files only
FOR_ACCURACY_AND_SOURCES = re.compile(r"FEP\+|Schr(?:ö|oe?)dinger", re.IGNORECASE)
ACCURACY_AND_SOURCE_FILES = (
    "benchmarks/*", "src/cufep/benchmarks/*", "src/cufep/cli.py", "tests/unit/test_benchmarks.py",
    "examples/*", "README.md", "README.ja.md", "CLAUDE.md", "PROGRESS.md", "docs/plans/*",
    "docs/LICENSING.md", "docs/wiki/*", "spikes/wiki_tables.py", "spikes/benchmark_progress.py",
    # entries of PROGRESS.md, moved there unchanged (docs/progress/archiving.md): results against the
    # published predictions stay what they were
    "docs/progress/archive/*",
)
THIS_CHECK = ("scripts/check_names.py", "tests/unit/test_harness.py")


def problems_in(path: str, text: str) -> list[str]:
    """The places in one file (repository path, text) where a name is not allowed."""
    if path in THIS_CHECK:
        return []
    allowed = any(fnmatch(path, pattern) for pattern in ACCURACY_AND_SOURCE_FILES)
    problems = []
    for n, line in enumerate(unicodedata.normalize("NFC", text).splitlines(), start=1):
        for m in NOWHERE.finditer(line):
            problems.append(f"{path}:{n}: {m.group(0)}")
        if not allowed:
            for m in FOR_ACCURACY_AND_SOURCES.finditer(line):
                problems.append(f"{path}:{n}: {m.group(0)} (only for comparing accuracy or citing a source, "
                                "in the files listed in scripts/check_names.py)")
    return problems


#: what .gitignore leaves out, for a checkout without git (the official image has none)
IGNORED_DIRS = {".git", "runs", "__pycache__", ".pytest_cache", ".ruff_cache", "worktrees"}
IGNORED_PATHS = ("spikes/out/", ".claude/worktrees/")


def tracked_files(root: Path = ROOT) -> list[str]:
    """The files of the repository: those git tracks, or, without git, every file outside the
    directories that .gitignore leaves out."""
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True)
        return [p for p in out.stdout.decode().split("\0") if p]
    except (OSError, subprocess.CalledProcessError):
        files = []
        for file in root.rglob("*"):
            rel = file.relative_to(root).as_posix()
            if file.is_file() and not set(file.relative_to(root).parts) & IGNORED_DIRS \
                    and not rel.startswith(IGNORED_PATHS) and not re.fullmatch(r"core(\..*)?", file.name) \
                    and ".egg-info/" not in rel:
                files.append(rel)
        return files


def main() -> int:
    problems = []
    for path in tracked_files():
        file = ROOT / path
        if not file.is_file():
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:          # binary files
            continue
        problems += problems_in(path, text)
    if problems:
        print("names of commercial FEP products outside the comparison of accuracy and the citation of "
              "sources (.claude/rules/fep-invariants.md):\n- " + "\n- ".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
