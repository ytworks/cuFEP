#!/usr/bin/env python3
"""Check PROGRESS.md against the size budget in .claude/rules/progress-log.md.

Run by the SessionStart hook in .claude/settings.json, so every agent session starts with a
warning when PROGRESS.md needs archiving, and by the `harness` job of CI. Standard library only:
the hook must also work in a session without Docker. Exits non-zero when the budget is exceeded
or the required section layout is broken.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MAX_TOTAL_LINES = 200
MAX_CHANGELOG_ENTRIES = 10
SECTIONS = ["## Product", "## Current Status", "## Change Log"]
#: lines of a section after its heading
MAX_SECTION_LINES = {"## Product": 25, "## Current Status": 80}
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def problems_in(text: str) -> list[str]:
    """Rule violations of a PROGRESS.md given as text (empty list: none). Lines inside fenced code
    blocks are neither headings nor entries."""
    lines = text.splitlines()
    sections, section_lines = [], {}
    current, fence = None, ""
    entries = 0
    for line in lines:
        m = FENCE.match(line)
        if fence:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) \
                    and not line.strip().strip(fence[0]):
                fence = ""
        elif m:
            fence = m.group(1)
        elif line.startswith("## "):
            current = line.rstrip()
            sections.append(current)
            section_lines[current] = 0
            continue
        elif current == "## Change Log" and line.startswith("### "):
            entries += 1
        if current is not None:
            section_lines[current] += 1

    problems = []
    if sections != SECTIONS:
        problems.append(f"sections {sections} (required: {SECTIONS})")
    if len(lines) > MAX_TOTAL_LINES:
        problems.append(f"{len(lines)} lines (budget: {MAX_TOTAL_LINES})")
    if entries > MAX_CHANGELOG_ENTRIES:
        problems.append(f"{entries} Change Log entries (budget: {MAX_CHANGELOG_ENTRIES})")
    for section, budget in MAX_SECTION_LINES.items():
        if section_lines.get(section, 0) > budget:
            problems.append(f"{section} has {section_lines[section]} lines (budget: {budget})")
    return problems


def main() -> int:
    progress = Path(__file__).resolve().parent.parent / "PROGRESS.md"
    if not progress.is_file():
        print(f"check_progress_size: {progress} not found", file=sys.stderr)
        return 1
    problems = problems_in(progress.read_text(encoding="utf-8"))
    if problems:
        print("PROGRESS.md violates its rules: " + "; ".join(problems)
              + ". See .claude/rules/progress-log.md; to archive, follow docs/progress/archiving.md"
              " before other work updates PROGRESS.md.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
