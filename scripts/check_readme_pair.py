#!/usr/bin/env python3
"""Check that README.md (English) and README.ja.md (Japanese) say the same thing.

The two cannot be compared word for word, so what can be compared is: the outline (the levels of
the `##` and `###` headings, in order), the commands and configuration in the code blocks (with
their comments removed, since the comments are translated), the links to files of the repository,
and that each names the other at the top. Every relative link and every anchor of both must
resolve. Run by the `harness` job of CI. Standard library only.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIR = ("README.md", "README.ja.md")
LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^\s*```")


def blocks_and_text(text: str) -> tuple[list[list[str]], list[str]]:
    """The code blocks (their lines) and the lines outside code blocks."""
    blocks, outside, current = [], [], None
    for line in text.splitlines():
        if FENCE.match(line):
            if current is None:
                current = []
            else:
                blocks.append(current)
                current = None
        elif current is not None:
            current.append(line)
        else:
            outside.append(line)
    return blocks, outside


def outline(text: str) -> list[int]:
    """The levels of the `##` and `###` headings, in order."""
    _, lines = blocks_and_text(text)
    return [len(m.group(1)) for m in (re.match(r"^(#{2,3}) ", ln) for ln in lines) if m]


def uncommented(block: list[str]) -> list[str]:
    """The lines of a code block without comments and blank lines (comments are translated)."""
    kept = []
    for line in block:
        line = re.sub(r"(^|\s)#.*$", "", line).rstrip()
        if line.strip():
            kept.append(line)
    return kept


def links(text: str) -> list[str]:
    """Link targets outside code blocks, in order."""
    _, lines = blocks_and_text(text)
    return [m.group(1) for ln in lines for m in LINK.finditer(ln)]


def is_external(target: str) -> bool:
    return bool(re.match(r"^[a-z][a-z0-9+.-]*:", target))


def slug(heading: str) -> str:
    """The anchor GitHub gives a heading: lower case; letters, numbers, spaces, hyphens and
    underscores kept, everything else dropped; spaces to hyphens."""
    text = heading.strip().lower()
    text = "".join(c for c in text if c in " -_" or unicodedata.category(c)[0] in "LNM")
    return text.replace(" ", "-")


def anchors(text: str) -> set[str]:
    _, lines = blocks_and_text(text)
    seen: dict[str, int] = {}
    found = set()
    for line in lines:
        m = re.match(r"^#{1,6} (.+?)\s*#*\s*$", line)
        if not m:
            continue
        base = slug(m.group(1))
        n = seen.get(base, 0)
        seen[base] = n + 1
        found.add(base if n == 0 else f"{base}-{n}")
    return found


def link_problems(name: str, text: str, root: Path) -> list[str]:
    """Relative links of a Markdown file whose file or anchor does not exist."""
    problems = []
    for target in links(text):
        if is_external(target):
            continue
        path, _, anchor = target.partition("#")
        if not path:                                   # an anchor of this file
            if anchor not in anchors(text):
                problems.append(f"{name}: link to {target}: no such heading")
            continue
        file = (root / name).parent / path
        if not file.exists():
            problems.append(f"{name}: link to {target}: no such file")
        elif anchor and file.suffix == ".md" and anchor not in anchors(file.read_text(encoding="utf-8")):
            problems.append(f"{name}: link to {target}: no such heading")
    return problems


def pair_problems(english: str, japanese: str, root: Path = ROOT) -> list[str]:
    """Differences between the two READMEs given as text (empty list: none)."""
    en_name, ja_name = PAIR
    problems = []
    if outline(english) != outline(japanese):
        problems.append(f"the headings differ: {en_name} has levels {outline(english)}, "
                        f"{ja_name} {outline(japanese)}")
    en_blocks = [uncommented(b) for b in blocks_and_text(english)[0]]
    ja_blocks = [uncommented(b) for b in blocks_and_text(japanese)[0]]
    if len(en_blocks) != len(ja_blocks):
        problems.append(f"{en_name} has {len(en_blocks)} code blocks, {ja_name} {len(ja_blocks)}")
    for i, (en, ja) in enumerate(zip(en_blocks, ja_blocks), start=1):
        if en != ja:
            first = next((a, b) for a, b in zip(en + [""], ja + [""]) if a != b)
            problems.append(f"code block {i} differs apart from comments: {first[0]!r} / {first[1]!r}")
    files = [[t.partition("#")[0] for t in links(x) if not is_external(t) and not t.startswith("#")]
             for x in (english, japanese)]
    en_files, ja_files = (sorted(set(f) - set(PAIR)) for f in files)
    if en_files != ja_files:
        problems.append(f"links to files differ: only in {en_name} {sorted(set(en_files) - set(ja_files))}, "
                        f"only in {ja_name} {sorted(set(ja_files) - set(en_files))}")
    if ja_name not in files[0]:
        problems.append(f"{en_name} does not link to {ja_name}")
    if en_name not in files[1]:
        problems.append(f"{ja_name} does not link to {en_name}")
    problems += link_problems(en_name, english, root) + link_problems(ja_name, japanese, root)
    return problems


def main() -> int:
    texts = []
    for name in PAIR:
        path = ROOT / name
        if not path.is_file():
            print(f"check_readme_pair: {path} not found", file=sys.stderr)
            return 1
        texts.append(path.read_text(encoding="utf-8"))
    problems = pair_problems(*texts)
    if problems:
        print("README.md and README.ja.md differ:\n- " + "\n- ".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
