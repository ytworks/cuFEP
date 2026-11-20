"""The development harness: the size budget of PROGRESS.md, and the check that the English and
the Japanese README say the same thing."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import check_progress_size  # noqa: E402
import check_readme_pair  # noqa: E402


def test_progress_log_budget():
    entry = "### 2026-01-01 — [progress] x\n- What: y\n"
    good = "# Progress\n\n## Product\n\n## Current Status\n\n## Change Log\n\n" + entry * 10
    assert check_progress_size.problems_in(good) == []
    assert any("11 Change Log entries" in p for p in check_progress_size.problems_in(good + entry))
    assert any("sections" in p for p in check_progress_size.problems_in(good.replace("## Product\n", "")))
    assert any("lines" in p for p in check_progress_size.problems_in(good + "x\n" * 200))


ENGLISH = """# tool
[English](README.md) | [日本語](README.ja.md)

- [Install](#install)

## Install

```bash
tool run config.yaml   # run or resume
```

See [the protocol](docs/protocol.md#adaptive-lambda-scheduling).

### Details
"""

JAPANESE = """# tool
[English](README.md) | [日本語](README.ja.md)

- [インストール](#インストール)

## インストール

```bash
tool run config.yaml   # 実行または再開
```

[プロトコル](docs/protocol.md#adaptive-lambda-scheduling) を参照。

### 詳細
"""


def test_the_two_readmes_are_compared_by_outline_commands_and_links(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "protocol.md").write_text("# The protocol\n\n### Adaptive lambda scheduling\n")
    for name, text in zip(check_readme_pair.PAIR, (ENGLISH, JAPANESE)):
        (tmp_path / name).write_text(text)

    def problems(japanese):
        return check_readme_pair.pair_problems(ENGLISH, japanese, tmp_path)

    # translated prose, headings and comments are allowed to differ; anchors of Japanese headings resolve
    assert problems(JAPANESE) == []
    assert any("headings differ" in p for p in problems(JAPANESE.replace("### 詳細\n", "")))
    assert any("code block 1 differs" in p for p in problems(JAPANESE.replace("config.yaml", "other.yaml")))
    assert any("links to files differ" in p for p in problems(JAPANESE.replace("docs/protocol.md", "LICENSE")))
    assert any("does not link to README.md" in p for p in problems(JAPANESE.replace("[English](README.md) | ", "")))
    assert any("no such file" in p
               for p in problems(JAPANESE.replace("docs/protocol.md#", "docs/missing.md#")))
    assert any("no such heading" in p for p in problems(JAPANESE.replace("#adaptive-lambda", "#fixed-lambda")))
    assert any("#インストール: no such heading" in p for p in problems(JAPANESE.replace("## インストール", "## 導入")))


def test_anchors_are_those_github_gives():
    assert check_readme_pair.slug("How cuFEP compares") == "how-cufep-compares"
    assert check_readme_pair.slug("`sampling.rest2` and REST2 (pocket)") == "samplingrest2-and-rest2-pocket"
    assert check_readme_pair.slug("結果の読み方") == "結果の読み方"
    assert check_readme_pair.anchors("## Notes\n\n## Notes\n") == {"notes", "notes-1"}
