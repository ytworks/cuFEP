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


def test_the_wiki_pages_are_compared_like_the_readmes_and_by_the_numbers_in_their_tables(tmp_path):
    """The accuracy page of the wiki in English and in Japanese: the same outline, figures and numbers in
    the tables (cells of prose are translated and are not compared); each links to the other's wiki page."""
    pair = check_readme_pair.PAIRS[1]
    wiki = check_readme_pair.WIKI
    (tmp_path / "docs" / "wiki" / "images").mkdir(parents=True)
    (tmp_path / "docs" / "wiki" / "images" / "tyk2.png").write_bytes(b"png")
    english = (f"# Accuracy\n[日本語]({wiki}Accuracy-ja)\n\n## TYK2\n\n| ligand | experiment | cuFEP | note |\n"
               "|---|---|---|---|\n| ejm_31 | -9.54 | -10.12 ± 0.50 | two windows |\n\n![TYK2](images/tyk2.png)\n")
    japanese = (f"# 精度\n[English]({wiki}Accuracy)\n\n## TYK2\n\n| リガンド | 実験値 | cuFEP | 注 |\n"
                "|---|---|---|---|\n| ejm_31 | −9.54 | -10.12 ± 0.50 | 2 つの窓 |\n\n![TYK2](images/tyk2.png)\n")

    def problems(japanese):
        return check_readme_pair.pair_problems(english, japanese, tmp_path, pair)

    assert problems(japanese) == []
    assert any("the numbers in the tables differ" in p and "'-10.12' / '-10.21'" in p
               for p in problems(japanese.replace("-10.12", "-10.21")))
    assert any("numbers in the tables differ" in p for p in problems(japanese.replace(" ± 0.50", "")))
    assert any(f"does not link to {wiki}Accuracy" in p for p in problems(japanese.replace(f"({wiki}Accuracy)", "")))
    assert any("links to files differ" in p for p in problems(japanese.replace("![TYK2](images/tyk2.png)", "")))
    assert any("no such file" in p for p in problems(japanese.replace("tyk2.png", "a2a.png")))


def test_anchors_are_those_github_gives():
    assert check_readme_pair.slug("How cuFEP compares") == "how-cufep-compares"
    assert check_readme_pair.slug("`sampling.rest2` and REST2 (pocket)") == "samplingrest2-and-rest2-pocket"
    assert check_readme_pair.slug("結果の読み方") == "結果の読み方"
    assert check_readme_pair.anchors("## Notes\n\n## Notes\n") == {"notes", "notes-1"}


def test_progress_log_reads_fenced_code_as_text_and_checks_the_budgets_of_the_sections():
    """Review of PR 2: a fenced `## Example` was taken for a section and a fenced `###` line for an
    entry; the budgets of Product (25 lines) and Current Status (80) were not checked."""
    entry = "### 2026-01-01 — [progress] x\n- What: y\n"
    good = "# Progress\n\n## Product\n\n## Current Status\n\n## Change Log\n\n" + entry * 10
    fenced = good.replace("## Current Status\n", "## Current Status\n```markdown\n## Example\n### 2026\n```\n")
    assert check_progress_size.problems_in(fenced) == []
    assert any("## Product has 26 lines" in p
               for p in check_progress_size.problems_in(good.replace("## Product\n", "## Product\n" + "x\n" * 25)))
    assert any("## Current Status has 81 lines" in p for p in check_progress_size.problems_in(
        good.replace("## Current Status\n", "## Current Status\n" + "x\n" * 80)))


def test_the_readme_check_reads_markdown_as_github_does(tmp_path):
    """Review of PR 2: inputs on which the check of the README pair was wrong."""
    c = check_readme_pair
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a (b).md").write_text("# A\n")
    # commands in ~~~ fences are compared; a `#` inside quotes is no comment
    assert c.blocks_and_text("~~~bash\ncufep run x\n~~~\n")[0] == [["cufep run x"]]
    assert c.uncommented(['cufep run --label "run #1"  # comment']) == ['cufep run --label "run #1"']
    # a line that opens with an inline code span is no fence and hides nothing after it
    assert c.outline("```cufep``` runs it\n\n## A\n\n#### B\n") == [2, 4]
    # link targets: angle brackets, titles, references, HTML; percent-encoded paths; not in code spans
    text = ("[x](<docs/missing file.md>) [y](f.md 'title') [z][r] <a href=\"g.md\">g</a> "
            "[w](docs/a%20(b).md) `[v](docs/nothing.md)`\n\n[r]: h.md\n")
    assert sorted(c.links(text)) == sorted(["docs/missing file.md", "f.md", "h.md", "g.md", "docs/a%20(b).md"])
    missing = c.link_problems("README.md", text, tmp_path)
    assert sorted(p.split(": ")[1] for p in missing) == sorted([
        "link to docs/missing file.md", "link to f.md", "link to h.md", "link to g.md"])
    # anchors: headings with links and code, setext, indented ATX, HTML ids, numbering of repeats
    doc = ("### See [protocol](docs/protocol.md)\n\nTitle\n=====\n\n   ## Indented\n\n<a id=\"here\"></a>\n\n"
           "## Notes\n\n## Notes 1\n\n## Notes\n\n## `sampling.rest2` and REST2\n")
    assert c.anchors(doc) == {"see-protocol", "title", "indented", "here", "notes", "notes-1", "notes-2",
                              "samplingrest2-and-rest2"}


def fake_docker(tmp_path, missing=""):
    """A `docker` that records its arguments; `image inspect` fails for the image named missing."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "docker.log"
    script = bin_dir / "docker"
    script.write_text(f"""#!/bin/bash
echo "$*" >> {log}
if [[ "$1 $2" == "image inspect" && "$3" == "{missing}" ]]; then exit 1; fi
if [[ "$1 $2" == "container inspect" ]]; then exit 1; fi
if [[ "$1" == run ]]; then printf '%s\\n' "$@"; fi
exit 0
""")
    script.chmod(0o755)
    return {**__import__("os").environ, "PATH": f"{bin_dir}:" + __import__("os").environ["PATH"]}, log


def test_spikes_run_sh_calls_the_image_as_it_did_before_the_wrapper(tmp_path):
    """Review of PR 2: GPUS=2 is a count of GPUs for docker, as in the old spikes/run.sh, and the
    script runs in /work from any directory."""
    import subprocess

    env, _ = fake_docker(tmp_path)
    run = lambda gpus=None: subprocess.run(  # noqa: E731
        ["bash", str(ROOT / "spikes" / "run.sh"), "x.py"], cwd=ROOT / "spikes", capture_output=True, text=True,
        env={**env, **({"GPUS": gpus} if gpus is not None else {})}).stdout.splitlines()
    args = run("2")
    assert args[args.index("--gpus") + 1] == "2" and args[args.index("-w") + 1] == "/work"
    assert args[-5:] == ["--entrypoint", "/usr/bin/env", "cufep:dev", "python", "x.py"]
    assert run()[run().index("--gpus") + 1] == "all"
    assert run('"device=1"')[run('"device=1"').index("--gpus") + 1] == '"device=1"'


def test_the_wrapper_never_builds_an_image_it_was_not_asked_to_build(tmp_path):
    """Review of PR 2: a missing tag was built from the working tree, so IMAGE=cufep:<commit> named a
    tree that was not that commit."""
    import subprocess

    env, log = fake_docker(tmp_path, missing="cufep:missing")
    wrapper = str(ROOT / "scripts" / "in_docker.sh")
    done = subprocess.run(["bash", wrapper, "--image", "cufep:missing", "python", "-V"], capture_output=True,
                          text=True, env=env)
    assert done.returncode == 2 and "not found" in done.stderr
    assert not any(line.startswith(("build", "run")) for line in log.read_text().splitlines())
    for bad in (["--gpus"], ["--gpus", '"device=1"'], ["--gpus", "0", "--docker-gpus", "2"]):
        assert subprocess.run(["bash", wrapper, *bad, "python"], capture_output=True, env=env).returncode == 2
    args = subprocess.run(["bash", wrapper, "--gpus", "6,7", "python"], capture_output=True, text=True,
                          env=env).stdout.splitlines()
    assert args[args.index("--gpus") + 1] == '"device=6,7"'


def test_names_of_commercial_fep_products_are_kept_to_accuracy_and_sources():
    """The owner's rule (2026-11-20): no text presents a decision of cuFEP as following a commercial FEP
    product; such names appear only where accuracy is compared and where a source is cited."""
    import check_names

    assert check_names.problems_in("docs/protocol.md", "as in FEP+ and Flare") == [
        "docs/protocol.md:1: Flare",
        "docs/protocol.md:1: FEP+ (only for comparing accuracy or citing a source, in the files listed in "
        "scripts/check_names.py)"]
    assert check_names.problems_in("src/cufep/mps.py", "\n# Desmond mps_factor") == ["src/cufep/mps.py:2: Desmond"]
    assert check_names.problems_in("src/cufep/benchmarks/runner.py", "| cuFEP | FEP+ |") == []
    assert check_names.problems_in("README.md", "the Schrödinger public binding free energy benchmark") == []
    assert check_names.problems_in("README.md", "the method of Cresset Flare") != []


def test_the_checks_hold_on_the_inputs_of_the_review_of_pr_3(tmp_path):
    """Independent review of PR 3: inputs on which the checks were still wrong."""
    import unicodedata

    import check_names

    c = check_readme_pair
    # PROGRESS.md: a line that opens with a code span is no fence (one reader of fences for both checks)
    progress = "# P\n\n## Product\n```x``` is a tool\n\n## Current Status\n\n## Change Log\n"
    assert check_progress_size.problems_in(progress) == []
    # names in any case and in decomposed Unicode
    assert len(check_names.problems_in("docs/protocol.md", "as desmond and flare and CRESSET do")) == 3
    assert check_names.problems_in("docs/protocol.md", unicodedata.normalize("NFD", "Schrödinger")) != []
    # README: fences inside list items, shortcut references, ids of any element, HTML comments,
    # links from the root and with a query, an apostrophe in a command, multi-line setext, entities
    assert c.blocks_and_text("1. run\n\n    ```bash\n    cufep run a.yaml\n    ```\n")[0] == [["    cufep run a.yaml"]]
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "p.md").write_text("# P\n")
    text = "[guide]\n\n[guide]: docs/missing.md\n\n[root](/docs/p.md) [raw](docs/p.md?plain=1)\n"
    assert [p.split(": ")[1] for p in c.link_problems("README.md", text, tmp_path)] == ["link to docs/missing.md"]
    assert c.anchors('<div id="x"></div>\n<h2 id="y">Y</h2>\n<a data-id="z"></a>\n') >= {"x", "y"}
    assert "z" not in c.anchors('<a data-id="z"></a>\n')
    assert c.outline("<!--\n## Hidden\n-->\n## Shown\n") == [2]
    assert c.uncommented(["echo it's fine # comment"]) == ["echo it's fine"]
    assert c.anchors("Line one\nline two\n========\n\n## A &amp; B\n") == {"line-one-line-two", "a--b"}
