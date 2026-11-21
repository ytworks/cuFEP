# Change Log archive

Change Log entries moved here from `PROGRESS.md` by `docs/progress/archiving.md`, newest first.
Frozen history: entries are inserted at the marker and never edited or deleted.

<!-- ARCHIVE-INSERT-POINT -->

### 2026-11-21 — [progress] Full TYK2 and A2A benchmarks; accuracy wiki
- What: TYK2 (16) and A2A (17) run once each with the example settings (5 ns per window, commit
  3b45bfe); recorded in `benchmarks/results/`; wiki pages in English and Japanese compare them
  with experiment, FEP+ and Felis. cuFEP: RMSE after removing the mean error 1.74 / 2.26
  kcal/mol, Kendall tau 0.12 / 0.38, mean signed error -0.62 / -4.12; FEP+ 0.47 / 1.08, 0.81 /
  0.59. The report gained Pearson r and a figure ligand by ligand.
- Refs: PR #4, issue #5, `docs/wiki/Accuracy.md`

### 2026-11-20 — [progress] Benchmarks with the settings of the examples
- What: a benchmark set reads its example's `config.yaml` and changes only the input files, the
  ligands and the output directory (smoke also shortens sampling, never lengthening a setting);
  both examples set 5 ns of production per window (default 10 ns). The benchmark report compares
  every ligand with experiment and FEP+, NOT RELIABLE ones included, with signed errors, where a
  leg is below the level of overlap (`overlap_below_level`) and a breakdown of the two groups.
- Why: the owner's order: benchmarks with the example settings, NOT RELIABLE results compared too.
- Refs: PR #4, `docs/plans/2026-11-20-full-benchmarks-and-accuracy-wiki.md`

### 2026-11-20 — [progress] Wording fixes and ruff
- What: the findings of the independent review of PR #2 (README, rules, wrappers, harness
  checks); no decision presented as following a commercial FEP product, enforced by
  `scripts/check_names.py`; ruff 0.17.5 with the rules of fmeta, the code changed to pass it
  (behaviour unchanged: the ASTs of the mechanical steps compared, 558 unit and 17 GPU tests).
- Why: the owner's rule on names of commercial products; lint left from PR #2.
- Refs: PR #3, `docs/plans/2026-11-20-wording-and-lint.md`

### 2026-11-20 — [progress] Development harness and OSS README
- What: `CLAUDE.md` / `AGENTS.md`, `.claude/rules/` (Docker execution, progress log, free-energy
  invariants), SessionStart hook and `PROGRESS.md` with its size check, `docs/plans/`,
  `scripts/in_docker.sh`, CI job `harness`; README in English and Japanese, `CONTRIBUTING.md`.
  `docs/PLAN.md` moved to `docs/plans/2026-11-18-abfe-engine.md`.
- Why: the project's rules lived only in agents' private notes; the README was not written for
  outside users. The owner's rule for these documents: no statement that a decision of cuFEP
  follows or copies a commercial FEP product; such names appear only in accuracy comparisons
  and in citations of sources.
- Refs: PR #2, `docs/plans/2026-11-20-oss-readme-and-harness.md`

### 2026-11-20 — [progress] PR #1 merged: the ABFE engine
- What: the engine of plan revisions 1 to 15 merged into `main` as 31a720e, after the review
  rounds of PR #1. Only published criteria decide (analysis version 10), ALS threshold 0.18,
  REST2 on the pocket by default.
- Refs: PR #1, `docs/plans/2026-11-18-abfe-engine.md`, `docs/protocol.md`

