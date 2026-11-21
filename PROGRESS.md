# Progress

Cross-session memory. Rules: `.claude/rules/progress-log.md`.
Older Change Log entries: `docs/progress/archive/change-log.md`.

## Product

**Goal** (plan `docs/plans/2026-11-18-abfe-engine.md`): absolute binding free energies (ABFE) on
OpenMM, on all GPUs of a machine, from two files.

- Inputs: a protein PDB and a ligand SDF with the ligand placed in the pocket. Every other
  setting has a recommended default with its basis and can be set in YAML
  (`docs/configuration.md`).
- Backends: NVIDIA's cuPotential plugin for OpenMM by default, stock OpenMM as the second
  backend and the fallback. cuPotential is a compute backend, not a force field; the force
  fields are ff14SB, TIP3P, Lipid21 and OpenFF 2.2.1 with AM1-BCC charges.
- Soluble proteins in water; membrane proteins in a POPC bilayer when `membrane.enabled` is set
  (never detected).
- Protocol: double decoupling with Boresch restraints chosen as OpenFE chooses them,
  Hamiltonian replica exchange with REST2 on the ligand and its pocket, adaptive lambda
  scheduling (Midgley et al. 2025), MBAR with the checks of the literature. Defaults and terms
  follow the literature and published open methods (`docs/configuration.md`); one calculation per
  ligand.
- Speed is a requirement: replicas spread over all GPUs in proportion to the work, several
  workers per GPU through NVIDIA MPS.
- Benchmarks against experiment and the FEP+ predictions of the Schrödinger public binding free
  energy benchmark (MIT licence): TYK2 (kinase) and A2A (GPCR in a bilayer).
- Version 0.1: neutral ligands only, no cofactors.
- Distribution: a Docker image built from `docker/Dockerfile`; an image with the plugin is not
  published before the licence question is settled (`docs/LICENSING.md`).

## Current Status

Snapshot date: 2026-11-21. `main` is cda6965 (PR #1, the engine of plan
`docs/plans/2026-11-18-abfe-engine.md`; PR #2, the README and the harness; PR #3, wording fixes
and ruff; PR #4, plan `docs/plans/2026-11-20-full-benchmarks-and-accuracy-wiki.md`: benchmarks
with the settings of the examples, the full sets, an accuracy wiki). Analysis version 10.
In progress: plan `docs/plans/2026-11-21-restraints-from-md-interactions.md` (the Boresch
restraint from the protein-ligand interactions of unrestrained MD, to become the default
`restraints.selection`); steps 1 to 9 of 14: `restraints.selection: interactions` can be chosen
and is tested on the CPU with trajectories made by hand; the default is still `auto`. Not run
on a GPU yet; no example run yet.

| item | status |
|---|---|
| Unit tests (CPU) | 565 passed (2026-11-21, the code of `main` in the image `cufep:dev` rebuilt with ProLIF 2.2.3 and MDAnalysis 2.11.0, `scripts/in_docker.sh python -m pytest`) |
| GPU tests | 17 passed (2026-11-21, one GPU, the same image and code, `scripts/in_docker.sh --gpus 0 python -m pytest tests/gpu`) |
| CI | `unit` (CPU tests in the conda environment of `docker/environment.yml`); `harness` (PROGRESS.md budget, README pair, names of commercial products, ruff) |
| TYK2 example (ejm_31, experiment -9.54 kcal/mol) | `config.yaml` sets 5 ns of production per window (PR #4; the default stays 10 ns). In the full TYK2 run: -9.84 ± 0.33 kcal/mol; one earlier calculation with the same settings: -8.94 ± 0.77 |
| A2A example (4e, experiment -12.07 kcal/mol) | in the full A2A run: -18.91 ± 0.52 kcal/mol (complex leg 44.68); one earlier calculation with the same settings: -11.95 ± 0.73 (complex leg 37.61) |
| Benchmarks | full sets run once with the settings of the examples, commit 3b45bfe (PR #4; `benchmarks/results/`, `docs/wiki/Accuracy.md`). TYK2 16/16: mean signed error -0.62, RMSE after removing the mean error 1.74, Kendall tau 0.12 (FEP+ 0.00, 0.47, 0.81). A2A 17/17: -4.12, 2.26, 0.38 (FEP+ -0.16, 1.08, 0.59). No result NOT RELIABLE; all legs ALS 0.18; a window with fewer than 50 independent samples in 12/16 and 16/17 ligands. Felis (arXiv:2603.22274 v3), same sets: TYK2 RMSE after offset 0.82, tau 0.68; A2A 1.16, 0.65 (issue #5) |
| Accuracy cross-check against OpenFE | stopped (plan of 2026-11-18, step 11) |
| Official image | built locally from `docker/Dockerfile`; not published |
| Development harness | PR #2; ruff 0.17.5 with the rules of fmeta (`scripts/lint.sh`) by PR #3 |

Open items:
- The wiki pages of PR #4 (`docs/wiki/`) are not yet on the GitHub Wiki: it has no page
  (checked 2026-11-21; the owner creates its first page). Issue #5: the differences from Felis
  and the measurements that would separate the causes of cuFEP's larger errors; nothing decided.
- The benchmarks and both example results above were made with `restraints.selection: auto`.
  The plan of 2026-11-21 changes the default at its step 12; a rerun of the benchmarks is not
  part of that plan.
- `membrane.orientation: auto` stops unless two of its three methods agree within 15 degrees and
  5 A: a stop of cuFEP's own without a published basis. Kept as it is; waits for the owner's
  decision (`.claude/rules/fep-invariants.md`).
- The conda environment is not locked (`docker/environment.yml` pins OpenMM 8.4, CUDA 13.4,
  ProLIF 2.2.3 and MDAnalysis 2.11 only); a rebuild of the image can change other packages.

## Change Log

### 2026-11-21 — [progress] `restraints.selection: interactions` can be chosen
- What: the complex leg can choose its Boresch restraint from four unrestrained NPT trajectories
  (3 ns, first 1 ns left out) that start from the end of the pre-equilibration and run as every
  plain MD of the leg does (backend, fallback, the settings of the YAML): ProLIF interactions
  and DSSP per frame, anchors by the rule of Ding et al. (thresholds of the paper as settings,
  the ranking constants of its source), the candidate also held to OpenFE's two rules,
  reference values from histograms. `interactions.json` / `.csv` per leg, `restraint_choice` in
  the results. Default unchanged (`auto`). CPU tests only so far.
- Refs: PR #6, `docs/protocol.md` ("Steps of one leg", step 2), `docs/configuration.md`
  (`restraints.interactions` and its basis), `.claude/rules/fep-invariants.md`

### 2026-11-21 — [amendment] The plain MD of a leg obeys `backend.fallback` and the precision rule
- What: the unrestrained MD of the pre-equilibration moved to stock OpenMM when the backend
  could not run it or failed, whatever `backend.fallback` said, and noted only a failure in the
  result. Now `none` stops the leg with the reason, a backend that cannot run the MD is a note
  of the result too, and the plugin gets mixed precision where double is asked, as in the
  alchemical run (it was given "double", refused it, and stock OpenMM ran the MD).
- Why: found while planning the MD of the restraint, which runs through the same function
  (`unrestrained_md`); a recovery is always in the log and the result.
- Refs: `tests/unit/test_plain_md.py`, `docs/protocol.md` ("Steps of one leg", step 1)

### 2026-11-21 — [design] Plan: the Boresch restraint from the interactions of unrestrained MD
- What: plan started. A new `restraints.selection: interactions`, to become the default: four
  unrestrained NPT runs of the complex (3 ns each, first 1 ns dropped), ProLIF interactions and
  DSSP over their frames, anchors and reference values by the rule of Ding et al.
  (arXiv:2603.22274 v3, sec. 2.1 and 2.4, and its source at commit d6dff91); the present search
  stays as `auto`. The MD runs by the settings of the YAML on the backend of the leg.
- Why: the owner's order of 2026-11-21 (the mode as the default, the source read where the
  paper is silent, both examples run once, a draft pull request).
- Refs: `docs/plans/2026-11-21-restraints-from-md-interactions.md`, issue #5

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
