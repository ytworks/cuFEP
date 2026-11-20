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
  follow established programs and the literature (`docs/protocol.md`); one calculation per
  ligand.
- Speed is a requirement: replicas spread over all GPUs in proportion to the work, several
  workers per GPU through NVIDIA MPS.
- Benchmarks against experiment and the FEP+ predictions of the Schrödinger public binding free
  energy benchmark (MIT licence): TYK2 (kinase) and A2A (GPCR in a bilayer).
- Version 0.1: neutral ligands only, no cofactors.
- Distribution: a Docker image built from `docker/Dockerfile`; an image with the plugin is not
  published before the licence question is settled (`docs/LICENSING.md`).

## Current Status

Snapshot date: 2026-11-20. `main` is 31a720e (PR #1 merged): the engine of plan
`docs/plans/2026-11-18-abfe-engine.md`, revisions 1 to 15. Analysis version 10.

| item | status |
|---|---|
| Unit tests (CPU) | 550 passed on the code of 31a720e (2026-11-20, image `cufep:dev`, `scripts/in_docker.sh python -m pytest`); PR #2 adds the tests of the harness |
| GPU tests | 17 passed on the code of 31a720e (2026-11-20, one GPU, `scripts/in_docker.sh --gpus 0 python -m pytest tests/gpu`) |
| CI | `unit` (CPU tests in the conda environment of `docker/environment.yml`); `harness` added by PR #2 |
| TYK2 example (ejm_31, experiment -9.54 kcal/mol) | one calculation with ALS threshold 0.18, REST2 on the pocket and 5 ns of production per window: -8.94 ± 0.77 kcal/mol (`docs/configuration.md`, `sampling.rest2.region`). Not run with the default of 10 ns |
| A2A example (4e, experiment -12.07 kcal/mol) | the same settings: -11.95 ± 0.73 kcal/mol. Not run with the default of 10 ns |
| Benchmarks | smoke and five-ligand subset tiers ran on earlier versions to check the pipeline; the full sets (TYK2 16 ligands, A2A 17 compounds) not run |
| Accuracy cross-check against OpenFE | stopped (plan of 2026-11-18, step 11) |
| Official image | built locally from `docker/Dockerfile`; not published |
| Development harness | PR #2 (plan `docs/plans/2026-11-20-oss-readme-and-harness.md`) |

Open items:
- The examples with the default settings (10 ns of production per window) in an image of the
  current code; the full benchmarks. Both wait for the owner's decision.
- Lint: ruff is not used yet. With fmeta's rules it reports about 1800 findings in the current
  tree, most of them line length; it is left to a pull request of its own.
- The conda environment is not locked (`docker/environment.yml` pins OpenMM 8.4 and CUDA 13.4
  only); a rebuild of the image can change other packages.

## Change Log

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
