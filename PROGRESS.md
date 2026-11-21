# Progress

Cross-session memory. Rules: `.claude/rules/progress-log.md`.
Older Change Log entries: `docs/progress/archive/change-log.md`.

## Product

**Goal** (plan `docs/plans/2026-11-18-abfe-engine.md`): absolute binding free energies (ABFE) on
OpenMM, on all GPUs of a machine, from two files.

- Inputs: a protein PDB and a ligand SDF with the ligand placed in the pocket. Every other
  setting has a recommended default with its basis and can be set in YAML.
- Backends: NVIDIA's cuPotential plugin for OpenMM by default, stock OpenMM as the second
  backend and the fallback. cuPotential is a compute backend, not a force field; the force
  fields are ff14SB, TIP3P, Lipid21 and OpenFF 2.2.1 with AM1-BCC charges.
- Soluble proteins in water; membrane proteins in a POPC bilayer when `membrane.enabled` is set
  (never detected).
- Protocol: double decoupling with Boresch restraints chosen from the protein-ligand
  interactions of unrestrained MD of the complex (Ding et al., arXiv:2603.22274; as OpenFE
  chooses them with `restraints.selection: auto`), Hamiltonian replica exchange with REST2 on
  the ligand and its pocket, adaptive lambda scheduling (Midgley et al. 2025), MBAR with the
  checks of the literature. Defaults and terms follow the literature and published open methods
  (`docs/configuration.md`); one calculation per ligand.
- Speed is a requirement: replicas spread over all GPUs in proportion to the work, several
  workers per GPU through NVIDIA MPS.
- Benchmarks against experiment and the FEP+ predictions of the Schrödinger public binding free
  energy benchmark (MIT licence): TYK2 (kinase) and A2A (GPCR in a bilayer).
- Version 0.1: neutral ligands only, no cofactors.
- Distribution: a Docker image built from `docker/Dockerfile`; an image with the plugin is not
  published before the licence question is settled (`docs/LICENSING.md`).

## Current Status

Snapshot date: 2026-11-21. `main` holds PR #1 (the engine of plan
`docs/plans/2026-11-18-abfe-engine.md`), PR #2 (the README and the harness), PR #3 (wording fixes
and ruff), PR #4 (plan `docs/plans/2026-11-20-full-benchmarks-and-accuracy-wiki.md`: benchmarks
with the settings of the examples, the full sets, an accuracy wiki) and PR #6, merged on the
owner's order of 2026-11-21 (plan `docs/plans/2026-11-21-restraints-from-md-interactions.md`,
done: the Boresch restraint from the protein-ligand interactions of unrestrained MD, the default
`restraints.selection`). Analysis version 10. PR #6 was tested
on the CPU with trajectories made by hand and on a GPU with the trajectories on both backends.
The whole diff was reviewed by readers who did not write it (an independent review and, on the
owner's order, Codex, whose later passes confirm the fixes): their findings are fixed, each
after it was reproduced, or answered on PR #6. Both examples ran once with the image of commit
720cdb1, in which the MD for the restraint starts from the pose that was given (below).

| item | status |
|---|---|
| Unit tests (CPU) | 654 passed (2026-11-21, the head of PR #6 in the image `cufep:dev` with ProLIF 2.2.3 and MDAnalysis 2.11.0, `scripts/in_docker.sh python -m pytest`; `main` before PR #6 in that image: 565) |
| GPU tests | 25 passed (2026-11-21, one GPU, the same image and commit, `scripts/in_docker.sh --gpus 0 python -m pytest tests/gpu`; `main` before PR #6: 17) |
| CI | `unit` (CPU tests in the conda environment of `docker/environment.yml`); `harness` (PROGRESS.md budget, README pair, names of commercial products, ruff) |
| TYK2 example (ejm_31, experiment -9.54 kcal/mol) | `config.yaml` sets 5 ns of production per window (PR #4; the default stays 10 ns). With `selection: auto` in the full TYK2 run: -9.84 ± 0.33 kcal/mol; one earlier calculation: -8.94 ± 0.77. With the default `interactions` and the MD for the restraint from the pose that was given, four calculations of the same YAML and seed (4 GPUs each, PR #6): -12.16 ± 0.32 (image of 720cdb1), -12.23 ± 0.64, -8.84 ± 0.91 and -8.33 ± 0.74 (image of 1b770da, the same code of the sampling); mean -10.39, standard deviation 2.09 kcal/mol. Complex leg 42.95, 42.91, 40.19, 39.34 (standard deviation 1.86); solvent leg 20.23, 20.13, 20.77, 20.44 (0.29); release of the restraint -10.55 to -10.57. The four chose the same six anchors (backbone of VAL 981, its hydrogen bond in 764 to 771 of 800 frames) with the same reference angles. Overlap criterion met in every leg (lowest 0.131); a window with fewer than 50 independent samples in five of the eight legs (fewest 6, a solvent leg). The 1 ns blocks of one complex leg range from 38.1 to 43.2 kcal/mol. The first version of the selection (MD for the restraint after 5 ns of free MD), four runs: -10.85, -11.99, -5.83, -9.77, standard deviation 2.7 |
| A2A example (4e, experiment -12.07 kcal/mol) | in the full A2A run: -18.91 ± 0.52 kcal/mol (complex leg 44.68); one earlier calculation with the same settings: -11.95 ± 0.73 (complex leg 37.61). With the default `interactions` (one calculation, image of commit 0f3aa32, 4 GPUs, PR #6): -12.14 ± 0.49 kcal/mol (complex leg 38.71); lowest overlap 0.102 and 0.178; a window with fewer than 50 independent samples in the complex leg; restraint from the hydrogen bond with ALA 59 (773 of 800 frames), all four trajectories on the plugin. A second calculation (fce6929): -16.85 ± 0.64 kcal/mol (complex leg 43.04), with the restraint on ASN 253 where the first had it on ALA 59. Both started the MD for the restraint after 10 ns of free MD. From the pose that was given, held through those 10 ns (one calculation, 720cdb1, 4 GPUs): -15.47 ± 1.00 kcal/mol (complex leg 41.83 ± 0.99, solvent leg 16.57 ± 0.14, release -9.80); lowest overlap 0.163 and 0.170; fewest independent samples 15 (complex leg, a window of the vanish stage, below 50) and 280; restraint from the hydrogen bond with GLU 169 (795 of 800 frames), the first of 54 ranked candidates; the hydrogen bonds with ASN 253 of the input pose are in 648 and 462 frames, the one with ALA 59 in 234; ligand RMSD 0.81 A; all four trajectories on the plugin; the same command again computed nothing again |
| Benchmarks | full sets run once with the settings of the examples, commit 3b45bfe (PR #4; `benchmarks/results/`, `docs/wiki/Accuracy.md`). TYK2 16/16: mean signed error -0.62, RMSE after removing the mean error 1.74, Kendall tau 0.12 (FEP+ 0.00, 0.47, 0.81). A2A 17/17: -4.12, 2.26, 0.38 (FEP+ -0.16, 1.08, 0.59). No result NOT RELIABLE; all legs ALS 0.18; a window with fewer than 50 independent samples in 12/16 and 16/17 ligands. Felis (arXiv:2603.22274 v3), same sets: TYK2 RMSE after offset 0.82, tau 0.68; A2A 1.16, 0.65 (issue #5) |
| Accuracy cross-check against OpenFE | stopped (plan of 2026-11-18, step 11) |
| Official image | built locally from `docker/Dockerfile`; not published |
| Development harness | PR #2; ruff 0.17.5 with the rules of fmeta (`scripts/lint.sh`) by PR #3 |

Open items:
- The wiki pages of PR #4 (`docs/wiki/`) are not yet on the GitHub Wiki: it has no page
  (checked 2026-11-21; the owner creates its first page). Issue #5: the differences from Felis
  and the measurements that would separate the causes of cuFEP's larger errors; nothing decided.
- The complex leg differs between runs of one protocol by more than its standard error (TYK2,
  four runs with the same anchors: standard deviation 2.1 kcal/mol where a run reports 0.3 to
  0.9; the restraint does not explain it and no cause is established). Plan
  `docs/plans/2026-11-21-als-initial-window-design.md` (in progress, a draft pull request; nothing
  implemented yet) tests the owner's hypothesis: more initial windows of adaptive lambda
  scheduling, placed where the sampling is slow.
- The benchmarks were made with `restraints.selection: auto`, the default until 2026-11-21; a
  rerun with the default `interactions` is not part of the plan. One calculation per example
  says nothing about a difference in accuracy (the spread between runs: the item above).
- `membrane.orientation: auto` stops unless two of its three methods agree within 15 degrees and
  5 A: a stop of cuFEP's own without a published basis. Kept as it is; waits for the owner's
  decision (`.claude/rules/fep-invariants.md`).
- The conda environment is not locked (`docker/environment.yml` pins OpenMM 8.4, CUDA 13.4,
  ProLIF 2.2.3 and MDAnalysis 2.11 only); a rebuild of the image can change other packages.

## Change Log

### 2026-11-21 — [progress] The start of adaptive lambda scheduling can be given as lambda values
- What: `lambda_schedule.adaptive.initial_windows` takes per stage a number of evenly spaced
  windows, as before, or the lambda values, which need not be evenly spaced and must be candidates;
  the start is written to `schedule.yaml` (`initial_lambda`). A start the precalculation cannot use
  (a value that is no candidate, a number that does not fit on the candidates, more windows than
  `max_windows`) is refused when the configuration is read; a number that did not fit used to stop
  the leg after its pre-equilibration. Defaults unchanged. In the image: unit tests 663 passed,
  GPU tests 25 passed (one GPU).
- Why: the owner's hypothesis (2026-11-21): more initial windows, not evenly spaced; step 3 of the plan.
- Refs: PR #7, `docs/configuration.md` (`initial_windows`), `src/cufep/alchemy/candidates.py`

### 2026-11-21 — [design] `lambda_schedule.adaptive.max_windows` is 100 in both legs
- What: the largest number of lambda windows of a leg in adaptive lambda scheduling goes from 48
  (complex) and 32 (solvent) to 100. Nothing else changes: with the threshold 0.18 the legs of
  the two full benchmark runs had 16 to 30 windows. A leg whose precalculation does not reach the
  threshold now grows to 100 before it ends with its warning.
  Unit tests in the image: 654 passed; GPU tests not run (they set the limit themselves).
- Why: the owner's order (2026-11-21); step 2 of the plan, whose candidates start from more windows.
- Refs: PR #7, `docs/plans/2026-11-21-als-initial-window-design.md`, `docs/configuration.md`
  (`max_windows`)

### 2026-11-21 — [design] Plan: the initial windows of adaptive lambda scheduling, three candidates
- What: the owner's decisions: `max_windows` 100; ALS stays on and starts from more windows that
  need not be evenly spaced; a higher threshold, for which a deficient pair is read from the overlap
  of the two neighbouring windows alone (cuFEP's own; the paper's matrix stays a setting); every
  window setting per leg in the YAML. Three candidates for the complex leg (OpenFE's values; a
  layout by where ALS put windows times the square root of the statistical inefficiency; that
  layout doubled), three runs of the TYK2 example each. Defaults change after the owner picks.
- Why: the complex leg differs between runs (standard deviation 1.9 kcal/mol); in the stored
  benchmark legs 31 of the 41 windows with fewer than 50 independent samples lie at vanish 0.7-1.0.
- Refs: `docs/plans/2026-11-21-als-initial-window-design.md` (not implemented yet), issue #5

### 2026-11-21 — [progress] PR #6 merged on the owner's order: the restraint from interactions
- What: plan `docs/plans/2026-11-21-restraints-from-md-interactions.md` is done (14 steps, an
  independent review, seven passes of Codex, both examples) and its pull request is merged into
  `main`: `restraints.selection: interactions` is the default there. The spread between runs
  (the entry below) was measured and is not explained.
- Why: the owner's order (2026-11-21).
- Refs: PR #6

### 2026-11-21 — [progress] The TYK2 example three more times: 3.9 kcal/mol between runs
- What: on the owner's order, three runs of the example as it is (image of 1b770da, 4 GPUs each,
  the same YAML and seed): -12.23 ± 0.64, -8.84 ± 0.91, -8.33 ± 0.74 kcal/mol; with the run of
  720cdb1 (-12.16, the same code of the sampling): mean -10.39, standard deviation 2.09. The four
  chose the same anchors with the same reference angles; the release of the restraint differs by
  0.02. The spread is in the complex leg (39.34 to 42.95; solvent leg 20.13 to 20.77): 0.8
  kcal/mol at the end of the restrain stage, 1.9 at the end of the discharge stage, 3.6 at the
  end. Within one run the 1 ns blocks of the complex leg range over 5 kcal/mol.
- Why: to measure the spread between runs of the method as it is (issue #5). No cause is established.
- Refs: PR #6, the plan (its section on these runs), `runs/tyk2_spread_1b770da_{1,2,3}` (untracked)

### 2026-11-21 — [progress] Step 14: both examples from the pose that was given
- What: one calculation each with the image of 720cdb1 (4 GPUs, 5 ns of production per window,
  the default `interactions`). TYK2 ejm_31: -12.16 ± 0.32 kcal/mol (experiment -9.54), restraint
  on VAL 981 as in the four earlier runs. A2A 4e: -15.47 ± 1.00 (experiment -12.07), restraint on
  GLU 169; the hydrogen bonds with ASN 253 of the input pose are kept in the MD (648 and 462 of
  800 frames). Overlap criterion met in all legs; a window with fewer than 50 independent
  samples in one leg of each. The same commands again computed nothing again.
- Why: the last step of the plan. With one calculation each and a spread between runs of 2.7
  kcal/mol (TYK2), the runs do not say whether the change of the start moved the values.
- Refs: PR #6, the plan (its section on these runs), `runs/*_interactions_720cdb1` (untracked)

### 2026-11-21 — [amendment] Confirmation passes of Codex (third to fifth): fixes
- What: three passes that confirm the fixes (reviews 6004258602, 6004435142, 6004511116 on PR #6;
  read only). After the last two: the record of a pre-equilibration that holds the solute has no
  `md_ns` (a change of that setting, which such a leg does not read, had refused the run), and the
  files of the two commits that recorded it (720cdb1, 48213b7: the example runs) stay usable;
  `equil.npz` names the precision of the platform where the platform has its own; the documents
  give the lengths with a bilayer and "at most 200 frames" for the RMSD. Not changed, reasons on
  the PR: 4, 16, 19, 29, 30 and two new points about a refusal after a changed setting.
- Why: the owner's order (a review by Codex after fixes; only what needs fixing is fixed).
- Refs: PR #6, the plan (its last section)

### 2026-11-21 — [design] The MD for the restraint starts from the pose that was given
- What: a complex leg with `restraints.selection: interactions` holds the heavy atoms of protein and
  ligand through NVT and NPT (and through `membrane.equilibration_ns` for a bilayer) and runs no
  free MD before its four trajectories; the alchemical run starts from the same configuration.
  `auto`, `manual` and the solvent leg are unchanged. The distance force constant stays 10
  kcal/mol/A^2, also the owner's decision (it is 0.1 to 0.2 of the 1.5 to 2.5 kcal/mol the restraint costs).
- Why: the owner's decision. After 10 ns of free MD two runs of the A2A example had the ligand in
  two states and chose two restraints (ALA 59, ASN 253; -12.14 and -16.85 kcal/mol); the method
  starts its trajectories from the minimised input.
- Refs: PR #6, the plan (its last section), `docs/protocol.md` (steps 1 and 2)

### 2026-11-21 — [amendment] Re-review by Codex: fixes; two runs of the TYK2 example 5 kcal/mol apart
- What: 36 findings by reading, each reproduced before it was changed (88965b8); seven not changed
  (reasons on the PR). The six coordinates of a restraint are measured as its force measures them
  (OpenMM's periodic rule for every displacement); OpenFE's flat-angle rule is asked once, with the
  constants the restraint gets, of the frames, the start and the reference angles; settings of
  records are compared without a tolerance. The TYK2 example ran twice with `interactions`: -10.85
  ± 0.65 (image 0f3aa32) and -5.83 ± 0.75 kcal/mol (fce6929): same anchors and force constants,
  the difference in the vanish stage; code of the alchemical run and analysis unchanged between them.
- Why: the owner's order (a review by Codex after fixes; only what needs fixing is fixed).
- Refs: PR #6 (review 6003995828), the plan (its last two sections), issue #5

### 2026-11-21 — [amendment] Second review (Codex) of the restraint from interactions: fixes
- What: 26 findings by reading (11 P1), each reproduced before it was changed. The nearest image in a
  skewed cell; the six coordinates of a restraint in the image its force measures in; the start held
  to the rules of one configuration; OpenFE's flat-angle rule with the force constants that
  `force_constants: auto` gives (all three selections); a candidate without reference values passed
  over; a fraction of the frames rounded down as written; whole numbers of records compared exactly;
  a restraint tied to its pre-equilibration; residues named as the input names them. Three findings
  not changed (reasons on the PR): a threshold of no frame stays a refusal by the owner's decision.
- Why: the owner's order of 2026-11-21 (a review by Codex after the push).
- Refs: PR #6 (review 6003453011; commits b4b26e6, 9b0f8fd), the plan (its last section)
