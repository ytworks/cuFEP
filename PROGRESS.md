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

Snapshot date: 2026-11-21. `main` is cda6965 (PR #1, the engine of plan
`docs/plans/2026-11-18-abfe-engine.md`; PR #2, the README and the harness; PR #3, wording fixes
and ruff; PR #4, plan `docs/plans/2026-11-20-full-benchmarks-and-accuracy-wiki.md`: benchmarks
with the settings of the examples, the full sets, an accuracy wiki). Analysis version 10.
In progress: plan `docs/plans/2026-11-21-restraints-from-md-interactions.md` (the Boresch
restraint from the protein-ligand interactions of unrestrained MD, to become the default
`restraints.selection`); steps 1 to 13 of 14: `interactions` is the default. Tested on the CPU
with trajectories made by hand and on a GPU with the trajectories on both backends. The whole
diff was reviewed twice by readers who did not write it (an independent review and, on the
owner's order, Codex): their findings are fixed, each after it was reproduced, or answered on
PR #6. Both examples ran once with the image of commit 0f3aa32 (below); step 14 ends with one
run of each with the image of the reviewed code.

| item | status |
|---|---|
| Unit tests (CPU) | 654 passed (2026-11-21, the head of PR #6 in the image `cufep:dev` with ProLIF 2.2.3 and MDAnalysis 2.11.0, `scripts/in_docker.sh python -m pytest`; `main` in that image: 565) |
| GPU tests | 25 passed (2026-11-21, one GPU, the same image and commit, `scripts/in_docker.sh --gpus 0 python -m pytest tests/gpu`; `main`: 17) |
| CI | `unit` (CPU tests in the conda environment of `docker/environment.yml`); `harness` (PROGRESS.md budget, README pair, names of commercial products, ruff) |
| TYK2 example (ejm_31, experiment -9.54 kcal/mol) | `config.yaml` sets 5 ns of production per window (PR #4; the default stays 10 ns). In the full TYK2 run: -9.84 ± 0.33 kcal/mol; one earlier calculation with the same settings: -8.94 ± 0.77. With the default `restraints.selection: interactions` (one calculation, image of commit 0f3aa32, 4 GPUs, PR #6): -10.85 ± 0.65 kcal/mol; lowest overlap 0.159 and 0.151; a window with fewer than 50 independent samples in both legs; restraint from the hinge hydrogen bond of VAL 981 (672 of 800 frames), all four trajectories on the plugin. A second calculation with the image of commit fce6929: -5.83 ± 0.75 kcal/mol (complex leg 36.33 against 41.47; same anchors; both published criteria met in both legs); the 5.1 kcal/mol between the two runs is in the vanish stage. Two more (0f3aa32 again, 88965b8): -11.99 ± 0.52 and -9.77 ± 0.60; the four: mean -9.61, standard deviation 2.7 kcal/mol. All of these started the MD for the restraint after 5 ns of free MD |
| A2A example (4e, experiment -12.07 kcal/mol) | in the full A2A run: -18.91 ± 0.52 kcal/mol (complex leg 44.68); one earlier calculation with the same settings: -11.95 ± 0.73 (complex leg 37.61). With the default `interactions` (one calculation, image of commit 0f3aa32, 4 GPUs, PR #6): -12.14 ± 0.49 kcal/mol (complex leg 38.71); lowest overlap 0.102 and 0.178; a window with fewer than 50 independent samples in the complex leg; restraint from the hydrogen bond with ALA 59 (773 of 800 frames), all four trajectories on the plugin. A second calculation (fce6929): -16.85 ± 0.64 kcal/mol (complex leg 43.04), with the restraint on ASN 253 where the first had it on ALA 59. Both started the MD for the restraint after 10 ns of free MD |
| Benchmarks | full sets run once with the settings of the examples, commit 3b45bfe (PR #4; `benchmarks/results/`, `docs/wiki/Accuracy.md`). TYK2 16/16: mean signed error -0.62, RMSE after removing the mean error 1.74, Kendall tau 0.12 (FEP+ 0.00, 0.47, 0.81). A2A 17/17: -4.12, 2.26, 0.38 (FEP+ -0.16, 1.08, 0.59). No result NOT RELIABLE; all legs ALS 0.18; a window with fewer than 50 independent samples in 12/16 and 16/17 ligands. Felis (arXiv:2603.22274 v3), same sets: TYK2 RMSE after offset 0.82, tau 0.68; A2A 1.16, 0.65 (issue #5) |
| Accuracy cross-check against OpenFE | stopped (plan of 2026-11-18, step 11) |
| Official image | built locally from `docker/Dockerfile`; not published |
| Development harness | PR #2; ruff 0.17.5 with the rules of fmeta (`scripts/lint.sh`) by PR #3 |

Open items:
- The wiki pages of PR #4 (`docs/wiki/`) are not yet on the GitHub Wiki: it has no page
  (checked 2026-11-21; the owner creates its first page). Issue #5: the differences from Felis
  and the measurements that would separate the causes of cuFEP's larger errors; nothing decided.
- The benchmarks were made with `restraints.selection: auto`, the default until 2026-11-21; a
  rerun with the default `interactions` is not part of the plan. One calculation per example
  says nothing about a difference in accuracy (the spread between runs is about 1.5 kcal/mol).
- `membrane.orientation: auto` stops unless two of its three methods agree within 15 degrees and
  5 A: a stop of cuFEP's own without a published basis. Kept as it is; waits for the owner's
  decision (`.claude/rules/fep-invariants.md`).
- The conda environment is not locked (`docker/environment.yml` pins OpenMM 8.4, CUDA 13.4,
  ProLIF 2.2.3 and MDAnalysis 2.11 only); a rebuild of the image can change other packages.

## Change Log

### 2026-11-21 — [design] The MD for the restraint starts from the pose that was given
- What: a complex leg with `restraints.selection: interactions` holds the heavy atoms of protein and
  ligand through NVT and NPT (and through `membrane.equilibration_ns` for a bilayer) and runs no
  free MD before its four trajectories; the alchemical run starts from the same configuration.
  `auto`, `manual` and the solvent leg are unchanged. The distance force constant stays 10
  kcal/mol/A^2 (measured: it is 0.1 to 0.2 of the 1.5 to 2.5 kcal/mol the restraint costs).
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

### 2026-11-21 — [amendment] Review of the restraint from interactions; plain MD in the precision asked
- What: an independent review of `main..f5a77ce` found no input with a wrong number that passes
  every check; its four P2 and four P3 are fixed. The ligand next to the protein is in the image
  of its nearest protein atom (one rule for the interactions, the search of `auto` and the pose
  RMSD); a run asks guest anchors of the ligands whose complex leg it runs; the record of a
  pre-equilibration holds `backend.fallback: none`; `host_max_distance_nm` is documented for every
  selection. Plain MD runs in the precision of the YAML, single too (it ran in mixed then).
- Why: step 13 of the plan; the precision is the owner's decision of 2026-11-21.
- Refs: PR #6 (1739162, 0db38f6, 8b7ba95, 75a1697), the plan (its last section), `docs/configuration.md`

### 2026-11-21 — [design] The default `restraints.selection` is `interactions`
- What: a configuration that names no selection chooses the Boresch restraint from the
  protein-ligand interactions of four unrestrained trajectories of the complex; `auto` (the
  search of OpenFE's protocol, the default until now) and `manual` are selectable. Both
  examples and with them both benchmark sets run so; the smoke tier of a benchmark shortens
  those trajectories. A restraint made with `auto` is refused by a run of the default (another
  selection): give `restraints.selection: auto` to go on with such a directory.
- Why: the owner's decision of 2026-11-21.
- Refs: PR #6, `docs/configuration.md` (`restraints.selection`), `docs/protocol.md` (step 2)

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
