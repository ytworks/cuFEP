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

Snapshot date: 2026-11-22. `main` holds PR #1 (the engine of plan
`docs/plans/2026-11-18-abfe-engine.md`), PR #2 (the README and the harness), PR #3 (wording fixes
and ruff), PR #4 (plan `docs/plans/2026-11-20-full-benchmarks-and-accuracy-wiki.md`: benchmarks
with the settings of the examples, the full sets, an accuracy wiki), PR #6 (plan
`docs/plans/2026-11-21-restraints-from-md-interactions.md`, done: the Boresch restraint from the
protein-ligand interactions of unrestrained MD, the default `restraints.selection`) and PR #7,
merged on the owner's order of 2026-11-22 (plan
`docs/plans/2026-11-21-als-initial-window-design.md`, done: adaptive lambda scheduling of the
complex leg starts from 30 windows and reads the overlap of a pair alone, window settings per
leg, `max_windows` 100). Analysis version 10. The whole diff of PR #7 was reviewed by readers
who did not write it (two independent reviews and six passes of Codex, the last without a
finding): their findings are fixed, each after it was reproduced, or answered on PR #7. Both
examples ran once with the image of commit 720cdb1, in which the MD for the restraint starts
from the pose that was given, and the TYK2 example three times with the present defaults of the
complex leg (below). The branch of PR #8 (draft) holds one run of both full benchmark sets with
the present defaults and the accuracy pages written from it (the row "Benchmarks").

| item | status |
|---|---|
| Unit tests (CPU) | 680 passed (2026-11-22, PR #7 at da052f6 in the image `cufep:dev`, with ProLIF 2.2.3 and MDAnalysis 2.11.0, `scripts/in_docker.sh python -m pytest`; its later commits change documents and a script of `spikes/` only) |
| GPU tests | 25 passed (2026-11-22, one GPU, PR #7 at d5f9c3f with the fixes of its independent review, in the same image, `scripts/in_docker.sh --gpus 0 python -m pytest tests/gpu`; its later commits do not touch what these tests cover) |
| CI | `unit` (CPU tests in the conda environment of `docker/environment.yml`); `harness` (PROGRESS.md budget, README pair, names of commercial products, ruff) |
| TYK2 example (ejm_31, experiment -9.54 kcal/mol) | `config.yaml` sets 5 ns of production per window (PR #4; the default stays 10 ns). With `selection: auto` in the full TYK2 run: -9.84 ± 0.33 kcal/mol; one earlier calculation: -8.94 ± 0.77. With the default `interactions` and the MD for the restraint from the pose that was given, four calculations of the same YAML and seed (4 GPUs each, PR #6): -12.16 ± 0.32 (image of 720cdb1), -12.23 ± 0.64, -8.84 ± 0.91 and -8.33 ± 0.74 (image of 1b770da, the same code of the sampling); mean -10.39, standard deviation 2.09 kcal/mol. Complex leg 42.95, 42.91, 40.19, 39.34 (standard deviation 1.86); solvent leg 20.23, 20.13, 20.77, 20.44 (0.29); release of the restraint -10.55 to -10.57. The four chose the same six anchors (backbone of VAL 981, its hydrogen bond in 764 to 771 of 800 frames) with the same reference angles. Overlap criterion met in every leg (lowest 0.131); a window with fewer than 50 independent samples in five of the eight legs (fewest 6, a solvent leg). The 1 ns blocks of one complex leg range from 38.1 to 43.2 kcal/mol. The first version of the selection (MD for the restraint after 5 ns of free MD), four runs: -10.85, -11.99, -5.83, -9.77, standard deviation 2.7. In the full TYK2 run with the present defaults (e77a6c5, PR #8): -10.08 ± 0.37 (complex leg 40.43, solvent leg 19.79; 33 and 18 windows; lowest overlap 0.154 and 0.189; a window with fewer than 50 independent samples in the complex leg) |
| Starts of the complex leg (plan `docs/plans/2026-11-21-als-initial-window-design.md`, PR #7) | A is the default since 2026-11-22 (the owner's choice), so its three runs are runs of the TYK2 example as it is now. TYK2 example, three starts of the complex leg, three calculations each with the same YAML and seed (image of 5e2cf33, 4 GPUs each; the overlap of a pair alone, threshold 0.30; solvent leg as the default), kcal/mol. A (30 windows, those of mode fixed): -9.36 ± 0.56, -9.63 ± 0.37, -9.33 ± 0.51; mean -9.44, standard deviation 0.17. B (30 windows, denser at vanish 0.7 to 0.9): -10.01 ± 0.34, -9.64 ± 0.52, -11.71 ± 0.53; -10.45, 1.11. C (59 windows placed as B): -11.43 ± 0.46, -6.41 ± 0.59, -7.64 ± 0.43; -8.50, 2.62. The four runs of the row above: -10.39, 2.09. Complex leg, standard deviation: 0.20, 1.23, 2.20 (1.86). Lowest overlap 0.141 to 0.153 (A), 0.128 to 0.129 (B), 0.081 to 0.085 (C); a window with fewer than 50 independent samples in the complex leg of 2, 1 and 1 of the three runs; 0 or 1 round trip in the complex leg of every run. Three runs give a standard deviation only roughly. Tables: the plan (`spikes/m19_window_runs.py`) |
| A2A example (4e, experiment -12.07 kcal/mol) | in the full A2A run: -18.91 ± 0.52 kcal/mol (complex leg 44.68); one earlier calculation with the same settings: -11.95 ± 0.73 (complex leg 37.61). With the default `interactions` (one calculation, image of commit 0f3aa32, 4 GPUs, PR #6): -12.14 ± 0.49 kcal/mol (complex leg 38.71); lowest overlap 0.102 and 0.178; a window with fewer than 50 independent samples in the complex leg; restraint from the hydrogen bond with ALA 59 (773 of 800 frames), all four trajectories on the plugin. A second calculation (fce6929): -16.85 ± 0.64 kcal/mol (complex leg 43.04), with the restraint on ASN 253 where the first had it on ALA 59. Both started the MD for the restraint after 10 ns of free MD. From the pose that was given, held through those 10 ns (one calculation, 720cdb1, 4 GPUs): -15.47 ± 1.00 kcal/mol (complex leg 41.83 ± 0.99, solvent leg 16.57 ± 0.14, release -9.80); lowest overlap 0.163 and 0.170; fewest independent samples 15 (complex leg, a window of the vanish stage, below 50) and 280; restraint from the hydrogen bond with GLU 169 (795 of 800 frames), the first of 54 ranked candidates; the hydrogen bonds with ASN 253 of the input pose are in 648 and 462 frames, the one with ALA 59 in 234; ligand RMSD 0.81 A; all four trajectories on the plugin; the same command again computed nothing again. In the full A2A run with the present defaults (e77a6c5, PR #8): -13.50 ± 0.34 (complex leg 39.76, solvent leg 16.54; 30 and 17 windows; lowest overlap 0.169 and 0.178; fewest independent samples 5, complex leg; restraint from the stacking with PHE 168, 785 of 800 frames) |
| Benchmarks | both full sets once with the present defaults, image of e77a6c5 on eight GPUs, 2026-11-22 (PR #8; `benchmarks/results/`, `docs/wiki/Accuracy.md`), both exit 0, one calculation per ligand, kcal/mol with 95 % bootstrap intervals over ligands. TYK2 16/16: mean signed error -1.20 [-1.83, -0.60], RMSE after removing the mean error 1.25 [0.75, 1.62], Kendall tau 0.47 [0.12, 0.75] (FEP+ 0.00, 0.47, 0.81). A2A 17/17: -4.14 [-5.36, -2.85], 2.68 [1.76, 3.24], 0.34 [-0.03, 0.67] (FEP+ -0.16, 1.08, 0.59). By the definitions of the Felis paper (R^2, offset, RMSE after offset, MAE, correct order ratio): TYK2 0.35, 1.20, 1.25, 0.94, 0.91; A2A 0.19, 4.14, 2.68, 2.16, 0.81 (Felis, arXiv:2603.22274 v3, three calculations averaged: 0.69, 0.04, 0.82, 0.56, 0.97 and 0.55, 2.74, 1.16, 0.88, 1.00; issue #5). No result NOT RELIABLE (lowest overlap 0.140 to 0.172 and 0.105 to 0.169); a window with fewer than 50 independent samples in 12/16 and 14/17 ligands (fewest 11 for TYK2; 4 and 5 in the complex legs of 4q and 4e). From the files of the runs: every restraint by `interactions` (TYK2: the hydrogen bond with VAL 981 in all 16; A2A: ASN 253 in 11, stacking with PHE 168 in 5, GLU 169 in 1; 736 to 797 of 800 frames); the 33 complex legs adaptive by the overlap of a pair alone with 0.30, 31 to 36 and 30 to 32 windows from 30 (7 A2A legs kept 30); the 33 solvent legs by the matrix of the stage with 0.18, 16 to 21 windows; all 66 legs on the plugin, no fallback; `cufep benchmark report` on the summaries gives the recorded reports again. The run of 3b45bfe (PR #4; `selection: auto`, the complex legs from 3, 5 and 9 windows by 0.18 of the stage matrix; record in the tree of cda6965): TYK2 -0.62, 1.74, 0.12; A2A -4.12, 2.26, 0.38; ligand by ligand the two runs differ by -3.76 to +2.27 (TYK2) and -4.78 to +5.41 (A2A). The subset tier is not run apart: its ligands are in the full sets with the same settings |
| Accuracy cross-check against OpenFE | stopped (plan of 2026-11-18, step 11) |
| Official image | built locally from `docker/Dockerfile`; not published |
| Development harness | PR #2; ruff 0.17.5 with the rules of fmeta (`scripts/lint.sh`) by PR #3 |

Open items:
- The wiki pages (`docs/wiki/`, rewritten in PR #8) are not yet on the GitHub Wiki: it has no
  page (checked 2026-11-22; the owner creates its first page). Issue #5: the differences from Felis
  and the measurements that would separate the causes of cuFEP's larger errors; nothing decided.
- The complex leg differs between runs of one protocol by more than its standard error (TYK2,
  four runs with the same anchors: standard deviation 2.1 kcal/mol where a run reports 0.3 to
  0.9; the restraint does not explain it and no cause is established). Plan
  `docs/plans/2026-11-21-als-initial-window-design.md` (done, PR #7) tested the
  owner's hypothesis, more initial windows of adaptive lambda scheduling, placed by design (the
  row "Starts of the complex leg"). Start A is the default of the complex leg since 2026-11-22:
  its three runs differ by 0.30 kcal/mol at most; B and C, with as many and more windows, by
  2.07 and 5.02, and that is not explained (C is not investigated, the owner's order).
- Plan `docs/plans/2026-11-22-full-benchmarks-with-the-present-defaults.md` (PR #8, draft): both
  full sets are recorded and the wiki pages, the READMEs, `docs/protocol.md` and
  `docs/configuration.md` give the run of e77a6c5. By the owner's order no unit tests, GPU tests
  or reviews were run for it (it changes no file of `src/`). Left: the owner's review and merge,
  the GitHub Wiki.
- Workers during the precalculation of adaptive lambda scheduling (seen in both runs of PR #8; no
  fix yet): a leg asks once for one worker per window of its start and keeps them through all
  iterations (`request_devices(..., pinned=True)` in `src/cufep/workflow.py`), also while an
  iteration runs one to five windows. With a start of 30 windows two complex legs hold 59 of the
  64 workers of eight GPUs, and a leg that starts meanwhile makes its whole precalculation with
  what is left: 6 of the 16 complex legs of TYK2 and 11 of the 17 of A2A began it with one or
  two workers, and the last leg of A2A made it on one worker while seven GPUs had no work. The
  free energies do not depend on it.
- Windows much closer than they need to be read low in the overlap matrix of a leg, which the
  analysis checks: 81 unit-width harmonic windows 0.05 widths apart overlap by 0.50 as pairs and
  read 0.016, below the level of 0.03, so such a result is `NOT RELIABLE` although its free
  energy is right, and the message advises more windows. Reachable with a start of many windows
  (`max_windows` 100). The owner's decision (2026-11-21): this may happen, what counts is the
  accuracy. The level, the verdict and its message are unchanged; such a result is reported and
  compared, not left out (`docs/protocol.md`, "Why the threshold cannot go much higher").
- The two runs of each set (3b45bfe and e77a6c5) are one calculation per ligand each and differ
  in two defaults at once (`restraints.selection`, the start of the complex leg): their
  difference does not by itself show the effect of either. How much one calculation varies is
  measured for ejm_31 only (the item on the complex leg above); for A2A it is not measured.
- `membrane.orientation: auto` stops unless two of its three methods agree within 15 degrees and
  5 A: a stop of cuFEP's own without a published basis. Kept as it is; waits for the owner's
  decision (`.claude/rules/fep-invariants.md`).
- The conda environment is not locked (`docker/environment.yml` pins OpenMM 8.4, CUDA 13.4,
  ProLIF 2.2.3 and MDAnalysis 2.11 only); a rebuild of the image can change other packages.

## Change Log

### 2026-11-22 — [progress] A2A full with the present defaults: 17 of 17; the wiki from this run
- What: one calculation per compound, image of e77a6c5, eight GPUs, exit 0; recorded as the run
  wrote it in `benchmarks/results/a2a/`. kcal/mol: mean signed error -4.14 [-5.36, -2.85], RMSE
  after removing the mean error 2.68 [1.76, 3.24], Kendall tau 0.34 [-0.03, 0.67] (FEP+ -0.16,
  1.08, 0.59; the run of 3b45bfe -4.12, 2.26, 0.38; Felis: offset 2.74, RMSE after offset 1.16,
  tau 0.65). No result NOT RELIABLE (lowest overlap 0.105); a window with fewer than 50
  independent samples in the complex leg of 14/17. 4e: -13.50 ± 0.34. `docs/wiki/Accuracy.md`
  and `Accuracy-ja.md` are rewritten from the run of both sets, with a section that puts the run
  of 3b45bfe beside it; both READMEs, `docs/protocol.md` and `docs/configuration.md` follow.
- Why: the owner's order (2026-11-22); steps 3 to 6 of the plan.
- Refs: PR #8, `benchmarks/results/README.md`, `docs/wiki/Accuracy.md`, `spikes/wiki_tables.py`

### 2026-11-22 — [progress] TYK2 full with the present defaults: 16 of 16 ligands
- What: one calculation per ligand, image of e77a6c5, eight GPUs, exit 0; recorded as the run
  wrote it in `benchmarks/results/tyk2/` (the record of 3b45bfe is in the tree of cda6965).
  kcal/mol: mean signed error -1.20 [-1.83, -0.60], RMSE after removing the mean error 1.25
  [0.75, 1.62], Kendall tau 0.47 [0.12, 0.75] (FEP+ 0.00, 0.47, 0.81; the run of 3b45bfe -0.62,
  1.74, 0.12; Felis, three calculations averaged: offset 0.04, RMSE after offset 0.82, tau 0.68).
  No result NOT RELIABLE (lowest overlap 0.140); a window with fewer than 50 independent samples
  in 12/16. Every restraint by `interactions` (VAL 981); complex legs 31 to 36 windows from 30.
  Seen in the run: how the precalculation holds workers (Open items). A2A is being calculated.
- Why: the owner's order (2026-11-22); step 3 and 4 of the plan.
- Refs: PR #8, `benchmarks/results/README.md`, `spikes/benchmark_progress.py` (Felis rows)

### 2026-11-22 — [progress] Plan started: both full benchmark sets once with the present defaults
- What: plan `docs/plans/2026-11-22-full-benchmarks-with-the-present-defaults.md` (in progress).
  TYK2 (16 ligands), then A2A (17 compounds): tier full, one calculation per ligand, eight GPUs,
  in the image built from e77a6c5 (`main` after PR #7; its conda environment is the layer of the
  image of 5e2cf33). No file of the code, the examples or the benchmarks changes: they follow the
  defaults (`restraints.selection: interactions`; the complex leg from 30 windows, the overlap of
  a pair alone, 0.30). The results will replace `benchmarks/results/` and the numbers of
  `docs/wiki/Accuracy.md`. By the owner's order no unit tests, GPU tests or reviews are run for
  this plan. No result yet.
- Why: the owner's order (2026-11-22).
- Refs: the plan, its pull request

### 2026-11-22 — [progress] PR #7 merged on the owner's order: the start of the complex leg
- What: plan `docs/plans/2026-11-21-als-initial-window-design.md` is done (10 steps, two
  independent reviews, six passes of Codex, the last without a finding; nine runs of the TYK2
  example) and its pull request is merged into `main`: there adaptive lambda scheduling of the
  complex leg starts from 30 windows and reads the overlap of a pair alone (threshold 0.30),
  `max_windows` is 100 and the window settings can differ between the legs. Not run with these
  defaults: the A2A example, the subset and full benchmarks.
- Why: the owner's order (2026-11-22).
- Refs: PR #7

### 2026-11-22 — [amendment] Review by Codex of the new defaults: fixes and a correction
- What: review 6007410417 (seven P2), each reproduced. Correction of `docs/configuration.md` and
  of the entry below: an element of the matrix of a stage can reach 0.30, at an end of a stage (at
  most 0.333 among 784 recorded pairs); 26 or 27 of the 29 pairs of the default start read at or
  below 0.30 there in the runs of A (6 by the pair alone), and the run to `max_windows` was of
  harmonic windows. `cufep benchmark run` prints the notes about the file of its example. The
  figure script counts the last exchange, reads results of earlier versions and draws moves rarer
  than 0.001. B and C came from 37 legs, not 33. Declined: an old YAML with `max_windows.complex`
  below 30 is refused with the setting to add. In the image: unit tests 680 passed.
- Why: the review after the push (the working agreement of PR #7).
- Refs: PR #7 (review 6007410417 and its replies), `docs/protocol.md`, `docs/configuration.md`

### 2026-11-22 — [design] The complex leg starts from the windows of mode fixed and reads the overlap of a pair
- What: defaults of `lambda_schedule.adaptive` for the complex leg: `initial_windows` restrain 6,
  discharge 11, vanish the 15 values of `windows.complex.vanish` (3, 5 and 9 until now);
  `overlap_matrix` pair with `overlap_threshold` 0.30 (stage with 0.18 until now). The solvent leg
  is unchanged (5 and 9 windows, stage, 0.18). The examples and the benchmarks set none of these
  and follow the defaults (`cufep plan` of both examples; the smoke tier of both sets, exit 0).
  Not run with them: the A2A example, the subset and full benchmarks. A leg whose threshold is
  not given has that of the matrix it reads (0.18 stage, 0.30 pair; an independent review:
  `overlap_matrix: stage` alone kept 0.30 and went to `max_windows`); a threshold without its
  matrix gets a note. In the image: unit tests 679 passed, GPU tests 25 passed.
- Why: the owner's choice (2026-11-22) of start A from the nine runs of the entry below.
- Refs: PR #7, `docs/configuration.md` (basis), `docs/protocol.md` ("The starts of the complex
  leg that were compared"), `.claude/rules/fep-invariants.md`

### 2026-11-22 — [progress] Three starts of the complex leg on the TYK2 example: nine runs
- What: step 8 of the plan. ejm_31, 5 ns, image of 5e2cf33, three calculations per start with the
  same YAML and seed; complex leg by the overlap of a pair alone, threshold 0.30; kcal/mol,
  experiment -9.54. A (30 windows, those of mode fixed): -9.36, -9.63, -9.33 (standard deviation
  0.17). B (30, denser at vanish 0.7 to 0.9): -10.01, -9.64, -11.71 (1.11). C (59, placed as B):
  -11.43, -6.41, -7.64 (2.62). The default, four runs: 2.09. Standard errors 0.34 to 0.59. Lowest
  overlap 0.081 (C) to 0.153 (A); 0 or 1 round trip in every complex leg. Three runs give a
  standard deviation only roughly. `spikes/m20_replica_exchange_figures.py` draws the moves of
  the replicas. The owner chose A (2026-11-22).
- Why: the owner's hypothesis of 2026-11-21: more initial windows, placed by design.
- Refs: PR #7, the plan (results), `spikes/m19_window_runs.py`, `runs/tyk2_windows_*` (untracked)
