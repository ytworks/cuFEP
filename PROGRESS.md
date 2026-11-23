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

Snapshot date: 2026-11-23. `main` holds PR #1 (the engine of plan
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
complex leg (below). PR #8, merged on the owner's order of 2026-11-22, added one run of both full
benchmark sets with the present defaults and the accuracy pages written from it (the row "Benchmarks").

| item | status |
|---|---|
| Unit tests (CPU) | 727 passed (2026-11-23, commit 0c9dcce of the branch `feat/relocation-mc`: PR #9 and the water relocation moves, in the image `cufep:dev`, `scripts/in_docker.sh python -m pytest`) |
| GPU tests | 30 passed (2026-11-23, two GPUs, commit ecf0e0e, `scripts/in_docker.sh --gpus 0,1 python -m pytest tests/gpu`); the four tests of the moves again on 0c9dcce, passed |
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
  and the measurements that would separate the causes of cuFEP's larger errors.
- Plan `docs/plans/2026-11-22-accuracy-causes-and-fixes.md` (in progress, PR #9): every remedy
  proposed for the errors gets a treatment; defaults change only by the owner's decision. One
  calculation varies by a standard deviation of 1.57 kcal/mol (ejm_49) and 1.72 (4k). What the
  windows start from moves the complex leg by about 5 (ejm_49 35.3 against 40.9; 4k 44.3 against
  39.0). The 16 TYK2 ligands with `alchemy.annihilate_electrostatics` (41760a1, once each): mean
  signed error -0.43 [-1.31, 0.41], RMSE after removing it 1.77 [1.16, 2.17], tau 0.20 [-0.12,
  0.48]; ligand by ligand +0.76 ± 0.51 from the run of e77a6c5. Water relocation moves (PR #10,
  default off): 4k -16.61 and ejm_49 -11.42 with them, inside the range of the runs without.
  Running since 2026-11-23T12:17Z: both full sets again with the image of 41760a1 (the owner).
- The complex leg differs between runs of one protocol by more than its standard error (TYK2,
  four runs with the same anchors: standard deviation 2.1 kcal/mol where a run reports 0.3 to
  0.9; the restraint does not explain it and no cause is established). Plan
  `docs/plans/2026-11-21-als-initial-window-design.md` (done, PR #7) tested the
  owner's hypothesis, more initial windows of adaptive lambda scheduling, placed by design (the
  row "Starts of the complex leg"). Start A is the default of the complex leg since 2026-11-22:
  its three runs differ by 0.30 kcal/mol at most; B and C, with as many and more windows, by
  2.07 and 5.02, and that is not explained (C is not investigated, the owner's order).
- Plan `docs/plans/2026-11-22-full-benchmarks-with-the-present-defaults.md` (done, PR #8): by the
  owner's order no unit tests, GPU tests or reviews were run for it (no file of `src/` changed).
- Workers during the precalculation of adaptive lambda scheduling (both runs of PR #8; no fix yet):
  a leg asks once for one worker per window of its start and keeps them through all iterations
  (`request_devices(..., pinned=True)` in `src/cufep/workflow.py`). With a start of 30 windows two
  complex legs hold 59 of the 64 workers of eight GPUs, and a leg that starts meanwhile makes its
  precalculation with what is left: 6 of the 16 complex legs of TYK2 and 11 of the 17 of A2A began
  it with one or two workers. A leg with one worker started every window from the end of the one
  before it, a leg with more from a 5 ps walk: 4 complex legs of TYK2 and 7 of A2A in the runs of
  e77a6c5. For 4k that leg is 5.86 kcal/mol above the mean of three runs started by the walk
  (standard deviation 1.78). The branch of PR #9 makes the way a setting (default: the walk).
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
  measured for ejm_31 (the item on the complex leg above), ejm_49 and 4k (the item on the plan).
- `membrane.orientation: auto` stops unless two of its three methods agree within 15 degrees and
  5 A: a stop of cuFEP's own without a published basis. Kept as it is; waits for the owner's
  decision (`.claude/rules/fep-invariants.md`).
- The conda environment is not locked (`docker/environment.yml` pins OpenMM 8.4, CUDA 13.4,
  ProLIF 2.2.3 and MDAnalysis 2.11 only); a rebuild of the image can change other packages.

## Change Log

### 2026-11-23 — [progress] E1 and E3 verified ligand by ligand; the full benchmarks run again with E1 alone
- What: E1 (the windows of a leg start from the walk): the complex legs of four A2A ligands
  that the run of e77a6c5 had started one window from another, made again, kcal/mol: 4k -20.32
  to -14.69 (mean of three), 4n -19.17 to -17.38, 4c -18.75 to -15.19, 4q -14.11 to -12.52; all
  towards experiment, by 3.1 on average. E3 (the decoupled-side windows from the relaxed
  decoupled end): the complex leg of ejm_49 changed by -5.65, that of 4k by +0.47 ± 0.63 (its
  pocket does not close). The owner stopped both verifications there and decided: no remedy for
  E3 in this plan; both full sets run again with E1 alone (image of 41760a1, the head of PR #9,
  A2A first), to be compared with the run of e77a6c5. No result yet.
- Why: the owner's order of 2026-11-23 (verify what had an effect, then end the session).
- Refs: the plan ("E1 の検証", "E3 の検証"), `runs/acc_e1_a2a`, `runs/acc_e3_a2a_4k`, `runs/configs/chain_full_benchmarks_e1.sh` (untracked)

### 2026-11-23 — [progress] Water relocation moves, one whole calculation per ligand; what is verified next
- What: with the moves (4096 trial poses, half of the anchors in the sphere), all eight GPUs,
  kcal/mol: 4k -16.61 ± 0.37 (without them -14.49, -13.07, -16.50; experiment -11.54), ejm_49
  -11.42 ± 0.52 (-11.84, -8.92, -9.40; -7.75). On ejm_49 the sections of the coupled side rise
  as moves are accepted, more into the sphere than out of it (the entry below said the reason
  was not established); checks on that leg found no bias (`spikes/m29_*`). A third ligand was
  stopped by the owner; the default stays off. The owner's order for the rest of the session:
  verify E1 (the walk for the legs the benchmark started one window from another) and E3 (the
  decoupled-side windows from the relaxed decoupled end), one ligand at a time on eight GPUs.
- Why: step 6 of the plan; the owner judges after every ligand.
- Refs: the plan ("評価 2"), PR #10, `runs/acc_reloc2_a2a_4k`, `runs/acc_moves_tyk2` (untracked)

### 2026-11-23 — [progress] Water relocation moves on ejm_49: the complex leg from two start constructions
- What: one run each, image of ecf0e0e, 4096 trial poses, half of the anchors in the sphere, the
  windows and start configurations of the runs they are compared with, kcal/mol. Started by the
  walk: 43.09 ± 0.75 (without the moves 40.91 ± 0.36); its 7 decoupled-side windows started from
  the relaxed decoupled end: 39.29 ± 0.35 (35.26 ± 0.48). The two differ by 3.80 (5.65), of which
  3.22 (4.55) between sterics 0.25 and 0, where at the end of the run the place of the ligand
  holds about 20 waters in the one and 8 to 10 protein atoms in the other. About 2 % of the
  moves were accepted (394 to 472 of 22 thousand each way). The sections up to the charges off
  are 2.1 to 2.8 larger with the moves in both; why is not established.
- Why: step 6 of the plan (the control of E3 again with the moves, the owner's order).
- Refs: the plan ("評価 1"), `spikes/m28_complex_leg_sections.py`, `runs/acc_reloc2_*` (untracked)

### 2026-11-23 — [amendment] Water relocation moves: the trial energies on the GPU; three corrections
- What: the stand-in energies of a proposal were computed by NumPy on a processor core, which
  made a leg with the moves several times slower; the three runs started with it were stopped
  (no result). They are now computed by kernels on the CUDA device of the Context (CuPy, added to
  the image; the same numbers to rounding; the NumPy function is the reference of the tests).
  Corrected, each with a test that fails before: with a pool, windows started one from another
  ran without moves; an interval shorter than an iteration gave one attempt per iteration; the
  cell was compared with the sphere only when the leg was built.
- Why: the owner's order of 2026-11-23 (an implementation that slows a leg is not usable).
- Refs: `docs/configuration.md`, `docs/protocol.md`, `tests/gpu/test_gpu.py`, the plan ("手順 6")

### 2026-11-23 — [progress] The TYK2 set with annihilated electrostatics: one calculation per ligand
- What: `alchemy.annihilate_electrostatics: true`, the rest as the full run of e77a6c5 (image of
  41760a1, eight GPUs, exit 0, 16 of 16 with a result, none NOT RELIABLE), kcal/mol, 95 %
  bootstrap intervals over ligands: mean signed error -0.43 [-1.31, 0.41] (e77a6c5: -1.20), RMSE
  after removing it 1.77 [1.16, 2.17] (1.25), Kendall tau 0.20 [-0.12, 0.48] (0.47). Ligand by
  ligand the two runs differ by +0.76 on average (standard deviation 2.03, standard error 0.51;
  -2.67 to +4.72). ejm_49: -9.33 ± 0.42 (the first run with this setting: -7.71). Both legs are
  about 197 larger. No default is changed. The runs with relocation moves have started.
- Why: the owner's order of 2026-11-23 (step 6 of the plan).
- Refs: the plan ("手順 6"), `spikes/m27_paired_runs.py`, `runs/acc_annihilate_tyk2_full` (untracked)
