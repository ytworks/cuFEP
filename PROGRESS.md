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

Snapshot date: 2026-11-25. `main` holds PR #1 (the engine of plan
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
PR #9, merged on the owner's order of 2026-11-23 (the accuracy plan, done): the windows of a leg
start from the walk whatever number of workers it has; three settings that stay off by default. PR #15, merged on the owner's order of 2026-11-25 (the plans of 2026-11-24, done): a bilayer with a 4-site water, `membrane.complex_leg` and `membrane.water`, water relocation moves and grand canonical moves of water, all off by default; no default changed.

| item | status |
|---|---|
| Unit tests (CPU) | 760 passed (2026-11-25, the branch `feat/thermodynamic-length-schedule` with `lambda_schedule.adaptive.method: thermodynamic_length`, in the image `cufep:dev`, `scripts/in_docker.sh python -m pytest`) |
| GPU tests | 57 passed (2026-11-25, one GPU, the same tree, image `cufep:dev`, `scripts/in_docker.sh --gpus 5 python -m pytest tests/gpu`); the tests of `rest2_leg`, of the water relocation moves and of the grand canonical moves run with TIP3P and with OPC; the precalculation in a pool of workers runs with both methods; none builds a bilayer |
| CI | `unit` (CPU tests in the conda environment of `docker/environment.yml`); `harness` (PROGRESS.md budget, README pair, names of commercial products, ruff) |
| TYK2 example (ejm_31, experiment -9.54 kcal/mol) | `config.yaml` sets 5 ns of production per window (PR #4; the default stays 10 ns). With `selection: auto` in the full TYK2 run: -9.84 ± 0.33 kcal/mol; one earlier calculation: -8.94 ± 0.77. With the default `interactions` and the MD for the restraint from the pose that was given, four calculations of the same YAML and seed (4 GPUs each, PR #6): -12.16 ± 0.32 (image of 720cdb1), -12.23 ± 0.64, -8.84 ± 0.91 and -8.33 ± 0.74 (image of 1b770da, the same code of the sampling); mean -10.39, standard deviation 2.09 kcal/mol. Complex leg 42.95, 42.91, 40.19, 39.34 (standard deviation 1.86); solvent leg 20.23, 20.13, 20.77, 20.44 (0.29); release of the restraint -10.55 to -10.57. The four chose the same six anchors (backbone of VAL 981, its hydrogen bond in 764 to 771 of 800 frames) with the same reference angles. Overlap criterion met in every leg (lowest 0.131); a window with fewer than 50 independent samples in five of the eight legs (fewest 6, a solvent leg). The 1 ns blocks of one complex leg range from 38.1 to 43.2 kcal/mol. The first version of the selection (MD for the restraint after 5 ns of free MD), four runs: -10.85, -11.99, -5.83, -9.77, standard deviation 2.7. In the full TYK2 run with the present defaults (e77a6c5, PR #8): -10.08 ± 0.37 (complex leg 40.43, solvent leg 19.79; 33 and 18 windows; lowest overlap 0.154 and 0.189; a window with fewer than 50 independent samples in the complex leg) |
| Starts of the complex leg (plan `docs/plans/2026-11-21-als-initial-window-design.md`, PR #7) | A is the default since 2026-11-22 (the owner's choice), so its three runs are runs of the TYK2 example as it is now. TYK2 example, three starts of the complex leg, three calculations each with the same YAML and seed (image of 5e2cf33, 4 GPUs each; the overlap of a pair alone, threshold 0.30; solvent leg as the default), kcal/mol. A (30 windows, those of mode fixed): -9.36 ± 0.56, -9.63 ± 0.37, -9.33 ± 0.51; mean -9.44, standard deviation 0.17. B (30 windows, denser at vanish 0.7 to 0.9): -10.01 ± 0.34, -9.64 ± 0.52, -11.71 ± 0.53; -10.45, 1.11. C (59 windows placed as B): -11.43 ± 0.46, -6.41 ± 0.59, -7.64 ± 0.43; -8.50, 2.62. The four runs of the row above: -10.39, 2.09. Complex leg, standard deviation: 0.20, 1.23, 2.20 (1.86). Lowest overlap 0.141 to 0.153 (A), 0.128 to 0.129 (B), 0.081 to 0.085 (C); a window with fewer than 50 independent samples in the complex leg of 2, 1 and 1 of the three runs; 0 or 1 round trip in the complex leg of every run. Three runs give a standard deviation only roughly. Tables: the plan (`spikes/m19_window_runs.py`) |
| A2A example (4e, experiment -12.07 kcal/mol) | in the full A2A run: -18.91 ± 0.52 kcal/mol (complex leg 44.68); one earlier calculation with the same settings: -11.95 ± 0.73 (complex leg 37.61). With the default `interactions` (one calculation, image of commit 0f3aa32, 4 GPUs, PR #6): -12.14 ± 0.49 kcal/mol (complex leg 38.71); lowest overlap 0.102 and 0.178; a window with fewer than 50 independent samples in the complex leg; restraint from the hydrogen bond with ALA 59 (773 of 800 frames), all four trajectories on the plugin. A second calculation (fce6929): -16.85 ± 0.64 kcal/mol (complex leg 43.04), with the restraint on ASN 253 where the first had it on ALA 59. Both started the MD for the restraint after 10 ns of free MD. From the pose that was given, held through those 10 ns (one calculation, 720cdb1, 4 GPUs): -15.47 ± 1.00 kcal/mol (complex leg 41.83 ± 0.99, solvent leg 16.57 ± 0.14, release -9.80); lowest overlap 0.163 and 0.170; fewest independent samples 15 (complex leg, a window of the vanish stage, below 50) and 280; restraint from the hydrogen bond with GLU 169 (795 of 800 frames), the first of 54 ranked candidates; the hydrogen bonds with ASN 253 of the input pose are in 648 and 462 frames, the one with ALA 59 in 234; ligand RMSD 0.81 A; all four trajectories on the plugin; the same command again computed nothing again. In the full A2A run with the present defaults (e77a6c5, PR #8): -13.50 ± 0.34 (complex leg 39.76, solvent leg 16.54; 30 and 17 windows; lowest overlap 0.169 and 0.178; fewest independent samples 5, complex leg; restraint from the stacking with PHE 168, 785 of 800 frames) |
| Benchmarks | both full sets once with the present defaults, image of e77a6c5 on eight GPUs, 2026-11-22 (PR #8; `benchmarks/results/`, `docs/wiki/Accuracy.md`), both exit 0, one calculation per ligand, kcal/mol with 95 % bootstrap intervals over ligands. TYK2 16/16: mean signed error -1.20 [-1.83, -0.60], RMSE after removing the mean error 1.25 [0.75, 1.62], Kendall tau 0.47 [0.12, 0.75] (FEP+ 0.00, 0.47, 0.81). A2A 17/17: -4.14 [-5.36, -2.85], 2.68 [1.76, 3.24], 0.34 [-0.03, 0.67] (FEP+ -0.16, 1.08, 0.59). By the definitions of the Felis paper (R^2, offset, RMSE after offset, MAE, correct order ratio): TYK2 0.35, 1.20, 1.25, 0.94, 0.91; A2A 0.19, 4.14, 2.68, 2.16, 0.81 (Felis, arXiv:2603.22274 v3, three calculations averaged: 0.69, 0.04, 0.82, 0.56, 0.97 and 0.55, 2.74, 1.16, 0.88, 1.00; issue #5). No result NOT RELIABLE (lowest overlap 0.140 to 0.172 and 0.105 to 0.169); a window with fewer than 50 independent samples in 12/16 and 14/17 ligands (fewest 11 for TYK2; 4 and 5 in the complex legs of 4q and 4e). From the files of the runs: every restraint by `interactions` (TYK2: the hydrogen bond with VAL 981 in all 16; A2A: ASN 253 in 11, stacking with PHE 168 in 5, GLU 169 in 1; 736 to 797 of 800 frames); the 33 complex legs adaptive by the overlap of a pair alone with 0.30, 31 to 36 and 30 to 32 windows from 30 (7 A2A legs kept 30); the 33 solvent legs by the matrix of the stage with 0.18, 16 to 21 windows; all 66 legs on the plugin, no fallback; `cufep benchmark report` on the summaries gives the recorded reports again. The run of 3b45bfe (PR #4; `selection: auto`, the complex legs from 3, 5 and 9 windows by 0.18 of the stage matrix; record in the tree of cda6965): TYK2 -0.62, 1.74, 0.12; A2A -4.12, 2.26, 0.38; ligand by ligand the two runs differ by -3.76 to +2.27 (TYK2) and -4.78 to +5.41 (A2A). The subset tier is not run apart: its ligands are in the full sets with the same settings |
| Accuracy cross-check against OpenFE | stopped (plan of 2026-11-18, step 11) |
| Official image | built locally from `docker/Dockerfile`; not published |
| Development harness | PR #2; ruff 0.17.5 with the rules of fmeta (`scripts/lint.sh`) by PR #3 |

Open items:
- Plans `docs/plans/2026-11-24-opc-water-model.md` and `docs/plans/2026-11-24-gcmc-water.md` (done; PR #15, merged on the owner's order of 2026-11-25): TYK2 with OPC (16 ligands); A2A stopped after 4k and 4e in each variant (OPC in the bilayer, the complex leg in water, water relocation moves, grand canonical moves with TIP3P and OPC: Change Log). The owner's decision of 2026-11-25: stop, and no default changes for now (water TIP3P; `sampling.gcmc`, `sampling.relocation_mc` off; `membrane.complex_leg: bilayer`).
  With `sampling.gcmc`: the error of the excess chemical potential of water enters the complex leg times the waters the sphere gains (0.05 kcal/mol and 7 waters: 0.35 kcal/mol). One short run under MPS ended with a CUDA launch failure on one GPU; the same run again and the four full runs did not (not explained). Open: with REST2 the backends differ in the dispersion correction (up to 0.55 kT between windows, 0 at both ends; the plan).
- The wiki pages (`docs/wiki/`, PR #8) are not on the GitHub Wiki yet (no page, checked 2026-11-22;
  the owner creates the first). Issue #5: the differences from Felis, measurements to separate causes.
- Plan `docs/plans/2026-11-22-accuracy-causes-and-fixes.md` (done, PR #9). Adopted: the walk as the
  one way of starting the windows (below). Not adopted, defaults unchanged: longer production,
  `alchemy.annihilate_electrostatics`, `alchemy.softcore_b` 2, `analysis.detect_equilibration`,
  REST2 off, a longer bilayer equilibration, water relocation moves (PR #10 closed; in the branch of PR #15, off). One
  calculation varies by a standard deviation of 1.57 kcal/mol (ejm_49) and 1.72 (4k). Not remedied:
  the decoupled-side windows of ejm_49 give 5.65 less from a relaxed decoupled end (4k: +0.47 ± 0.63).
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
  e77a6c5. PR #9 made the way a setting (default: the walk): five such A2A ligands come 0.2 to
  5.6 kcal/mol nearer experiment (mean 2.6; the plan). The full sets are not run again with it.
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
- `membrane.orientation: auto` stops unless two of its three methods agree within 15 degrees and 5 A: a stop
  of cuFEP's own without a published basis, kept until the owner decides (`.claude/rules/fep-invariants.md`).
- The conda environment is not locked (`docker/environment.yml` pins OpenMM 8.4, CUDA 13.4, ProLIF 2.2.3 and MDAnalysis 2.11 only); a rebuild of the image can change other packages.
- Plan `docs/plans/2026-11-25-thermodynamic-length-schedule.md` (in progress, PR #16): `lambda_schedule.adaptive.method: thermodynamic_length`, off by default: the rule of adaptive lambda scheduling as it is, and windows added to its result until neighbours are also a step of the thermodynamic length apart or less (Change Log).
  Verification runs done (image `cufep:eeb2351`, each against a run of the present method from the same image; Change Log). No default changed: the owner decides. A prepared system differs in every run of the same code (water and ions; a bilayer also in its waters and lipids: OpenMM's `Modeller.addMembrane` relaxes it with an unseeded Langevin integrator), and so can the restraint chosen from its MD (4e: PHE168 or GLU169).

## Change Log

### 2026-11-25 — [progress] Thermodynamic-length method: verification runs, each with a control of the same image
- What: image `cufep:eeb2351`, 5 ns per window, method minus the present method of the same image (kcal/mol):
  ejm_31 TIP3P (3 runs) complex leg +1.37 to +1.41, solvent -0.21 to -0.38, dG -11.48 to -11.64 against
  -9.88 ± 0.66; ejm_31 OPC complex +0.11, solvent -0.81, dG -8.97 ± 0.44 against -8.02 ± 0.77; A2A 4e complex
  +0.68, solvent +0.07, dG -15.02 ± 0.37 against -14.39 ± 0.61 (another restraint was chosen). Experiment -9.54, -12.07.
  The code of 22 November and of today build the same Systems, worker settings and analysis (checked from the code).
- Why: the owner's order: a control of the same image first, so that two runs differ by the method alone.
- Refs: the plan ("記録", stage 4), `spikes/m36_candidates/`, `runs/*_eeb2351_*` (untracked), PR #16

### 2026-11-25 — [design] `lambda_schedule.adaptive.method: thermodynamic_length` (off by default)
- What: the rule of adaptive lambda scheduling runs as it is; then windows are added wherever neighbouring
  windows are more than a step of the thermodynamic length apart (2 erfcinv(0.35) = 1.32 kT, the owner's
  choice; ceil(d / step) - 1 windows per interval), the rule reads the overlaps again, and the two take turns
  until neither adds. No window of the rule is taken away or moved: the windows of `als` are among those of the
  method. The design of the entry "Plan: lambda windows placed by thermodynamic length" below (all windows
  placed anew, fewer than the start possible) is not what was built: the owner rejected it.
  From the stored legs: a complex leg would have 40 [38, 45] windows where it has 31 [30, 35].
- Why: the owner's orders: both conditions, and what the present method gives is kept.
- Refs: the plan ("記録", stage 3), `docs/configuration.md`, `docs/protocol.md` ("Windows added by thermodynamic length"), PR #16

### 2026-11-25 — [progress] Thermodynamic length: measured on the stored runs, and recorded as numbers from now on
- What: `cufep.alchemy.thermodynamic_length` and `spikes/m36`; the lengths are numbers of the record of the
  precalculation (`thermodynamic_length` in `schedule.yaml`) and of a result (`neighbour_distance`,
  `stage_length`), read by no rule; analysis version and golden numbers unchanged. From the production samples of the 66 legs of the runs of e77a6c5 (1581 pairs of neighbouring windows; median
  [5 %, 95 %]): the overlap of a pair alone over the one a normally distributed energy difference gives is
  1.00 to 1.07 by stage (95 %: up to 1.37, vanish), 4 pairs at or below 0.878; the recorded acceptance over
  the mean over the sample pairs 1.00 [0.96, 1.04]. Windows for an acceptance of 0.35: complex leg 28
  [23, 30] (now 31 [30, 35]), solvent leg 23 [21, 25] (now 18 [17, 19]). The length of a stage did not depend
  on its number of windows (ejm_31, vanish: 19.6 to 20.8 kT with 18 to 39 windows). The windows are chosen as before.
- Refs: `docs/plans/2026-11-25-thermodynamic-length-schedule.md` ("記録"), PR #16, `spikes/out/m36/` (untracked)

### 2026-11-25 — [design] Plan: lambda windows placed by thermodynamic length, the present rule as the repair
- What: the owner's order: a development plan, no implementation yet. Opt-in
  (`lambda_schedule.adaptive.method: thermodynamic_length`, defaults unchanged): proposal windows,
  placed again at equal thermodynamic length from the candidate energies the precalculation already
  computes (Clark et al., JCTC 2024, 20, 7806, eq 19), then repaired by the present rule on the overlap
  of a pair; the spacing from a target exchange acceptance (provisional). Five stages in one pull
  request (the owner), the first two without a GPU. Seven questions wait for the owner. Not run.
- Why: the present rule only adds windows and leaves their spacing uneven; with replica exchange the
  spacing is bounded by the exchange, so the paper's case for windows as wide as possible does not carry over.
- Refs: `docs/plans/2026-11-25-thermodynamic-length-schedule.md`, this pull request

### 2026-11-25 — [design] The owner's decision: the runs stop, no default changes for now
- What: after the four runs with grand canonical moves of water (the entry below) the owner stops the
  evaluation on A2A; the remaining 15 ligands are not run. No default changes for now: the water
  stays TIP3P, `sampling.gcmc` and `sampling.relocation_mc` stay off. The two plans are closed.
- Why: the owner's decision. In the four runs the binding stayed 5 to 8 kcal/mol too strong.
- Refs: `docs/plans/2026-11-24-gcmc-water.md`, `docs/plans/2026-11-24-opc-water-model.md` (`Status:`)

### 2026-11-25 — [progress] A2A with grand canonical moves of water: 4k and 4e, TIP3P and OPC
- What: image of 8252ccd, one calculation each on eight GPUs, the set's settings with
  `sampling.gcmc.enabled: true` (kcal/mol; experiment -11.54 and -12.07). 4k: TIP3P -17.32 ± 0.42
  (without the moves -20.32 ± 0.52 in the run of e77a6c5; three other runs -14.49, -13.07, -16.50),
  OPC -19.96 ± 0.29 (-19.15 ± 0.35). 4e: TIP3P -17.21 ± 0.36 with the restraint on ASN 253 (without,
  on ASN 253, -16.85 ± 0.64; on PHE 168, e77a6c5, -13.50 ± 0.34), OPC -18.26 ± 0.54 (-20.61 ± 0.47,
  both on PHE 168). The moves were made in every run (5 to 20 thousand insertions and as many
  deletions); the sphere holds 7 to 9 waters where the ligand is coupled and 16 to 21 where it is
  decoupled. No GPU error, no worker restarted. The rest of the set waits for the owner.
- Refs: `docs/plans/2026-11-24-gcmc-water.md` ("記録"), `runs/benchmark_a2a_full_gcmc_{tip3p,opc}_8252ccd`
