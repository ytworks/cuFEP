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
- Version 0.1: neutral ligands in the ABFE; the RBFE takes charges and changes of one charge. No cofactors.
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
| Unit tests (CPU) | 774 passed (2026-11-25, branch `feat/rbfe`, image `cufep:dev` with the RBFE packages, `scripts/in_docker.sh python -m pytest`; the last one, the reserved names of the output root, passed again after its fix with the RBFE tests) |
| GPU tests | 65 passed (2026-11-25, two GPUs, branch `feat/rbfe`, `scripts/in_docker.sh --gpus 0,1 python -m pytest tests/gpu`): the 55 of before (REST2, water relocation and grand canonical moves with TIP3P and OPC) and 10 of hybrid legs (each state against a tight Ewald sum on both backends, with and without REST2; the evaluation of all states; replica exchange); none builds a bilayer |
| CI | `unit` (CPU tests in the conda environment of `docker/environment.yml`); `harness` (PROGRESS.md budget, README pair, names of commercial products, ruff) |
| TYK2 example (ejm_31, experiment -9.54 kcal/mol) | `config.yaml` sets 5 ns of production per window (PR #4; the default stays 10 ns). With `selection: auto` in the full TYK2 run: -9.84 ± 0.33 kcal/mol; one earlier calculation: -8.94 ± 0.77. With the default `interactions` and the MD for the restraint from the pose that was given, four calculations of the same YAML and seed (4 GPUs each, PR #6): -12.16 ± 0.32 (image of 720cdb1), -12.23 ± 0.64, -8.84 ± 0.91 and -8.33 ± 0.74 (image of 1b770da, the same code of the sampling); mean -10.39, standard deviation 2.09 kcal/mol. Complex leg 42.95, 42.91, 40.19, 39.34 (standard deviation 1.86); solvent leg 20.23, 20.13, 20.77, 20.44 (0.29); release of the restraint -10.55 to -10.57. The four chose the same six anchors (backbone of VAL 981, its hydrogen bond in 764 to 771 of 800 frames) with the same reference angles. Overlap criterion met in every leg (lowest 0.131); a window with fewer than 50 independent samples in five of the eight legs (fewest 6, a solvent leg). The 1 ns blocks of one complex leg range from 38.1 to 43.2 kcal/mol. The first version of the selection (MD for the restraint after 5 ns of free MD), four runs: -10.85, -11.99, -5.83, -9.77, standard deviation 2.7. In the full TYK2 run with the present defaults (e77a6c5, PR #8): -10.08 ± 0.37 (complex leg 40.43, solvent leg 19.79; 33 and 18 windows; lowest overlap 0.154 and 0.189; a window with fewer than 50 independent samples in the complex leg) |
| Starts of the complex leg (plan `docs/plans/2026-11-21-als-initial-window-design.md`, PR #7) | A is the default since 2026-11-22 (the owner's choice), so its three runs are runs of the TYK2 example as it is now. TYK2 example, three starts of the complex leg, three calculations each with the same YAML and seed (image of 5e2cf33, 4 GPUs each; the overlap of a pair alone, threshold 0.30; solvent leg as the default), kcal/mol. A (30 windows, those of mode fixed): -9.36 ± 0.56, -9.63 ± 0.37, -9.33 ± 0.51; mean -9.44, standard deviation 0.17. B (30 windows, denser at vanish 0.7 to 0.9): -10.01 ± 0.34, -9.64 ± 0.52, -11.71 ± 0.53; -10.45, 1.11. C (59 windows placed as B): -11.43 ± 0.46, -6.41 ± 0.59, -7.64 ± 0.43; -8.50, 2.62. The four runs of the row above: -10.39, 2.09. Complex leg, standard deviation: 0.20, 1.23, 2.20 (1.86). Lowest overlap 0.141 to 0.153 (A), 0.128 to 0.129 (B), 0.081 to 0.085 (C); a window with fewer than 50 independent samples in the complex leg of 2, 1 and 1 of the three runs; 0 or 1 round trip in the complex leg of every run. Three runs give a standard deviation only roughly. Tables: the plan (`spikes/m19_window_runs.py`) |
| A2A example (4e, experiment -12.07 kcal/mol) | in the full A2A run: -18.91 ± 0.52 kcal/mol (complex leg 44.68); one earlier calculation with the same settings: -11.95 ± 0.73 (complex leg 37.61). With the default `interactions` (one calculation, image of commit 0f3aa32, 4 GPUs, PR #6): -12.14 ± 0.49 kcal/mol (complex leg 38.71); lowest overlap 0.102 and 0.178; a window with fewer than 50 independent samples in the complex leg; restraint from the hydrogen bond with ALA 59 (773 of 800 frames), all four trajectories on the plugin. A second calculation (fce6929): -16.85 ± 0.64 kcal/mol (complex leg 43.04), with the restraint on ASN 253 where the first had it on ALA 59. Both started the MD for the restraint after 10 ns of free MD. From the pose that was given, held through those 10 ns (one calculation, 720cdb1, 4 GPUs): -15.47 ± 1.00 kcal/mol (complex leg 41.83 ± 0.99, solvent leg 16.57 ± 0.14, release -9.80); lowest overlap 0.163 and 0.170; fewest independent samples 15 (complex leg, a window of the vanish stage, below 50) and 280; restraint from the hydrogen bond with GLU 169 (795 of 800 frames), the first of 54 ranked candidates; the hydrogen bonds with ASN 253 of the input pose are in 648 and 462 frames, the one with ALA 59 in 234; ligand RMSD 0.81 A; all four trajectories on the plugin; the same command again computed nothing again. In the full A2A run with the present defaults (e77a6c5, PR #8): -13.50 ± 0.34 (complex leg 39.76, solvent leg 16.54; 30 and 17 windows; lowest overlap 0.169 and 0.178; fewest independent samples 5, complex leg; restraint from the stacking with PHE 168, 785 of 800 frames) |
| Benchmarks | both full sets once with the present defaults, image of e77a6c5 on eight GPUs, 2026-11-22 (PR #8; `benchmarks/results/`, `docs/wiki/Accuracy.md`), both exit 0, one calculation per ligand, kcal/mol with 95 % bootstrap intervals over ligands. TYK2 16/16: mean signed error -1.20 [-1.83, -0.60], RMSE after removing the mean error 1.25 [0.75, 1.62], Kendall tau 0.47 [0.12, 0.75] (FEP+ 0.00, 0.47, 0.81). A2A 17/17: -4.14 [-5.36, -2.85], 2.68 [1.76, 3.24], 0.34 [-0.03, 0.67] (FEP+ -0.16, 1.08, 0.59). By the definitions of the Felis paper (R^2, offset, RMSE after offset, MAE, correct order ratio): TYK2 0.35, 1.20, 1.25, 0.94, 0.91; A2A 0.19, 4.14, 2.68, 2.16, 0.81 (Felis, arXiv:2603.22274 v3, three calculations averaged: 0.69, 0.04, 0.82, 0.56, 0.97 and 0.55, 2.74, 1.16, 0.88, 1.00; issue #5). No result NOT RELIABLE (lowest overlap 0.140 to 0.172 and 0.105 to 0.169); a window with fewer than 50 independent samples in 12/16 and 14/17 ligands (fewest 11 for TYK2; 4 and 5 in the complex legs of 4q and 4e). From the files of the runs: every restraint by `interactions` (TYK2: the hydrogen bond with VAL 981 in all 16; A2A: ASN 253 in 11, stacking with PHE 168 in 5, GLU 169 in 1; 736 to 797 of 800 frames); the 33 complex legs adaptive by the overlap of a pair alone with 0.30, 31 to 36 and 30 to 32 windows from 30 (7 A2A legs kept 30); the 33 solvent legs by the matrix of the stage with 0.18, 16 to 21 windows; all 66 legs on the plugin, no fallback; `cufep benchmark report` on the summaries gives the recorded reports again. The run of 3b45bfe (PR #4; `selection: auto`, the complex legs from 3, 5 and 9 windows by 0.18 of the stage matrix; record in the tree of cda6965): TYK2 -0.62, 1.74, 0.12; A2A -4.12, 2.26, 0.38; ligand by ligand the two runs differ by -3.76 to +2.27 (TYK2) and -4.78 to +5.41 (A2A). The subset tier is not run apart: its ligands are in the full sets with the same settings |
| Accuracy cross-check against OpenFE | stopped (plan of 2026-11-18, step 11) |
| Official image | built locally from `docker/Dockerfile`; not published |
| Development harness | PR #2; ruff 0.17.5 with the rules of fmeta (`scripts/lint.sh`) by PR #3 |

Open items:
- Plan `docs/plans/2026-11-26-rbfe-features-and-benchmarks.md` (PR #19): steps 1-7 done with `region: perturbed`; m39 (published edges, LOMAP among them): pairwise RMSE TYK2 0.72 (22 edges) vs 0.70 (24), A2A 1.52 (43) vs 1.52 (46), the owner: no problem. Full relative TYK2 benchmark (image 08d58d5, 22 LOMAP edges, 5 ns per window, `benchmarks/results/tyk2_rbfe/`): 16 of 16 ligands, 22 of 22 edges; pairwise ddG RMSE 0.86 [0.57, 1.03] kcal/mol (FEP+ 0.69 [0.41, 0.92], cuFEP ABFE e77a6c5 1.83 [1.10, 2.36]), R^2 0.81 [0.63, 0.94] (0.86, 0.35), Kendall tau 0.68 [0.41, 0.89] (0.81, 0.47); edgewise RMSE own 0.79, network 0.75 (22 edges); one complex leg with a lowest overlap 0.022 (ejm_31~ejm_45). Full A2A (29 states): started 2026-11-26 (37 LOMAP edges; legs start as their edge is prepared, 0cd6147; pinned precalculations get a worker per window, a607be8), stopped by the owner after 15 of 74 legs; 4d~4f is to be prepared again with the mapping for the plugin (b53f969: 5 TYK2 and 1 A2A edges had hydrogens SHAKE cannot constrain and ran on stock OpenMM). The five TYK2 edges on that mapping (431eee3, `runs/rbfe_tyk2_plugin_check`, all legs on cuPotential): within 1.8 SE of 08d58d5 except ejm_42~ejm_54 (-2.6 SE), whose cause led to hybrid version 3 (dummy atoms by Fleck et al. 2021). TYK2 with version 3 (8e2a945, `runs/rbfe_bench_tyk2_8e2a945`, all 44 legs on cuPotential): pairwise ddG RMSE 1.31 [0.79, 1.69] kcal/mol, R^2 0.62, Kendall tau 0.57, edge RMSE 0.88 (own) and 0.81 (network); the 17 edges of an unchanged mapping agree with 08d58d5. Versions 3 and 4 and the mapping for the plugin were reverted (2026-11-28): Kartograf's mapping is kept, and the hydrogens SHAKE cannot constrain move on their bond terms (05d3ff0). The five TYK2 edges of 08d58d5 on stock OpenMM, on cuPotential with 05d3ff0 (`runs/rbfe_tyk2_05d3ff0_five`, 10 legs, no fallback): diff/SE against 08d58d5 -0.6, +1.9, 0.0, -0.1, -2.7; a control on image 08d58d5 (`runs/rbfe_tyk2_08d58d5_control`) moved the legs of ejm_42~ejm_54 by 0.65 to 0.69 kcal/mol and ejm_50~ejm_55 by -1.24 (ddG); the two hybrids differ only in the N-H bond term and one mass, with equal energies in all 9 states (`spikes/out/m47`). The remaining 17 TYK2 edges are running (05d3ff0); A2A goes on in `runs/rbfe_bench_a2a_0cd6147` with 05d3ff0 after them (4d~4f anew). Reviews of PR #19: seventeen rounds answered (findings fixed or rejected by the quality of the product) (the eleventh, a re-review of the whole PR with 17 findings, changed the centre from which the ghosts of the start of a hybrid are counted: midway between the end states, the same both ways; legs of earlier versions keep their ghosts, a leg with samples those of the System its samples recorded; the twelfth, 16 findings, checks the final mapping of every leg for atoms that join mapped atoms; the thirteenth, 14 findings, checks the record of a prepared edge before it is read and takes the heavy atoms of a hybrid from both ends; the fourteenth, 14 findings, takes legs only with the record they were made from, and checks windows before a run) (reuse, resumption and analysis of run directories of earlier versions; the GPU stand-in held a context on GPU 0, which stopped the run of 2ecbcd2, fixed in d512824). The A2A RBFE example's 4e~4b and 4b~4o were computed with the REST2 torsion term of B unscaled (from PR #17). Issue #20: ABFE repeats are not compared by GCMC. Unit suite 05d3ff0: 902 passed; GPU tests 05d3ff0: 81 passed; ABFE fingerprint (spikes/m38) of 59af960 identical with main.
- Plan `docs/plans/2026-11-25-rbfe.md` (done: PR #17 with PR #18 merged into main on 2026-11-26): `cufep rbfe` (`docs/rbfe.md`). TYK2 example (image f6ac3b4): ddG -1.86 ± 0.29, +0.56 ± 0.32, +2.39 ± 0.32 kcal/mol (experiment -1.77, +1.79, +3.56), cycle closure -0.03 ± 0.54, dG RMSE 0.57 against experiment. A2A example in the bilayer (image f6ac3b4): ddG +0.91 ± 0.30, +3.26 ± 0.12, +1.38 ± 0.26 (experiment +2.12, +0.72, -1.40), cycle closure -0.97 ± 0.41, dG RMSE 1.35 (FEP+ 0.83); both in the plan's record. No long-range correction of the ligand LJ in RBFE (OpenFE's default), where the plan had one at both ends. Plan `docs/plans/2026-11-25-leg-engine.md` (done, PR #18): one engine of a leg for both (`src/cufep/engine/`), steps 1-6 done; ABFE fingerprint and analysis of a stored run unchanged; the TYK2 relative example on its image agrees with image f6ac3b4 within the errors (the plan's record).
- Plans `docs/plans/2026-11-24-opc-water-model.md` and `docs/plans/2026-11-24-gcmc-water.md` (done; PR #15, merged on the owner's order of 2026-11-25): TYK2 with OPC (16 ligands); A2A stopped after 4k and 4e in each variant (OPC in the bilayer, the complex leg in water, water relocation moves, grand canonical moves with TIP3P and OPC: Change Log). The owner's decision of 2026-11-25: stop, and no default changes for now (water TIP3P; `sampling.gcmc`, `sampling.relocation_mc` off; `membrane.complex_leg: bilayer`).
  With `sampling.gcmc`: the error of the excess chemical potential of water enters the complex leg times the waters the sphere gains (0.05 kcal/mol and 7 waters: 0.35 kcal/mol). One short run under MPS ended with a CUDA launch failure on one GPU; the same run again and the four full runs did not (not explained). Open: with REST2 the backends differ in the dispersion correction (up to 0.55 kT between windows, 0 at both ends; the plan); `spikes/m36` (2026-11-25): stock OpenMM does not update it when an epsilon offset changes (3.50 kT at s = 0.6 on a TYK2 complex), the plugin gives the energy of the System with the scaled parameters written in.
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
- Plan `docs/plans/2026-11-22-full-benchmarks-with-the-present-defaults.md` (done, PR #8): no tests or reviews by order.
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
- The two runs of each set (3b45bfe and e77a6c5), one calculation per ligand each, differ in two defaults at once
  (`restraints.selection`, the start of the complex leg); how much one calculation varies: ejm_31, ejm_49, 4k (above).
- `membrane.orientation: auto` stops unless two of its three methods agree within 15 degrees and
  5 A: a stop of cuFEP's own without a published basis. Kept as it is; waits for the owner's
  decision (`.claude/rules/fep-invariants.md`).
- The conda environment is not locked (`docker/environment.yml` pins OpenMM 8.4, CUDA 13.4,
  ProLIF 2.2.3 and MDAnalysis 2.11 only); a rebuild of the image can change other packages.

## Change Log

### 2026-11-28 — [progress] RBFE: the five TYK2 edges on cuPotential with Kartograf's mapping (05d3ff0)
- What: all 10 legs on cuPotential, no fallback; ddG against 08d58d5 within 0.6 SE for three edges, ejm_42~ejm_54
  +1.9 and ejm_50~ejm_55 -2.7 SE. Control on image 08d58d5: ejm_50~ejm_55 -1.37 +- 0.31 (recorded -0.13); both legs of
  ejm_42~ejm_54 moved 0.65-0.69 kcal/mol between two runs of the same code. The two hybrids differ only in one N-H
  bond term and one mass; energies equal in all 9 states (code check, `spikes/out/m47`). Unit 902, GPU 81 passed.
- Refs: `docs/plans/2026-11-26-rbfe-features-and-benchmarks.md` (2026-11-28), `runs/rbfe_tyk2_05d3ff0_five`

### 2026-11-28 — [amendment] RBFE: hydrogens SHAKE cannot constrain move on their bond terms (mapping kept)
- What: where Kartograf's mapping puts more than three hydrogens, or hydrogens of two lengths, on one mapped atom, the
  hybrid keeps the largest group of one length (at most 3) as constraints and moves the others on the force field's
  bond terms (`src/cufep/rbfe/constraints.py`); that atom gets the larger of its masses in A and B. Every such leg runs
  on cuPotential at 4 fs; before, it fell back to stock OpenMM (CCMA).
- Why: the owner's goal (accuracy on cuPotential at speed) without changing the mapping, which changed the accuracy.
  Checked: the plugin runs three such hybrids (tests/gpu); the bonds at 4 fs within 7 +- 1.5 % of 0.5 fs (CH2 onto NH).
- Refs: `docs/rbfe.md` ("Hybrid topology", constraints), `docs/plans/2026-11-26-rbfe-features-and-benchmarks.md`

### 2026-11-28 — [amendment] RBFE: the mapping for the plugin and hybrid versions 3 and 4 reverted
- What: b53f969, 5a87243, 3fd1516, 9e473f9, 431eee3 (the mapping changed for the plugin) and 99afe76, 8e2a945, f04851e,
  94b8c16 (dummy atoms of Fleck et al. 2021, hybrid 3 and 4) reverted; the code is that of b31e551 (Kartograf's
  mapping, OpenFE's hybrid, version 2). Next: the hydrogens the plugin cannot constrain move on bond terms instead.
- Why: the owner's goal, accuracy kept on cuPotential at speed. TYK2 (8e2a945): the 17 edges of Kartograf's mapping
  agree with 08d58d5, the 5 of the changed mapping do not (pairwise ddG RMSE 0.86 -> 1.31); versions 3, 4 and the
  REST2 trials chased a conformer problem the changed mapping made (ejm_42~ejm_54), not the plugin's.
- Refs: `docs/plans/2026-11-26-rbfe-features-and-benchmarks.md` (2026-11-28)

### 2026-11-28 — [amendment] RBFE: the terms of those dummy atoms change about the middle of the leg (hybrid 4)
- What: the bonded terms of the dummy atoms of Fleck et al. 2021 change from 0.4 to 0.5 (A) and from 0.5 to 0.6 (B) of
  lambda (`lambda_junction_delete`, `lambda_junction_insert`), not with the steric lambdas; REST2 stays at 1.5.
- Why: TYK2 with version 3 (8e2a945, 44 legs on cuPotential): pairwise ddG RMSE 1.31 [0.79, 1.69] kcal/mol (08d58d5
  0.86); the 17 edges of an unchanged mapping agree with 08d58d5 (mean +0.07, RMS of difference/SE 1.11), the 5 of the
  mapping for the plugin do not (-0.80, 2.04): their linker conformers in water. In most windows a dummy group was half
  anchored to each side of a bond of its mapped atom (220-640 kJ/mol to turn the urea C-N bond of ejm_54); owner:
  REST2 and windows only, then "進めて" for moving the change of the terms.
- Refs: `docs/rbfe.md` ("Dummy atoms ..."), `docs/plans/2026-11-26-rbfe-features-and-benchmarks.md`

### 2026-11-27 — [amendment] RBFE: dummy atoms by Fleck et al. 2021 where both ligands have unmapped atoms on one atom
- What: hybrid version 3 (`src/cufep/rbfe/junction.py`): where a mapped atom carries unmapped atoms of both ligands, the
  dummy atoms get the bonded terms of a terminal, dual (bond, angle and dihedral) or triple junction of Fleck, Wieder
  and Boresch, JCTC 17, 4403 (2021), changed with the steric lambda of their ligand; other atoms keep OpenFE's rule. Legs of versions 1 and 2 at such
  edges are left out of the analysis, and a run refuses to continue them. 57 such atoms in the prepared TYK2 and A2A edges.
- Why: the five TYK2 edges on the mapping for the plugin (431eee3, 10 legs on cuPotential) agree with 08d58d5 within
  1.8 SE except ejm_42~ejm_54 (-2.03 +- 0.22 vs -0.24 +- 0.64, -2.6 SE): the dummies at C17 raised the urea C-N barrier of
  ejm_54 at lambda 1 from 53.6 to 88.4 kJ/mol, and its solvent leg never reached its H-bonded conformer (owner: "1で進めて").
- Refs: `docs/rbfe.md` ("Dummy atoms where both ligands ..."), `docs/plans/2026-11-26-rbfe-features-and-benchmarks.md`

### 2026-11-27 — [amendment] RBFE: the mapping for the plugin decided by constraint lengths; earlier edges
- What: a pair is left out of the mapping where its hydrogens are constrained to other lengths in the end states
  (other elements, carbons of four and of three bonds, rings); a run on the plugin prepares again an edge prepared
  with a mapping the plugin cannot run where no leg has a result (kept in `<edge>.earlier_mapping_<n>`), and refuses
  one with results before it writes.
- Why: the owner: every case automatic; the element-only decision missed ethylbenzene -> styrene (5 hydrogens of
  0.1086 and 0.1094 nm on one carbon, reproduced on 5a87243).
- Refs: 3fd1516, `docs/rbfe.md` ("Atom mapping")

### 2026-11-27 — [amendment] RBFE: every leg on the cuPotential plugin (the mapping of six edges)
- What: a mapped pair of heavy atoms of other elements that both carry hydrogens is not mapped (Kartograf again,
  as its `filter_element_changes`); hydrogens of a mapped atom that would carry more than three are mapped closest
  first (as LOMAP); an edge whose hybrid still has hydrogens the plugin cannot constrain is refused before writing.
- Why: the owner's requirement that every leg runs on cuPotential; the plugin's integrator has SHAKE only, and 5
  TYK2 edges and A2A 4d~4f ran on stock OpenMM. cuFEP's own combination of two published steps.
- Verified: unit test; the six edges prepared again, every hydrogen in a SHAKE cluster. Not run: their ddG.
- Refs: `docs/rbfe.md` ("Atom mapping"), `docs/plans/2026-11-26-rbfe-features-and-benchmarks.md`

### 2026-11-26 — [amendment] RBFE: the fifteenth and sixteenth reviews of PR #19 (15 + 3 and 4 findings)
- What: a run checks before it writes that the legs it runs can go on (results of the record's preparation, samples
  of the System this version makes, else refused with the way that recovers them) and their windows (REST2 scale of
  every schedule, also one borrowed from the solvent leg); the analysis takes each leg result only in its place and
  leaves out only the repeat that fails; the report names the ligands the dG were set on.
- Why: findings of the owner's reviews (issue comments 7191879225, 7192277055); each reproduced on the code before.
- Verified: the unit tests of the files changed; the TYK2 run of 08d58d5 analysed again with 3f8209a (22 of 22).
- Refs: `docs/plans/2026-11-26-rbfe-features-and-benchmarks.md` (15th, 16th reviews), `docs/rbfe.md`
