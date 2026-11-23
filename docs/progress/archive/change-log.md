# Change Log archive

Change Log entries moved here from `PROGRESS.md` by `docs/progress/archiving.md`, newest first.
Frozen history: entries are inserted at the marker and never edited or deleted.

<!-- ARCHIVE-INSERT-POINT -->

### 2026-11-23 — [amendment] Review of the three settings: windows are held to the System they were made for
- What: independent review of dd075d6..0815fd4 (five findings, each reproduced), and a second
  reader of the fix (none at that level; two lesser points closed). The start configurations
  record the settings of the System (`alchemy`, `sampling.rest2`) and the windows they are of: a
  run that gives other settings to a leg with windows and no samples is refused before it writes
  anything; start configurations of other windows are made again. Correction of the entry below:
  a leg of an earlier version has no `start_configurations` in its result. Settings of `alchemy`
  added since no longer make the same YAML "another configuration". Unit tests 706 passed, GPU
  tests 26 passed (the image `cufep:dev`).
- Why: the review before the push (`.claude/rules/fep-invariants.md`).
- Refs: PR #9, `docs/protocol.md` (adaptive lambda scheduling), `tests/unit/test_workflow.py`

### 2026-11-23 — [progress] Accuracy plan, steps 1 and 3: repeats, REST2 off, 10 ns, MD at the decoupled end
- What: image of e77a6c5, the recorded settings of the full runs, kcal/mol. ejm_49 (experiment
  -7.75): -11.84 ± 0.57 (the benchmark run), -8.92 ± 0.44, -9.40 ± 0.29, standard deviation 1.57
  (complex leg 1.29); REST2 off: -9.76 ± 0.39; the first repeat continued to 10 ns: -9.21 ± 0.43.
  4k (-11.54), three runs started by the walk: -14.49 ± 0.51, -13.07 ± 0.56, -16.50 ± 0.40 (1.72);
  the benchmark run, whose windows were started one from another: -20.32 ± 0.52 (complex leg 44.87
  against 38.78, 37.36, 40.90); to 10 ns: -14.38 ± 0.60. Plain MD: the bilayer of 4k with the solute
  held stops shrinking at about 70 ns (62.1 nm2; about 66 after the default 10 ns); at the
  decoupled end of 4k the water in the place of the ligand rises from 10.7 to about 15 molecules in
  20 to 30 ns, at that of ejm_49 it does not change.
- Why: steps 1 and 3 of the plan.
- Refs: the plan ("手順 1 と 3 の結果"), `spikes/m22_decoupled_end_md.py`, `runs/acc_*` (untracked)

### 2026-11-23 — [design] Three settings: the start of the windows, annihilated electrostatics, soft-core power
- What: `lambda_schedule.adaptive.start_configurations` (walk, the default, or sequential): what
  the windows of the precalculation start from no longer follows from the number of workers a leg
  got; it is recorded with `start_configs.npz`, in the log and the result.
  `alchemy.annihilate_electrostatics` (default false): the electrostatics inside the ligand go with
  its charges by the rule of openmmtools with exact PME, which OpenFE's ABFE protocol runs; legs
  that differ in it are not combined. `alchemy.softcore_b` (1 or 2, default 1). With the defaults
  the System and the recorded conditions of the stored benchmark legs are reproduced. Unit tests
  706 passed, GPU tests 26 passed. Not run on a benchmark; no default changed.
- Why: steps 4a to 4c of the plan (remedies as settings; defaults are the owner's decision).
- Refs: the plan, PR #9, `docs/configuration.md` (basis of the three), `docs/protocol.md`

### 2026-11-22 — [progress] Accuracy plan, step 2: what the stored full runs say
- What: from the files of the runs of e77a6c5 and 3b45bfe, no new calculation; kcal/mol
  (`spikes/m21_stored_run_measurements.py`, reviewed by a reader who did not write it). A2A: the XY
  area of the bilayer falls through all 10 ns of its equilibration (16 of 17 legs, -0.33 nm2/ns)
  and on, from 69 to about 62 nm2 in 40 ns; in the 7 legs whose precalculation had one worker the
  last windows start at 62.05 nm2 and the first at 66.05. Second minus first half of the
  production, complex leg: TYK2 restrain +0.51, discharge +0.40, vanish -0.06; A2A +0.66, -0.11,
  -0.97, with water entering the place of the ligand (+2.02 molecules, 16 of 17). The benchmark
  inputs hold no crystal water. The entry below: -1.12, not -1.13 (unrounded errors).
- Why: step 2 of the plan: what the stored runs decide needs no run.
- Refs: the plan ("手順 2 の結果"), PR #9

### 2026-11-22 — [progress] Plan started: the causes of the errors measured, the remedies implemented
- What: plan `docs/plans/2026-11-22-accuracy-causes-and-fixes.md` (in progress). Step 1 runs in
  the image of e77a6c5 with the recorded settings of the full runs: ejm_49 twice, 4k three times,
  ejm_49 once with `sampling.rest2.scale: 1.0`; no result yet. PR #8 is merged (its plan: done).
  Correction of the open item on the workers of the precalculation ("The free energies do not
  depend on it"): with one worker the start configurations are made in another way (4 complex
  legs of TYK2, 7 of A2A); their mean signed errors are -1.42 and -5.85 kcal/mol against -1.13
  and -2.95 of the other legs (other ligands, found afterwards: no cause follows from it).
- Why: the owner's order (2026-11-22): find what improves the accuracy and implement it, few runs.
- Refs: the plan, its pull request, issue #5

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

### 2026-11-21 — [amendment] Reviews of PR #7 before its runs: fixes; the start of candidate C made again
- What: an independent review (no must-fix) and Codex (ten P2 by reading), each finding reproduced
  first. A start is a candidate to the six decimals cuFEP writes (a recorded start was refused with
  100 candidates). A leg that keeps its `schedule.yaml` says so in its log. One count of independent
  samples for the check and for tables (`counted_samples`). The design script moved windows that
  fell on one candidate upwards: 14 of the 37 inner vanish values of candidate C were up to 0.03 too
  high; a candidate step now holds at most one interval, the rest is shared by weight (B unchanged).
  Documents corrected. Not changed: `cufep plan`, the four decimals of records. In the image: unit
  tests 675 passed, GPU tests 25 passed.
- Why: the whole diff is read by someone who did not write it before the runs of the plan.
- Refs: PR #7 (review 6005834599 and its replies), `spikes/m18_window_design.py`, `docs/protocol.md`

### 2026-11-21 — [progress] The overlap of a pair alone for adaptive lambda scheduling; settings per leg
- What: `lambda_schedule.adaptive.overlap_matrix`: `stage` (the paper's matrix of all windows of a
  stage; the default until the candidates of the plan have run) or `pair` (the two neighbouring
  windows alone; cuFEP's own, the owner's decision). `mode`, `overlap_threshold`, `overlap_matrix`
  and `candidates` take one value or one per leg. `schedule.yaml` keeps the matrix and, with `pair`,
  the overlap of each pair beside the element of the stage. The documents now say that the rule
  spaces a bad section evenly again, so given start values inside one move (the commit of the entry
  below said none is removed). In the image: unit tests 674 passed, GPU tests 25 passed.
- Why: an element of the stage matrix falls when windows get close (harmonic windows: at most
  0.242), so many initial windows or a higher threshold run to `max_windows`; legs set separately.
- Refs: PR #7, the plan, `docs/configuration.md`, `docs/protocol.md` (adaptive lambda scheduling)

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

