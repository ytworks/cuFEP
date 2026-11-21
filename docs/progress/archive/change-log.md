# Change Log archive

Change Log entries moved here from `PROGRESS.md` by `docs/progress/archiving.md`, newest first.
Frozen history: entries are inserted at the marker and never edited or deleted.

<!-- ARCHIVE-INSERT-POINT -->

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

