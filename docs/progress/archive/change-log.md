# Change Log archive

Change Log entries moved here from `PROGRESS.md` by `docs/progress/archiving.md`, newest first.
Frozen history: entries are inserted at the marker and never edited or deleted.

<!-- ARCHIVE-INSERT-POINT -->

### 2026-11-26 — [amendment] RBFE: the nine findings of the review of PR #17
- What: each end state in a force field with its own ligand (one template served A and B of one bonded graph:
  System A had the charges of B in the test); the legs of an edge are checked as the ABFE checks them
  (temperature, pressure, prepared systems) and a leg without a standard error does not count, the other
  edges are combined; pH in the record of a prepared edge; the force field of the lipid patch for a 4-site
  water; prepared edges checked before a run writes anything; exit code from the jobs of the run; the legs in
  edges.csv from the repeats of the ddG; initial windows within max_windows. Each reproduced first, as a test.
- Why: the review of PR #17 (2026-11-25); the examples were not affected (no such inputs, no reruns).
- Refs: `docs/plans/2026-11-25-rbfe.md` (record); `docs/rbfe.md`

### 2026-11-25 — [progress] One engine of a leg: the ABFE and the RBFE as two protocols
- What: `src/cufep/engine/` (protocols, steps of a leg, analysis of a path, jobs); `CommonConfig`; the relative
  leg runs on the engine without `shared()`, renamed results or the ABFE job. Found on the way: the timing of a
  relative run read a ligand per leg (KeyError after all legs had finished), and the test of an edge was not
  repeatable (CPU platform without deterministic forces, hydrogens placed at random); both fixed with tests.
- Verified: ABFE fingerprint (`spikes/m38`, Reference platform) identical before and after; analysis of a stored
  TYK2 ligand identical; unit suite and GPU tests; TYK2 relative example (image 5cc23eb, resumed on c3665a3).
- Refs: `docs/plans/2026-11-25-leg-engine.md`; branch `refactor/leg-engine`

### 2026-11-25 — [design] Plan: one engine of a leg for the ABFE and the RBFE
- What: `src/cufep/engine/` (protocols of a leg, the steps of a leg, jobs); the ABFE and the RBFE become two
  protocols on it; `RbfeConfig.shared()`, the renaming of leg results, the job subclass and the dispatch on the
  keys of a state go. ABFE numbers, files and defaults unchanged, checked with `spikes/m38` before and after.
- Why: the owner's question (is the sharing forced, is it the right architecture?): the RBFE was fitted into the
  entrances of the ABFE (an ABFE configuration made from its settings, its pre-equilibration steered by the
  setting of the ABFE restraint, result keys renamed, command hints replaced as text).
- Refs: `docs/plans/2026-11-25-leg-engine.md`; branch `refactor/leg-engine`

### 2026-11-25 — [progress] RBFE: TYK2 example; the findings of the review of the diff
- What: TYK2 example on image f6ac3b4 (plan record; Current Status). Review of PR #17: a torsion that a System
  has twice counts twice (it became one, as in OpenFE; the end at lambda 0 was 16.6 kJ/mol off in the test);
  bonds and angles of the environment that differ between A and B are refused; `--leg complex` with
  `adaptive.legs: solvent` before the solvent leg has windows is refused (it waited for ever); the complex
  pre-equilibration is stated in `shared()`. One finding rejected: OpenFE also separates unique old and new atoms.
- Refs: `docs/rbfe.md` ("Hybrid topology", "Sampling"); `test_rbfe_hybrid.py`, `test_rbfe_config.py`

### 2026-11-25 — [design] RBFE: each leg chooses its own lambda windows (`lambda_schedule.adaptive.legs: each`)
- What: TYK2 example, image of 58bef64, 8 GPUs, kcal/mol (experiment; FEP+): ddG ejm_31->ejm_46 -1.43 ± 0.32
  (-1.77; -0.91), ejm_31->ejm_49 +0.74 ± 1.28 NOT RELIABLE (+1.79; +1.86), ejm_46->ejm_49 +2.10 ± 0.45 (+3.56);
  cycle closure -0.08 ± 1.40; dG (mean of the experimental values) -9.28, -10.71, -8.61 (-9.54, -11.31, -7.75).
  The windows of the solvent leg left a pair of the ejm_31->ejm_49 complex leg at 0.013 (λ 0.12, 0.25). The owner:
  each leg its own precalculation (complex 500 ps after 500 ps, as the ABFE); `cufep rbfe run --leg`.
- Why: the owner's choice of 2026-11-25 after that result.
- Refs: `docs/rbfe.md` ("Lambda windows"), `runs/rbfe_tyk2_58bef64` (untracked)

### 2026-11-25 — [amendment] RBFE: no ABFE; every window that a leg does not choose starts from its pre-equilibration
- What: the reference of the dG by an ABFE in the same run (`dg.references: abfe`, the entry below) is removed:
  the dG come from the edges alone (MLE), referenced to the mean of the experimental values, else to the mean
  of the network. The windows the complex leg takes from the solvent leg, and fixed windows, start all from the
  end of the pre-equilibration (OpenFE's start) instead of a walk on one GPU; `lambda_schedule.fixed_walk_ps`
  of RBFE is gone. The ABFE run driver is back as on main but for `execute_jobs`, shared with RBFE.
- Why: the owner's order of 2026-11-25 ("ABFE は不要", "RBFE だけで DG を出す方法をちゃんと実現せよ"; no
  serial step in the GPU parallelism).
- Refs: `docs/rbfe.md`, `docs/plans/2026-11-25-rbfe.md` ("記録")

### 2026-11-25 — [progress] `cufep rbfe`: relative binding free energies, implemented, not run yet
- What: hybrid topology by the rules of OpenFE's relative protocol with only charges on the plugin (`spikes/m36`:
  within 0.037 kT of the charges written in), Kartograf mapping, LOMAP networks, adaptive lambda scheduling of the
  paper (solvent leg decides), REST2, the bilayer; ddG, cycle closures, dG by cinnabar's MLE with the experimental
  mean or an ABFE in the same run. Shared code: one state dispatch, `sample_leg`, `prepare_run`/`finish_run`; the
  ABFE tests unchanged. RBFE has no long-range correction of the ligand LJ (OpenFE's default; the plan had both ends).
- Why: the owner's order (the plan).
- Refs: `docs/rbfe.md`, `docs/plans/2026-11-25-rbfe.md` ("記録"), PR #17

### 2026-11-25 — [design] Plan: relative binding free energies, `cufep rbfe`
- What: the owner's order. A second product in the same package and image: hybrid topology with the
  lambda functions and soft-core of OpenFE's protocol, Kartograf mapping, adaptive lambda scheduling
  as the paper sets it for relative transformations (9 windows, 0.04), 5 ns per window, REST2, the
  bilayer as the ABFE has it; ddG per edge, DG by the MLE of the network with the experimental mean
  (default) or a cuFEP ABFE of a reference ligand. Its own YAML. Shared code generalised so that the
  ABFE gives the same numbers. Examples: TYK2 and A2A in the bilayer, three compounds each.
- Why: the owner's order of 2026-11-25; the two choices (DG reference, place) are the owner's answers.
- Refs: `docs/plans/2026-11-25-rbfe.md`

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

### 2026-11-25 — [progress] Grand canonical moves of water in the complex leg (`sampling.gcmc`, default off)
- What: ghost waters switched in the live Context, the acceptance of Ross 2017 (eq 1, 2, 7), batches of
  trials with the stand-in energy and a second test with the energy of the simulation, every replica
  with the list of its waters, the alchemical run of the leg without a barostat in the mean cell of
  its pre-equilibration; TIP3P and OPC. The two numbers of the water: TIP3P -6.09 kcal/mol and 30.345
  A^3 (grand's; measured here -6.125 ± 0.053, 30.319 ± 0.009), OPC -8.05 and 30.000 (measured). A short
  run of 4k end to end: the sphere holds about 7 waters where the ligand is coupled and 14 where it is
  decoupled. Unit tests 750, GPU tests 55 passed; a short review: no P1, two P2 fixed.
- Why: the owner's order (plan `docs/plans/2026-11-24-gcmc-water.md`); no benchmark result with it yet.
- Refs: the plan ("記録"), `docs/configuration.md` (`sampling.gcmc`), `docs/protocol.md` ("Grand canonical moves of water")

### 2026-11-25 — [design] Plan: grand canonical Monte Carlo of water in the complex leg; the relocation run is stopped
- What: A2A in the bilayer with OPC and the water relocation moves (image of 9989cdb): 4k -19.16 ± 0.59
  kcal/mol, without the moves -19.15 ± 0.35 (experiment -11.54); moves accepted 117 of 19,247 into
  the sphere and 67 of 19,503 out of it. Stopped on the owner's order during 4e. New plan: GCMC with
  ghost waters (grand), batches of trials on the GPU and a second test with the energy of the
  simulation, the complex leg at constant volume, TIP3P and OPC; default off. Risk check passed
  (`spikes/m33`): waters switched off and on in a live Context give the energies of a System built
  so, on both backends. Found by it: with REST2 the two backends treat the dispersion correction
  differently, up to 0.55 kT between windows (TIP3P), 0 at both ends; the GPU tests have it off.
- Why: the owner's order of 2026-11-24, after the moves that keep the number of waters changed nothing.
- Refs: `docs/plans/2026-11-24-gcmc-water.md`, `docs/plans/2026-11-24-opc-water-model.md` ("記録")

### 2026-11-24 — [progress] A2A in water with OPC stopped after two; the water relocation moves are merged and take a 4-site water
- What: restraint from the bilayer in TIP3P, alchemical run in water with OPC (image of 02f3d7a): 4k
  -17.50 ± 0.73 and 4e -20.16 ± 0.57 kcal/mol (experiment -11.54, -12.07; in water with TIP3P -12.11,
  -16.17); stopped on the owner's order. `feat/relocation-mc` (PR #10) is merged into this branch
  (`sampling.relocation_mc`, default off). With a 4-site water an accepted move gave the virtual site a
  velocity: fixed, only the atoms with a mass are turned. The energy change of a moved water against a
  tight Ewald sum: the plugin within 0.035 kT, stock OpenMM 0.225 kT with OPC (GPU test, m32). A short
  review: no P1 or P2. Unit tests 738, GPU tests 43 passed. Next: A2A in the bilayer with OPC and the moves.
- Why: the owner's order of 2026-11-24: all of A2A back to the bilayer with OPC, with the moves.
- Refs: the plan ("記録"), `docs/protocol.md` ("Water relocation moves"), `tests/unit/test_relocation_mc.py`

### 2026-11-24 — [design] `membrane.water`: the restraint from the bilayer in TIP3P, the alchemical run in water with OPC
- What: the owner's order after A2A with OPC in the bilayer, stopped after two ligands: 4k
  -19.15 ± 0.35 and 4e -20.61 ± 0.47 kcal/mol (experiment -11.54 and -12.07; the TIP3P run of
  e77a6c5 -20.32 and -13.50), the box area 4 to 7 % larger than with TIP3P. `membrane.complex_leg`
  of the closed PR #14 is merged into this branch, and `membrane.water` (cuFEP's own, default null:
  `forcefield.water`) gives the system with the bilayer a water model of its own, accepted only
  with `complex_leg: water`. 4k is prepared so: complex in water 18,989 OPC waters, bilayer 14,729
  TIP3P waters. A short review: no P1. Unit tests 716 passed; GPU tests 35 passed before its fixes.
- Why: the owner: OPC may not go with the bilayer. Lipid21 is validated with TIP3P.
- Refs: the plan ("記録"), `docs/configuration.md` (`membrane.water`), `tests/unit/test_complex_leg_in_water.py`

### 2026-11-24 — [progress] TYK2 with OPC: 16 ligands, one calculation each
- What: image of 29fb9df, eight GPUs, one ligand at a time, the settings of the set with
  `forcefield.water: amber14/opc.xml`. All 32 legs on the plugin, none NOT RELIABLE (lowest
  overlap 0.099), a window below 50 independent samples in 15 of 16. Against experiment, OPC and
  the run of e77a6c5 (TIP3P), 95 % bootstrap intervals: mean signed error +1.33 [0.67, 2.01] and
  -1.20 [-1.83, -0.60]; RMSE 1.92 [1.27, 2.53] and 1.73 [1.10, 2.30]; after removing the mean
  1.38 [0.85, 1.72] and 1.25 [0.75, 1.62]; Kendall tau 0.33 [-0.10, 0.69] and 0.47 [0.12, 0.75].
  OPC binds 2.53 kcal/mol weaker on average: solvent leg +1.32 (all 16), complex leg -1.20.
- Why: the plan. The owner decides about A2A and the default; nothing is changed.
- Refs: the plan ("記録"), `runs/benchmark_tyk2_full_opc_29fb9df` (untracked)

### 2026-11-24 — [design] A bilayer takes a 4-site water; the plugin's MD with virtual sites checked; ejm_31
- What: `membrane.enabled` no longer refuses a 4-site water: the bilayer is built with 3-site waters
  and every water gets its extra site afterwards (cuFEP's own way; A2A 4k prepared with OPC, no MD
  of it yet). A review of 29fb9df asked what the plugin's own integrator and barostat do to a
  virtual site: in 39 stored configurations of ejm_31 the sites are within 5e-10 nm of their place,
  and the NPT volume is that of stock OpenMM (OPC -0.008 ± 0.053 of 25.8 nm3; test box +0.09 ± 0.08 %).
  A GPU test holds sites and constraints after NPT MD; the four tests of the backends share one
  function. ejm_31 with OPC: -9.30 ± 0.57 kcal/mol (TIP3P -10.08 ± 0.37; experiment -9.54).
- Why: the plan (A2A needs the bilayer); the reviews before the push.
- Refs: the plan ("記録"), `docs/configuration.md`, `docs/protocol.md` ("Systems"), `tests/unit/test_membrane.py`

### 2026-11-24 — [design] Plan: the water model OPC against TIP3P on the benchmark sets
- What: the owner's order. Only the water changes (`forcefield.water: amber14/opc.xml`, ff14SB
  kept); TYK2, then A2A, one ligand at a time against the run of e77a6c5; the default stays TIP3P
  until the owner decides. The GPU fixture `rest2_leg` is built with TIP3P and OPC, and
  `test_rest2_backends_agree` holds each backend to stock OpenMM with an Ewald tolerance of 1e-6
  (the plugin to 0.1 kT, stock at 5e-4 to 0.3 kT; the owner's decision): on 31 configurations the
  plugin is within 0.022 (TIP3P) and 0.044 kT (OPC) of it, stock within 0.118 and 0.156.
- Why: Li & Minkara 2024 (protein-glycan ABFE: RMSE 4.17 with TIP3P, 1.85 with OPC). The plugin
  against stock was 0.114 kT with OPC on the one configuration of the test: stock PME's error.
- Refs: `docs/plans/2026-11-24-opc-water-model.md`, `spikes/m31_water_models_on_the_backends.py`
### 2026-11-24 — [design] The plan of `membrane.complex_leg` is dropped: PR #14 closed without a merge
- What: the owner's decision after the two ligands of the entry below. The setting, its tests and
  its documents stay on the branch `feat/complex-leg-in-water` and are not in `main`.
- Why: the owner's decision.
- Refs: the plan (`Status:`), PR #14

### 2026-11-24 — [progress] The A2A runs with `membrane.complex_leg: water` were stopped after two ligands
- What: the owner's order, during the third ligand (4g, no result). In water against the bilayer
  run of e77a6c5, kcal/mol: 4k -12.11 ± 0.53 against -20.32 ± 0.52 (its windows there were started
  one from another; with the walk -14.49, -13.07, -16.50), 4e -16.17 ± 0.47 against -13.50 ± 0.34
  (experiment -11.54 and -12.07). No statistics of the set. The setting is in PR #14, off by
  default; nothing is merged, and what becomes of it waits for the owner's decision.
- Why: the owner's order.
- Refs: the plan ("記録"), `runs/benchmark_a2a_full_water_a34bbe2`, `runs/a2a_water_a34bbe2.log` (untracked)

### 2026-11-24 — [progress] `membrane.complex_leg: water` on A2A: the short run, and 4k
- What: image of a34bbe2, eight GPUs, the settings of the set (5 ns per window, one calculation).
  A short run of 4k passed the checks of the plan from its files (systems of 61,452 and 73,275
  atoms, the restraint from the bilayer, the same command again computes nothing). 4k (experiment
  -11.54): -12.11 ± 0.53 kcal/mol; complex leg 36.60 ± 0.50, solvent leg 14.90 ± 0.18; restraint
  ASN 253 (784 of 800 frames); lowest overlap 0.154 and 0.171; fewest independent samples 12
  (complex leg) and 193. In the bilayer: -20.32 ± 0.52 (e77a6c5, windows started one from another)
  and -14.49, -13.07, -16.50 (the walk; complex leg 38.78, 37.36, 40.90). The other 16 follow.
- Why: the plan (one ligand at a time, each reported).
- Refs: the plan ("記録"), `runs/benchmark_a2a_full_water_a34bbe2` (untracked), PR #14

### 2026-11-23 — [design] `membrane.complex_leg`: the restraint from the bilayer, the alchemical run in water
- What: a setting, default `bilayer` (no default, number of the analysis or stored result changes).
  With `water` the preparation writes two complexes of one pose (`prep/complex.*` in water,
  `prep/complex_bilayer.*`); the pre-equilibration of the bilayer (`restraint_equil.npz`) and the
  MD of the restraint run in the bilayer, the alchemical run on the complex in water; the anchors
  are the same atoms, held to the rules at the start of the alchemical run; only with
  `restraints.selection: interactions`. cuFEP's own, off by default. A review by a reader who
  did not write it: no wrong number found, five points, four fixed. Unit tests 714 passed, GPU
  tests 26 passed (the image `cufep:dev`). No run with the setting yet.
- Why: the plan of this day (the owner's order).
- Refs: PR #14, `docs/configuration.md`, `docs/protocol.md` ("Systems"), `tests/unit/test_complex_leg_in_water.py`

### 2026-11-23 — [design] Plan: the bilayer for the MD of the restraint only, the complex leg in water
- What: the owner's order. A setting `membrane.complex_leg: bilayer | water` (default `bilayer`,
  the behaviour of today): with `water` the MD from which the restraint is chosen runs in the
  bilayer and the alchemical run of the complex leg in water. To be run on the A2A set one ligand
  at a time and compared with the run of e77a6c5 (the owner's decision). Not implemented yet.
- Why: to see whether the bilayer in the alchemical run is behind the larger errors of A2A.
- Refs: `docs/plans/2026-11-23-complex-leg-without-the-bilayer.md`

### 2026-11-23 — [design] The accuracy plan is closed: the walk is the one way of starting the windows
- What: the owner's decision. Adopted: `lambda_schedule.adaptive.start_configurations`, default
  the walk (E1). The A2A ligands whose complex leg the run of e77a6c5 had started one window
  from another, with the walk, kcal/mol: 4k -20.32 to -14.69 (mean of three), 4n -19.17 to
  -17.38, 4c -18.75 to -15.19, 4q -14.11 to -12.52, 4r -16.17 to -15.95. Not adopted (off by
  default): annihilated electrostatics (16 TYK2 ligands: mean signed error -0.43, RMSE after
  removing it 1.77, tau 0.20), the squared soft-core, equilibration detection; water relocation
  moves stay in PR #10 (4k -16.61, ejm_49 -11.42 with them). No remedy for the relaxation of the
  pocket at the decoupled end. The full sets were not run again (stopped after one ligand).
- Why: only E1 changed results beyond the scatter of one calculation (1.57 and 1.72).
- Refs: the plan (steps 5 and 6, "結論"), PR #9, PR #10, `spikes/m27_paired_runs.py`, `spikes/m30_ligands_against_a_run.py`

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

### 2026-11-23 — [design] Water relocation moves in the complex leg (`sampling.relocation_mc`), default off
- What: RelocationMC of Ding et al. (arXiv:2603.22274 v3), written from the paper: a whole water
  is moved between a sphere around the ligand and the rest of the cell; the move, its acceptance
  ratio and the velocities are the paper's. cuFEP's own: the energies of the trial poses and the
  scores are a stand-in (Lennard-Jones and reaction-field Coulomb on the processor) and the
  acceptance uses the potential energy of the Context in the state of the replica; the anchors
  can be drawn with a share in the sphere (`inner_fraction`; default the paper's uniform draw);
  any cell. Attempted in the replica exchange and in the precalculation; counted by window in the
  result. Unit tests 725 passed, the three GPU tests of the moves passed. No calculation yet.
- Why: step 4d of the plan; the owner's order of 2026-11-23 (implement it, then decide the rest).
- Refs: `docs/protocol.md` ("Water relocation moves"), `docs/configuration.md`, branch `feat/relocation-mc`

### 2026-11-23 — [progress] The owner's decisions after step 5; the TYK2 set with annihilated electrostatics started
- What: the owner (2026-11-23): implement RelocationMC first and decide the rest by its result;
  `membrane.equilibration_ns` stays 10 ns (100 ns is too long); no run with a softer distance
  restraint and none with crystal waters; run the 16 TYK2 ligands once with
  `alchemy.annihilate_electrostatics: true`; after RelocationMC, the control of E3 again with it.
  Started: that TYK2 set, image of 41760a1, eight GPUs (`runs/acc_annihilate_tyk2_full`). No
  result yet.
- Why: step 6 of the plan (the owner decides the defaults and what is implemented next).
- Refs: the plan, PR #9, `runs/configs/chain_acc_annihilate_tyk2_full.sh` (untracked)

### 2026-11-23 — [amendment] Fourth review by Codex (93d42ee): two more paths of a given schedule
- What: review 6016033556 (two P2), both fixed. With `--reschedule` a given file inside an
  archive of the leg that was begun and not completed is refused too. A schedule a leg has
  without start configurations gets the record of this leg before the walk through its windows.
- Why: the review after the push (the working agreement of PR #9).
- Refs: PR #9 (the review and its answer), `tests/unit/test_workflow.py`

### 2026-11-23 — [amendment] Third review by Codex (6b1a609): a given schedule is read, not copied first
- What: review 6015941311 (two P2), both fixed. A leg that is given its windows
  (`lambda_schedule.file`) reads the schedule and writes its `schedule.yaml` with the start
  configurations of the walk, not before it: a run stopped during the walk leaves no windows
  that look like those of an earlier version, and the schedule names this leg. With
  `--reschedule` only a file that is set aside is refused, not the one in `previous_<n>/`.
  Unit tests 707 passed, GPU tests 26 passed (the image `cufep:dev`).
- Why: the review after the push (the working agreement of PR #9).
- Refs: PR #9 (the review and its answer), `tests/unit/test_workflow.py`

### 2026-11-23 — [amendment] Second review by Codex (8dc042e): the origin of windows, a given schedule of the leg itself
- What: review 6015842320 (four P2); three fixed, one answered. `--reschedule` with
  `lambda_schedule.file` naming a file of the leg that is set aside is refused before anything is
  moved. Windows of adaptive lambda scheduling that a leg had without start configurations get
  the note that their origin is not recorded also when a file is named. That note is read from
  the files of the leg whenever its result is made (`cufep analyze` too). Not changed: a leg whose
  windows 2861b93, a commit inside this pull request, sampled first (no such directory exists).
  Unit tests 707 passed, GPU tests 26 passed (the image `cufep:dev`).
- Why: the review after the push (the working agreement of PR #9).
- Refs: PR #9 (the review and its answer), `tests/unit/test_workflow.py`

### 2026-11-23 — [amendment] Review by Codex of 2861b93: records that hold through other versions and interruptions
- What: review 6015499088 (one P1, seven P2), each fixed. A result that lacks
  `annihilate_electrostatics` takes it from its samples, before a run and in the combination; a
  stored ligand result is shown again only if it compared the conditions of the cycle this
  version compares. Samples without recorded conditions refuse a treatment their version did not
  have. A given schedule of another REST2 scale is refused before the leg takes it. A leg whose
  archive was not completed is refused until `--reschedule` completes it. Windows of an earlier
  version that a run samples first get its record, and the result keeps saying that their origin
  is not recorded. `windows_run_concurrently` is true only if windows ran at the same time.
- Why: the review after the push (the working agreement of PR #9).
- Refs: PR #9 (the review and its answer), `docs/protocol.md`, `tests/unit/test_workflow.py`

### 2026-11-23 — [progress] Accuracy plan, step 5: the bilayer equilibrated 100 ns; the control of E3 to 10 ns
- What: one run each, image of e77a6c5, kcal/mol. 4k (experiment -11.54) with
  `membrane.equilibration_ns: 100`: -19.84 ± 0.41, complex leg 44.32 ± 0.40 against 38.78, 37.36,
  40.90 of the three runs with 10 ns (the walk) and 44.87 of the benchmark run (windows started
  one from another); the area of the bilayer no longer changes at the end of the 100 ns (-0.008 %
  between its last two quarters; -0.9 to -2.5 % after 10 ns). The control of E3 continued to
  10 ns: complex leg 35.39 ± 0.66 against 40.74 of the first repeat continued to 10 ns; the two
  ways of starting the windows do not come closer. No cause is established by one run each.
- Why: step 5 of the plan (controls whose condition was met).
- Refs: the plan ("手順 5 の結果", "手順 5 までのまとめ"), `runs/acc_*` (untracked)

### 2026-11-23 — [progress] Accuracy plan, step 5: annihilated electrostatics, the squared soft-core, the control of E3
- What: one run each on ejm_49 (experiment -7.75; the defaults three times: -11.84, -8.92, -9.40;
  kcal/mol; everything else as the full run). `alchemy.annihilate_electrostatics`: -7.71 ± 0.41;
  two torsions of the ligand that stayed as in the input are turned at the end in 23 and 17 of 36
  complex windows and 16 and 11 of 17 solvent windows. `alchemy.softcore_b` 2 with alpha 1:
  -8.78 ± 0.34 (40 and 24 windows). Control of E3, the first repeat's complex leg with its 7
  windows of sterics 0.25 or less started from the decoupled end after 50 ns there: 35.26 ± 0.48
  against 40.91 ± 0.36, the whole difference between sterics 0.30 and 0.10; it rises during the
  production (halves 34.31, 36.23); binding free energy -3.28 ± 0.54. One run each: no cause is
  established. Not finished: 4k with 100 ns of bilayer equilibration; that control to 10 ns.
- Why: step 5 of the plan (one run per new setting; controls whose condition was met).
- Refs: the plan ("手順 5 の結果"), `runs/acc_*` (untracked), `spikes/m25_*.py`

### 2026-11-23 — [design] `analysis.detect_equilibration`: automatic equilibration detection as a setting
- What: off by default. On, the analysis of a leg leaves out, after `sampling.equilibration_ns`,
  the iterations before the time origin from which the effective energy has the most uncorrelated
  samples (the rule of openmmtools, which OpenFE's protocols run); the free energy, its parts and
  the numbers of the structure are of the rest, and the result records what was left out. Only
  the start is taken from the detection; the samples are decorrelated window by window as before
  (cuFEP's own). The analysis of the default is unchanged (version 10, the pinned numbers). Unit
  tests 707 passed, GPU tests 26 passed (the image `cufep:dev`).
- Why: step 4e of the plan; what it does on the stored benchmark legs is in the entry below.
- Refs: `docs/configuration.md` (basis), `docs/protocol.md`, `tests/unit/test_equilibration.py`

### 2026-11-23 — [progress] Accuracy plan: equilibration detection on the stored legs, acceptance of water relocation
- What: no run. Automatic equilibration detection (the rule of openmmtools) on the production of
  the 66 stored legs of e77a6c5 leaves out 1.04 ns (TYK2) and 2.30 ns (A2A) of the complex leg on
  average; kcal/mol: TYK2 mean signed error -1.20 to -1.29, RMSE after removing it 1.25 to 1.12,
  Kendall tau 0.47 to 0.52; A2A -4.14 to -3.79, 2.68 to 2.73, 0.34 to 0.38. Water relocation
  (Ding et al.) at the decoupled end of 4k, 200 trial moves each way, both tests of the rule (the
  second with the energy of the simulation): accepted 0.012 into and 0.004 out of the sphere with
  1024 trial poses, 0.046 and 0.010 with 4096. The control of E3 is defined and queued.
- Why: steps 4e and 4d of the plan (measured before a setting is wired or a move implemented).
- Refs: the plan ("手順 4e の評価と、E3 の対照の作り方"), `spikes/m21_*.py detect`, `spikes/m24_*.py`

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

