# Free-energy invariants and project rules

cuFEP is worth something only if its binding free energies and their uncertainties can be
trusted, and if what it says about them rests on published knowledge. Every change to
`src/cufep/` keeps these contracts; the tests named here check them.

## The only verdicts: published criteria

A verdict (stop, refuse, warning, note, a flag in a result) needs a published basis. These are the
published criteria:

| criterion | source | checked by |
|---|---|---|
| neighbouring lambda windows overlap by at least 0.03 (`analysis.min_overlap`), among all iterations and among the samples of the estimate | Klimovich, Shirts and Mobley 2015, sec. 3.5 | `test_checks_of_a_leg_use_the_levels_of_the_literature`, `test_the_overlap_is_also_taken_from_the_samples_of_the_estimate` (`tests/unit/test_sampling.py`) |
| at least 50 uncorrelated samples per lambda window for MBAR (`analysis.min_effective_samples`) | Mey et al. 2020, sec. 8.2 | `test_the_samples_check_counts_what_mbar_gets` (`test_sampling.py`) |
| a free energy needs windows that the samples connect (`windows_determined`) | Shirts and Chodera 2008 (MBAR) | `test_connected_windows_are_determined_whatever_the_level_and_unconnected_ones_never` (`test_sampling.py`), `test_windows_without_connecting_samples_give_no_free_energy_and_no_binding_free_energy` (`test_workflow.py`) |
| box height at least 2x (stock OpenMM) or 3x (cuPotential) the cutoff | limits of OpenMM and of the plugin | `test_solvation_box_meets_the_minimum_height`, `test_a_stock_openmm_box_is_at_least_twice_the_cutoff_high` (`test_workflow.py`) |
| adaptive lambda scheduling inserts a window where neighbouring windows overlap by `lambda_schedule.adaptive.overlap_threshold` or less: by default 0.18 of the overlap matrix of the stage in the solvent leg and 0.30 of that of the two windows alone (`overlap_matrix: pair`) in the complex leg; a leg whose threshold is not given has the one of the matrix it reads | the method of Midgley et al. 2025; the values 0.18 and 0.30 are the project's choice, and the overlap of the pair alone is cuFEP's own by the owner's decisions of 2026-11-21 and 2026-11-22, all recorded in `docs/configuration.md` | `test_a_pair_is_deficient_at_or_below_the_threshold`, `test_the_rule_ends_with_every_pair_above_the_threshold`, `test_the_overlap_of_a_pair_alone_rises_as_windows_get_closer_and_the_matrix_of_the_stage_falls`, `test_many_initial_windows_and_a_raised_threshold_converge_by_the_overlap_of_a_pair_alone`, `test_a_leg_that_is_only_told_its_overlap_matrix_gets_the_threshold_of_that_matrix` (`test_schedule.py`) |
| Boresch anchors: theta_A and theta_B 10 kT from 0 and 180 degrees in every frame, by the force constant each angle is restrained with (also the one `restraints.force_constants: auto` gives it); refused if H2-H1-H0-G0-G1-G2 has three atoms in a row on a line (abs(cos) > 0.9) in every frame | OpenFE `EvaluateBoreschAtoms` (`check_angle_not_flat`, `is_collinear`), quoted in `docs/configuration.md` | `test_anchors_on_a_straight_line_are_refused_only_if_they_are_in_every_frame`, `test_a_guest_on_a_straight_line_is_refused_and_would_give_nan` (`test_periodic_images.py`), `test_force_constants_from_fluctuations_are_held_to_the_rule_of_the_flat_angle_themselves`, `test_anchors_on_a_line_in_every_frame_are_refused_though_the_start_is_bent` (`test_interaction_anchors.py`) |
| `restraints.selection: interactions`: an interaction is kept in at least 0.40 of the analysed frames (the number of frames rounded down); anchors are candidates if L1-L2-L3 and P1-L1-L2 lie within 45 to 135 degrees in as many of the frames of the interaction; without one, the residues whose backbone is nearest to the ligand, within 1.0 nm, in at least 0.15 of the frames; without anchors the calculation stops | Ding et al., arXiv:2603.22274 v3, sec. 2.1, quoted in `docs/configuration.md` | `test_an_interaction_is_kept_in_two_fifths_of_the_frames_rounded_down`, `test_anchors_are_candidates_if_two_angles_are_in_range_often_enough_among_the_frames_of_the_interaction`, `test_the_distance_method_takes_the_residues_whose_backbone_is_nearest_to_the_ligand`, `test_the_best_candidate_that_can_be_used_is_chosen_and_without_one_there_is_no_restraint` (`test_interaction_anchors.py`) |

Not verdicts, and kept:

- rules of computation: a part of the production gets an estimate of its own only with at least
  4 independent samples per lambda window (`MIN_INDEPENDENT_SAMPLES` in
  `src/cufep/analysis/mbar.py`); the tolerances to which the MBAR equations are solved and a
  part counts as determined (`SOLVER_TOLERANCE`, `NEGLIGIBLE_FRACTION`, `RESOLUTION_KT`); the
  constants that rank the candidates of `restraints.selection: interactions` and the 20 bins of
  its reference values (`src/cufep/alchemy/interaction_anchors.py`, `REFERENCE_BINS`), taken
  from the source of the method and cited in `docs/configuration.md`;
- what version 0.1 cannot treat, where it stops with the reason: a ligand with a net charge,
  unknown or modified residues, a ligand without explicit hydrogens (`docs/protocol.md`,
  "Inputs" and "Limits of version 0.1").

One exception to the rule remains, a stop of cuFEP's own without a published basis, kept as it
is until the owner decides: `membrane.orientation: auto` stops unless two of its three methods
agree within 15 degrees and 5 A (`consensus_orientation` in `src/cufep/prep/orient.py`).

A refusal of the configuration that stays by the owner's decision (2026-11-21), where the paper
only rounds down: `restraints.selection: interactions` with a frame threshold that would be no
frame (`configuration_problems` in `src/cufep/config/schema.py`, `docs/configuration.md`).

Everything else is reported as a number without a verdict and without words such as
"converged", "settled" or "OK": the forward and reverse series, the halves, the blocks, BAR,
`ligand_rmsd_from_input_A`, `structure`, `bilayer`, the differences of `cufep validate`
(`test_halves_below_the_level_are_a_note_of_the_analysis_not_a_verdict`,
`test_how_the_bilayer_changed_is_given_as_numbers_without_a_verdict`,
`test_validate_gives_the_differences_of_every_pair_without_a_verdict`). Before adding any
threshold that decides something, quote the sentence or the source code it comes from in
`docs/configuration.md`; without one, report the number. Rules of construction are not verdicts
(chain-break distances, removal of overlapping waters when a bilayer is trimmed, the box margin,
the window caps of adaptive lambda scheduling, the 2 fs and stock-OpenMM fallbacks).

Never relax a criterion to make a run pass. A criterion changes only with its source, in
`docs/configuration.md`, `docs/protocol.md` and `PROGRESS.md`.

## The numbers of the analysis

- The reported free energy, its standard error and every partial estimate (halves, series,
  blocks, jackknife replicates) are computed by one estimator on one quantity (MBAR minus the
  tail correction); a partial estimate counts only if its windows are determined.
- The numbers are pinned: `test_the_numbers_of_the_analysis_cannot_change_without_a_new_version`
  (`test_sampling.py`, `tests/unit/data/analysis_golden.json`). Changing them raises
  `ANALYSIS_VERSION`; results of an earlier version are analysed again or marked as such.
- The energies of the alchemical and REST2 states equal those of a System set to each state
  (`test_scaled_hamiltonian_is_that_of_the_rest2_rules`, `test_evaluator_matches_setting_each_state`
  in `test_rest2.py`; `test_rest2_backends_agree` in `tests/gpu/test_gpu.py`).

## Specification and defaults

- Features, defaults and terms follow the literature and published open methods (for example
  OpenFE's ABFE protocol). `docs/configuration.md` ("Basis of the recommended values") gives the
  source of each default and says which values are cuFEP's own. Use the field's words
  (`docs/protocol.md`, "Terms"); do not coin new ones.
- One calculation per ligand by default (`sampling.repeats: 1`). Repeats are
  an option and a tool for validating the protocol; they are not offered as the remedy for a
  result, nor used for the examples or the benchmarks. A problem of one run is solved within one
  run: longer production, more lambda windows, better sampling (REST2).
- Every setting can be set in the YAML and has a recommended default with its basis in
  `docs/configuration.md` (`test_every_setting_has_a_default`,
  `test_documented_defaults_are_the_defaults_of_the_code` in `test_config.py`).
- `auto` only where a value has to be computed from the input (restraint atoms, GPUs, box shape,
  membrane orientation, oversubscription factor). No automatic behaviour that was not asked
  for: the membrane is a setting and is never detected (`test_membrane_is_an_explicit_switch`),
  production is never extended automatically, nothing is run again automatically.
- A recovery from a failure (stock OpenMM when the plugin fails, 2 fs after a non-finite energy,
  a restarted worker) is always written to the log and the result.
- The flagship features (the cuPotential backend, adaptive lambda scheduling, REST2) are never
  configured so that they are switched on but do nothing. A fix stays within the published
  method.
- Before implementing a published method, read the source in full and state its rule exactly;
  where the source is silent, say so and record the choice as cuFEP's own.

## How changes are made

- Close a finding for its whole class: grep every place that uses the concept and make them go
  through one shared function (one estimator, one rounding, one check of what an analysis needs,
  one file resolver, one writer that publishes atomically). Add cross-cutting tests: agreement
  with synthetic truth, failure injected at every write step, the same directory state through
  every entry point.
- Read every inherited threshold and sign before keeping it.
- Before a push, have the whole diff reviewed by someone who did not write it, with the aim of
  constructing a legal input with a known answer for which cuFEP gives a wrong number while
  every check passes.
- No speed or run time of the cuPotential plugin in a tracked file, a commit message or text on
  GitHub (`docs/LICENSING.md`).
- The README (both languages), `CONTRIBUTING.md` and the files of this harness do not say that a
  decision of cuFEP follows, copies or was made by looking at a commercial FEP product or its
  vendor. Such a name appears only where accuracy is compared (the benchmark report against the
  published predictions) and where a source is cited correctly (the benchmark set, licence
  notices). Names of the software cuFEP depends on and of development tools are fine.
  `scripts/check_names.py` (CI job `harness`) holds the names to the files that compare accuracy
  or cite a source.
