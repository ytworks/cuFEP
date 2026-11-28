# Recorded benchmark results

One run of each full set, recorded as it came out: nothing was left out, rerun or adjusted.
How to read them, and the comparison with experiment, FEP+ and another ABFE method:
[docs/wiki/Accuracy.md](../../docs/wiki/Accuracy.md) (Japanese: [Accuracy-ja.md](../../docs/wiki/Accuracy-ja.md)).

| set | directory | ligands with a result | calculation | report |
|---|---|---|---|---|
| TYK2 (`tyk2`) | [tyk2](tyk2) | 16 of 16 | 2026-11-22, commit e77a6c5 | the one the run wrote |
| A2A (`a2a`) | [a2a](a2a) | 17 of 17 | 2026-11-22, commit e77a6c5 | the one the run wrote |

- Calculation: `cufep benchmark run <set> --tier full` in a Docker image built from the commit
  named (`docker/Dockerfile`, with NVIDIA's cuPotential plugin), on one machine with eight GPUs.
  The set runs with the settings of its example (5 ns of production per lambda window, and for A2A
  the POPC bilayer, `membrane.enabled`; every other setting the default of that commit);
  `resolved_config.yaml` holds all of them. One calculation per ligand.
- The defaults of that commit: the Boresch restraint from the protein-ligand interactions of
  unrestrained MD of the complex (`restraints.selection: interactions`); adaptive lambda
  scheduling of the complex leg from 30 windows, by the overlap of a pair alone with the
  threshold 0.30; of the solvent leg from 5 and 9 windows, by the overlap matrix of the stage
  with 0.18.
- Report: the one the run wrote at its end (`cufep benchmark report <set> <run directory>` of the
  same commit gives it again from the run's `summary.json`).
- Files: `benchmark_report.md` (statistics and tables), `benchmark_report.csv` (per ligand),
  `benchmark_report.png` (calculated against experiment), `benchmark_report_ligands.png` (each
  ligand, and the signed errors), `summary.csv` and `summary.json` (the result of every ligand
  with its legs, checks and messages), `resolved_config.yaml` (every setting of the run).
- No run time is recorded here (`docs/LICENSING.md`).
- The run before this one (2026-11-20/21, commit 3b45bfe, with `restraints.selection: auto` and
  the complex leg from 3, 5 and 9 windows by the overlap matrix of the stage with 0.18) is in the
  tree of commit cda6965; the wiki page puts its statistics and every ligand beside this run.

## Relative binding free energies

One run of each full set by relative calculations (`cufep benchmark run <set> --tier full --protocol rbfe`), recorded
as it came out, in an image built from the commit named, on one machine with eight GPUs: the settings of the set's
relative example (5 ns of production per lambda window, adaptive lambda scheduling of each leg, REST2, grand canonical
moves of water in the complex leg), every
compound of the set on the LOMAP network of `cufep rbfe`, one calculation per edge; `resolved_config.yaml` holds
every setting. The report compares the run with experiment, FEP+ (Ross et al. 2023) and OpenFE (IndustryBenchmarks2024,
TYK2 only); `cufep benchmark report <set> <run directory> --protocol rbfe` gives it again from the run's files, and
from the files kept here (a unit test holds the recorded reports to that). The uncertainty of an OpenFE edge is the
published one: the standard deviation propagated from the three repeats of its complex and of its solvent leg.

| set | directory | edges with a ddG | compounds with a dG | calculation | report |
|---|---|---|---|---|---|
| TYK2 (`tyk2`) | [tyk2_rbfe](tyk2_rbfe) | 22 of 22 | 16 of 16 | 2026-11-28, commit 957a9cf (grand canonical moves of water in the complex leg; on three of the GPUs) | of the run |
| A2A (`a2a`) | [a2a_rbfe](a2a_rbfe) | 21 of 21 | 17 of 17 | 2026-11-28, commit 957a9cf (grand canonical moves of water in the complex leg; started on three of the GPUs, resumed from its checkpoints on six with the same image) | of the run |

- Both sets were run again with the image of 957a9cf (the owner's decision of 2026-11-28, after the reviews of PR #21):
  the runs that were recorded before (TYK2 of faa65e0, A2A of cd15025) had been sampled before the fix of the
  velocities of an inserted water (c2f6903). Edge by edge the new ddG differ from those of the earlier run by +0.16
  kcal/mol in the mean, 0.44 RMS and at most 1.08 (ejm_31 -> ejm_45) for TYK2, and by -0.27 in the mean, 0.59 RMS and
  at most 1.70 (4b -> 4e) for A2A; one calculation each, so that the differences hold the scatter between two runs
  too. The records of the earlier runs are in the tree of commit 957a9cf.
- Files: `rbfe_report.md` (statistics and tables), `rbfe_report_ligands.csv`, `rbfe_report_edges.csv`,
  `rbfe_report.png` (dG of the compounds), `rbfe_report_edges.png` (ddG of the edges), and the run's `summary.json`,
  `edges.csv`, `ligands.csv`, `network.json`, `resolved_config.yaml`.
- The run of commit 39ee4ac (2026-11-27, without the moves of water; its 4 edges with core proper torsions of B before the
  fix of d61c215) is in the tree of commit d9afc69.
- TYK2 (faa65e0) and A2A (cd15025) compute the same: they differ only in the cell of a leg with the moves, which for TYK2's
  legs (no free MD in their pre-equilibration) is the same cell.
- A2A: both legs of 4d~4f ran on stock OpenMM (as below). OpenFE published no result for this set.
- TYK2: the legs of 5 edges ran on stock OpenMM (their hybrid has hydrogens the plugin's integrator cannot constrain;
  the owner's decision of 2026-11-27 to keep the fallback); the report names them.

