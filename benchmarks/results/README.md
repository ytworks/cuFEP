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
relative example (5 ns of production per lambda window, adaptive lambda scheduling of each leg, REST2), every
compound of the set on the LOMAP network of `cufep rbfe`, one calculation per edge; `resolved_config.yaml` holds
every setting. The report compares the run with experiment, FEP+ (Ross et al. 2023) and OpenFE (IndustryBenchmarks2024,
TYK2 only); `cufep benchmark report <set> <run directory> --protocol rbfe` gives it again from the run's files.

| set | directory | edges with a ddG | compounds with a dG | calculation | report |
|---|---|---|---|---|---|
| TYK2 (`tyk2`) | [tyk2_rbfe](tyk2_rbfe) | 22 of 22 | 16 of 16 | 2026-11-27, commit 39ee4ac | made from the run's files by the code of 383758f (39ee4ac had no report; the calculation is the same code) |

- Files: `rbfe_report.md` (statistics and tables), `rbfe_report_ligands.csv`, `rbfe_report_edges.csv`,
  `rbfe_report.png` (dG of the compounds), `rbfe_report_edges.png` (ddG of the edges), and the run's `summary.json`,
  `edges.csv`, `ligands.csv`, `network.json`, `resolved_config.yaml`.
- TYK2: the legs of 5 edges ran on stock OpenMM (their hybrid has hydrogens the plugin's integrator cannot constrain;
  the owner's decision of 2026-11-27 to keep the fallback); the report names them.

