# Recorded benchmark results

One run of each full set, recorded as it came out: nothing was left out, rerun or adjusted.
How to read them, and the comparison with experiment, FEP+ and another ABFE method:
[docs/wiki/Accuracy.md](../../docs/wiki/Accuracy.md) (Japanese: [Accuracy-ja.md](../../docs/wiki/Accuracy-ja.md)).

| set | directory | ligands with a result | calculation | report |
|---|---|---|---|---|
| TYK2 (`tyk2`) | [tyk2](tyk2) | 16 of 16 | 2026-11-22, commit e77a6c5 | the one the run wrote |
| A2A (`a2a`) | [a2a](a2a) | 17 of 17 | 2026-11-21, commit 3b45bfe | commit f6464ce |

The two records are of different commits at present. The TYK2 record is of the defaults of
2026-11-22; the A2A record is of the defaults before, and the A2A set is being calculated with
the present ones. Until that run is recorded, the wiki pages describe the run of commit 3b45bfe
for both sets; the TYK2 record of that commit is in the tree of commit cda6965.

- Calculation: `cufep benchmark run <set> --tier full` in a Docker image built from the commit
  named (`docker/Dockerfile`, with NVIDIA's cuPotential plugin), on one machine with eight GPUs.
  The set runs with the settings of its example (5 ns of production per lambda window, and for A2A
  the POPC bilayer, `membrane.enabled`; every other setting the default of that commit);
  `resolved_config.yaml` holds all of them. One calculation per ligand.
- The defaults of commit e77a6c5 (TYK2): the Boresch restraint from the protein-ligand
  interactions of unrestrained MD of the complex (`restraints.selection: interactions`); adaptive
  lambda scheduling of the complex leg from 30 windows, by the overlap of a pair alone with the
  threshold 0.30; of the solvent leg from 5 and 9 windows, by the overlap matrix of the stage
  with 0.18.
- The defaults of commit 3b45bfe (A2A): `restraints.selection: auto`; adaptive lambda scheduling
  with the threshold 0.18 of the overlap matrix of the stage in both legs, the complex leg from
  3, 5 and 9 windows. Its `resolved_config.yaml` was written before
  `lambda_schedule.adaptive.overlap_matrix` existed: reading it with the present version needs
  `overlap_matrix: stage` beside the threshold.
- Report: TYK2, the report the run wrote at its end (`cufep benchmark report` of the same
  commit). A2A, `cufep benchmark report <set> <run directory>` from the run's `summary.json`,
  with the report of the commit named (the same numbers as the report the run wrote, plus
  Pearson r and the figure ligand by ligand).
- Files: `benchmark_report.md` (statistics and tables), `benchmark_report.csv` (per ligand),
  `benchmark_report.png` (calculated against experiment), `benchmark_report_ligands.png` (each
  ligand, and the signed errors), `summary.csv` and `summary.json` (the result of every ligand
  with its legs, checks and messages), `resolved_config.yaml` (every setting of the run).
- No run time is recorded here (`docs/LICENSING.md`).
