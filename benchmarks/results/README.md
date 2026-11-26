# Recorded benchmark results

One run of each full set, recorded as it came out: nothing was left out, rerun or adjusted.
How to read them, and the comparison with experiment, FEP+ and another ABFE method:
[docs/wiki/Accuracy.md](../../docs/wiki/Accuracy.md) (Japanese: [Accuracy-ja.md](../../docs/wiki/Accuracy-ja.md)).

| set | directory | ligands with a result | calculation | report |
|---|---|---|---|---|
| TYK2 (`tyk2`) | [tyk2](tyk2) | 16 of 16 | 2026-11-22, commit e77a6c5 | the one the run wrote |
| A2A (`a2a`) | [a2a](a2a) | 17 of 17 | 2026-11-22, commit e77a6c5 | the one the run wrote |
| TYK2, relative (`tyk2`, `--protocol rbfe`) | [tyk2_rbfe](tyk2_rbfe) | 16 of 16 ligands, 22 of 22 edges | 2026-11-26, commit 08d58d5 | the one the run wrote |

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

## The relative calculation of TYK2 (`tyk2_rbfe`)

- Calculation: `cufep benchmark run tyk2 --tier full --protocol rbfe` in a Docker image built from commit 08d58d5,
  on one machine with eight GPUs: the 22 edges of the LOMAP network of the 16 ligands (`network.json`), with the
  settings of the relative example and the defaults of that commit (hybrid topology, REST2 on the ligand and the
  pocket, grand canonical moves of water in the complex leg, adaptive lambda scheduling); `resolved_config.yaml`
  holds all of them. One calculation per edge.
- Report: the one the run wrote (`rbfe_report.md`: the dG of the ligands against experiment, the published FEP+
  predictions and the absolute calculation of e77a6c5; the ddG of every edge, its own and the network's, against
  experiment and the published prediction of the same edge). Files: `rbfe_report_ligands.csv`,
  `rbfe_report_edges.csv`, the two figures, `summary.json`, `edges.csv`, `ligands.csv`, `network.json`.
- The later commits of PR #19 change how existing run directories are reused, resumed and analysed, not the numbers
  of a run made from the start; `cufep benchmark report tyk2 <run directory>` of a later commit gives the same
  tables from the legs.
