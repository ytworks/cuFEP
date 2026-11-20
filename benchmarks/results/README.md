# Recorded benchmark results

One run of each full set, recorded as it came out: nothing was left out, rerun or adjusted.
How to read them, and the comparison with experiment, FEP+ and another ABFE method:
[docs/wiki/Accuracy.md](../../docs/wiki/Accuracy.md) (Japanese: [Accuracy-ja.md](../../docs/wiki/Accuracy-ja.md)).

| set | directory | ligands with a result | calculation | report |
|---|---|---|---|---|
| TYK2 (`tyk2`) | [tyk2](tyk2) | 16 of 16 | 2026-11-20, commit 3b45bfe | commit f6464ce |

- Calculation: `cufep benchmark run <set> --tier full` in a Docker image built from the commit
  named (`docker/Dockerfile`, with NVIDIA's cuPotential plugin), on one machine with eight GPUs.
  The set runs with the settings of its example (5 ns of production per lambda window; every other
  setting the default); `resolved_config.yaml` holds all of them. One calculation per ligand.
- Report: `cufep benchmark report <set> <run directory>` from the run's `summary.json`, with the
  report of the commit named (the same numbers as the report the run wrote, plus Pearson r, the
  figure ligand by ligand and where a leg is below the level of overlap).
- Files: `benchmark_report.md` (statistics and tables), `benchmark_report.csv` (per ligand),
  `benchmark_report.png` (calculated against experiment), `benchmark_report_ligands.png` (each
  ligand, and the signed errors), `summary.csv` and `summary.json` (the result of every ligand
  with its legs, checks and messages), `resolved_config.yaml` (every setting of the run).
- No run time is recorded here (`docs/LICENSING.md`).
