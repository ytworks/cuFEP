# cuFEP benchmarks

| Set | Target | Source subset | Ligands |
|---|---|---|---|
| `tyk2` | TYK2 kinase (PDB 4GIH) | `jacs_set/tyk2` | 16 |
| `a2a` | Adenosine A2A receptor (PDB 5OLZ), membrane | `gpcrs/a2a_hip278` | 17 compounds (29 states) |

A set runs with the settings of its example (`examples/kinase_tyk2/config.yaml`,
`examples/membrane_a2a/config.yaml`; `src/cufep/benchmarks/registry.yaml`): everything in that
configuration but the input files, the choice of ligands and the output directory (the smoke tier
also shortens the sampling, never lengthening a setting of the example). The example's protein is the set's
protein file and its ligand is one of the set's records.

By relative calculations (`--protocol rbfe`, tiers `smoke` and `full`; `docs/rbfe.md`, "Benchmarks") a set runs with
the settings of its relative example (`examples/rbfe_tyk2/config.yaml`, `examples/rbfe_a2a_membrane/config.yaml`)
on every compound of the set and the LOMAP network of `cufep rbfe` (TYK2 22 edges, A2A 21); its report compares
the dG and the ddG with experiment, FEP+ (the published map, `edge_predictions_*_out.csv`) and OpenFE (TYK2 only).

`*/inputs/` holds unmodified files from the Schrödinger
[public binding free energy benchmark](https://github.com/schrodinger/public_binding_free_energy_benchmark)
(MIT licence, see `THIRD_PARTY/`): protein PDB, ligand SDF with experimental dG
(`r_exp_dg`, kcal/mol), perturbation edges, and the FEP+ per-ligand predictions
(`*_out.csv`, columns `Exp. dG` / `Pred. dG`). `THIRD_PARTY/SHA256SUMS` pins the
downloaded files (retrieved 2026-11-18 from the `main` branch). The FEP+ per-edge predictions
(`edge_predictions_*_out.csv`, from `21_4_results/edge_predictions/`, columns `Exp. ddG`, `Bennett ddG`,
`CCC ddG` and their errors) were retrieved on 2026-11-25 for the relative examples (`examples/rbfe_*`).

`tyk2/inputs/openfe_dg_tyk2.csv` and `openfe_edges_tyk2.csv` are the published results of OpenFE on this
set (OpenFE 1.0.1, relative hybrid topology protocol, 11 lambda windows of 5 ns, three repeats per edge, the
22 edges of a LOMAP network): the header and the rows whose `dataset_name` is
`tyk2_genentech/jacs_set_tyk2/results_data`, unchanged, of `combined_pymbar3_calculated_dg_data.csv` (SHA256
f2c9b6e30a5b9d2ba51b51813d8bdf5f3c6c2c42281a2d2eb555c96e25edf27d) and `combined_pymbar3_edge_data.csv` (SHA256
7df72fba96afaf57669a0e3f60003efd902d1bc37e04f956be788e5991b73ddd) in
`industry_benchmarks/analysis/processed_results/` of
[IndustryBenchmarks2024](https://github.com/OpenFreeEnergy/IndustryBenchmarks2024) v1.0.0 (Baumann et al.,
Zenodo, doi:10.5281/zenodo.17245550; MIT licence, copyright 2024 Open Free Energy, see `THIRD_PARTY/`),
retrieved on 2026-11-27. Their dG are referenced to the mean of the experimental values. The project published
no result for A2A, so the relative A2A benchmark is compared with experiment and FEP+ alone.
