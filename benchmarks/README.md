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

`*/inputs/` holds unmodified files from the Schrödinger
[public binding free energy benchmark](https://github.com/schrodinger/public_binding_free_energy_benchmark)
(MIT licence, see `THIRD_PARTY/`): protein PDB, ligand SDF with experimental dG
(`r_exp_dg`, kcal/mol), perturbation edges, and the FEP+ per-ligand predictions
(`*_out.csv`, columns `Exp. dG` / `Pred. dG`). `THIRD_PARTY/SHA256SUMS` pins the
downloaded files (retrieved 2026-11-18 from the `main` branch).
