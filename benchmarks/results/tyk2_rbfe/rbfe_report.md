# cuFEP relative benchmark report: tyk2

TYK2 kinase (PDB 4GIH), Schrödinger public benchmark jacs_set/tyk2

Edges with a ddG: 22 of 22 (planned by LOMAP); compounds with a dG: 16 of 16.

Compared with:

- cuFEP: `cufep rbfe` with the settings of the set's relative example, 5.0 ns of production per lambda window, calculations per edge: 1, lambda windows adaptive, backend cupotential (`resolved_config.yaml`)
- FEP+: the published FEP+ predictions of the public benchmark (Ross et al., Commun. Chem. 2023; `ligands.csv` and `edge_predictions_*_out.csv` of the set)
- OpenFE: the published results of the OpenFE benchmark (IndustryBenchmarks2024 v1.0.0, Baumann et al.): OpenFE 1.0.1, relative hybrid topology protocol, 11 lambda windows of 5 ns, three repeats per edge averaged, the LOMAP network of the set

The table of the compounds gives the dG as estimated and as published: cuFEP's maximum likelihood estimate from its edges referenced to the mean of the experimental values of the compounds of this run, FEP+ and OpenFE referenced to that of their own sets. The statistics and the figure of the dG centre every method on the compounds compared. The numbers are given without a verdict.

## dG of the compounds against experiment

The same 16 compounds for every method, each method centred on them (its mean set to the mean of their experimental values, as cinnabar's plots of dG centre them), so that RMSE and MUE compare like with like; pairwise RMSE, R^2 and Kendall tau do not depend on it. 95 % bootstrap intervals over compounds in brackets.

| statistic | cuFEP | FEP+ | OpenFE |
|---|---|---|---|
| RMSE (kcal/mol) | 0.67 [0.34, 1.00] | 0.47 [0.29, 0.65] | 0.59 [0.42, 0.74] |
| MUE (kcal/mol) | 0.48 [0.27, 0.73] | 0.37 [0.24, 0.53] | 0.48 [0.32, 0.65] |
| pairwise ddG RMSE (kcal/mol) | 0.98 [0.45, 1.42] | 0.69 [0.41, 0.92] | 0.86 [0.56, 1.06] |
| Pearson R^2 | 0.73 [0.33, 0.95] | 0.86 [0.73, 0.96] | 0.80 [0.65, 0.93] |
| Kendall tau | 0.70 [0.37, 0.95] | 0.81 [0.61, 0.96] | 0.68 [0.43, 0.88] |

## ddG of the edges against experiment

cuFEP: the ddG of the edge itself; from the network: the difference of the two dG of the estimate. FEP+ Bennett: the ddG of the edge as published; cycle closure: the published ddG after the correction of the cycles. OpenFE: the mean of its three repeats (± their standard deviation, as published). Every ddG in the direction of the edge of this run; 95 % bootstrap intervals over edges in brackets.

| edges | n | statistic | cuFEP | cuFEP from the network | FEP+ Bennett | FEP+ cycle closure | OpenFE |
|---|---|---|---|---|---|---|---|
| all 22 edges of the run | 22 | RMSE (kcal/mol) | 0.93 [0.51, 1.29] | 0.91 [0.47, 1.28] |  |  |  |
| all 22 edges of the run | 22 | MUE (kcal/mol) | 0.67 [0.42, 0.95] | 0.63 [0.38, 0.91] |  |  |  |
| the 11 edges also in the published map | 11 | RMSE (kcal/mol) | 1.23 [0.64, 1.73] |  | 0.58 [0.27, 0.81] | 0.58 [0.29, 0.81] |  |
| the 11 edges also in the published map | 11 | MUE (kcal/mol) | 0.98 [0.58, 1.45] |  | 0.44 [0.23, 0.68] | 0.45 [0.24, 0.68] |  |
| the 22 edges also in OpenFE's network | 22 | RMSE (kcal/mol) | 0.93 [0.51, 1.29] |  |  |  | 0.61 [0.48, 0.72] |
| the 22 edges also in OpenFE's network | 22 | MUE (kcal/mol) | 0.67 [0.42, 0.95] |  |  |  | 0.53 [0.40, 0.66] |

## Compounds (kcal/mol)

| compound | experiment | cuFEP | SE | FEP+ | OpenFE | cuFEP - exp | FEP+ - exp | OpenFE - exp |
|---|---|---|---|---|---|---|---|---|
| jmc_23 | -11.70 | -11.83 | 0.25 | -11.27 | -11.10 | -0.13 | +0.43 | +0.60 |
| ejm_46 | -11.31 | -11.06 | 0.23 | -10.82 | -11.02 | +0.25 | +0.49 | +0.29 |
| jmc_27 | -11.28 | -11.00 | 0.28 | -11.10 | -11.19 | +0.28 | +0.18 | +0.09 |
| jmc_28 | -10.98 | -10.40 | 0.28 | -10.34 | -10.40 | +0.58 | +0.64 | +0.58 |
| jmc_30 | -10.94 | -11.41 | 0.25 | -11.13 | -11.43 | -0.47 | -0.19 | -0.49 |
| ejm_54 | -10.53 | -8.49 | 0.40 | -10.38 | -10.33 | +2.04 | +0.15 | +0.20 |
| ejm_42 | -9.78 | -10.13 | 0.23 | -9.95 | -9.34 | -0.35 | -0.17 | +0.44 |
| ejm_47 | -9.70 | -10.71 | 0.26 | -9.94 | -9.77 | -1.01 | -0.24 | -0.07 |
| ejm_45 | -9.56 | -8.99 | 0.30 | -9.66 | -9.56 | +0.57 | -0.10 | +0.00 |
| ejm_31 | -9.54 | -9.73 | 0.14 | -9.91 | -9.83 | -0.19 | -0.37 | -0.29 |
| ejm_55 | -9.21 | -9.77 | 0.36 | -10.34 | -10.33 | -0.56 | -1.13 | -1.12 |
| ejm_48 | -9.00 | -9.11 | 0.30 | -8.86 | -9.13 | -0.11 | +0.14 | -0.13 |
| ejm_50 | -8.98 | -9.06 | 0.23 | -9.40 | -9.66 | -0.08 | -0.42 | -0.68 |
| ejm_43 | -8.26 | -8.49 | 0.30 | -8.30 | -7.62 | -0.23 | -0.04 | +0.64 |
| ejm_49 | -7.75 | -8.45 | 0.24 | -8.05 | -8.79 | -0.70 | -0.30 | -1.04 |
| ejm_44 | -7.42 | -7.33 | 0.38 | -6.48 | -6.45 | +0.09 | +0.94 | +0.97 |

## Edges (kcal/mol)

Windows, lowest overlap of neighbouring windows (all iterations) and backend of the complex and the solvent leg; SE: standard error of cuFEP's ddG; OpenFE: mean ± standard deviation of its three repeats; NOT RELIABLE names a leg whose overlap is below the level of the analysis (Klimovich et al. 2015). `-`: the edge is not in that network.

| edge | LOMAP score | experiment | cuFEP | SE | from the network | FEP+ Bennett | FEP+ cycle closure | OpenFE | windows | lowest overlap | backend | NOT RELIABLE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ejm_31 -> ejm_42 | 0.90 | -0.24 | -0.18 | 0.36 | -0.40 | - | - | +0.55 ± 0.05 | 9, 9 | 0.075, 0.097 | cupotential, cupotential |  |
| ejm_31 -> ejm_44 | 0.61 | +2.12 | +2.42 | 0.62 | +2.40 | - | - | +2.71 ± 0.26 | 10, 9 | 0.095, 0.060 | cupotential, cupotential |  |
| ejm_31 -> ejm_45 | 0.74 | -0.02 | +0.74 | 0.29 | +0.74 | +0.21 | +0.25 | +0.27 ± 0.11 | 9, 9 | 0.026, 0.079 | openmm, openmm | complex |
| ejm_31 -> ejm_46 | 0.37 | -1.77 | -1.30 | 0.31 | -1.33 | -0.84 | -0.91 | -1.38 ± 0.12 | 9, 9 | 0.065, 0.097 | cupotential, cupotential |  |
| ejm_31 -> ejm_48 | 0.30 | +0.54 | +0.33 | 0.40 | +0.62 | +1.03 | +1.05 | +0.65 ± 0.19 | 10, 9 | 0.041, 0.066 | cupotential, cupotential |  |
| ejm_42 -> ejm_54 | 0.86 | -0.75 | +1.46 | 0.54 | +1.64 | -0.44 | -0.43 | -0.92 ± 0.11 | 9, 9 | 0.092, 0.158 | openmm, openmm |  |
| ejm_43 -> ejm_42 | 0.90 | -1.52 | -1.63 | 0.28 | -1.64 | - | - | -1.96 ± 0.15 | 9, 9 | 0.062, 0.100 | openmm, openmm |  |
| ejm_44 -> ejm_43 | 0.74 | -0.84 | -1.15 | 0.39 | -1.16 | - | - | -1.27 ± 0.10 | 9, 9 | 0.054, 0.098 | cupotential, cupotential |  |
| ejm_47 -> ejm_31 | 0.33 | +0.16 | +1.03 | 0.33 | +0.98 | +0.06 | +0.03 | -0.10 ± 0.37 | 10, 9 | 0.049, 0.089 | cupotential, cupotential |  |
| ejm_47 -> ejm_50 | 0.30 | +0.72 | +1.58 | 0.38 | +1.65 | - | - | +0.12 ± 0.21 | 10, 9 | 0.048, 0.084 | cupotential, cupotential |  |
| ejm_49 -> ejm_31 | 0.27 | -1.79 | -1.32 | 0.24 | -1.28 | -1.84 | -1.86 | -0.84 ± 0.14 | 10, 10 | 0.041, 0.086 | cupotential, cupotential |  |
| ejm_49 -> ejm_50 | 0.25 | -1.23 | -0.25 | 0.72 | -0.61 | -1.38 | -1.35 | -1.08 ± 0.14 | 10, 10 | 0.075, 0.089 | cupotential, cupotential |  |
| ejm_50 -> ejm_42 | 0.95 | -0.80 | -1.52 | 0.44 | -1.07 | -0.58 | -0.55 | -0.06 ± 0.24 | 9, 9 | 0.095, 0.124 | openmm, openmm |  |
| ejm_50 -> ejm_48 | 0.27 | -0.02 | +0.30 | 0.43 | -0.05 | - | - | +0.56 ± 0.15 | 10, 9 | 0.077, 0.053 | cupotential, cupotential |  |
| ejm_50 -> ejm_55 | 0.90 | -0.23 | -0.61 | 0.41 | -0.72 | - | - | -0.82 ± 0.17 | 9, 9 | 0.080, 0.148 | openmm, openmm |  |
| ejm_55 -> ejm_54 | 0.86 | -1.32 | +1.40 | 0.42 | +1.29 | -0.03 | -0.04 | -0.30 ± 0.23 | 9, 9 | 0.080, 0.144 | cupotential, cupotential |  |
| jmc_23 -> ejm_31 | 0.33 | +2.16 | +2.14 | 0.35 | +2.10 | - | - | +1.24 ± 0.04 | 9, 9 | 0.059, 0.086 | cupotential, cupotential |  |
| jmc_23 -> jmc_27 | 0.95 | +0.42 | +0.75 | 0.43 | +0.83 | +0.19 | +0.17 | -0.02 ± 0.14 | 9, 9 | 0.098, 0.220 | cupotential, cupotential |  |
| jmc_23 -> jmc_28 | 0.95 | +0.72 | +1.45 | 0.33 | +1.43 | - | - | +0.76 ± 0.08 | 9, 9 | 0.101, 0.147 | cupotential, cupotential |  |
| jmc_27 -> ejm_46 | 0.90 | -0.03 | -0.08 | 0.23 | -0.06 | - | - | +0.18 ± 0.06 | 9, 9 | 0.076, 0.126 | cupotential, cupotential |  |
| jmc_28 -> jmc_30 | 0.78 | +0.04 | -1.00 | 0.22 | -1.01 | -0.78 | -0.80 | -1.01 ± 0.05 | 9, 9 | 0.109, 0.160 | cupotential, cupotential |  |
| jmc_30 -> ejm_46 | 0.82 | -0.37 | +0.35 | 0.17 | +0.35 | - | - | +0.53 ± 0.12 | 9, 9 | 0.079, 0.140 | cupotential, cupotential |  |

Legs on another backend than the cuPotential plugin (the fallback to stock OpenMM, whose reason the log of the leg gives): ejm_31~ejm_45, ejm_42~ejm_54, ejm_43~ejm_42, ejm_50~ejm_42, ejm_50~ejm_55.

## Cycle closures (kcal/mol)

| cycle | sum of ddG | standard error |
|---|---|---|
| ejm_31 -> jmc_23 -> jmc_27 -> ejm_46 | -0.17 | 0.68 |
| jmc_28 -> jmc_23 -> jmc_27 -> ejm_46 -> jmc_30 | -0.14 | 0.65 |
| ejm_47 -> ejm_50 -> ejm_49 -> ejm_31 | -0.52 | 0.91 |
| ejm_42 -> ejm_50 -> ejm_49 -> ejm_31 | +0.27 | 0.95 |
| ejm_48 -> ejm_50 -> ejm_49 -> ejm_31 | -1.04 | 0.96 |
| ejm_42 -> ejm_54 -> ejm_55 -> ejm_50 | -0.84 | 0.91 |
| ejm_42 -> ejm_43 -> ejm_44 -> ejm_31 | +0.19 | 0.86 |
