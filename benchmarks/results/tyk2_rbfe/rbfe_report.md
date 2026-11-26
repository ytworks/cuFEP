# cuFEP relative benchmark report: tyk2

TYK2 kinase (PDB 4GIH), Schrödinger public benchmark jacs_set/tyk2

Ligands with a dG: 16 of 16; edges with a ddG: 22 of 22 (edges planned by LOMAP).

The dG of cuFEP come from the edges alone, their mean over the ligands with an experimental value set to the mean of those values (`docs/rbfe.md`, "The network"); the published FEP+ values are relative calculations shifted to the experimental mean of the whole published set, with 20 ns per lambda window (Ross et al. 2023), and cuFEP's absolute values (`benchmarks/results/`) are not shifted. The pairwise RMSE, R^2 and Kendall tau do not depend on a shift.

## Per ligand (kcal/mol)

| ligand | experiment | cuFEP RBFE | standard error | FEP+ | cuFEP ABFE |
|---|---|---|---|---|---|
| jmc_23 | -11.70 | -11.60 | 0.23 | -11.27 | -13.36 |
| ejm_46 | -11.31 | -11.08 | 0.16 | -10.82 | -10.40 |
| jmc_27 | -11.28 | -10.93 | 0.19 | -11.10 | -12.41 |
| jmc_28 | -10.98 | -10.58 | 0.25 | -10.34 | -10.39 |
| jmc_30 | -10.94 | -11.60 | 0.25 | -11.13 | -12.42 |
| ejm_54 | -10.53 | -9.87 | 0.33 | -10.38 | -13.51 |
| ejm_42 | -9.78 | -9.50 | 0.14 | -9.95 | -10.62 |
| ejm_47 | -9.70 | -10.02 | 0.27 | -9.94 | -11.44 |
| ejm_45 | -9.56 | -9.43 | 0.39 | -9.66 | -12.04 |
| ejm_31 | -9.54 | -9.80 | 0.11 | -9.91 | -10.08 |
| ejm_55 | -9.21 | -10.09 | 0.28 | -10.34 | -9.02 |
| ejm_48 | -9.00 | -8.49 | 0.27 | -8.86 | -10.21 |
| ejm_50 | -8.98 | -9.98 | 0.17 | -9.40 | -9.08 |
| ejm_43 | -8.26 | -7.75 | 0.21 | -8.30 | -9.59 |
| ejm_49 | -7.75 | -8.72 | 0.27 | -8.05 | -11.84 |
| ejm_44 | -7.42 | -6.50 | 0.25 | -6.48 | -8.72 |

## Agreement of the dG with experiment

16 ligands with a dG and an experimental value; 95 % bootstrap intervals over ligands in brackets.

| statistic | cuFEP RBFE | FEP+ | cuFEP ABFE |
|---|---|---|---|
| mean signed error (kcal/mol) | 0.00 [-0.31, 0.29] | 0.00 [-0.23, 0.23] | -1.20 [-1.83, -0.60] |
| RMSE, raw (kcal/mol) | 0.59 [0.43, 0.73] | 0.47 [0.29, 0.65] | 1.73 [1.10, 2.30] |
| MUE, raw (kcal/mol) | 0.51 [0.36, 0.66] | 0.37 [0.24, 0.53] | 1.41 [0.94, 1.93] |
| RMSE after removing the mean error (kcal/mol) | 0.59 [0.39, 0.70] | 0.47 [0.28, 0.63] | 1.25 [0.75, 1.62] |
| pairwise ddG RMSE (kcal/mol) | 0.86 [0.57, 1.03] | 0.69 [0.41, 0.92] | 1.83 [1.10, 2.36] |
| Kendall tau | 0.68 [0.41, 0.89] | 0.81 [0.61, 0.96] | 0.47 [0.12, 0.75] |
| Spearman rho | 0.84 [0.54, 0.97] | 0.93 [0.75, 0.99] | 0.65 [0.19, 0.90] |
| Pearson r | 0.90 [0.79, 0.97] | 0.93 [0.85, 0.98] | 0.59 [0.15, 0.87] |
| Pearson R^2 | 0.81 [0.63, 0.94] | 0.86 [0.73, 0.96] | 0.35 [0.03, 0.75] |

## Per edge (kcal/mol)

ddG = dG(B) - dG(A). Own: the edge's two legs; network: dG(B) - dG(A) of the estimate from all edges (what the cycle closure correction gives an edge); FEP+: the published prediction of the same edge (Bennett and cycle closure corrected), where the published map has it. 

| edge | experiment | cuFEP own | cuFEP network | FEP+ Bennett | FEP+ CCC | lowest overlap complex, solvent | NOT RELIABLE |
|---|---|---|---|---|---|---|---|
| ejm_31~ejm_42 | -0.24 | +0.48 ± 0.17 | +0.30 ± 0.15 |  |  | 0.073, 0.097 |  |
| ejm_31~ejm_44 | +2.12 | +3.28 ± 0.47 | +3.30 ± 0.27 |  |  | 0.032, 0.061 |  |
| ejm_31~ejm_45 | -0.02 | +0.37 ± 0.40 | +0.37 ± 0.40 | +0.21 | +0.25 | 0.022, 0.083 | complex |
| ejm_31~ejm_46 | -1.77 | -1.34 ± 0.19 | -1.27 ± 0.18 | -0.84 | -0.91 | 0.076, 0.101 |  |
| ejm_31~ejm_48 | +0.54 | +0.54 ± 0.47 | +1.32 ± 0.30 | +1.03 | +1.05 | 0.040, 0.070 |  |
| ejm_42~ejm_54 | -0.75 | -0.24 ± 0.64 | -0.37 ± 0.37 | -0.44 | -0.43 | 0.071, 0.159 |  |
| ejm_43~ejm_42 | -1.52 | -1.75 ± 0.22 | -1.75 ± 0.20 |  |  | 0.052, 0.100 |  |
| ejm_44~ejm_43 | -0.84 | -1.25 ± 0.20 | -1.25 ± 0.19 |  |  | 0.048, 0.096 |  |
| ejm_47~ejm_31 | +0.16 | +0.28 ± 0.32 | +0.22 ± 0.28 | +0.06 | +0.03 | 0.037, 0.083 |  |
| ejm_47~ejm_50 | +0.72 | -0.09 ± 0.48 | +0.04 ± 0.31 |  |  | 0.043, 0.086 |  |
| ejm_49~ejm_31 | -1.79 | -0.74 ± 0.38 | -1.09 ± 0.29 | -1.84 | -1.86 | 0.042, 0.054 |  |
| ejm_49~ejm_50 | -1.23 | -1.57 ± 0.36 | -1.26 ± 0.28 | -1.38 | -1.35 | 0.072, 0.044 |  |
| ejm_50~ejm_42 | -0.80 | +0.14 ± 0.23 | +0.48 ± 0.19 | -0.58 | -0.55 | 0.072, 0.129 |  |
| ejm_50~ejm_48 | -0.02 | +1.82 ± 0.30 | +1.49 ± 0.26 |  |  | 0.035, 0.045 |  |
| ejm_50~ejm_55 | -0.23 | -0.13 ± 0.30 | -0.11 ± 0.28 |  |  | 0.072, 0.142 |  |
| ejm_55~ejm_54 | -1.32 | +0.20 ± 0.27 | +0.22 ± 0.25 | -0.03 | -0.04 | 0.086, 0.144 |  |
| jmc_23~ejm_31 | +2.16 | +1.39 ± 0.48 | +1.80 ± 0.26 |  |  | 0.059, 0.088 |  |
| jmc_23~jmc_27 | +0.42 | +0.95 ± 0.26 | +0.68 ± 0.21 | +0.19 | +0.17 | 0.097, 0.218 |  |
| jmc_23~jmc_28 | +0.72 | +0.87 ± 0.27 | +1.02 ± 0.23 |  |  | 0.069, 0.148 |  |
| jmc_27~ejm_46 | -0.03 | -0.08 ± 0.13 | -0.15 ± 0.13 |  |  | 0.079, 0.125 |  |
| jmc_28~jmc_30 | +0.04 | -1.05 ± 0.14 | -1.02 ± 0.13 | -0.78 | -0.80 | 0.118, 0.162 |  |
| jmc_30~ejm_46 | -0.37 | +0.32 ± 0.31 | +0.52 ± 0.24 |  |  | 0.084, 0.135 |  |

## Agreement of the ddG of the edges with experiment

| ddG | edges | RMSE | MUE |
|---|---|---|---|
| cuFEP, own | 22 | 0.79 | 0.63 |
| cuFEP, network | 22 | 0.75 | 0.61 |
| cuFEP, own, on the edges FEP+ also has | 11 | 0.77 | 0.63 |
| FEP+ Bennett, the same edges | 11 | 0.58 | 0.44 |
| FEP+ Bennett, its whole map among these ligands | 24 | 0.83 | 0.65 |
