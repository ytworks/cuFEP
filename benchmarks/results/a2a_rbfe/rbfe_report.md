# cuFEP relative benchmark report: a2a

Adenosine A2A receptor (PDB 5OLZ) in a POPC bilayer, Schrödinger public benchmark gpcrs/a2a_hip278

Edges with a ddG: 21 of 21 (planned by LOMAP); compounds with a dG: 17 of 17.

Compared with:

- cuFEP: `cufep rbfe` with the settings of the set's relative example in a POPC bilayer, 5.0 ns of production per lambda window, calculations per edge: 1, lambda windows adaptive, backend cupotential (`resolved_config.yaml`)
- FEP+: the published FEP+ predictions of the public benchmark (Ross et al., Commun. Chem. 2023; `ligands.csv` and `edge_predictions_*_out.csv` of the set)
- OpenFE: the OpenFE benchmark published no result for this set; its columns are empty

The table of the compounds gives the dG as estimated and as published: cuFEP's maximum likelihood estimate from its edges referenced to the mean of the experimental values of the compounds of this run, FEP+ and OpenFE referenced to that of their own sets. The statistics and the figure of the dG centre every method on the compounds compared. The numbers are given without a verdict.

## dG of the compounds against experiment

The same 17 compounds for every method, each method centred on them (its mean set to the mean of their experimental values, as cinnabar's plots of dG centre them), so that RMSE and MUE compare like with like; pairwise RMSE, R^2 and Kendall tau do not depend on it. 95 % bootstrap intervals over compounds in brackets.

| statistic | cuFEP | FEP+ |
|---|---|---|
| RMSE (kcal/mol) | 1.59 [0.95, 2.14] | 1.08 [0.63, 1.45] |
| MUE (kcal/mol) | 1.22 [0.77, 1.72] | 0.79 [0.47, 1.14] |
| pairwise ddG RMSE (kcal/mol) | 2.33 [1.16, 2.99] | 1.58 [0.85, 2.05] |
| Pearson R^2 | 0.51 [0.07, 0.81] | 0.50 [0.08, 0.81] |
| Kendall tau | 0.57 [0.31, 0.77] | 0.59 [0.28, 0.86] |

## ddG of the edges against experiment

cuFEP: the ddG of the edge itself; from the network: the difference of the two dG of the estimate. FEP+ Bennett: the ddG of the edge as published; cycle closure: the published ddG after the correction of the cycles. OpenFE: the mean of its three repeats (± their standard deviation, as published). Every ddG in the direction of the edge of this run; 95 % bootstrap intervals over edges in brackets.

| edges | n | statistic | cuFEP | cuFEP from the network | FEP+ Bennett | FEP+ cycle closure | OpenFE |
|---|---|---|---|---|---|---|---|
| all 21 edges of the run | 21 | RMSE (kcal/mol) | 1.57 [1.11, 1.97] | 1.54 [1.11, 1.95] |  |  |  |
| all 21 edges of the run | 21 | MUE (kcal/mol) | 1.27 [0.89, 1.67] | 1.26 [0.89, 1.65] |  |  |  |
| the 10 edges also in the published map | 10 | RMSE (kcal/mol) | 1.97 [1.29, 2.55] |  | 1.42 [0.97, 1.81] | 1.39 [0.97, 1.76] |  |
| the 10 edges also in the published map | 10 | MUE (kcal/mol) | 1.66 [1.03, 2.33] |  | 1.27 [0.89, 1.67] | 1.26 [0.89, 1.64] |  |

## Compounds (kcal/mol)

| compound | experiment | cuFEP | SE | FEP+ | OpenFE | cuFEP - exp | FEP+ - exp | OpenFE - exp |
|---|---|---|---|---|---|---|---|---|
| 4e | -12.07 | -12.33 | 0.20 | -11.01 | - | -0.25 | +1.06 | - |
| 4n | -11.55 | -13.26 | 0.10 | -13.93 | - | -1.71 | -2.38 | - |
| 4k | -11.54 | -12.73 | 0.16 | -12.00 | - | -1.18 | -0.46 | - |
| 4r | -11.46 | -13.60 | 0.21 | -12.31 | - | -2.14 | -0.85 | - |
| 4c | -11.46 | -11.42 | 0.13 | -12.09 | - | +0.04 | -0.63 | - |
| 4f | -11.45 | -11.77 | 0.29 | -11.72 | - | -0.32 | -0.27 | - |
| 4l | -11.38 | -10.83 | 0.14 | -11.32 | - | +0.55 | +0.06 | - |
| 4o | -11.36 | -7.98 | 0.20 | -9.44 | - | +3.38 | +1.92 | - |
| 4g | -11.06 | -11.84 | 0.11 | -11.83 | - | -0.78 | -0.77 | - |
| 4j | -10.89 | -12.41 | 0.11 | -11.54 | - | -1.52 | -0.65 | - |
| 4h | -10.65 | -11.22 | 0.22 | -11.13 | - | -0.57 | -0.48 | - |
| 4m | -10.55 | -11.74 | 0.12 | -11.71 | - | -1.19 | -1.16 | - |
| 4d | -10.51 | -9.11 | 0.15 | -9.98 | - | +1.40 | +0.53 | - |
| 4i | -10.31 | -11.03 | 0.12 | -10.97 | - | -0.72 | -0.66 | - |
| 4b | -9.96 | -9.70 | 0.16 | -10.06 | - | +0.26 | -0.10 | - |
| 4a | -9.41 | -5.63 | 0.17 | -7.15 | - | +3.78 | +2.26 | - |
| 4q | -8.89 | -7.91 | 0.13 | -8.96 | - | +0.98 | -0.07 | - |

## Edges (kcal/mol)

Windows, lowest overlap of neighbouring windows (all iterations) and backend of the complex and the solvent leg; SE: standard error of cuFEP's ddG; OpenFE: mean ± standard deviation of its three repeats; NOT RELIABLE names a leg whose overlap is below the level of the analysis (Klimovich et al. 2015). `-`: the edge is not in that network.

| edge | LOMAP score | experiment | cuFEP | SE | from the network | FEP+ Bennett | FEP+ cycle closure | OpenFE | windows | lowest overlap | backend | NOT RELIABLE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4a -> 4o | 0.90 | -1.94 | -2.32 | 0.19 | -2.35 | -1.37 | -1.51 |  | 9, 9 | 0.104, 0.158 | cupotential, cupotential |  |
| 4a -> 4q | 0.95 | +0.53 | -2.30 | 0.16 | -2.27 | -1.87 | -1.83 |  | 9, 9 | 0.137, 0.158 | cupotential, cupotential |  |
| 4b -> 4e | 0.90 | -2.12 | -2.35 | 0.45 | -2.62 | -1.03 | -1.02 |  | 9, 9 | 0.115, 0.157 | cupotential, cupotential |  |
| 4b -> 4m | 0.95 | -0.59 | -2.06 | 0.12 | -2.04 | -1.70 | -1.66 |  | 9, 9 | 0.114, 0.161 | cupotential, cupotential |  |
| 4c -> 4d | 0.90 | +0.96 | +2.36 | 0.13 | +2.32 | +2.00 | +2.11 |  | 9, 9 | 0.170, 0.178 | cupotential, cupotential |  |
| 4c -> 4n | 0.95 | -0.09 | -1.89 | 0.13 | -1.84 | -1.74 | -1.84 |  | 9, 9 | 0.130, 0.160 | cupotential, cupotential |  |
| 4d -> 4f | 0.90 | -0.94 | -3.42 | 0.58 | -2.66 | -0.58 | -0.43 |  | 9, 9 | 0.150, 0.147 | openmm, openmm |  |
| 4f -> 4e | 0.86 | -0.63 | -0.76 | 0.30 | -0.56 | - | - |  | 9, 9 | 0.084, 0.128 | cupotential, cupotential |  |
| 4g -> 4d | 0.95 | +0.56 | +2.42 | 0.25 | +2.74 | +1.82 | +1.85 |  | 9, 9 | 0.157, 0.164 | cupotential, cupotential |  |
| 4g -> 4i | 0.82 | +0.75 | +0.94 | 0.06 | +0.81 | - | - |  | 9, 9 | 0.109, 0.120 | cupotential, cupotential |  |
| 4g -> 4j | 0.90 | +0.18 | -0.61 | 0.04 | -0.57 | - | - |  | 9, 9 | 0.126, 0.123 | cupotential, cupotential |  |
| 4g -> 4k | 0.74 | -0.48 | -0.86 | 0.14 | -0.88 | - | - |  | 9, 9 | 0.105, 0.116 | cupotential, cupotential |  |
| 4g -> 4l | 0.67 | -0.31 | +1.01 | 0.11 | +1.02 | - | - |  | 9, 9 | 0.124, 0.111 | cupotential, cupotential |  |
| 4g -> 4n | 0.90 | -0.49 | -1.32 | 0.14 | -1.42 | - | - |  | 9, 9 | 0.180, 0.184 | cupotential, cupotential |  |
| 4h -> 4l | 0.74 | -0.72 | +0.45 | 0.24 | +0.40 | - | - |  | 9, 9 | 0.084, 0.113 | cupotential, cupotential |  |
| 4h -> 4r | 0.70 | -0.81 | -2.44 | 0.27 | -2.38 | - | - |  | 9, 9 | 0.134, 0.152 | cupotential, cupotential |  |
| 4j -> 4i | 0.90 | +0.57 | +1.15 | 0.08 | +1.38 | - | - |  | 9, 9 | 0.100, 0.120 | cupotential, cupotential |  |
| 4k -> 4r | 0.86 | +0.08 | -0.84 | 0.20 | -0.87 | - | - |  | 9, 9 | 0.176, 0.187 | cupotential, cupotential |  |
| 4m -> 4n | 0.90 | -1.00 | -1.58 | 0.15 | -1.53 | -1.94 | -1.85 |  | 9, 9 | 0.125, 0.124 | cupotential, cupotential |  |
| 4m -> 4q | 0.90 | +1.66 | +3.84 | 0.08 | +3.83 | - | - |  | 9, 9 | 0.129, 0.126 | cupotential, cupotential |  |
| 4o -> 4e | 0.90 | -0.72 | -4.34 | 0.05 | -4.35 | -2.99 | -2.79 |  | 9, 9 | 0.136, 0.130 | cupotential, cupotential |  |

Legs on another backend than the cuPotential plugin (the fallback to stock OpenMM, whose reason the log of the leg gives): 4d~4f.

## Cycle closures (kcal/mol)

| cycle | sum of ddG | standard error |
|---|---|---|
| 4b -> 4e -> 4o -> 4a -> 4q -> 4m | +0.24 | 0.53 |
| 4d -> 4f -> 4e -> 4o -> 4a -> 4q -> 4m -> 4n -> 4g | -1.48 | 0.78 |
| 4d -> 4c -> 4n -> 4g | -0.51 | 0.34 |
| 4h -> 4l -> 4g -> 4k -> 4r | +0.19 | 0.45 |
| 4i -> 4j -> 4g | +0.40 | 0.11 |
