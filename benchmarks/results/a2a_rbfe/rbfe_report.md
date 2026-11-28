# cuFEP relative benchmark report: a2a

Adenosine A2A receptor (PDB 5OLZ) in a POPC bilayer, Schrödinger public benchmark gpcrs/a2a_hip278

Edges with a ddG: 21 of 21 (planned by LOMAP); compounds with a dG: 17 of 17.

Compared with:

- cuFEP: `cufep rbfe` with the settings of the set's relative example in a POPC bilayer, 5.0 ns of production per lambda window, calculations per edge: 1, lambda windows adaptive, backend cupotential (`resolved_config.yaml`)
- FEP+: the published FEP+ predictions of the public benchmark (Ross et al., Commun. Chem. 2023; `ligands.csv` and `edge_predictions_*_out.csv` of the set)
- OpenFE: the OpenFE benchmark published no result for this set; its columns are empty

The table of the compounds gives the dG as estimated and as published: cuFEP's maximum likelihood estimate from its edges referenced to the mean of the experimental values of the compounds of this run, FEP+ and OpenFE referenced to that of their own sets. The statistics and the figure of the dG centre every method on the compounds compared. The numbers are given without a verdict.

## dG of the compounds against experiment

The same 17 compounds for every method, each method centred on them (its mean set to the mean of their experimental values, as cinnabar's plots of dG centre them), so that RMSE and MUE compare like with like; pairwise RMSE, R^2 and Kendall tau do not depend on it. 95 % bootstrap intervals over compounds in brackets, each resample centred on its own compounds.

| statistic | cuFEP | FEP+ |
|---|---|---|
| RMSE (kcal/mol) | 1.51 [0.79, 1.98] | 1.08 [0.58, 1.40] |
| MUE (kcal/mol) | 1.15 [0.63, 1.70] | 0.79 [0.41, 1.17] |
| pairwise ddG RMSE (kcal/mol) | 2.21 [1.16, 2.89] | 1.58 [0.85, 2.05] |
| Pearson R^2 | 0.68 [0.29, 0.90] | 0.50 [0.08, 0.81] |
| Kendall tau | 0.69 [0.44, 0.89] | 0.59 [0.28, 0.86] |

## ddG of the edges against experiment

cuFEP: the ddG of the edge itself; from the network: the difference of the two dG of the estimate. FEP+ Bennett: the ddG of the edge as published; cycle closure: the published ddG after the correction of the cycles. OpenFE: the mean of its three repeats (± the standard deviation propagated from the three repeats of the complex and of the solvent leg, as published). Every ddG in the direction of the edge of this run; 95 % bootstrap intervals over edges in brackets.

| edges | n | statistic | cuFEP | cuFEP from the network | FEP+ Bennett | FEP+ cycle closure | OpenFE |
|---|---|---|---|---|---|---|---|
| all 21 edges of the run | 21 | RMSE (kcal/mol) | 1.66 [1.19, 2.11] | 1.65 [1.18, 2.09] |  |  |  |
| all 21 edges of the run | 21 | MUE (kcal/mol) | 1.35 [0.95, 1.78] | 1.35 [0.97, 1.78] |  |  |  |
| the 10 edges also in the published map | 10 | RMSE (kcal/mol) | 2.16 [1.53, 2.80] |  | 1.42 [0.97, 1.81] | 1.39 [0.97, 1.76] |  |
| the 10 edges also in the published map | 10 | MUE (kcal/mol) | 1.95 [1.40, 2.58] |  | 1.27 [0.89, 1.67] | 1.26 [0.89, 1.64] |  |

## Compounds (kcal/mol)

| compound | experiment | cuFEP | SE | FEP+ | OpenFE | cuFEP - exp | FEP+ - exp | OpenFE - exp |
|---|---|---|---|---|---|---|---|---|
| 4e | -12.07 | -13.68 | 0.15 | -11.01 | - | -1.61 | +1.06 | - |
| 4n | -11.55 | -13.09 | 0.10 | -13.93 | - | -1.54 | -2.38 | - |
| 4k | -11.54 | -12.43 | 0.20 | -12.00 | - | -0.89 | -0.46 | - |
| 4r | -11.46 | -13.44 | 0.24 | -12.31 | - | -1.98 | -0.85 | - |
| 4c | -11.46 | -11.09 | 0.12 | -12.09 | - | +0.37 | -0.63 | - |
| 4f | -11.45 | -12.63 | 0.18 | -11.72 | - | -1.18 | -0.27 | - |
| 4l | -11.38 | -11.44 | 0.17 | -11.32 | - | -0.06 | +0.06 | - |
| 4o | -11.36 | -8.93 | 0.16 | -9.44 | - | +2.43 | +1.92 | - |
| 4g | -11.06 | -11.40 | 0.10 | -11.83 | - | -0.34 | -0.77 | - |
| 4j | -10.89 | -11.86 | 0.11 | -11.54 | - | -0.97 | -0.65 | - |
| 4h | -10.65 | -11.07 | 0.20 | -11.13 | - | -0.42 | -0.48 | - |
| 4m | -10.55 | -11.20 | 0.10 | -11.71 | - | -0.65 | -1.16 | - |
| 4d | -10.51 | -8.98 | 0.13 | -9.98 | - | +1.52 | +0.53 | - |
| 4i | -10.31 | -10.48 | 0.11 | -10.97 | - | -0.16 | -0.66 | - |
| 4b | -9.96 | -9.81 | 0.13 | -10.06 | - | +0.15 | -0.10 | - |
| 4a | -9.41 | -5.40 | 0.15 | -7.15 | - | +4.01 | +2.26 | - |
| 4q | -8.89 | -7.57 | 0.12 | -8.96 | - | +1.31 | -0.07 | - |

## Edges (kcal/mol)

Windows, lowest overlap of neighbouring windows (all iterations), fewest independent samples in a window and backend, each of the complex and the solvent leg; SE: standard error of cuFEP's ddG; OpenFE: mean of its three repeats ± the standard deviation propagated from the three repeats of the complex and of the solvent leg. Flags, the two checks of the analysis: NOT RELIABLE names a leg whose overlap is below its level (Klimovich et al. 2015); few independent samples names a leg with a window below the level of the analysis (50 by default, Mey et al. 2020), for which more sampling is advisable. `-`: the edge is not in that network.

| edge | LOMAP score | experiment | cuFEP | SE | from the network | FEP+ Bennett | FEP+ cycle closure | OpenFE | windows | lowest overlap | fewest independent samples | backend | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4a -> 4o | 0.90 | -1.94 | -3.49 | 0.17 | -3.53 | -1.37 | -1.51 |  | 9, 9 | 0.112, 0.158 | 286, 1000 | cupotential, cupotential |  |
| 4a -> 4q | 0.95 | +0.53 | -2.20 | 0.14 | -2.17 | -1.87 | -1.83 |  | 9, 9 | 0.146, 0.159 | 400, 1000 | cupotential, cupotential |  |
| 4b -> 4e | 0.90 | -2.12 | -4.05 | 0.24 | -3.87 | -1.03 | -1.02 |  | 9, 9 | 0.101, 0.157 | 167, 1000 | cupotential, cupotential |  |
| 4b -> 4m | 0.95 | -0.59 | -1.36 | 0.10 | -1.39 | -1.70 | -1.66 |  | 9, 9 | 0.136, 0.159 | 334, 1000 | cupotential, cupotential |  |
| 4c -> 4d | 0.90 | +0.96 | +2.07 | 0.18 | +2.11 | +2.00 | +2.11 |  | 9, 9 | 0.182, 0.177 | 36, 1000 | cupotential, cupotential | few independent samples: complex |
| 4c -> 4n | 0.95 | -0.09 | -1.98 | 0.13 | -2.00 | -1.74 | -1.84 |  | 9, 9 | 0.134, 0.162 | 72, 1000 | cupotential, cupotential |  |
| 4d -> 4f | 0.90 | -0.94 | -3.59 | 0.17 | -3.64 | -0.58 | -0.43 |  | 9, 9 | 0.147, 0.147 | 58, 1000 | openmm, openmm |  |
| 4f -> 4e | 0.86 | -0.63 | -0.86 | 0.33 | -1.06 | - | - |  | 9, 9 | 0.089, 0.127 | 96, 1000 | cupotential, cupotential |  |
| 4g -> 4d | 0.95 | +0.56 | +2.52 | 0.19 | +2.42 | +1.82 | +1.85 |  | 9, 9 | 0.109, 0.160 | 223, 1000 | cupotential, cupotential |  |
| 4g -> 4i | 0.82 | +0.75 | +0.91 | 0.08 | +0.92 | - | - |  | 9, 9 | 0.145, 0.117 | 334, 1000 | cupotential, cupotential |  |
| 4g -> 4j | 0.90 | +0.18 | -0.44 | 0.07 | -0.46 | - | - |  | 9, 9 | 0.118, 0.121 | 667, 1000 | cupotential, cupotential |  |
| 4g -> 4k | 0.74 | -0.48 | -0.95 | 0.22 | -1.03 | - | - |  | 9, 9 | 0.115, 0.117 | 200, 1000 | cupotential, cupotential |  |
| 4g -> 4l | 0.67 | -0.31 | -0.09 | 0.18 | -0.04 | - | - |  | 9, 9 | 0.081, 0.110 | 200, 1000 | cupotential, cupotential |  |
| 4g -> 4n | 0.90 | -0.49 | -1.77 | 0.18 | -1.69 | - | - |  | 9, 9 | 0.165, 0.177 | 667, 1000 | cupotential, cupotential |  |
| 4h -> 4l | 0.74 | -0.72 | -0.33 | 0.15 | -0.37 | - | - |  | 9, 9 | 0.094, 0.115 | 223, 1000 | cupotential, cupotential |  |
| 4h -> 4r | 0.70 | -0.81 | -2.49 | 0.27 | -2.37 | - | - |  | 9, 9 | 0.130, 0.155 | 27, 1000 | cupotential, cupotential | few independent samples: complex |
| 4j -> 4i | 0.90 | +0.57 | +1.38 | 0.05 | +1.38 | - | - |  | 9, 9 | 0.111, 0.121 | 667, 1000 | cupotential, cupotential |  |
| 4k -> 4r | 0.86 | +0.08 | -0.88 | 0.29 | -1.01 | - | - |  | 9, 9 | 0.152, 0.188 | 143, 1000 | cupotential, cupotential |  |
| 4m -> 4n | 0.90 | -1.00 | -1.87 | 0.09 | -1.88 | -1.94 | -1.85 |  | 9, 9 | 0.122, 0.123 | 334, 1000 | cupotential, cupotential |  |
| 4m -> 4q | 0.90 | +1.66 | +3.64 | 0.08 | +3.63 | - | - |  | 9, 9 | 0.129, 0.126 | 286, 1000 | cupotential, cupotential |  |
| 4o -> 4e | 0.90 | -0.72 | -4.75 | 0.05 | -4.75 | -2.99 | -2.79 |  | 9, 9 | 0.121, 0.128 | 500, 1000 | cupotential, cupotential |  |

Edges with a flag: NOT RELIABLE: none; few independent samples: 4c~4d, 4h~4r.

Legs on another backend than the cuPotential plugin (the fallback to stock OpenMM, whose reason the log of the leg gives): 4d~4f.

## Cycle closures (kcal/mol)

| cycle | sum of ddG | standard error |
|---|---|---|
| 4b -> 4e -> 4o -> 4a -> 4q -> 4m | -0.29 | 0.35 |
| 4d -> 4f -> 4e -> 4o -> 4a -> 4q -> 4m -> 4n -> 4g | +0.38 | 0.52 |
| 4d -> 4c -> 4n -> 4g | +0.24 | 0.34 |
| 4h -> 4l -> 4g -> 4k -> 4r | +0.42 | 0.51 |
| 4i -> 4j -> 4g | -0.04 | 0.11 |
