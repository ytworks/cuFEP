# cuFEP benchmark report: tyk2

TYK2 kinase (PDB 4GIH), Schrödinger public benchmark jacs_set/tyk2

Ligands with a result: 16 of 16 run (16 in the set).

## Per ligand (kcal/mol)

Every ligand with a binding free energy, NOT RELIABLE ones included (marked). Standard error: of one calculation, from its own trajectories (- for a result of an earlier version that did not record it). SD over repeats: only where several independent calculations were run. The last two columns are the signed errors against experiment.

| ligand | experiment | cuFEP | standard error | SD over repeats | repeats | FEP+ | cuFEP - experiment | FEP+ - experiment |
|---|---|---|---|---|---|---|---|---|
| jmc_23 | -11.70 | -13.36 | 0.30 |  | 1 | -11.27 | -1.66 | +0.43 |
| ejm_46 | -11.31 | -10.40 | 0.43 |  | 1 | -10.82 | +0.91 | +0.49 |
| jmc_27 | -11.28 | -12.41 | 0.43 |  | 1 | -11.10 | -1.13 | +0.18 |
| jmc_28 | -10.98 | -10.39 | 0.63 |  | 1 | -10.34 | +0.59 | +0.64 |
| jmc_30 | -10.94 | -12.42 | 0.79 |  | 1 | -11.13 | -1.48 | -0.19 |
| ejm_54 | -10.53 | -13.51 | 0.37 |  | 1 | -10.38 | -2.98 | +0.15 |
| ejm_42 | -9.78 | -10.62 | 0.34 |  | 1 | -9.95 | -0.84 | -0.17 |
| ejm_47 | -9.70 | -11.44 | 0.43 |  | 1 | -9.94 | -1.74 | -0.24 |
| ejm_45 | -9.56 | -12.04 | 0.61 |  | 1 | -9.66 | -2.48 | -0.10 |
| ejm_31 | -9.54 | -10.08 | 0.37 |  | 1 | -9.91 | -0.54 | -0.37 |
| ejm_55 | -9.21 | -9.02 | 0.45 |  | 1 | -10.34 | +0.19 | -1.13 |
| ejm_48 | -9.00 | -10.21 | 0.74 |  | 1 | -8.86 | -1.21 | +0.14 |
| ejm_50 | -8.98 | -9.08 | 0.54 |  | 1 | -9.40 | -0.10 | -0.42 |
| ejm_43 | -8.26 | -9.59 | 0.41 |  | 1 | -8.30 | -1.33 | -0.04 |
| ejm_49 | -7.75 | -11.84 | 0.57 |  | 1 | -8.05 | -4.09 | -0.30 |
| ejm_44 | -7.42 | -8.72 | 0.47 |  | 1 | -6.48 | -1.30 | +0.94 |

## Lambda windows and checks of the analysis

Windows per leg, how many of them adaptive lambda scheduling added to its start, the lowest overlap of neighbouring windows in the production (from all iterations with equal numbers of samples per window, and among the samples the free energy is estimated from), and the legs that do not meet a level of the checks (first repeat). The checks are the two of the literature: NOT RELIABLE (overlap below 0.03, Klimovich et al. 2015) says that the free energy of a leg rests on too little overlap of neighbouring windows; few independent samples (fewer than 50 in a window, Mey et al. 2020) says that more sampling is advisable. A result that an earlier version of cuFEP analysed carries the flag that the checks of this version were not made; they are made when its legs are analysed again. The last column says where a NOT RELIABLE result is below the level: leg, pair of lambda windows, stage, lambda values and overlap.

| ligand | windows complex (added) | windows solvent (added) | lowest overlap | of the samples used | flags | overlap below the level |
|---|---|---|---|---|---|---|
| jmc_23 | 32 (2) | 19 (6) | 0.172 | 0.171 |  |  |
| ejm_46 | 33 (3) | 17 (4) | 0.153 | 0.154 | few independent samples: complex leg |  |
| jmc_27 | 33 (3) | 17 (4) | 0.140 | 0.136 | few independent samples: complex leg, solvent leg |  |
| jmc_28 | 33 (3) | 19 (6) | 0.144 | 0.143 |  |  |
| jmc_30 | 35 (5) | 17 (4) | 0.159 | 0.162 | few independent samples: complex leg, solvent leg |  |
| ejm_54 | 34 (4) | 21 (8) | 0.143 | 0.145 | few independent samples: complex leg |  |
| ejm_42 | 33 (3) | 16 (3) | 0.155 | 0.154 |  |  |
| ejm_47 | 35 (5) | 17 (4) | 0.146 | 0.148 |  |  |
| ejm_45 | 33 (3) | 19 (6) | 0.155 | 0.155 | few independent samples: complex leg |  |
| ejm_31 | 33 (3) | 18 (5) | 0.154 | 0.154 | few independent samples: complex leg |  |
| ejm_55 | 31 (1) | 19 (6) | 0.153 | 0.153 | few independent samples: complex leg |  |
| ejm_48 | 34 (4) | 17 (4) | 0.157 | 0.158 | few independent samples: complex leg |  |
| ejm_50 | 33 (3) | 18 (5) | 0.162 | 0.164 | few independent samples: solvent leg |  |
| ejm_43 | 35 (5) | 18 (5) | 0.149 | 0.149 | few independent samples: complex leg, solvent leg |  |
| ejm_49 | 36 (6) | 18 (5) | 0.155 | 0.155 | few independent samples: complex leg |  |
| ejm_44 | 34 (4) | 17 (4) | 0.153 | 0.153 | few independent samples: complex leg |  |

Adaptive lambda scheduling added windows in 32 of 32 legs. 12 of 16 ligands have a flag.

## Agreement with experiment

Values in brackets are 95 % bootstrap intervals over ligands. All numbers are for the ligands of this report: every ligand with a binding free energy, NOT RELIABLE ones included. FEP+ values come from relative calculations that were shifted as a whole to the experimental mean of the published set; over a part of that set their mean signed error is not zero. Only the rows below the first three compare the two methods like for like.

| statistic | cuFEP | FEP+ |
|---|---|---|
| mean signed error (kcal/mol) | -1.20 [-1.83, -0.60] | 0.00 [-0.23, 0.23] |
| RMSE, raw (kcal/mol) | 1.73 [1.10, 2.30] | 0.47 [0.29, 0.65] |
| MUE, raw (kcal/mol) | 1.41 [0.94, 1.93] | 0.37 [0.24, 0.53] |
| RMSE after removing the mean error (kcal/mol) | 1.25 [0.75, 1.62] | 0.47 [0.28, 0.63] |
| pairwise ddG RMSE (kcal/mol) | 1.83 [1.10, 2.36] | 0.69 [0.41, 0.92] |
| Kendall tau | 0.47 [0.12, 0.75] | 0.81 [0.61, 0.96] |
| Spearman rho | 0.65 [0.19, 0.90] | 0.93 [0.75, 0.99] |
| Pearson r | 0.59 [0.15, 0.87] | 0.93 [0.85, 0.98] |
| Pearson R^2 | 0.35 [0.03, 0.75] | 0.86 [0.73, 0.96] |

## Agreement with experiment by the check of overlap

No ligand with a binding free energy is NOT RELIABLE.
