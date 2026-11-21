# cuFEP benchmark report: tyk2

TYK2 kinase (PDB 4GIH), Schrödinger public benchmark jacs_set/tyk2

Ligands with a result: 16 of 16 run (16 in the set).

## Per ligand (kcal/mol)

Every ligand with a binding free energy, NOT RELIABLE ones included (marked). Standard error: of one calculation, from its own trajectories (- for a result of an earlier version that did not record it). SD over repeats: only where several independent calculations were run. The last two columns are the signed errors against experiment.

| ligand | experiment | cuFEP | standard error | SD over repeats | repeats | FEP+ | cuFEP - experiment | FEP+ - experiment |
|---|---|---|---|---|---|---|---|---|
| jmc_23 | -11.70 | -9.60 | 0.37 |  | 1 | -11.27 | +2.10 | +0.43 |
| ejm_46 | -11.31 | -11.26 | 0.48 |  | 1 | -10.82 | +0.05 | +0.49 |
| jmc_27 | -11.28 | -9.48 | 0.52 |  | 1 | -11.10 | +1.80 | +0.18 |
| jmc_28 | -10.98 | -9.76 | 0.33 |  | 1 | -10.34 | +1.22 | +0.64 |
| jmc_30 | -10.94 | -14.69 | 0.91 |  | 1 | -11.13 | -3.75 | -0.19 |
| ejm_54 | -10.53 | -12.47 | 0.57 |  | 1 | -10.38 | -1.94 | +0.15 |
| ejm_42 | -9.78 | -11.97 | 0.41 |  | 1 | -9.95 | -2.19 | -0.17 |
| ejm_47 | -9.70 | -9.41 | 0.31 |  | 1 | -9.94 | +0.29 | -0.24 |
| ejm_45 | -9.56 | -10.43 | 0.48 |  | 1 | -9.66 | -0.87 | -0.10 |
| ejm_31 | -9.54 | -9.84 | 0.33 |  | 1 | -9.91 | -0.30 | -0.37 |
| ejm_55 | -9.21 | -10.23 | 0.46 |  | 1 | -10.34 | -1.02 | -1.13 |
| ejm_48 | -9.00 | -10.72 | 0.36 |  | 1 | -8.86 | -1.72 | +0.14 |
| ejm_50 | -8.98 | -11.04 | 0.54 |  | 1 | -9.40 | -2.06 | -0.42 |
| ejm_43 | -8.26 | -8.21 | 0.41 |  | 1 | -8.30 | +0.05 | -0.04 |
| ejm_49 | -7.75 | -11.07 | 0.60 |  | 1 | -8.05 | -3.32 | -0.30 |
| ejm_44 | -7.42 | -5.73 | 0.36 |  | 1 | -6.48 | +1.69 | +0.94 |

## Lambda windows and checks of the analysis

Windows per leg, how many of them adaptive lambda scheduling added to its start, the lowest overlap of neighbouring windows in the production (from all iterations with equal numbers of samples per window, and among the samples the free energy is estimated from), and the legs that do not meet a level of the checks (first repeat). The checks are the two of the literature: NOT RELIABLE (overlap below 0.03, Klimovich et al. 2015) says that the free energy of a leg rests on too little overlap of neighbouring windows; few independent samples (fewer than 50 in a window, Mey et al. 2020) says that more sampling is advisable. A result that an earlier version of cuFEP analysed carries the flag that the checks of this version were not made; they are made when its legs are analysed again. The last column says where a NOT RELIABLE result is below the level: leg, pair of lambda windows, stage, lambda values and overlap.

| ligand | windows complex (added) | windows solvent (added) | lowest overlap | of the samples used | flags | overlap below the level |
|---|---|---|---|---|---|---|
| jmc_23 | 25 (10) | 17 (4) | 0.145 | 0.142 |  |  |
| ejm_46 | 22 (7) | 17 (4) | 0.158 | 0.161 | few independent samples: solvent leg |  |
| jmc_27 | 25 (10) | 17 (4) | 0.154 | 0.153 | few independent samples: solvent leg |  |
| jmc_28 | 24 (9) | 17 (4) | 0.140 | 0.148 | few independent samples: complex leg |  |
| jmc_30 | 30 (15) | 18 (5) | 0.164 | 0.165 | few independent samples: solvent leg |  |
| ejm_54 | 25 (10) | 18 (5) | 0.155 | 0.161 | few independent samples: complex leg |  |
| ejm_42 | 21 (6) | 20 (7) | 0.139 | 0.123 | few independent samples: solvent leg |  |
| ejm_47 | 27 (12) | 18 (5) | 0.168 | 0.166 |  |  |
| ejm_45 | 22 (7) | 18 (5) | 0.170 | 0.171 | few independent samples: complex leg |  |
| ejm_31 | 20 (5) | 19 (6) | 0.165 | 0.174 | few independent samples: complex leg |  |
| ejm_55 | 21 (6) | 19 (6) | 0.156 | 0.158 | few independent samples: solvent leg |  |
| ejm_48 | 24 (9) | 17 (4) | 0.168 | 0.165 |  |  |
| ejm_50 | 21 (6) | 21 (8) | 0.180 | 0.178 | few independent samples: complex leg |  |
| ejm_43 | 23 (8) | 18 (5) | 0.166 | 0.166 | few independent samples: solvent leg |  |
| ejm_49 | 21 (6) | 17 (4) | 0.139 | 0.147 | few independent samples: complex leg, solvent leg |  |
| ejm_44 | 21 (6) | 20 (7) | 0.162 | 0.162 |  |  |

Adaptive lambda scheduling added windows in 32 of 32 legs. 12 of 16 ligands have a flag.

## Agreement with experiment

Values in brackets are 95 % bootstrap intervals over ligands. All numbers are for the ligands of this report: every ligand with a binding free energy, NOT RELIABLE ones included. FEP+ values come from relative calculations that were shifted as a whole to the experimental mean of the published set; over a part of that set their mean signed error is not zero. Only the rows below the first three compare the two methods like for like.

| statistic | cuFEP | FEP+ |
|---|---|---|
| mean signed error (kcal/mol) | -0.62 [-1.44, 0.29] | 0.00 [-0.23, 0.23] |
| RMSE, raw (kcal/mol) | 1.85 [1.33, 2.34] | 0.47 [0.29, 0.65] |
| MUE, raw (kcal/mol) | 1.52 [1.03, 2.05] | 0.37 [0.24, 0.53] |
| RMSE after removing the mean error (kcal/mol) | 1.74 [1.22, 2.12] | 0.47 [0.28, 0.63] |
| pairwise ddG RMSE (kcal/mol) | 2.55 [1.78, 3.10] | 0.69 [0.41, 0.92] |
| Kendall tau | 0.12 [-0.36, 0.54] | 0.81 [0.61, 0.96] |
| Spearman rho | 0.21 [-0.40, 0.69] | 0.93 [0.75, 0.99] |
| Pearson r | 0.44 [-0.19, 0.77] | 0.93 [0.85, 0.98] |
| Pearson R^2 | 0.19 [0.00, 0.59] | 0.86 [0.73, 0.96] |

## Agreement with experiment by the check of overlap

No ligand with a binding free energy is NOT RELIABLE.
