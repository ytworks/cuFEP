# cuFEP benchmark report: a2a

Adenosine A2A receptor (PDB 5OLZ) in a POPC bilayer, Schrödinger public benchmark gpcrs/a2a_hip278

Ligands with a result: 17 of 17 run (17 in the set).

## Per ligand (kcal/mol)

Every ligand with a binding free energy, NOT RELIABLE ones included (marked). Standard error: of one calculation, from its own trajectories (- for a result of an earlier version that did not record it). SD over repeats: only where several independent calculations were run. The last two columns are the signed errors against experiment.

| ligand | experiment | cuFEP | standard error | SD over repeats | repeats | FEP+ | cuFEP - experiment | FEP+ - experiment |
|---|---|---|---|---|---|---|---|---|
| 4e | -12.07 | -18.91 | 0.52 |  | 1 | -11.01 | -6.83 | +1.06 |
| 4n | -11.55 | -18.50 | 0.65 |  | 1 | -13.93 | -6.95 | -2.38 |
| 4k | -11.54 | -15.54 | 0.34 |  | 1 | -12.00 | -4.00 | -0.46 |
| 4r | -11.46 | -16.43 | 0.63 |  | 1 | -12.31 | -4.96 | -0.85 |
| 4c | -11.46 | -15.14 | 0.72 |  | 1 | -12.09 | -3.68 | -0.63 |
| 4f | -11.45 | -14.52 | 0.38 |  | 1 | -11.72 | -3.08 | -0.27 |
| 4l | -11.38 | -13.82 | 0.65 |  | 1 | -11.32 | -2.44 | +0.06 |
| 4o | -11.36 | -10.19 | 0.85 |  | 1 | -9.44 | +1.16 | +1.92 |
| 4g | -11.06 | -13.57 | 0.69 |  | 1 | -11.83 | -2.51 | -0.77 |
| 4j | -10.89 | -14.63 | 0.47 |  | 1 | -11.54 | -3.75 | -0.65 |
| 4h | -10.65 | -18.36 | 1.02 |  | 1 | -11.13 | -7.70 | -0.48 |
| 4m | -10.55 | -16.30 | 0.30 |  | 1 | -11.71 | -5.75 | -1.16 |
| 4d | -10.51 | -13.30 | 0.34 |  | 1 | -9.98 | -2.80 | +0.53 |
| 4i | -10.31 | -15.41 | 0.66 |  | 1 | -10.97 | -5.10 | -0.66 |
| 4b | -9.96 | -16.68 | 0.62 |  | 1 | -10.06 | -6.72 | -0.10 |
| 4a | -9.41 | -10.15 | 0.45 |  | 1 | -7.15 | -0.74 | +2.26 |
| 4q | -8.89 | -13.09 | 0.36 |  | 1 | -8.96 | -4.20 | -0.07 |

## Lambda windows and checks of the analysis

Windows per leg, how many of them adaptive lambda scheduling added to its start, the lowest overlap of neighbouring windows in the production (from all iterations with equal numbers of samples per window, and among the samples the free energy is estimated from), and the legs that do not meet a level of the checks (first repeat). The checks are the two of the literature: NOT RELIABLE (overlap below 0.03, Klimovich et al. 2015) says that the free energy of a leg rests on too little overlap of neighbouring windows; few independent samples (fewer than 50 in a window, Mey et al. 2020) says that more sampling is advisable. A result that an earlier version of cuFEP analysed carries the flag that the checks of this version were not made; they are made when its legs are analysed again. The last column says where a NOT RELIABLE result is below the level: leg, pair of lambda windows, stage, lambda values and overlap.

| ligand | windows complex (added) | windows solvent (added) | lowest overlap | of the samples used | flags | overlap below the level |
|---|---|---|---|---|---|---|
| 4e | 18 (3) | 17 (4) | 0.035 | 0.037 | few independent samples: complex leg |  |
| 4n | 20 (5) | 19 (6) | 0.155 | 0.155 | few independent samples: complex leg |  |
| 4k | 20 (5) | 18 (5) | 0.171 | 0.184 |  |  |
| 4r | 21 (6) | 18 (5) | 0.166 | 0.166 | few independent samples: complex leg |  |
| 4c | 18 (3) | 19 (6) | 0.137 | 0.162 | few independent samples: complex leg |  |
| 4f | 21 (6) | 20 (7) | 0.154 | 0.162 | few independent samples: complex leg |  |
| 4l | 17 (2) | 17 (4) | 0.140 | 0.128 | few independent samples: complex leg |  |
| 4o | 18 (3) | 16 (3) | 0.169 | 0.140 | few independent samples: complex leg |  |
| 4g | 19 (4) | 17 (4) | 0.175 | 0.166 | few independent samples: complex leg |  |
| 4j | 18 (3) | 17 (4) | 0.121 | 0.121 | few independent samples: complex leg |  |
| 4h | 17 (2) | 19 (6) | 0.102 | 0.108 | few independent samples: complex leg |  |
| 4m | 18 (3) | 17 (4) | 0.173 | 0.174 | few independent samples: complex leg |  |
| 4d | 18 (3) | 18 (5) | 0.182 | 0.170 | few independent samples: complex leg |  |
| 4i | 18 (3) | 17 (4) | 0.107 | 0.139 | few independent samples: complex leg |  |
| 4b | 18 (3) | 17 (4) | 0.118 | 0.143 | few independent samples: complex leg |  |
| 4a | 17 (2) | 18 (5) | 0.115 | 0.118 | few independent samples: complex leg |  |
| 4q | 18 (3) | 18 (5) | 0.164 | 0.167 | few independent samples: complex leg |  |

Adaptive lambda scheduling added windows in 34 of 34 legs. 16 of 17 ligands have a flag.

## Agreement with experiment

Values in brackets are 95 % bootstrap intervals over ligands. All numbers are for the ligands of this report: every ligand with a binding free energy, NOT RELIABLE ones included. FEP+ values come from relative calculations that were shifted as a whole to the experimental mean of the published set; over a part of that set their mean signed error is not zero. Only the rows below the first three compare the two methods like for like.

| statistic | cuFEP | FEP+ |
|---|---|---|
| mean signed error (kcal/mol) | -4.12 [-5.15, -3.01] | -0.16 [-0.68, 0.36] |
| RMSE, raw (kcal/mol) | 4.70 [3.73, 5.53] | 1.09 [0.68, 1.43] |
| MUE, raw (kcal/mol) | 4.26 [3.30, 5.19] | 0.84 [0.53, 1.18] |
| RMSE after removing the mean error (kcal/mol) | 2.26 [1.47, 2.85] | 1.08 [0.58, 1.40] |
| pairwise ddG RMSE (kcal/mol) | 3.30 [2.14, 4.15] | 1.58 [0.85, 2.05] |
| Kendall tau | 0.38 [-0.04, 0.74] | 0.59 [0.28, 0.86] |
| Spearman rho | 0.48 [-0.09, 0.85] | 0.73 [0.35, 0.95] |
| Pearson r | 0.39 [-0.19, 0.74] | 0.71 [0.28, 0.90] |
| Pearson R^2 | 0.15 [0.00, 0.55] | 0.50 [0.08, 0.81] |

## Agreement with experiment by the check of overlap

No ligand with a binding free energy is NOT RELIABLE.
