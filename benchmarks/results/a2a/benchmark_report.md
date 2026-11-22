# cuFEP benchmark report: a2a

Adenosine A2A receptor (PDB 5OLZ) in a POPC bilayer, Schrödinger public benchmark gpcrs/a2a_hip278

Ligands with a result: 17 of 17 run (17 in the set).

## Per ligand (kcal/mol)

Every ligand with a binding free energy, NOT RELIABLE ones included (marked). Standard error: of one calculation, from its own trajectories (- for a result of an earlier version that did not record it). SD over repeats: only where several independent calculations were run. The last two columns are the signed errors against experiment.

| ligand | experiment | cuFEP | standard error | SD over repeats | repeats | FEP+ | cuFEP - experiment | FEP+ - experiment |
|---|---|---|---|---|---|---|---|---|
| 4e | -12.07 | -13.50 | 0.34 |  | 1 | -11.01 | -1.43 | +1.06 |
| 4n | -11.55 | -19.17 | 0.33 |  | 1 | -13.93 | -7.62 | -2.38 |
| 4k | -11.54 | -20.32 | 0.52 |  | 1 | -12.00 | -8.78 | -0.46 |
| 4r | -11.46 | -16.17 | 0.26 |  | 1 | -12.31 | -4.70 | -0.85 |
| 4c | -11.46 | -18.75 | 0.88 |  | 1 | -12.09 | -7.29 | -0.63 |
| 4f | -11.45 | -14.96 | 0.29 |  | 1 | -11.72 | -3.51 | -0.27 |
| 4l | -11.38 | -14.37 | 0.52 |  | 1 | -11.32 | -2.99 | +0.06 |
| 4o | -11.36 | -11.05 | 0.51 |  | 1 | -9.44 | +0.31 | +1.92 |
| 4g | -11.06 | -16.30 | 0.45 |  | 1 | -11.83 | -5.23 | -0.77 |
| 4j | -10.89 | -15.12 | 0.28 |  | 1 | -11.54 | -4.23 | -0.65 |
| 4h | -10.65 | -13.74 | 0.30 |  | 1 | -11.13 | -3.08 | -0.48 |
| 4m | -10.55 | -17.17 | 0.42 |  | 1 | -11.71 | -6.62 | -1.16 |
| 4d | -10.51 | -11.56 | 0.39 |  | 1 | -9.98 | -1.06 | +0.53 |
| 4i | -10.31 | -16.22 | 0.41 |  | 1 | -10.97 | -5.91 | -0.66 |
| 4b | -9.96 | -14.13 | 0.47 |  | 1 | -10.06 | -4.17 | -0.10 |
| 4a | -9.41 | -8.32 | 0.46 |  | 1 | -7.15 | +1.09 | +2.26 |
| 4q | -8.89 | -14.11 | 0.27 |  | 1 | -8.96 | -5.22 | -0.07 |

## Lambda windows and checks of the analysis

Windows per leg, how many of them adaptive lambda scheduling added to its start, the lowest overlap of neighbouring windows in the production (from all iterations with equal numbers of samples per window, and among the samples the free energy is estimated from), and the legs that do not meet a level of the checks (first repeat). The checks are the two of the literature: NOT RELIABLE (overlap below 0.03, Klimovich et al. 2015) says that the free energy of a leg rests on too little overlap of neighbouring windows; few independent samples (fewer than 50 in a window, Mey et al. 2020) says that more sampling is advisable. A result that an earlier version of cuFEP analysed carries the flag that the checks of this version were not made; they are made when its legs are analysed again. The last column says where a NOT RELIABLE result is below the level: leg, pair of lambda windows, stage, lambda values and overlap.

| ligand | windows complex (added) | windows solvent (added) | lowest overlap | of the samples used | flags | overlap below the level |
|---|---|---|---|---|---|---|
| 4e | 30 (0) | 17 (4) | 0.169 | 0.143 | few independent samples: complex leg |  |
| 4n | 31 (1) | 17 (4) | 0.147 | 0.148 | few independent samples: complex leg |  |
| 4k | 31 (1) | 18 (5) | 0.148 | 0.147 | few independent samples: complex leg |  |
| 4r | 31 (1) | 17 (4) | 0.151 | 0.148 |  |  |
| 4c | 30 (0) | 17 (4) | 0.145 | 0.145 | few independent samples: complex leg |  |
| 4f | 32 (2) | 18 (5) | 0.160 | 0.161 | few independent samples: complex leg |  |
| 4l | 31 (1) | 20 (7) | 0.150 | 0.149 | few independent samples: complex leg |  |
| 4o | 31 (1) | 19 (6) | 0.158 | 0.158 | few independent samples: complex leg |  |
| 4g | 30 (0) | 18 (5) | 0.147 | 0.146 | few independent samples: complex leg |  |
| 4j | 31 (1) | 17 (4) | 0.150 | 0.151 | few independent samples: complex leg |  |
| 4h | 30 (0) | 16 (3) | 0.145 | 0.146 |  |  |
| 4m | 31 (1) | 19 (6) | 0.152 | 0.152 | few independent samples: complex leg |  |
| 4d | 31 (1) | 18 (5) | 0.145 | 0.147 | few independent samples: complex leg |  |
| 4i | 30 (0) | 17 (4) | 0.105 | 0.104 | few independent samples: complex leg |  |
| 4b | 30 (0) | 18 (5) | 0.145 | 0.145 |  |  |
| 4a | 31 (1) | 18 (5) | 0.142 | 0.145 | few independent samples: complex leg |  |
| 4q | 30 (0) | 18 (5) | 0.151 | 0.151 | few independent samples: complex leg |  |

Adaptive lambda scheduling added windows in 27 of 34 legs. 14 of 17 ligands have a flag.

## Agreement with experiment

Values in brackets are 95 % bootstrap intervals over ligands. All numbers are for the ligands of this report: every ligand with a binding free energy, NOT RELIABLE ones included. FEP+ values come from relative calculations that were shifted as a whole to the experimental mean of the published set; over a part of that set their mean signed error is not zero. Only the rows below the first three compare the two methods like for like.

| statistic | cuFEP | FEP+ |
|---|---|---|
| mean signed error (kcal/mol) | -4.14 [-5.36, -2.85] | -0.16 [-0.68, 0.36] |
| RMSE, raw (kcal/mol) | 4.94 [3.82, 5.89] | 1.09 [0.68, 1.43] |
| MUE, raw (kcal/mol) | 4.31 [3.20, 5.40] | 0.84 [0.53, 1.18] |
| RMSE after removing the mean error (kcal/mol) | 2.68 [1.76, 3.24] | 1.08 [0.58, 1.40] |
| pairwise ddG RMSE (kcal/mol) | 3.91 [2.56, 4.72] | 1.58 [0.85, 2.05] |
| Kendall tau | 0.34 [-0.03, 0.67] | 0.59 [0.28, 0.86] |
| Spearman rho | 0.43 [-0.07, 0.82] | 0.73 [0.35, 0.95] |
| Pearson r | 0.44 [-0.02, 0.77] | 0.71 [0.28, 0.90] |
| Pearson R^2 | 0.19 [0.00, 0.60] | 0.50 [0.08, 0.81] |

## Agreement with experiment by the check of overlap

No ligand with a binding free energy is NOT RELIABLE.
