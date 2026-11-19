# The cuFEP protocol

cuFEP computes absolute binding free energies by double decoupling with Boresch restraints,
Hamiltonian replica exchange and MBAR. Its features, defaults and vocabulary follow
established FEP software; the table in "How cuFEP compares" says which one each choice
follows. Every number below is a default that can be changed in the configuration
([configuration.md](configuration.md)).

## Terms

| term | meaning | also called |
|---|---|---|
| lambda window | one alchemical intermediate state | lambda state |
| replica | one simulation in replica exchange; it moves between lambda windows | |
| leg | one side of the cycle: the complex leg (protein and ligand) and the solvent leg (ligand in water) | bound / free (Flare), phase (YANK) |
| stage | part of a leg in which one kind of interaction changes: restrain, discharge, vanish | names of Flare and A3FE |
| decoupling | switching off the ligand's interactions with its environment while keeping those within the ligand | |
| Boresch restraints | one distance, two angles and three dihedrals between three ligand atoms and three protein atoms | Flare also writes "constraint strengths" for their force constants |
| constraints | fixed bond lengths to hydrogen, which with hydrogen mass repartitioning allow a 4 fs time step | |
| iteration | one cycle of replica exchange: every replica runs for `time_per_iteration_ps`, energies are evaluated, lambda windows are exchanged | exchange attempt interval |
| repeat | a completely independent run of the whole calculation with another seed | replicate, trial; `protocol_repeats` in OpenFE |
| pre-equilibration | minimisation and plain MD of the fully interacting system before the alchemical run | |
| equilibration, production | per lambda window: the part that is discarded, and the part used for the estimate | |
| overlap matrix | probability of a sample of one lambda window being found in another; its superdiagonal (first off-diagonal) holds neighbouring windows | |
| adaptive lambda scheduling | a precalculation that chooses the lambda windows | ALS, adaptive lambda windows (Flare) |
| oversubscription factor | number of workers that share one GPU at the same time under NVIDIA MPS | SOMD2; `mps_factor` in Desmond |
| backend | the code that computes the forces: the cuPotential plugin or stock OpenMM. Not a force field | |

## Inputs

A protein PDB and a ligand SDF with the ligand already placed in the pocket (an SDF with
several molecules gives one calculation per molecule). cuFEP does not dock, move or
re-protonate the ligand.

What happens to the protein (all of it is reported in `auto_report.md`):

- hydrogens in the file are kept and define protonation and tautomer states; if there are
  none they are added at pH 7.4;
- missing heavy atoms are rebuilt; chain breaks split the chain and get ACE/NME caps;
  missing loops are never modelled;
- explicitly charged chain ends stay charged; hydrogen-capped ends get ACE/NME caps;
- water and monatomic ions are kept; waters overlapping the ligand are removed.

cuFEP stops, with the reason, on: unknown or modified residues, a ligand without explicit
hydrogens, a ligand with a net charge, a ligand outside the protein, a severe clash or a
covalent contact, and a metal ion within 3 A of the ligand.

## Systems

| | default |
|---|---|
| protein / water / lipids | AMBER ff14SB / TIP3P / Lipid21 |
| ligand | OpenFF Sage 2.2.1, AM1-BCC charges (AmberTools sqm) on the input conformer |
| nonbonded | PME, 0.9 nm cutoff, Ewald tolerance 5e-4 |
| dynamics | Langevin middle integrator, 4 fs, hydrogen mass 3 amu, bonds to hydrogen constrained |
| ensemble | 298.15 K, 1 bar (isotropic; semi-isotropic with zero surface tension for bilayers) |
| solvent | 0.15 M NaCl; complex in a rhombic dodecahedron with 1.0 nm padding |

A lipid bilayer is built only when the configuration says `membrane: {enabled: true}`;
the default is a system in water. cuFEP does not decide whether a protein is a membrane
protein. With the membrane on, the membrane plane is taken from the structure if at least
two of three methods agree on it (a transfer-energy model, a hydrophobic-slab search and
`memembed`); otherwise the run stops and asks for an oriented structure
(`membrane.orientation: input`). The complex is embedded in a POPC bilayer with
`Modeller.addMembrane`, the bilayer is cut to the smallest sufficient box, and cavities
inside the protein at the depth of the bilayer are filled with water, because the builder
adds bulk water only above and below the lipids.

## Thermodynamic cycle

```
complex leg:  restrain  ->  discharge  ->  vanish
solvent leg:                discharge  ->  vanish

dG_bind = dG(solvent leg) - dG(complex leg) - dG(release)
```

In the restrain stage the Boresch restraint is switched on, in the discharge stage the
ligand's charges are switched off, and in the vanish stage its Lennard-Jones interactions
are switched off with a soft-core potential (Beutler form, alpha 0.5). `dG(release)` is the
standard state correction: the free energy of releasing the restraint on the decoupled
ligand into the standard-state volume. It is computed by numerical integration of the
restraint's partition function; Boresch's analytical formula is reported next to it as a
check.

The ligand's own nonbonded interactions are kept at every lambda window (decoupling), so
the decoupled ligand is the same in both legs and no vacuum leg is needed.

### Interactions inside the ligand

To keep them, every pair of ligand atoms is an exception of the `NonbondedForce` with its
full charge product and Lennard-Jones parameters. An exception has no cutoff. In the original
System the pairs of ligand atoms that are four or more bonds apart are ordinary pairs, cut
off at `dynamics.cutoff_nm`; OpenFE (openmmtools) keeps that cutoff for them. The coupled end
of a cuFEP leg therefore differs from the original System by the Lennard-Jones energy of
ligand atoms that are further apart than the cutoff. Electrostatics are not affected beyond
the tolerance of the Ewald sum.

The difference is small and it is the same at both ends of both legs, so it enters a binding
free energy only through how differently the ligand is folded when bound and when free. In
the input poses of the benchmark ligands, with OpenFF 2.2.1 and a cutoff of 0.9 nm, the
energy in question is -0.016 to -0.043 kcal/mol for the 16 TYK2 ligands (longest distance
inside the ligand 1.3 to 1.5 nm) and at most -0.003 kcal/mol for the 17 A2A ligands (about
1.0 nm). `spikes/m12_intraligand_lj.py` gives the numbers per ligand.

## Steps of one leg

1. Pre-equilibration: minimise, 0.25 ns NVT and 0.5 ns NPT on stock OpenMM, then 5 ns NPT
   on the backend of the leg (solvent leg: 0.1, 0.2 and 0.5 ns). A bilayer system is held
   by position restraints during NVT and NPT and then runs unrestrained for
   `membrane.equilibration_ns`. At the end of that run cuFEP checks whether the bilayer has
   settled: between the last two quarters the mean area in the membrane plane must not change
   by more than `membrane.area_tolerance` (1 %) and the mean number of water molecules within
   0.6 nm of the ligand by more than `membrane.pocket_water_tolerance` (1.5). If it has not,
   the result carries a warning with the two changes; the run is not made longer. In the A2A
   example the area was still shrinking after 10 ns, by 1.5 % between the last two quarters.
2. Complex leg only: choose the Boresch restraint from that trajectory. Ligand anchor:
   three bonded atoms of an aromatic ring near the centre of the molecule. Protein anchor:
   three bonded backbone atoms in stable secondary structure, RMSF below 0.1 nm, 0.5-1.5 nm
   from the ligand anchor. Candidates with a restrained angle near 0 or 180 degrees are
   rejected and the one with the smallest fluctuations is taken. The reference distance and
   angles are those of the final frame, from which the alchemical run starts, as in OpenFE.
   Force constants: 10 kcal/mol/A^2 and 80 kcal/mol/rad^2. If the ligand has moved away from
   the pose that was given (symmetry-corrected heavy-atom RMSD against the prepared input
   coordinates, in the frame of the binding site, above `restraints.pose_rmsd_warning_nm` over
   the second half of the trajectory), the run goes on, as in OpenFE, and the result carries
   a warning: it is then the free energy of the pose the ligand moved to.
3. Lambda windows: adaptive lambda scheduling, or the windows given in the configuration.
4. Hamiltonian replica exchange, one replica per lambda window, 2.5 ps per iteration: 1 ns
   equilibration and 10 ns production per window.
5. MBAR on samples that are decorrelated lambda window by lambda window ("Checks of the
   analysis").

One calculation is run by default (`sampling.repeats: 1`), as in FEP+ and Flare, and the
reported uncertainty is then the MBAR standard error. With more repeats, every repeat is a
completely independent run of steps 1-5 with a different seed, and the reported uncertainty
is the standard deviation over the repeats, as in OpenFE.

The lengths are fixed: nothing extends a run automatically. The analysis checks every leg,
the overlap of its lambda windows first, and the result says what it found ("Checks of the
analysis").

## Lambda windows

### Fixed windows

`lambda_schedule.mode: fixed` uses the windows of the configuration: per leg and stage a
number of windows or the lambda values. The recommended values, which are the defaults, are
those of OpenFE's ABFE protocol: 30 windows for the complex leg and 14 for the solvent leg.

### Adaptive lambda scheduling

`lambda_schedule.mode: adaptive` is the default. It is the method of Midgley, Bariami,
Habgood and Mackey, "Adaptive Lambda Scheduling: A Method for Computational Efficiency in
Free Energy Perturbation Simulations", J. Chem. Inf. Model. 2025, 65, 512-516 (the adaptive
lambda windows of Cresset Flare). The aim is the smallest set of windows that still overlap
enough. A short precalculation decides the windows; its data are not used for the free
energy.

Definitions. Lambda can take `candidates` (101) equally spaced values. The windows of a
stage are ordered by lambda. A pair of neighbouring windows is *deficient* if its element
of the overlap matrix (the superdiagonal) is at or below `overlap_threshold` (0.04). A *bad
section* is a run of consecutive deficient pairs that cannot be extended; its length is its
number of pairs.

The rule:

1. Start with a few evenly spaced windows.
2. Run `window_ps` of MD (50 ps in the paper) in every window that has not been run yet and record the
   energy of every sample at all candidate lambda values.
3. Compute the overlap matrix of the current windows. If no pair is deficient, the schedule
   is final.
4. Take the longest bad section; among equally long ones the one that contains the lowest
   overlap.
5. Keep the two end windows of that section, add one window between them, and space all
   windows of the section evenly between the two ends. Go to 2. Only lambda values that
   have not been simulated are run; the data of the others are reused.

So windows are added one per iteration, only between windows that overlap too little, and
only in the longest such stretch at that moment. The spacing there shrinks step by step
until every pair is above the threshold. Windows are never added where the overlap is
sufficient, and never removed.

Example from the paper: a bad section with lambda 0.00, 0.06 and 0.12 becomes 0.00, 0.04,
0.08, 0.12; the data at 0.00 and 0.12 are kept and only 0.04 and 0.08 are run.

A longer example with threshold 0.04 (the overlaps are made up for illustration):

| iteration | windows | overlaps of neighbours | bad sections | action |
|---|---|---|---|---|
| 1 | 0.00 0.12 0.25 0.38 0.50 0.62 0.75 0.88 1.00 | 0.20 0.15 0.09 0.03 0.02 0.06 0.03 0.10 | 0.38-0.62 (2 pairs), 0.75-0.88 (1 pair) | the longer one: 0.38 0.50 0.62 becomes 0.38 0.46 0.54 0.62 |
| 2 | 0.00 0.12 0.25 0.38 0.46 0.54 0.62 0.75 0.88 1.00 | 0.20 0.15 0.09 0.08 0.05 0.07 0.06 0.03 0.10 | 0.75-0.88 | 0.75 0.88 becomes 0.75 0.82 0.88 |
| 3 | 0.00 0.12 0.25 0.38 0.46 0.54 0.62 0.75 0.82 0.88 1.00 | all above 0.04 | none | final: 11 windows |

How cuFEP applies it to absolute binding. The paper treats relative binding with a single
lambda coordinate and runs the precalculation for the ligand in solution only. In cuFEP
every stage of every leg is treated as one such coordinate, with its own candidates, overlap
matrix and bad sections; at most one window is added per stage and iteration. Flare's
absolute FEP likewise holds separate numbers of windows for restrain (bound), discharge
(bound and free) and vanish (bound and free). The start (`initial_windows`) is 3 windows
for restrain, 5 for discharge and 9 for vanish; since windows are never removed, this is also
the smallest schedule.

Points the paper does not specify, and cuFEP's rule for each:

| point | cuFEP |
|---|---|
| an evenly spaced value that is not a candidate | rounded to the nearest candidate, a half to the even one (0.125 becomes 0.12, the value that appears in the paper) |
| a bad section whose windows are already one candidate apart | skipped; if no section can be refined the precalculation ends with a warning |
| upper limit of the number of windows | `max_windows`; reaching it ends the precalculation with a warning |
| a pair exactly at the threshold | deficient (as in the analysis scripts published with the paper) |
| data used for the overlap matrix | the windows currently in the schedule |
| energy samples per window | every `sample_interval_ps` (0.5 ps): 100 per window in the solvent leg, 1000 in the complex leg |
| start of a new window | last configuration of the nearest window that has been run |

The windows, the threshold and one record per iteration (windows, overlaps, the chosen
section, the new lambda values) are written to `schedule.yaml` in the leg directory. The
last configuration of each window is the start of its replica; replica exchange then runs
its own equilibration before the production.

### Where cuFEP follows Cresset and where it differs

| point | Cresset (the paper, and Flare) | cuFEP |
|---|---|---|
| when the windows are chosen | in a precalculation before the production: "Once converged, all results from the ALS are discarded, and the converged bespoke λ schedule is used in a production run" | the same. No window is added during the equilibration or the production of the replica-exchange run |
| rule and values | threshold 0.04, 101 candidates, one window per iteration into the longest bad section | the same |
| MD per new window of the precalculation | 50 ps | solvent leg: 50 ps. Complex leg: 500 ps recorded after 500 ps of equilibration, because the pocket adapts to a partly decoupled ligand over hundreds of picoseconds ("What was measured on TYK2") |
| initial windows | 9, for the one lambda coordinate of a relative transformation | per stage: restrain 3, discharge 5, vanish 9 |
| where the precalculation runs | the ligand in solution; the schedule is then used for the bound and the free state. For absolute FEP, Flare states only that the schedule is adaptive "for each stage of the calculation" | each leg has its own precalculation, stage by stage (the restrain stage exists only in the complex leg) |
| replica exchange | none in the paper's runs | Hamiltonian replica exchange |
| after the production | Flare shows the overlap matrix (warning below 0.03) and a convergence analysis; the user runs a further instance of the link with more lambda windows or a longer simulation | the analysis checks the overlap, the number of independent samples and the two halves of the production ("Checks of the analysis"); the user runs a leg again with more windows or a longer production ("When a leg carries a message") |
| Fast / Normal / Accurate | an option of Flare; its values are not published | `overlap_threshold` is set as a number (the paper tested 0.03, 0.04 and 0.05) |

### What was measured on TYK2

The paper's values were set for relative transformations of a ligand in solution, run
without replica exchange. For absolute binding they were checked on TYK2 with ligand ejm_31
(5 ns of production per window, replica exchange).

*The solvent leg.* The overlaps of the precalculation hold in the production: a pair that
the precalculation read at 0.042 had 0.038 in the production. On five other TYK2 ligands
the precalculation (50 ps per window, the paper's value) added one window to the vanish stage
of every solvent leg, and the lowest overlap in the production was 0.033 to 0.051.

*The complex leg: why its precalculation is longer.* In the vanish stage, between lambda
0.62 and 0.88, side chains of the pocket (VAL981, ILE960) change rotamer once the ligand is
mostly decoupled, and the overlap of the windows falls while they do. That takes hundreds of
picoseconds, and a precalculation of 50 ps per window does not see it:

- Eight precalculations of 50 to 200 ps per window on ejm_31 read overlaps between 0.04 and
  0.18 for the pairs 0.62-0.75 and 0.75-0.88, differently each time.
- On five other ligands (one run each, 50 ps per window) the precalculation added no window
  to any complex leg. In all five the lowest pair of the production was 0.75-0.88: the
  precalculation had read 0.058 to 0.129 there, the production had 0.008 to 0.031, a replica
  moved between the two windows in at most 0.5 % of the iterations, and four of the five
  results were `NOT RELIABLE`.
- The A2A example in a bilayer behaved the same with 50 ps per window: 0.209 in the
  precalculation and 0.053 in the production for the pair 0.75-0.88, no window added.
- A precalculation of 500 ps per window after 500 ps of equilibration on ejm_31 read 0.105
  and 0.015 for the two pairs, as the production does (0.084 and 0.022 in one run), added a
  window at 0.82, and then read at least 0.10 everywhere.

So the rule is the paper's in both legs, and the complex leg gives each new window 500 ps of
equilibration and 500 ps of recorded MD (`lambda_schedule.adaptive.equilibration_ps` and
`window_ps`, one value per leg).

The five ligands again with this precalculation (one run each): four of the five complex legs
got a window, each in the stretch 0.62-0.88 of the vanish stage, where the first iteration
read 0.011 to 0.035, and their lowest overlap in the production was 0.038 to 0.105. In the
fifth leg the precalculation read 0.061 for the pair 0.75-0.88 and added nothing, and the
production had 0.024 there, so that result was `NOT RELIABLE`. The longer precalculation
finds the region in most legs, not in all.

If a pair of windows still ends below the warning level in the production, the result says
`NOT RELIABLE` and names the pair, and the remedy is the one of Flare: more windows in that
stage (`lambda_schedule.adaptive.initial_windows`, or `lambda_schedule.windows` with
`mode: fixed`) and a new run of that leg ("When a leg carries a message"). For orientation (50 ps
per window in the precalculation unless stated):

| TYK2, complex leg | windows | lowest overlap in production |
|---|---|---|
| start 3 / 5 / 9, ejm_31, four runs | 15 | 0.022, 0.041, 0.055, 0.070 |
| start 3 / 5 / 9, five other ligands | 15 | 0.008-0.031 |
| start 3 / 5 / 9 with the default precalculation of 500 + 500 ps, five ligands | 15-16 | 0.024 (no window added), 0.038-0.105 (one window added) |
| start 3 / 5 / 13, ejm_31, four runs | 19 | 0.092-0.145 |
| start 3 / 5 / 13, four other ligands | 19 | 0.067-0.139 |
| fixed windows (`lambda_schedule.windows`), ejm_31, three runs | 30 | 0.119-0.127 |

*Why the threshold stays at 0.04.* An element of the overlap matrix is shared among all
windows that overlap. Windows that are close to each other therefore have superdiagonal
elements near 1 / (number of such windows), however close they are: eleven discharge
windows of toluene, which has almost no charge, read 0.065 each. Above that level a
threshold cannot be met and each added window lowers the overlaps further: with a threshold
of 0.20 the TYK2 complex leg grew to `max_windows`.

*What windows do not cure.* Runs of TYK2 ejm_31 differ by the same amount with 13 to 34
windows per leg. In the solvent leg the free energy follows the orientation of the ligand's
acetamide group relative to its pyridine ring, which turns over only once in a few
nanoseconds: about 20.0 kcal/mol in runs where it stays as in the bound pose, about 20.9
where it has turned (13 runs, correlation 0.8). The complex leg (38.9 to 43.6 kcal/mol over
21 runs, standard deviation 1.3) is correlated with the chi1 rotamers of ILE960 and VAL981
(0.4 to 0.5), which explain part of its spread. These are limits of the sampling, not of the
lambda windows; see "Limits of version 0.1".

## REST2

REST2 is replica exchange with solute scaling (Wang, Friesner and Berne, J. Phys. Chem. B
2011, 115, 9431). FEP+ runs it between its lambda windows (FEP/REST) and SOMD2, the engine
of Flare, offers it as `rest2_scale`. It is for degrees of freedom that change too slowly to
be sampled within one window, such as a ligand torsion or a side-chain rotamer in the
pocket: in the windows in the middle of a leg a *hot region* is given a higher effective
temperature, and replica exchange carries what is sampled there to the physical windows at
the ends.

The Hamiltonian. With s = T / T_hot, the energy inside the hot region is scaled by s, its
interaction with the rest by sqrt(s), and the rest is unchanged:

    E = s E(hot, hot) + sqrt(s) E(hot, rest) + E(rest, rest)

cuFEP applies it as Sire / SOMD2 do:

| term | scaling |
|---|---|
| charge of a hot atom | sqrt(s) |
| Lennard-Jones epsilon of a hot atom | s (so a pair of two hot atoms scales by s, a pair with one hot atom by sqrt(s)) |
| 1-4 and other exception pairs | s if both atoms are hot, sqrt(s) if one is |
| proper torsion with all four atoms hot | s |
| bonds, angles, improper torsions, the Boresch restraint | not scaled |

The hot region is the ligand and, in the complex leg, the protein residues chosen with
`sampling.rest2.region` and `sampling.rest2.residues` (whole residues).

The scale along a leg. `sampling.rest2.scale` is T_hot / T in the middle of the leg, as
SOMD2's `rest2_scale`; the default is 1.5 (447 K at 298 K; its basis is in
configuration.md) and 1 switches REST2 off. A high scale is not harmless: REST2 lowers every
torsion barrier of the hot region, and at 2.5 the amide bonds of the ligand isomerised in the
hot windows. The leg is treated as one path from 0 to 1 on
which every stage takes an equal part, and a window at position x has

    T_hot / T = 1 + (scale - 1) (1 - 2 |x - 0.5|)

which is SOMD2's rule: 1 at the first and at the last window, so both end states are the
physical ones, and `scale` in the middle. Because the scale is a function of the position,
the candidates of adaptive lambda scheduling have it too: the precalculation runs and
compares windows with their REST2 scaling, and adds windows where the scaling lowers the
overlap.

What was measured. On five TYK2 ligands the complex leg with REST2 (the ligand as the hot
region) was compared with the mean of three runs without REST2. At a scale of 1.5 it came
out higher by 1.2, 1.1 and 0.7 kcal/mol on average in three sets of runs (in 5, 5 and 3 of
the 5 ligands), at a scale of 2.5 by 2.3 kcal/mol (4 of 5). The solvent leg did not change
(+0.1 kcal/mol at 1.5). Runs of one complex leg without REST2 differ among themselves with a
standard deviation of 1.8 kcal/mol (pooled over the five ligands, three runs each). Against
that spread the shift at 1.5, 1.0 kcal/mol over 15 runs with and 15 without REST2, is about
1.5 standard errors: an indication, not an established effect. The mean signed error of the
binding free energies against experiment was -1.8 to -2.3 kcal/mol in the three sets with
REST2 and -1.2 without, which is the same observation and carries the same weight.

Both ends of a leg are physical states, so the free energy of a converged leg cannot depend
on the scale. If the shift is real, it says that the windows in between are not in
equilibrium with each other. The replicas of the complex leg make no round trip between the
ends, with or without REST2: with the few windows that adaptive lambda scheduling chooses,
neighbours exchange too rarely for what is sampled in the hot windows to reach the ends. FEP+
runs REST2 with 80 windows in the complex leg. REST2 has not been shown to improve cuFEP's
results on TYK2.

Only the free energy of a whole leg is a physical quantity then. The scale is not 1 where one
stage ends and the next begins, so the free energies of the single stages in the results
(`stage_delta_g`) contain the free energy of scaling the hot region up and down again: the
restrain stage of a TYK2 complex leg reads about 59 kcal/mol with the default scale, the three
stages together the 38 kcal/mol of the leg.

How FEP+ differs: Desmond takes the temperatures of the windows from a ladder predicted for
an exchange probability of 0.3 between neighbours (the predictor of Patriksson and van der
Spoel on the atoms of the hot region), rising over the first half of the windows and falling
over the second; its authors give about 1000 K at the top for a hot region of about 20 heavy
atoms. Desmond scales a torsion by sqrt(s) for each of its two end atoms in the hot region.
cuFEP follows SOMD2 in both points, because a ladder that depends on the number of windows
does not go together with windows chosen adaptively.

In the System the scaling is carried by two global parameters, sqrt(s) - 1 and s - 1 (both
zero in the physical state): parameter offsets on the charges and epsilons of hot protein
atoms and on exception pairs in the `NonbondedForce`, a factor per hot atom in the soft-core
force, and a `CustomTorsionForce` for the scaled torsions. The ligand charges get sqrt(s)
through `lambda_electrostatics`. The energies of all states are still obtained from a few
evaluations: the main force group is a quadratic polynomial in the ligand charge factor and
sqrt(s). One difference between the backends: stock OpenMM does not update the long-range
dispersion correction when an epsilon offset changes, the cuPotential plugin does. The two
therefore differ by a constant per window in the scaled windows; the end states and the
free energy are not affected, and a leg is sampled by one backend.

## Checks of the analysis

### Free energy and uncertainty of a leg

MBAR assumes uncorrelated samples. The samples of every lambda window are therefore thinned
by the statistical inefficiency g of that window. The series is the energy difference
between the window and the next one (the previous one for the last window), over the
configurations the window holds from iteration to iteration; the window keeps every
ceil(g)-th iteration. This is the default of alchemlyb (`decorrelate_u_nk`, method `dE`,
uniform intervals) and the procedure of Klimovich, Shirts and Mobley 2015 (section 2.5) and
of Mey et al. 2020 (section 8.2) for simulations whose states have different correlation
times. The free energy of the leg and its standard error are those of MBAR on the samples
that are kept. A part of the production (a point of the forward or reverse series, a half, a
block) is estimated from those kept samples that lie in it.

Of replica-exchange runs Mey et al. say that "the replicas should undergo equivalent random
walks in alchemical space, and the replicas can be subsampled with the same g". cuFEP's
replicas do not undergo such walks: with the 15 to 19 windows that adaptive lambda scheduling
chooses for a complex leg, the replicas made no round trip between the ends in 51 of 53 legs
of 5 ns on TYK2, and one in the other two. One g for all windows, from the reduced potential
summed over the replicas (the series of openmmtools 0.26, which OpenFE uses), read about 2
iterations there, where the slowest window of a complex leg has 22 to 412 (median 74; an
iteration is 2.5 ps). cuFEP used that g at first. The standard errors of the binding free
energy were then about half of what they are with one g per window (median 0.20 against 0.37
kcal/mol in 15 runs on five TYK2 ligands).

The overlap matrix is computed with the same number of samples in every window, so that its
elements mean the same for every window.

### The checks

After the production the analysis of each leg makes four checks. They change nothing in what
is run. What they find is written to the result line, to `result.json` (`quality`), to
`summary.csv` and to the benchmark report. Every level is one that the literature gives:
Klimovich, Shirts and Mobley, "Guidelines for the analysis of free energy calculations", J.
Comput. Aided Mol. Des. 2015, 29, 397, and Mey et al., "Best Practices for Alchemical Free
Energy Calculations", LiveCoMS 2020. Flare shows the same quantities after a run:
the overlap matrix with a warning below 0.03, a convergence analysis, and a comparison of TI
and MBAR.

| check | level | message | source |
|---|---|---|---|
| overlap | every pair of neighbouring lambda windows overlaps by at least `analysis.min_overlap` (0.03) in the production | `NOT RELIABLE` | Klimovich et al., section 3.5: "values as low as 0.03 [seem] to be tolerable to yield a reliable free energy estimate (as long as the resulting error is sufficiently low)". "Anything below that number should serve as a caution sign ...; in this case, not only will the estimated variance increase but the free energy estimate itself will likely be substantially incorrect" |
| independent samples | the lambda window with the fewest has at least `analysis.min_effective_samples` (50); the number is the iterations of the production divided by g of the window | few independent samples | Mey et al., section 8.2: "it is recommended that the effective number of uncorrelated samples Neff >= ~50 if the BAR or MBAR estimators ... are used" |
| halves | the free energies from the first and from the second half of the production differ by at most `analysis.halves_tolerance_kT` (1 kT, 0.59 kcal/mol at 298 K) | the two halves of the production differ | Mey et al., figure 9: the difference between the forward and the reverse free energy "at the midpoint of the simulation time gives an indication of the overall convergence of the simulation, with differences over 1 kBT indicating poor convergence". The figure applies it to each step of a relative perturbation, cuFEP to a leg |
| estimators | BAR between neighbouring windows and MBAR differ by at most `analysis.estimators_tolerance_kcal_mol` (1 kcal/mol) | a warning | Mey et al.: "differences between methods that are larger than 1 kcal/mol ... would indicate poor convergence". The guides compare TI with BAR-type estimators; cuFEP records no dU/dlambda, and BAR against MBAR is a weaker comparison |

The overlap comes first. The overlap matrix is "a quantitative estimator of the phase-space
overlap" (Klimovich et al., section 3.4), "trustworthy results should in general have at
least a tridiagonal overlap matrix" (section 3.5), and the other checks presuppose it:
without overlap the estimate and its standard error mean nothing. `NOT RELIABLE` is
therefore the only message that says a result cannot be used.

The other messages say that more sampling is advisable, with the number and the level. They
do not say that a result is wrong:

- A free energy that is converged still fluctuates: estimates from parts of the production
  differ by their statistical error. The message about the halves gives the difference in kT
  and in standard errors of the difference. A difference above 1 kT that is no more than
  about twice its standard error is what that number of samples allows; it says that the leg
  is not yet determined to 1 kT, not that it drifts.
- With few independent samples the comparison of the halves is itself uncertain; the message
  about the samples then matters more than the one about the halves.
- The two halves are the first and the second half of the production in time and share no
  iteration. A part of the production gets an estimate only if every lambda window holds at
  least four kept samples in it. If a half does not, the result says that the halves could
  not be compared and why.

What no check of one run can show. "It is difficult to establish with certainty that good
sampling has been achieved, but it is not difficult to rule out high-quality sampling"
(Grossfield et al. 2018): the checks can rule convergence out, they cannot prove it. A slow change that occurs once in
a run, or not at all, leaves no trace in the statistical inefficiency. "The variance in free
energy afforded by any estimator will always be an underestimate of the true uncertainty"
(Mey et al., section 8.4), and Rizzi et al. 2020 found the MBAR error from an analysis of
the autocorrelation at the low end of the spread between independent replica-exchange runs. The guides' way to measure the precision of a
protocol is "two or three runs of the same perturbation" (Mey et al.): `sampling.repeats`.

### What the checks say about TYK2 and A2A

86 legs of earlier TYK2 runs with 5 ns of production per window (most with 19 and 21
windows), read again with this analysis:

| | complex legs (43) | solvent legs (43) |
|---|---|---|
| overlap below 0.03 | 1 | 0 |
| statistical inefficiency of the slowest window, in iterations of 2.5 ps | 22-412, median 74 | 1-281, median 25 |
| independent samples in that window | 5-89, median 27; fewer than 50 in 37 legs | 7-1392, median 80; fewer than 50 in 16 legs |
| standard error of the leg, kcal/mol | 0.16-0.58, median 0.33 | 0.11-0.44, median 0.13 |
| halves more than 1 kT apart | 27 legs; in 13 of them by more than 2 standard errors | 8 legs; in 6 of them by more than 2 standard errors |
| BAR and MBAR more than 1 kcal/mol apart | 0 | 0 |

The overlap holds in all legs but one. What limits these legs is the number of independent
samples: in 5 ns the slowest window of a complex leg holds about 27 where 50 are recommended,
and at that number the halves of most complex legs are more than 1 kT apart, about half of
them by no more than twice the standard error of the difference. In the solvent leg the
slowest window is the first one, the fully interacting ligand in water, in 38 of the 43 legs.

One run against several. The same complex leg of five TYK2 ligands, run three times with
different seeds, differs with a standard deviation of 0.5 to 3.0 kcal/mol (pooled 1.8), the
binding free energy with 0.3 to 3.0 (pooled 1.6); the standard error of a binding free energy
from one run is 0.3 to 0.5 kcal/mol. The standard error of one run does not contain the slow
changes described under "What windows do not cure".

Longer runs, and more windows:

| leg | production | free energy, kcal/mol | fewest independent samples | halves apart |
|---|---|---|---|---|
| TYK2 ejm_31, complex, 16 windows | 5 ns | 36.77 ± 0.26 | 41 | 3.0 kT (3.4 standard errors) |
| the same leg continued | 10 ns | 36.25 ± 0.22 | 30 | 0.3 kT (0.4) |
| TYK2 jmc_23, complex, 15 windows | 5 ns | 37.34 ± 0.53 | 38 | 0.1 kT (0.0) |
| the same leg continued | 10 ns | 37.12 ± 0.58 | 45 | 0.6 kT (0.3) |
| A2A 4e, complex, 15 windows | 5 ns | 37.61 ± 0.69 | 9 | 1.3 kT (0.5) |
| the same leg continued | 10 ns | 36.09 ± 0.47 | 32 | 2.6 kT (1.7) |
| TYK2 ejm_31, complex, vanish stage started with 13 windows: 19 windows, lowest overlap 0.145 | 5 ns | 36.75 ± 0.26 | 38 | 3.2 kT (3.6) |

The halves of the ejm_31 leg, 3.0 kT apart after 5 ns, agree after 10 ns: what looked like a
drift was a fluctuation over a few nanoseconds. The number of independent samples can fall
when a run is continued, because a longer run shows slower changes (in that leg the window
at which the discharge stage begins, with a statistical inefficiency of 132 iterations). More lambda windows
raise the overlap; they do not add independent samples. In the A2A leg a window near the end
of the vanish stage is correlated over 126 to 217 iterations (0.3 to 0.5 ns), so 5 ns give 9
independent samples and 10 ns give 32; the bilayer of that system had also not settled when
the alchemical run began (`membrane_settled`).

Why the default production is 10 ns per window. It is the default of OpenFE's ABFE protocol;
FEP+ uses 5 ns. Of 55 TYK2 complex legs with 5 ns, 12 had 50 independent samples in their
slowest window; the statistical inefficiencies of that window (quartiles 44, 68 and 116
iterations) put 33 of them there with 10 ns, if they stay as they are, and 46 with 20 ns. Of
54 solvent legs, 33 had 50 with 5 ns. A longer run can also show a slower change, as the
ejm_31 leg above did, so 10 ns do not guarantee the 50.

`analysis.png` in each repeat directory shows, for both legs, the forward and reverse series
(the free energy from growing parts of the production, counted from its start and from its
end, at up to `analysis.convergence_points` points, each drawn at the fraction of the
production it really uses), the two halves, the estimates from `analysis.convergence_blocks`
consecutive blocks that do not overlap, the band of the standard error and the overlap
matrix. The number of round trips of the replicas, the exchange between neighbouring windows
and the statistical inefficiency and the independent samples of every window are in
`result.json`; no level is applied to round trips and exchange.

### When a leg carries a message

cuFEP does not run anything again by itself. The user runs the leg concerned again, as a
Flare user adds a further instance of a link "with longer simulation time/more Lambda
windows"; the other legs and ligands keep their results and the binding free energy is put
together again.

- `NOT RELIABLE`: more lambda windows in the stage that the message names
  (`lambda_schedule.adaptive.initial_windows`, a different `overlap_threshold`, or
  `lambda_schedule.windows` with `mode: fixed`), then `cufep run CONFIG --ligand NAME --leg
  complex --reschedule`. The leg starts anew from its pre-equilibration and restraint with
  windows determined again; the windows, samples and result it had are moved to
  `previous_<n>/` in the leg directory.
- Few independent samples, halves that differ, or halves that could not be compared: a
  longer production. `cufep run CONFIG --ligand NAME --leg complex --production-ns 20` (or a
  larger `sampling.production_ns`) continues the replica-exchange run of that leg from its
  last iteration; the analysis then covers the whole production. The number is the length of
  the production in total, not what is added: a leg that has 10 ns runs 10 ns more.

A run restricted with `--ligand` or `--leg` writes its timing to `timing_rerun.json` and
leaves `timing.json` of the whole run in place.

### Running again in the same output directory

A leg that runs again goes on from what it has. So that this cannot give a wrong number:

- What a leg gave before is set aside as `result.previous.json` when it runs again, for
  instance its result at 10 ns when it is continued to 20 ns. If the run fails, the leg is
  `FAILED`: it has no result, the ligand has no binding free energy from it, and the command
  ends with an error. The earlier result is a record and is never used in its place.
- A leg gets a result only if it has reached the length that was asked for, with every
  iteration stored once. A run that was interrupted before its first checkpoint starts again
  from iteration 0 with an empty storage.
- Samples are continued only under the conditions they were made under: the lambda states,
  temperature, pressure and barostat, friction, time step, length of an iteration, backend,
  precision, and the System with its restraint and REST2 region. If a run gives something
  else, it stops before anything is written and names what differs. A leg that has samples
  keeps their backend.
- The prepared systems, the pre-equilibration and the restraint are used again only if the
  input files and the settings that went into them are the same; otherwise the run stops and
  asks for a new `output.dir`. Only the lambda windows of a leg can be determined again in
  place, with `--reschedule`.
- A lambda schedule from a file (`lambda_schedule.file`) must be one of the same leg: its
  stages and its two end states are checked. Give a directory as written by `cufep benchmark
  freeze`, or the file of one leg together with `--leg`.

Output directories written before these checks existed do not hold the records to compare
with. They can still be continued: what they do hold (states, temperature, backend, length
of an iteration) is compared, and the log says which files could not be compared.

## Long-range dispersion

During sampling the soft-core ligand-environment interaction is truncated at the cutoff.
The missing energy is evaluated explicitly on coupled-state configurations out to 2 nm
(plus a homogeneous estimate beyond) and its free energy is added at the coupled end
state of each leg. An isotropic tail correction would assume the same atom density
around the ligand in the protein and in water.

## Use of GPUs

A job is one leg of one repeat of one ligand. Jobs are independent processes. Within a job
the replicas are independent during an iteration; they meet only when lambda windows are
exchanged.

- A worker is a process with one OpenMM Context on one GPU. It advances the replicas
  assigned to it one after another: it loads a replica's coordinates from shared memory,
  sets its lambda window, runs the MD of one iteration, evaluates the energy at all lambda
  windows and writes the result back.
- When all workers have finished, the job exchanges lambda windows between replicas and
  commits the iteration. Only energies and window indices pass between processes. If a
  worker dies, its input is still intact; it is restarted once and the iteration repeated.
- Several workers share one GPU at the same time through NVIDIA MPS; cuFEP starts its own
  MPS daemon. The oversubscription factor is the number of workers of a job per GPU; it is
  chosen from the size of the system (small systems leave most of a GPU idle). Without MPS
  it is 1.
- All GPUs form one pool. A job that is in pre-equilibration or in the restraint search
  holds one worker. The rest is divided among the jobs that are in the lambda precalculation
  (its new windows of an iteration run at the same time) or sampling, in proportion to the
  work they still have to do and never more than one worker per window or replica. When a job starts sampling or ends, the shares are recomputed, and jobs start
  or stop workers at their next iteration. With one ligand the solvent leg therefore uses
  the free GPUs while the complex leg is still in pre-equilibration, and the complex leg
  takes all of them afterwards.
- With more jobs than can run at once, the others wait; freed capacity goes to waiting
  jobs first. The same code runs on one GPU.

`timing.json` in the output directory lists the time per phase and job, and the GPU
utilization during the run.

## How cuFEP compares

Absolute binding protocols of three established programs, from their documentation,
source code and papers (see "Sources").

| | Schrödinger FEP+ (absolute binding) | Cresset Flare (absolute FEP) | OpenFE | cuFEP |
|---|---|---|---|---|
| lambda windows are given as | a number | a number per stage and side | lambda values | a number or lambda values, per leg and stage |
| default windows | complex 80, solvent 68 | presets Quick / Normal / Accurate; adaptive optional | complex 30, solvent 14 | adaptive; fixed mode: complex 30, solvent 14 |
| time per lambda window | 5 ns | set by the preset | 1 ns equilibration + 10 ns | 1 ns equilibration + 10 ns |
| runs per calculation | 1, no repeat setting | 1, no repeat setting | 3 repeats | 1; `sampling.repeats` for more |
| reported uncertainty | Bennett error | predicted activity error; overlap warning below 0.03 | standard deviation over repeats | MBAR standard error on samples decorrelated window by window; standard deviation over repeats if more than one |
| checks after the run | convergence plots and sliding window, exchange between replicas (FEP+ checklist) | overlap matrix with a warning below 0.03, convergence analysis, TI against MBAR | plots of the MBAR overlap matrix, the replica exchange matrix, the forward and reverse convergence and the replica states; no verdict | overlap below 0.03, independent samples per window, the two halves of the production, BAR against MBAR; messages in the result |
| a calculation that is not good enough | the user: "Try increasing the length of the simulations (i.e., 10 ns)", "try increasing the number of lambda windows and/or increasing the sampling" | the user runs a further instance of the link "with longer simulation time/more Lambda windows" | left to the user | the user runs the leg concerned again, longer or with more lambda windows |
| replica exchange | FEP/REST (REST2), exchange every 1.2 ps | none in absolute FEP (REST2 in relative FEP since V12) | Hamiltonian, 2.5 ps per iteration | Hamiltonian, 2.5 ps per iteration; REST2 with `sampling.rest2` |
| MD before the alchemical run | 1 ns and a short equilibration | minimisation, equilibration, MD for the restraint search | 0.25 + 0.5 + 5 ns | 0.25 + 0.5 + 5 ns |
| restraint | Boresch-type, atoms chosen from hydrogen bonds and salt bridges | Boresch, atoms and strengths chosen automatically | Boresch, atoms chosen automatically, fixed force constants | as OpenFE |
| water sampling | grand canonical Monte Carlo by default | grand canonical option | none | none |
| GPUs | replicas spread over GPUs, several per GPU under MPS | lambda windows on separate GPUs | one GPU per run | replicas spread over all GPUs, several per GPU under MPS |
| membrane | chosen by the user | chosen by the user | prepared by the user | switched on in the configuration; oriented, built and hydrated by cuFEP |

Further differences from OpenFE, whose protocol cuFEP follows most closely:

| | OpenFE | cuFEP |
|---|---|---|
| compute backend | stock OpenMM | cuPotential plugin for the nonbonded force and integrator; stock OpenMM as fallback |
| standard state correction | analytical | numerical, analytical as a check |
| ligand-environment dispersion tail | isotropic | explicit, by reweighting the coupled state |
| Lennard-Jones between atoms of the ligand | cut off as in the original System | no cutoff (hundredths of a kcal/mol; "Interactions inside the ligand") |
| decorrelation of the samples | one statistical inefficiency for all windows, from the reduced potential summed over the replicas (openmmtools) | one per lambda window, from its energy difference to the next window (alchemlyb's default) |

Sources: Schrödinger `fep_plus` and `fep_absolute_binding` command help and Python API
documentation, the "FEP+ troubleshooting" checklist of the FEP+ user manual, Ross et al., Commun. Chem. 2023, 6, 222; Cresset Flare release notes for
versions 10 to 12 and the pyflare scripts `feprun.py` and `abferun.py`, Midgley et al. 2025;
OpenFE 1.13 source and documentation (the plots are those of `multistate_analysis.py`); SOMD2 documentation (oversubscription factor); Mey et
al., "Best Practices for Alchemical Free Energy Calculations", LiveCoMS 2020; Grossfield et
al., "Best Practices for Quantification of Uncertainty and Sampling Quality in Molecular
Simulations", LiveCoMS 2018; Rizzi et al., "The SAMPL6 SAMPLing challenge: assessing the
reliability and efficiency of binding free energy calculations", J. Comput. Aided Mol. Des.
2020, 34, 601; openmmtools 0.26 source (`multistateanalyzer.py`); alchemlyb documentation
(`decorrelate_u_nk`). Flare's preset values are not public.

## Limits of version 0.1

Neutral ligands only. No cofactors yet. No symmetry correction (the ligand's graph
symmetry number is reported); a small symmetric ligand that turns over in its pocket leaves
the unrestrained window without overlap, which the result line reports. Symmetric POPC-type
bilayers only. No grand canonical water sampling. REST2 covers the ligand by default; slow
rearrangements of the pocket, such as a side-chain rotamer that differs with and without the
ligand, are sampled faster only if their residues are put into the hot region
(`sampling.rest2.region: pocket` or `sampling.rest2.residues`).
