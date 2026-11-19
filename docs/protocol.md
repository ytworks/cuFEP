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

## Steps of one leg

1. Pre-equilibration: minimise, 0.25 ns NVT and 0.5 ns NPT on stock OpenMM, then 5 ns NPT
   on the backend of the leg (solvent leg: 0.1, 0.2 and 0.5 ns). A bilayer system is held
   by position restraints during NVT and NPT and then runs unrestrained for
   `membrane.equilibration_ns`.
2. Complex leg only: choose the Boresch restraint from that trajectory. Ligand anchor:
   three bonded atoms of an aromatic ring near the centre of the molecule. Protein anchor:
   three bonded backbone atoms in stable secondary structure, RMSF below 0.1 nm, 0.5-1.5 nm
   from the ligand anchor. Candidates with a restrained angle near 0 or 180 degrees are
   rejected and the one with the smallest fluctuations is taken. The reference distance and
   angles are those of the final frame, from which the alchemical run starts, as in OpenFE.
   Force constants: 10 kcal/mol/A^2 and 80 kcal/mol/rad^2. If the ligand moves away from its starting pose in
   this trajectory (symmetry-corrected heavy-atom RMSD above `restraints.pose_rmsd_warning_nm`
   over the second half), the run goes on, as in OpenFE, and the result carries a warning: it
   is then the free energy of the pose the ligand moved to.
3. Lambda windows: adaptive lambda scheduling, or the windows given in the configuration.
4. Hamiltonian replica exchange, one replica per lambda window, 2.5 ps per iteration: 1 ns
   equilibration and 5 ns production per window.
5. MBAR on decorrelated samples.

One calculation is run by default (`sampling.repeats: 1`), as in FEP+ and Flare, and the
reported uncertainty is then the MBAR uncertainty. With more repeats, every repeat is a
completely independent run of steps 1-5 with a different seed, and the reported uncertainty
is the standard deviation over the repeats, as in OpenFE.

The lengths are fixed: nothing extends a run automatically. The analysis checks the overlap
of the lambda windows and the convergence of every leg and flags a leg that fails
("Checks of the analysis").

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
| after the production | Flare shows the overlap matrix (warning below 0.03) and a convergence analysis; the user runs a further instance of the link with more lambda windows or a longer simulation | the analysis checks overlap and convergence ("Checks of the analysis"); the user runs a flagged leg again with more windows or a longer production ("When a leg is flagged") |
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
- A precalculation of 500 ps per window after 500 ps of equilibration on ejm_31 read 0.105
  and 0.015 for the two pairs, as the production does (0.084 and 0.022 in one run), added a
  window at 0.82, and then read at least 0.10 everywhere.

So the rule is the paper's in both legs, and the complex leg gives each new window 500 ps of
equilibration and 500 ps of recorded MD (`lambda_schedule.adaptive.equilibration_ps` and
`window_ps`, one value per leg).

If a pair of windows still ends below the warning level in the production, the result says
`NOT RELIABLE` and names the pair, and the remedy is the one of Flare: more windows in that
stage (`lambda_schedule.adaptive.initial_windows`, or `lambda_schedule.windows` with
`mode: fixed`) and a new run of that leg ("When a leg is flagged"). For orientation, with a
precalculation of 50 ps per window in the complex leg:

| TYK2, complex leg, 50 ps per window | windows | lowest overlap in production |
|---|---|---|
| start 3 / 5 / 9, ejm_31, four runs | 15 | 0.022, 0.041, 0.055, 0.070 |
| start 3 / 5 / 9, five other ligands | 15 | 0.008-0.031 |
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

What was measured. On five TYK2 ligands, REST2 with the ligand as the hot region raised the
free energy of the complex leg relative to runs without REST2: by 1.0 kcal/mol at a scale
of 1.5 (in five of five ligands, in two independent sets of runs) and by 2.2 kcal/mol at 2.5.
The solvent leg did not change at 1.5. The binding free energies moved away from experiment
(mean signed error -2.1 against -1.1 kcal/mol). Both ends of a leg are physical states, so a
converged leg cannot depend on the scale; the shift says that the windows in between are not
in equilibrium with each other. The replicas of the complex leg make no round trip between
the ends, with or without REST2: with the few windows that adaptive lambda scheduling chooses,
neighbours exchange too rarely for what is sampled in the hot windows to reach the ends. FEP+
runs REST2 with 80 windows in the complex leg. As it stands, REST2 does not improve cuFEP's
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

After the production the analysis of each leg makes three checks. They change nothing in
what is run. A leg that fails is flagged in the result line, in `result.json` (`quality`),
in `summary.csv` and in the benchmark report. The checks follow Klimovich, Shirts and Mobley,
"Guidelines for the analysis of free energy calculations", J. Comput. Aided Mol. Des. 2015,
29, 397. Flare shows the same quantities after a run: the overlap matrix with a warning
below 0.03, a convergence analysis, and a comparison of TI and MBAR.

| check | rule | flag | what the guideline says |
|---|---|---|---|
| overlap | every pair of neighbouring lambda windows overlaps by at least `analysis.min_overlap` (0.03) in the production | `NOT RELIABLE` | "values as low as 0.03 ... tolerable"; "Anything below that number should serve as a caution sign" |
| convergence | the free energy from the first half of the production alone, and from the second half alone, lies within `analysis.convergence_tolerance_sigma` (2) MBAR standard errors of the free energy from all of it | `NOT CONVERGED` | "In a favorable case, both sets of data should converge rapidly to within uncertainty of the final value." It gives no factor; 2 is cuFEP's own choice |
| estimators | BAR between neighbouring windows and MBAR agree within the same tolerance | a warning | "we primarily look for disagreement between TI-based methods and BAR-based methods". cuFEP does not record dU/dlambda and has no TI estimate; BAR against MBAR is a weaker comparison |

With independent samples a half of the data differs from the whole by about one standard
error, so at a tolerance of 2 about one leg in twenty is flagged by chance.

What the checks say about TYK2. Read again with these checks, 88 legs of earlier runs with
5 ns of production per window (most with 19 and 21 windows) gave: overlap below 0.03 in 1
complex leg; `NOT CONVERGED` in 30 of 44 complex legs and in 16 of 44 solvent legs; BAR and
MBAR never apart. In the complex legs a half of the production typically differs from the
whole by 0.3 to 0.8 kcal/mol where the tolerance is about 0.3. Longer runs do not simply
cure this: one complex leg that was continued from 5 to 10 ns moved by 1.4 kcal/mol and its
uncertainty rose from 0.2 to 0.7 kcal/mol, because a slow change occurred in the second half.

`analysis.png` in each repeat directory shows, for both legs, the forward and reverse series
(the free energy from growing parts of the production, counted from its start and from its
end, at `analysis.convergence_points` points), the two halves, the band of the tolerance and
the overlap matrix. The number of round trips of the replicas and the exchange between
neighbouring windows are in `result.json`; no threshold is applied to them.

### When a leg is flagged

cuFEP does not run anything again by itself. The user runs the leg concerned again, as a
Flare user adds a further instance of a link "with longer simulation time/more Lambda
windows"; the other legs and ligands keep their results and the binding free energy is put
together again.

- `NOT CONVERGED`: a longer production. `cufep run CONFIG --ligand NAME --leg complex
  --production-ns 10` (or a larger `sampling.production_ns`) continues the replica-exchange
  run of that leg from its last iteration; the analysis then covers the whole production.
- `NOT RELIABLE`: more lambda windows in the stage that the message names
  (`lambda_schedule.adaptive.initial_windows`, a different `overlap_threshold`, or
  `lambda_schedule.windows` with `mode: fixed`), then `cufep run CONFIG --ligand NAME --leg
  complex --reschedule`. The leg starts anew from its pre-equilibration and restraint with
  windows determined again; the windows, samples and result it had are moved to
  `previous_<n>/` in the leg directory.

A run restricted with `--ligand` or `--leg` writes its timing to `timing_rerun.json` and
leaves `timing.json` of the whole run in place.

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
| time per lambda window | 5 ns | set by the preset | 1 ns equilibration + 10 ns | 1 ns equilibration + 5 ns |
| runs per calculation | 1, no repeat setting | 1, no repeat setting | 3 repeats | 1; `sampling.repeats` for more |
| reported uncertainty | Bennett error | predicted activity error; overlap warning below 0.03 | standard deviation over repeats | MBAR uncertainty; standard deviation over repeats if more than one |
| checks after the run | convergence plots and sliding window, exchange between replicas (FEP+ checklist) | overlap matrix with a warning below 0.03, convergence analysis, TI against MBAR | plots of the MBAR overlap matrix, the replica exchange matrix, the forward and reverse convergence and the replica states; no verdict | overlap below 0.03, halves of the production against the whole, BAR against MBAR; flags in the result |
| a calculation that is not good enough | the user: "Try increasing the length of the simulations (i.e., 10 ns)", "try increasing the number of lambda windows and/or increasing the sampling" | the user runs a further instance of the link "with longer simulation time/more Lambda windows" | left to the user | the user runs the flagged leg again, longer or with more lambda windows |
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

Sources: Schrödinger `fep_plus` and `fep_absolute_binding` command help and Python API
documentation, the "FEP+ troubleshooting" checklist of the FEP+ user manual, Ross et al., Commun. Chem. 2023, 6, 222; Cresset Flare release notes for
versions 10 to 12 and the pyflare scripts `feprun.py` and `abferun.py`, Midgley et al. 2025;
OpenFE 1.13 source and documentation (the plots are those of `multistate_analysis.py`); SOMD2 documentation (oversubscription factor); Mey et
al., "Best Practices for Alchemical Free Energy Calculations", LiveCoMS 2020. Flare's preset
values are not public.

## Limits of version 0.1

Neutral ligands only. No cofactors yet. No symmetry correction (the ligand's graph
symmetry number is reported); a small symmetric ligand that turns over in its pocket leaves
the unrestrained window without overlap, which the result line reports. Symmetric POPC-type
bilayers only. No grand canonical water sampling. REST2 covers the ligand by default; slow
rearrangements of the pocket, such as a side-chain rotamer that differs with and without the
ligand, are sampled faster only if their residues are put into the hot region
(`sampling.rest2.region: pocket` or `sampling.rest2.residues`).
