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

The lengths are fixed: nothing extends a run automatically. A leg whose neighbouring
lambda windows overlap by less than `analysis.min_overlap` (0.03) gets a warning.

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
2. Run `window_ps` (50 ps) of MD in every window that has not been run yet and record the
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
for restrain, 5 for discharge and 13 for vanish in the complex leg, and 9 for discharge and
13 for vanish in the solvent leg; since windows are never removed, this is also the smallest
schedule. The next section gives the reasons for these numbers.

Points the paper does not specify, and cuFEP's rule for each:

| point | cuFEP |
|---|---|
| an evenly spaced value that is not a candidate | rounded to the nearest candidate, a half to the even one (0.125 becomes 0.12, the value that appears in the paper) |
| a bad section whose windows are already one candidate apart | skipped; if no section can be refined the precalculation ends with a warning |
| upper limit of the number of windows | `max_windows`; reaching it ends the precalculation with a warning |
| a pair exactly at the threshold | deficient (as in the analysis scripts published with the paper) |
| data used for the overlap matrix | the windows currently in the schedule |
| energy samples per window | every `sample_interval_ps` (0.5 ps), 100 per window |
| start of a new window | last configuration of the nearest window that has been run |

The windows, the threshold and one record per iteration (windows, overlaps, the chosen
section, the new lambda values) are written to `schedule.yaml` in the leg directory. The
last configuration of each window is the start of its replica; replica exchange then runs
its own equilibration before the production.

### Initial windows and threshold: what was measured

The paper's values (9 initial windows, threshold 0.04, 50 ps per window) were set for
relative transformations of a ligand in solution, run without replica exchange. For absolute
binding they were checked on TYK2 with ligand ejm_31 and on T4 lysozyme L99A with toluene
(5 ns of production per window, replica exchange). "Exchange" below is the fraction of
iterations in which a replica moved between the two windows.

*What the precalculation cannot see.* In the vanish stage of the TYK2 complex leg, between
lambda 0.62 and 0.88, side chains of the pocket (VAL981, ILE960) change rotamer once the
ligand is mostly decoupled, within nanoseconds. Eight precalculations of 50 to 200 ps per
window gave overlaps between 0.04 and 0.18 for the pairs 0.62-0.75 and 0.75-0.88,
differently each time; the production gave 0.084 and 0.022 in one run and 0.116 and 0.055
in another, with exchange of 0.1-4 %. A precalculation of 1 ns per window read 0.105 and
0.015 and took as long as several more windows of production. So a short precalculation
does not reliably find this region, whatever the threshold.

*Why the threshold stays at 0.04.* An element of the overlap matrix is shared among all
windows that overlap. Windows that are close to each other therefore have superdiagonal
elements near 1 / (number of such windows), however close they are: eleven discharge
windows of toluene, which has almost no charge, read 0.065 each. Above that level a
threshold cannot be met and each added window lowers the overlaps further. With a
threshold of 0.20 the TYK2 complex leg grew to `max_windows`; with 0.15 it ended at 18
windows, all pairs above 0.13 with at least 12 % exchange, but which windows it added
depended on the noise of the precalculation, and with 0.10 one precalculation added nothing
in the region above. 0.04 is far below the level that windows reach when they are close,
so the rule cannot run away, and it still adds windows where a stage truly lacks them (for
toluene in T4 lysozyme it added four windows at the start of the restrain stage).

*Why the vanish stage starts with 13 windows and the solvent discharge stage with 9.*

| TYK2 ejm_31, complex leg | windows | lowest overlap in production | lowest exchange |
|---|---|---|---|
| start 3 / 5 / 9, threshold 0.04 (two runs) | 15 | 0.022 and 0.055 | 0.1 % and 0.6 % |
| start 3 / 5 / 13, threshold 0.04 (four runs) | 19 | 0.092-0.145 | 2.4-7 % |
| start 3 / 5 / 9, threshold 0.15 | 18 | 0.135 | 12 % |
| fixed windows (three runs) | 30 | 0.119-0.127 | 13-15 % |

With 13 initial windows the vanish stage stays above the warning level of 0.03
(`analysis.min_overlap`) by a factor of three or more without relying on the precalculation
to find the slow region, at two thirds of the windows of the fixed schedule.

In the solvent leg, with the start of 5 discharge and 9 vanish windows, the first pair of
the discharge stage (lambda 0 to 0.25) had production overlaps of 0.06-0.13 with exchange of
1-5 %, and the vanish stage 0.04-0.05; the precalculation read values just above the
threshold for both. The solvent leg is a small part of the cost, so it starts with 9
discharge and 13 vanish windows; three runs then had overlaps of at least 0.13 and exchange
of at least 5 % everywhere.

*What more windows do not cure.* The repeats of TYK2 ejm_31 still differ, by the same amount
with 13 to 34 windows per leg. In the solvent leg the free energy follows the orientation of
the ligand's acetamide group relative to its pyridine ring, which turns over only once in a
few nanoseconds: about 20.0 kcal/mol in runs where it stays as in the bound pose, about 20.9
where it has turned (13 runs, correlation 0.8). In the complex leg it follows the rotamers
of ILE960 and VAL981. These are limits of the sampling, not of the lambda windows; see
"Limits of version 0.1".

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
| replica exchange | FEP/REST (REST2), exchange every 1.2 ps | none in absolute FEP | Hamiltonian, 2.5 ps per iteration | Hamiltonian, 2.5 ps per iteration |
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
documentation, Ross et al., Commun. Chem. 2023, 6, 222; Cresset Flare release notes for
versions 10 to 12 and the pyflare scripts `feprun.py` and `abferun.py`, Midgley et al. 2025;
OpenFE 1.13 source and documentation; SOMD2 documentation (oversubscription factor); Mey et
al., "Best Practices for Alchemical Free Energy Calculations", LiveCoMS 2020. Flare's preset
values are not public.

## Limits of version 0.1

Neutral ligands only. No cofactors yet. No symmetry correction (the ligand's graph
symmetry number is reported). Symmetric POPC-type bilayers only. No enhanced sampling of the
binding site or of the ligand (REST2) and no grand canonical water sampling: slow
rearrangements, such as a pocket side-chain rotamer that differs with and without the ligand
or a ligand torsion that turns over once in a few nanoseconds in water, are sampled only as
far as replica exchange between lambda windows carries them. Where they matter, repeats of a
calculation differ by more than the reported MBAR uncertainty; `sampling.repeats` shows it.
