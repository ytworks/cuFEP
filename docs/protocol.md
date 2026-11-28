# The cuFEP protocol

cuFEP computes absolute binding free energies by double decoupling with Boresch restraints,
Hamiltonian replica exchange and MBAR. Its defaults and vocabulary follow the literature and
published open methods; [configuration.md](configuration.md) gives the source of each default,
and "How cuFEP compares" sets the protocol beside OpenFE's. Every number below is a default
that can be changed in the configuration.

## Terms

| term | meaning | also called |
|---|---|---|
| lambda window | one alchemical intermediate state | lambda state |
| replica | one simulation in replica exchange; it moves between lambda windows | |
| leg | one side of the cycle: the complex leg (protein and ligand) and the solvent leg (ligand in water) | bound / free, phase (YANK) |
| stage | part of a leg in which one kind of interaction changes: restrain, discharge, vanish | names of A3FE |
| decoupling | switching off the ligand's interactions with its environment while keeping those within the ligand | |
| annihilation | switching off interactions within the ligand too; `alchemy.annihilate_electrostatics` does it for the electrostatics | |
| Boresch restraints | one distance, two angles and three dihedrals between three ligand atoms and three protein atoms | |
| constraints | fixed bond lengths to hydrogen, which with hydrogen mass repartitioning allow a 4 fs time step | |
| iteration | one cycle of replica exchange: every replica runs for `time_per_iteration_ps`, energies are evaluated, lambda windows are exchanged | exchange attempt interval |
| repeat | a completely independent run of the whole calculation with another seed | replicate, trial; `protocol_repeats` in OpenFE |
| pre-equilibration | minimisation and plain MD of the fully interacting system before the alchemical run | |
| equilibration, production | per lambda window: the part that is discarded, and the part used for the estimate | |
| overlap matrix | probability of a sample of one lambda window being found in another; its superdiagonal (first off-diagonal) holds neighbouring windows | |
| adaptive lambda scheduling | a precalculation that chooses the lambda windows | ALS |
| oversubscription factor | number of workers that share one GPU at the same time under NVIDIA MPS | SOMD2 |
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
- water and monatomic ions are kept; waters overlapping the ligand are removed;
- one input gives one prepared protein: the small minimisations that place missing atoms,
  caps and hydrogens run on OpenMM's Reference platform with a fixed seed (before revision
  15 the caps of one input landed up to 1.9 A apart from one preparation to the next).

cuFEP stops, with the reason, on what it cannot treat: unknown or modified residues, a
ligand without explicit hydrogens, and a ligand with a net charge. Where the ligand sits is
not judged: earlier versions also stopped on a ligand outside the protein, a clash, a
covalent contact or a metal ion within 3 A of the ligand, and warned of a metal ion within
6 A, levels of cuFEP's own that are gone (owner, 2026-11-20).

## Systems

| | default |
|---|---|
| protein / water / lipids | AMBER ff14SB / TIP3P / Lipid21 |
| ligand | OpenFF Sage 2.2.1, AM1-BCC charges (AmberTools sqm) on the input conformer |
| nonbonded | PME, 0.9 nm cutoff, Ewald tolerance 5e-4 |
| dynamics | Langevin middle integrator, 4 fs, hydrogen mass 3 amu, bonds to hydrogen constrained |
| ensemble | 298.15 K, 1 bar (isotropic; semi-isotropic with zero surface tension for bilayers) |
| solvent | 0.15 M NaCl; complex in a rhombic dodecahedron with 1.0 nm padding |

Every height of a box (the distance between opposite faces) is at least 3 x cutoff for the
cuPotential plugin and 2 x cutoff for stock OpenMM (OpenMM's rule: the cutoff may be at most
half a height), with a margin of 8 % for the box relaxing under NPT; the padding is raised
where the solute is small. With the defaults (cutoff 0.9 nm, solvent padding 1.5 nm) this
changes no box of stock OpenMM. A prepared system, also one stored by an earlier version,
whose box is lower than OpenMM's rule allows is refused before any MD.

A lipid bilayer is built only when the configuration says `membrane: {enabled: true}`;
the default is a system in water. cuFEP does not decide whether a protein is a membrane
protein. With the membrane on, the membrane plane is taken from the structure if at least
two of three methods agree on it (a transfer-energy model, a hydrophobic-slab search and
`memembed`); otherwise the run stops and asks for an oriented structure
(`membrane.orientation: input`). The complex is embedded in a POPC bilayer with
`Modeller.addMembrane`, the bilayer is cut to the smallest sufficient box, and cavities
inside the protein at the depth of the bilayer are filled with water, because the builder
adds bulk water only above and below the lipids. After the cut, every water that comes
within 0.24 nm of another molecule through the new periodic boundary is removed (before
revision 15 only waters next to a lipid were, and two waters 0.09 A apart made the
pre-equilibration of the A2A example end in NaN). The numbers of the bilayer (lipids in
each leaflet, waters in the hydrophobic core away from the protein) are recorded in
`prep/prep.json` without a verdict; earlier versions stopped below 40 lipids, warned of
leaflets that differ by more than 10 % and of such waters, and refused a prepared system with
a water or an ion closer than 0.15 nm to another molecule, levels of cuFEP's own that are gone.
`Modeller.addMembrane` relaxes the lipids with a short MD whose random seed OpenMM does not
let the caller set, so two preparations of one membrane system are not identical.
The bilayer is built with 3-site waters whatever its water model is: `Modeller.addMembrane`
makes a System of its lipid patch, whose waters have 3 sites. Waters that came with the input
lose their extra sites for the build in every case. With a water model of more sites (OPC,
TIP4P-Ew) the build uses the force field with `amber14/tip3p.xml`, and at the end every water
gets the extra site of the model, at the place its virtual site has; the pre-equilibration
then runs with the model itself.

With `membrane.complex_leg: water` (cuFEP's own, off by default: an option to see what the
bilayer does to a result) the bilayer is built for the choice of the restraint alone. The
preparation then writes two complexes of the pose that was given: the complex in the bilayer
(`prep/complex_bilayer.*`) and the complex in water (`prep/complex.*`), built as a run without
a bilayer builds it. The complex in the bilayer is pre-equilibrated as every bilayer system is
(`restraint_equil.npz` in the directory of the leg), and the MD from which the restraint is
chosen (`restraints.selection: interactions`, the only selection this can be used with) starts
from there. The alchemical run of the complex leg is made on the complex in water, after the
pre-equilibration of a complex in water (`equil.npz`): nothing stands in for the bilayer
around the protein. The six anchors are the same atoms in both systems; the build of a
bilayer can remove a crystal water or an ion that comes before the ligand, so the two number
the ligand differently, and the anchors are named again as the complex in water numbers them.
The preparation stops, before it writes anything, unless protein and ligand are the same
atoms with the same bonds in both. Every candidate for the anchors is held to the rules of the
anchors at the configuration the alchemical run starts from, as in every other calculation.
With this setting the system with the bilayer can have a water model of its own
(`membrane.water`, cuFEP's own; by default it is `forcefield.water`): no free energy is
computed in the bilayer, so the complex of the alchemical run and the solvent leg stay in one
water model, `forcefield.water`, and the bilayer can stay in the one Lipid21 was validated with
(TIP3P). The restraint, its six atoms and its reference values, then comes from an MD in
another water than the alchemical run (with `restraints.force_constants: auto` its force
constants too). With the alchemical run in the bilayer another water
model than `forcefield.water` is refused when the configuration is read.
The result of the complex leg says in which system the MD of its restraint was made
(`restraint_choice`). Its `bilayer` numbers are those of the pre-equilibration in the bilayer,
and `ligand_rmsd_from_input_A` is measured over the MD in the bilayer, as it is measured over
the MD of the restraint in every other calculation. The solvent leg is prepared and run as
without the setting; on stock OpenMM `solvation.box_shape: auto` gives its box the shape of
the complex in water (a dodecahedron) where a bilayer in the alchemical run gives a cube, and
with the cuPotential plugin it is a cube in both cases.

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
the decoupled ligand is the same in both legs and no vacuum leg is needed. With
`alchemy.annihilate_electrostatics: true` the electrostatics inside the ligand are switched off
with its charges (annihilation) and only its Lennard-Jones interactions are kept; the decoupled
ligand is then again the same in both legs, a ligand without electrostatics of its own.

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

Annihilated electrostatics (`alchemy.annihilate_electrostatics: true`, off by default) follow
the rule of openmmtools with exact PME, which OpenFE's ABFE protocol runs: the charges of the
ligand atoms are scaled by lambda, so a pair of ligand atoms that the force field treats as an
ordinary pair interacts with lambda^2, and the charge product of a 1-4 pair is scaled by lambda.
In cuFEP's System the exceptions of the ligand pairs then carry no charge product, and a
`CustomBondForce` of the main force group holds the Coulomb energy of every pair with its factor
(configuration.md gives the rule as it was read in the source). With the electrostatics kept, the
decoupled ligand keeps its internal hydrogen bonds at every lambda window, in both legs; with
them annihilated it does not. A complex leg and a solvent leg are combined only if both were
run the same way.

## Steps of one leg

The steps that do not depend on what a leg transforms (pre-equilibration, backend, lambda windows and the start
configurations of their replicas, replica exchange, fallback to stock OpenMM, analysis of the path) are those of
the engine of a leg (`src/cufep/engine/`), which `cufep rbfe` shares (`docs/rbfe.md`); the restraint (step 2)
and the correction for releasing it are the ABFE's.

1. Pre-equilibration: minimise, 0.25 ns NVT and 0.5 ns NPT on stock OpenMM, then 5 ns NPT
   on the backend of the leg (solvent leg: 0.1, 0.2 and 0.5 ns), with a frame every 20 ps
   and one at the end. A complex leg whose restraint is chosen from interactions
   (`restraints.selection: interactions`, the default) keeps the pose that was given
   instead: the heavy atoms of protein and ligand are held by position restraints through
   NVT and NPT while the solvent and the box settle, and no free MD follows, because the
   trajectories of step 2 are that MD and start from the pose that was given, as in the
   method they come from. A bilayer that was just built still runs
   `membrane.equilibration_ns` to close around the protein, with the solute held (the
   owner's decision of 2026-11-21: after 10 ns of free MD two runs of the A2A example had
   the ligand in two states and chose two restraints). The alchemical run starts where the
   pre-equilibration ends, in every case. If that backend cannot run the system, or the run fails on it, stock
   OpenMM runs it from the same start and the log and the result say so; with
   `backend.fallback: none` it does not, and the leg stops with the reason, as its alchemical
   run does. This MD and the MD from which a restraint is chosen run in the precision of
   the alchemical run (`backend.precision`), single too; where double is asked, the plugin,
   which does not have it, runs in mixed precision. Versions before 2026-11-21 ran them in
   mixed precision where single was asked, and a pre-equilibration they made with
   `backend.precision: single` is not used again. The whole length asked for is integrated, also where it is not a
   whole number of frame intervals, and the length integrated is recorded; versions before
   revision 15 integrated whole intervals only (40 ps of 50 ps), and such a file holding less
   than it was made for is not used again. A bilayer system is held
   by position restraints during NVT and NPT, the NPT twice as long as for a soluble system
   (1.0 ns), and then runs for `membrane.equilibration_ns`: unrestrained with
   `restraints.selection: auto` and `manual`, with the heavy atoms of protein and ligand still
   held with `interactions` (above). How the bilayer changed at the end of that run is reported with
   the result of the complex leg, as numbers without a verdict (`bilayer`): the relative change
   of the mean area in the membrane plane and the change of the mean number of water molecules
   within 0.6 nm of the ligand between the last two quarters. The run is never made longer for
   it. In the A2A example the area was still shrinking after 10 ns, by 1.5 % between the last
   two quarters.
2. Complex leg only: choose the Boresch restraint (`restraints.selection`). By default
   (`interactions`) the anchors come from the interactions between protein and ligand in
   unrestrained MD of the complex, as described further down. With `auto` they come from
   the trajectory of the pre-equilibration, as in OpenFE's protocol. Ligand anchor:
   three bonded atoms of an aromatic ring near the centre of the molecule. Protein anchor:
   three bonded backbone atoms in stable secondary structure, RMSF below 0.1 nm, 0.5-1.5 nm
   from the ligand anchor. Candidates are held to two rules of OpenFE's `EvaluateBoreschAtoms`
   and to nothing of cuFEP's own: theta_A and theta_B must stay, in every frame, where their
   restraint energy is at least 10 kT from that at 0 and at 180 degrees
   (`check_angle_not_flat`, 22 degrees with the default force constant), and anchors whose
   H2-H1-H0-G0-G1-G2 have three in a row on a straight line (|cos| of the bond vectors above
   0.9, `is_collinear`) in every frame are refused. Of the candidates left, the one with the
   smallest fluctuations is taken. The reference distance and angles are those of the final
   frame, from which the alchemical run starts, as in OpenFE. Anchors given with
   `restraints.selection: manual` are held to the same two rules, also in the prepared pose
   before any job. Force constants: 10 kcal/mol/A^2 and 80 kcal/mol/rad^2. How far the ligand
   moved from the pose that was given (symmetry-corrected heavy-atom RMSD against the prepared
   input coordinates, in the frame of the binding site, over the second half of the
   trajectory) is reported with the result as a number without a level
   (`ligand_rmsd_from_input_A`), as OpenFE reports the ligand RMSD; the run goes on in any
   case, and the result is the free energy of the pose the ligand is in. The same is measured
   for manual anchors and for a restraint chosen from interactions, over the free MD the
   selection has: the second half of the frames of the pre-equilibration for `auto` and
   `manual`; for `interactions`, whose pre-equilibration holds the pose, the analysed frames
   of the four trajectories, of which at most 200 are taken, evenly spaced (800 analysed
   frames give 200; fewer than 200 are all taken), and all of those counted. The frames are superposed on the backbone atoms (N, CA, C)
   within `restraints.host_max_distance_nm` + 0.5 nm of the centre of the ligand's heavy
   atoms in the first frame, or on all C-alpha atoms if those are fewer than 12. The RMSD and the drift of the centre of the ligand in the coupled state
   during the production are numbers of the result too (`structure`).

   With `restraints.selection: interactions`, the default, the anchors come from the interactions
   between protein and ligand in unrestrained MD of the complex, the method of Ding et al.
   (arXiv:2603.22274 v3, sec. 2.1 and 2.4; `docs/configuration.md` quotes its rules and says
   which numbers are those of its published source and what is cuFEP's own):
   - Four NPT trajectories of 3 ns start from the end of the pre-equilibration, which for
     this selection is the pose that was given in settled surroundings (step 1), and differ by
     their random numbers only. Each is a plain MD of the leg: it runs by the settings of the
     YAML and on the backend of the leg, with the recoveries of step 1 (stock OpenMM, 2 fs),
     and the result names the backend that ran each. With several GPUs they run at the same
     time. A frame is kept every 10 ps; those of the first 1 ns of each are not analysed, which
     leaves 800 frames. A trajectory that is complete is kept and not run again.
   - ProLIF gives, for every frame, the hydrogen bonds, ionic interactions, halogen bonds,
     cation-pi and aromatic stacking interactions between the ligand and the residues within
     6 A. Each interaction is given to one heavy atom of the ligand (the interacting atom, or
     the first suitable one within three bonds) and is kept if it is there in at least 40% of
     the frames (the number of frames rounded down).
   - The host anchors are the backbone N, CA and C of the residue of a kept interaction, in
     each of their six orders; the guest anchors are that ligand atom and two bonded heavy
     atoms that are no chain ends. A set of anchors is a candidate if the angles L1-L2-L3 and
     P1-L1-L2 lie within 45 to 135 degrees in at least 40% of the frames, counted among the
     frames of the interaction. The candidates are ranked by how often P1-L1-L2 and L1-P1-P2
     lie in that range, by how far those angles are from 90 degrees, by the secondary structure
     of the residue (DSSP: helix, strand or bridge in more than half of the frames), by the
     kind of interaction and by whether the residue has two interactions.
   - If the interactions give no anchors, the residues whose backbone is the nearest to the
     ligand (within 1.0 nm) in at least 15% of the frames are taken, with the same condition
     on the angles at 15%. If that gives none either, the calculation stops before anything is
     sampled. A ligand that has no three bonded heavy atoms for guest anchors is refused before
     any MD.
   - The best candidate that passes the two rules of OpenFE above is taken, over the analysed
     frames and at the configuration the alchemical run starts from; a candidate that does not
     pass is passed over and the result says how many were. The reference value of each of the
     six coordinates is the centre of the fullest of 20 bins of its histogram over the frames
     in which both anchor angles lie within 45 to 135 degrees. The force constants are those
     of `restraints.force_constants`, as for every selection.
   - The numbers of the analysis are written to `interactions.json` and `interactions.csv` in
     the directory of the leg: how often every interaction is there (in all analysed frames and
     in each trajectory), the secondary structure of the residues, the candidates with what
     they were ranked by, and the six coordinates of the chosen anchors frame by frame. What
     the restraint rests on is a number of the result of the leg and of the ligand
     (`restraint_choice`), without a verdict.
3. Lambda windows: adaptive lambda scheduling, or the windows given in the configuration.
4. Hamiltonian replica exchange, one replica per lambda window, 2.5 ps per iteration: 1 ns
   equilibration and 10 ns production per window.
5. MBAR on samples that are decorrelated lambda window by lambda window ("Checks of the
   analysis").

One calculation is run by default (`sampling.repeats: 1`), and the
reported uncertainty is then the standard error of that calculation, from blocks of its
production ("Checks of the analysis"). With more repeats, every repeat is a
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
Free Energy Perturbation Simulations", J. Chem. Inf. Model. 2025, 65, 512-516. The aim is the
smallest set of windows that still overlap enough. A short precalculation decides the windows;
its data are not used for the free energy.

Definitions. Lambda can take `candidates` (101) equally spaced values. The windows of a
stage are ordered by lambda. A pair of neighbouring windows is *deficient* if its element
of the overlap matrix (the superdiagonal) is at or below `overlap_threshold` (the paper uses
0.04; the defaults, the owner's choices, are 0.18 in the solvent leg and 0.30 in the complex
leg: configuration.md). The matrix is that of all windows of the stage (`overlap_matrix` set to
`stage`, the paper's and the default of the solvent leg) or that of the two windows of the pair
alone (`pair`, the default of the complex leg; "Why the threshold cannot go much higher"). A
*bad section* is a run of consecutive deficient pairs that cannot be extended; its length is
its number of pairs.

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
matrix and bad sections; at most one window is added per stage and iteration. The start
(`initial_windows`) of the solvent leg is 5 evenly spaced windows for discharge and 9 for
vanish. The complex leg starts from the recommended windows of mode fixed (their defaults,
whatever `lambda_schedule.windows` is set to): 6 for restrain, 11 for discharge and 15 for
vanish, the last denser from lambda 0.6 on ("The starts of the complex leg that were
compared"). The number of windows of a stage never falls, so the start is also the
smallest schedule. A stage is given a number of evenly spaced windows or the lambda values of
its start. They must be candidates and need not be evenly spaced (cuFEP's own: the paper starts
from evenly spaced windows). The rule treats them as any start: where it adds a
window, step 5 spaces the windows of that bad section evenly, so given values inside a bad
section move, and those outside every bad section stay.

Points the paper does not specify, and cuFEP's rule for each:

| point | cuFEP |
|---|---|
| an evenly spaced value that is not a candidate | rounded to the nearest candidate, a half to the even one (0.125 becomes 0.12, the value that appears in the paper) |
| a bad section whose windows are already one candidate apart | skipped; if no section can be refined the precalculation ends with a warning |
| upper limit of the number of windows | `max_windows` (100 per leg by default); reaching it ends the precalculation with a warning |
| a pair exactly at the threshold | deficient (as in the analysis scripts published with the paper) |
| data used for the overlap matrix | the windows currently in the schedule (`overlap_matrix: stage`, the default of the solvent leg); with `overlap_matrix: pair`, the default of the complex leg, the two windows of each pair alone |
| energy samples per window | every `sample_interval_ps` (0.5 ps): 100 per window in the solvent leg, 1000 in the complex leg |
| start of the windows of the first iteration | `start_configurations: walk` (the default): from the end of the pre-equilibration a walk through the windows in order, `walk_ps` (5 ps) of MD in each, and every window then runs from the configuration the walk left it. `sequential`: each window from the last configuration of the window before it (configuration.md) |
| start of a window that a later iteration adds | last configuration of the nearest window that had been run before that iteration |
| windows run at the same time | those that have their start configurations, where the leg has workers for them; otherwise one after another. What a window starts from and how much MD it gets does not depend on it |

The windows, the start (`initial_windows` as numbers, `initial_lambda` as values), the
threshold, the matrix it was compared with and one record per iteration (windows, overlaps, the
chosen section, the new lambda values) are written to `schedule.yaml` in the leg directory. A
record always holds the elements of the overlap matrix of the windows of the stage
(`superdiagonal`), and with `overlap_matrix: pair` also the overlap of each pair alone
(`pair_overlap`), which the rule read. Both are of the samples of the precalculation and of the
windows of one stage. The analysis of the production checks another number: the element of the
matrix of all windows of the leg, from the samples of the production. The
last configuration of each window is the start of its replica; replica exchange then runs
its own equilibration before the production. How the windows were started is written with the
start configurations (`start_configs.npz`, which is written before `schedule.yaml`, so that a
leg with a schedule of its own has the start configurations made with it), to the log, and to
the result of the leg (`start_configurations`). With them are written the windows they are of,
whether this precalculation chose the windows or they were given, and the settings of the
System they were made with (`alchemy`, `sampling.rest2`): a run that gives other settings to a
leg that has its windows is refused before it writes anything (`--reschedule` starts the leg
anew), and start configurations that are not those of the windows a leg has are made again. A
leg of an earlier version has no such record: its result has no `start_configurations` (the
record of its schedule tells whether the precalculation had workers, not whether the start
configurations the leg has were made by it). A run that samples such windows for the first
time writes its own record with them, which says that their origin is not recorded, and the
result of the leg says so from then on; so does the result of a leg that found windows of
adaptive lambda scheduling without start configurations (an earlier version stopped between
the two) and walked through them.

### Where cuFEP follows the paper and where it differs

| point | the paper | cuFEP |
|---|---|---|
| when the windows are chosen | in a precalculation before the production: "Once converged, all results from the ALS are discarded, and the converged bespoke λ schedule is used in a production run" | the same. No window is added during the equilibration or the production of the replica-exchange run |
| rule and values | threshold 0.04 (0.03 and 0.05 also tested), 101 candidates, one window per iteration into the longest bad section | the same rule and 101 candidates; by default the threshold 0.18 in the solvent leg (the owner's choice of 2026-11-20) and 0.30 in the complex leg (2026-11-22; configuration.md) |
| the overlap compared with the threshold | the element of the overlap matrix of the current windows | the same in the solvent leg (`overlap_matrix: stage`); in the complex leg by default the overlap of the two windows of a pair alone (`overlap_matrix: pair`, cuFEP's own, the owner's decisions of 2026-11-21 and 2026-11-22) |
| settings of the two legs | one precalculation, for the ligand in solution | each leg has its own: the mode, the threshold, the overlap matrix, the candidates, the start, the limit and the MD per new window can differ between the legs (configuration.md) |
| MD per new window of the precalculation | 50 ps | solvent leg: 50 ps. Complex leg: 500 ps recorded after 500 ps of equilibration, because the pocket adapts to a partly decoupled ligand over hundreds of picoseconds ("What was measured on TYK2") |
| initial windows | 9 evenly spaced ones, for the one lambda coordinate of a relative transformation | solvent leg: discharge 5 and vanish 9, evenly spaced. Complex leg: the recommended windows of mode fixed, restrain 6, discharge 11 and vanish 15, the last denser from lambda 0.6 on (the owner's choice of 2026-11-22). A stage is given a number of evenly spaced windows or the lambda values of its start, not necessarily evenly spaced |
| where the precalculation runs | the ligand in solution; the schedule is then used for the bound and the free state | each leg has its own precalculation, stage by stage (the restrain stage exists only in the complex leg) |
| replica exchange | none in the paper's runs | Hamiltonian replica exchange |
| after the production | not part of the method | the analysis checks the overlap and the number of independent samples, the levels of the literature, and shows the forward and reverse series and the halves without a verdict ("Checks of the analysis"); the user runs a leg again with more windows or a longer production ("When a leg carries a message") |

### What was measured on TYK2

The paper's values were set for relative transformations of a ligand in solution, run
without replica exchange. For absolute binding they were checked on TYK2 with ligand ejm_31
(5 ns of production per window, replica exchange). The measurements of this section were made
with the paper's threshold of 0.04, the default until 2026-11-20, unless they say otherwise.
The default was 0.18 in both legs from then until 2026-11-22; since then it is 0.18 in the
solvent leg and 0.30 of the overlap of a pair alone in the complex leg (configuration.md, where
the runs at 0.10 to 0.20 are given; "The starts of the complex leg that were compared" below).

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

So the insertion rule is the paper's in both legs, and the complex leg gives each new window 500 ps of
equilibration and 500 ps of recorded MD (`lambda_schedule.adaptive.equilibration_ps` and
`window_ps`, one value per leg).

The five ligands again with this precalculation (one run each): four of the five complex legs
got a window, each in the stretch 0.62-0.88 of the vanish stage, where the first iteration
read 0.011 to 0.035, and their lowest overlap in the production was 0.038 to 0.105. In the
fifth leg the precalculation read 0.061 for the pair 0.75-0.88 and added nothing, and the
production had 0.024 there, so that result was `NOT RELIABLE`. The longer precalculation
finds the region in most legs, not in all.

If a pair of windows still ends below the warning level in the production, the result says
`NOT RELIABLE` and names the pair, and the remedy is more windows in that
stage (`lambda_schedule.adaptive.initial_windows`, or `lambda_schedule.windows` with
`mode: fixed`) and a new run of that leg ("When a leg carries a message"). For orientation (threshold
0.04, 50 ps per window in the precalculation unless stated):

| TYK2, complex leg | windows | lowest overlap in production |
|---|---|---|
| start 3 / 5 / 9, ejm_31, four runs | 15 | 0.022, 0.041, 0.055, 0.070 |
| start 3 / 5 / 9, five other ligands | 15 | 0.008-0.031 |
| start 3 / 5 / 9 with the default precalculation of 500 + 500 ps, five ligands | 15-16 | 0.024 (no window added), 0.038-0.105 (one window added) |
| start 3 / 5 / 13, ejm_31, four runs | 19 | 0.092-0.145 |
| start 3 / 5 / 13, four other ligands | 19 | 0.067-0.139 |
| fixed windows (`lambda_schedule.windows`), ejm_31, three runs | 30 | 0.119-0.127 |

*Why the threshold cannot go much higher.* An element of the overlap matrix is shared among
all windows that overlap. Windows that are close to each other therefore have superdiagonal
elements near 1 / (number of such windows), however close they are: eleven discharge
windows of toluene, which has almost no charge, read 0.065 each. Above that level a
threshold cannot be met and each added window lowers the overlaps further: with a threshold
of 0.20 the TYK2 complex leg grew to `max_windows`, 48 then (measured while 0.04 was the
default), and with 5 ns of production the TYK2 solvent leg grew to 32 windows, its limit then
(2026-11-20). The default of 0.18, of both legs then and of the solvent leg now, is the
owner's choice from those runs.

For the same reason a start with many windows reads as deficient where its windows are closer
than they need to be, and the rule then adds windows there. `overlap_matrix: pair` (the
owner's decision of 2026-11-21, cuFEP's own) lets the rule read the overlap of a pair from the
overlap matrix of its two windows alone. That number lies between 0 and 0.5 and rises as the
two windows get closer, whatever other windows the stage has, so the rule can end with every
pair above a threshold above 0.2, also from a start with many windows. `max_windows` and the
candidates end it earlier where they are reached, each with its warning. For unit-width harmonic windows
the two numbers are, by the distance of neighbours in widths: 2: 0.21 and 0.22; 1: 0.22 and
0.40; 0.5: 0.13 and 0.47; 0.25: 0.07 and 0.49 (matrix of the stage, pair alone; measured in the
unit tests). The same threshold therefore asks for another spacing under the two matrices, and
a leg whose threshold is not given has the default of the matrix it reads: 0.18 for the stage,
0.30 for the pair. With 0.30 on the matrix of a stage nearly every pair of the default start of
the complex leg reads as deficient (26, 26 and 27 of 29 in the three runs of start A below, 6 by
the pair alone; only pairs at an end of a stage read above 0.30, at most 0.333), and harmonic
windows grow to `max_windows`. The
analysis of a result keeps the matrix of all windows and its level of 0.03.

That level can be missed by windows that are much closer than they need to be, by the sharing
alone. 81 unit-width harmonic windows 0.05 widths apart overlap by 0.50 as pairs, so the rule
ends at any threshold, and read 0.016 in the matrix of the leg: their result is
`NOT RELIABLE`, although the free energy is the true one within its error (41 such windows 0.1
widths apart read 0.031; measured 2026-11-21). The message of that verdict advises more windows
in the stage, which lowers the number further. Neither is changed (the level is the
literature's). The owner's decision of 2026-11-21: a result may fall below the level for this
reason, since what counts is the accuracy of the binding free energy; such a result keeps its
verdict and is reported and compared with it, not left out. The lowest overlap of a leg is part
of every result.

*The starts of the complex leg that were compared.* On the owner's order of 2026-11-21 three
starts of the complex leg were run three times each on the TYK2 example (ejm_31, 5 ns of
production per window, one calculation per run; the runs of a start use the same file and
seed), all with the overlap of a pair alone and the threshold 0.30, the solvent leg as it was
(plan `docs/plans/2026-11-21-als-initial-window-design.md`, tables by
`spikes/m19_window_runs.py`). A: the recommended windows of mode fixed, the values of OpenFE's
ABFE protocol. B: as many windows, with a density proportional to that of the windows the rule
chose in 37 stored complex legs (the 33 of the two full benchmark runs and the four runs of
ejm_31 that are the comparison here; `spikes/m18_window_design.py`) times the square root of
the statistical inefficiency there, which puts half of the vanish windows at lambda 0.7 to 0.9. C: twice the windows of B.
The four runs of the default until then (3, 5 and 9 evenly spaced windows, threshold 0.18 of
the matrix of the stage) are the comparison. Free energies in kcal/mol; the experimental
binding free energy is -9.54.

| start of the complex leg | windows of the complex leg | binding free energy of each run ± its standard error | mean | standard deviation | complex leg of each run | lowest overlap of the complex leg in production | fewest independent samples of a window of the complex leg | round trips in the complex leg |
|---|---|---|---|---|---|---|---|---|
| 3 / 5 / 9, four runs | 21-25 | -12.16 ± 0.32, -12.23 ± 0.64, -8.84 ± 0.91, -8.33 ± 0.74 | -10.39 | 2.09 | 42.95, 42.91, 40.19, 39.34 | 0.123-0.145 | 75, 18, 21, 16 | 0, 0, 0, 0 |
| A, 30 windows | 35, 33, 35 | -9.36 ± 0.56, -9.63 ± 0.37, -9.33 ± 0.51 | -9.44 | 0.17 | 39.97, 40.29, 40.35 | 0.141-0.153 | 106, 35, 40 | 1, 0, 0 |
| B, 30 windows | 35, 35, 35 | -10.01 ± 0.34, -9.64 ± 0.52, -11.71 ± 0.53 | -10.45 | 1.11 | 40.59, 40.71, 42.77 | 0.128-0.129 | 125, 17, 143 | 1, 0, 0 |
| C, 59 windows | 59, 59, 59 | -11.43 ± 0.46, -6.41 ± 0.59, -7.64 ± 0.43 | -8.50 | 2.62 | 41.91, 37.68, 38.72 | 0.081-0.085 | 29, 167, 200 | 0, 0, 0 |

In A and B the rule added 3 to 5 windows to the start; in C it added none, every pair of the
start being above the threshold. The solvent leg, which the starts do not change, gave 19.90 to
20.77 in the thirteen runs. The comparison does not show why the runs of a start differ: B and
C have as many and more windows than A. Three runs give a standard deviation only roughly (the
95 % interval of a standard deviation of three values reaches from about half of it to about
six times it), and this is one ligand of one target. The owner's choice (2026-11-22): A is the
default start of the complex leg, with the overlap of a pair alone and the threshold 0.30. Both
full benchmark sets ran once with it on 2026-11-22 (commit e77a6c5; `benchmarks/results/`): the
complex legs had 31 to 36 windows for TYK2 and 30 to 32 for A2A, where 7 of the 17 kept the 30
of the start, and no result is below the level of overlap of 0.03 (the lowest overlap of a
ligand is 0.140 to 0.172 for TYK2 and 0.105 to 0.169 for A2A).

*What windows do not cure.* Before that comparison, runs of TYK2 ejm_31 differed by the same
amount with 13 to 34 windows per leg. In the solvent leg the free energy follows the orientation of the ligand's
acetamide group relative to its pyridine ring, which turns over only once in a few
nanoseconds: about 20.0 kcal/mol in runs where it stays as in the bound pose, about 20.9
where it has turned (13 runs, correlation 0.8). The complex leg (38.9 to 43.6 kcal/mol over
21 runs, standard deviation 1.3) is correlated with the chi1 rotamers of ILE960 and VAL981
(0.4 to 0.5), which explain part of its spread. These are limits of the sampling, not of the
lambda windows; see "Limits of version 0.1".

## REST2

REST2 is replica exchange with solute scaling (Wang, Friesner and Berne, J. Phys. Chem. B
2011, 115, 9431). SOMD2 offers it as `rest2_scale`. It is for degrees of freedom that change too slowly to
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
`sampling.rest2.region` and `sampling.rest2.residues` (whole residues). By default
(`region: pocket`, the owner's choice of 2026-11-20) these are the residues with a heavy atom
within 0.4 nm of a heavy atom of the ligand in the prepared structure;
SOMD2 heats the ligand alone by default (`region: ligand`).

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
out higher by 1.1, 0.9 and 0.7 kcal/mol on average in three sets of runs (in 5, 5 and 3 of
the 5 ligands), at a scale of 2.5 by 2.3 kcal/mol (4 of 5). The solvent leg did not change
(+0.1 kcal/mol at 1.5). Runs of one complex leg without REST2 differ among themselves with a
standard deviation of 1.8 kcal/mol (pooled over the five ligands, three runs each). Against
that spread the shift at 1.5, 0.9 kcal/mol over 15 runs with and 15 without REST2, is about
1.5 standard errors: an indication, not an established effect. The mean signed error of the
binding free energies against experiment was -1.8 to -2.2 kcal/mol in the three sets with
REST2 and -1.2 without, which is the same observation and carries the same weight.

Both ends of a leg are physical states, so the free energy of a converged leg cannot depend
on the scale. If the shift is real, it says that the windows in between are not in
equilibrium with each other. The replicas of the complex leg make no round trip between the
ends, with or without REST2: with the few windows that adaptive lambda scheduling chooses,
neighbours exchange too rarely for what is sampled in the hot windows to reach the ends. REST2 has not been shown to improve cuFEP's
results on TYK2.

With the pocket in the hot region (the residues within 0.4 nm of the ligand), an ALS threshold
of 0.18 and 5 ns of production per window, one calculation of each example gave -8.94 ± 0.77
kcal/mol for TYK2 ejm_31 (experiment -9.54; with the ligand alone as the hot region and
otherwise the same settings -10.56) and -11.95 ± 0.73 for A2A 4e (experiment -12.07; -17.02
with the ligand alone). These are single runs: the complex legs of these examples differ from
run to run by more than their standard errors (the discharge stage of TYK2 from 15.3 to 18.4
kcal/mol over runs with thresholds of 0.10 to 0.18). The owner made the pocket the default on
2026-11-20. The full benchmark runs of 2026-11-20/21 (commit 3b45bfe, the same settings) gave
-9.84 ± 0.33 for ejm_31 and -18.91 ± 0.52 for 4e, whose complex leg was 44.68 kcal/mol against
37.61 in the earlier run (the record of those runs is in the tree of commit cda6965). The full
benchmark runs of 2026-11-22 (commit e77a6c5; the restraint from interactions and the complex
leg from 30 windows, the defaults since) gave -10.08 ± 0.37 for ejm_31 and -13.50 ± 0.34 for 4e,
whose complex leg was 39.76 (`benchmarks/results/`).

Only the free energy of a whole leg is a physical quantity then. The scale is not 1 where one
stage ends and the next begins, so the free energies of the single stages in the results
(`stage_delta_g`) contain the free energy of scaling the hot region up and down again: the
restrain stage of a TYK2 complex leg reads about 59 kcal/mol with the default scale, the three
stages together the 38 kcal/mol of the leg.

The temperatures of the windows follow from one scale per leg, as in SOMD2, and not from a
ladder of temperatures that depends on the number of windows: such a ladder does not go
together with windows chosen adaptively.

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

## Water relocation moves

Water exchanges slowly between a buried binding site and the bulk. At the decoupled end of the
A2A complex leg of 4k the number of water molecules in the place of the ligand kept rising for 20
to 30 ns of plain MD, far longer than a lambda window is sampled (plan
`docs/plans/2026-11-22-accuracy-causes-and-fixes.md`). With `sampling.relocation_mc.enabled: true`
the complex leg makes water relocation moves: RelocationMC of Ding et al. (arXiv:2603.22274 v3,
"RelocationMC water-sampling protocol" and its appendix), a Monte Carlo move that takes a whole
water molecule from the rest of the cell into a sphere around the ligand, or out of it. The number
of particles and the cell do not change, so no chemical potential is needed, and the replicas,
their exchange and MBAR are what they are without the moves. The default is off.

The sphere is centred on the geometric centre of the heavy atoms of the ligand and has the radius
of the paper: half the largest distance between two heavy atoms of the ligand in the prepared
pose, plus 0.3 nm (`radius_margin_nm`). One move, with x the configuration and beta = 1 / kT:

1. A direction is chosen: out of the sphere or into it. The source region S is where the water
   is taken, the target region T the other one.
2. Every water M of S has a score s_M, its interaction with everything else. The water D that is
   moved is chosen with probability exp(beta s_D) / Z_S(x), Z_S(x) the sum of exp(beta s_M) over
   the waters of S: a water that is held weakly is preferred.
3. K anchors are drawn (`trial_poses`), each with an independent uniform orientation of the
   conformation of D, and sorted into the two regions afterwards. A trial pose i has the weight
   w_i = exp(-beta u_i) / h(i): u_i is its interaction with everything but D, h the density its
   anchor was drawn with. Among the poses of T one, C, is chosen with probability w_C / W_f, W_f
   the sum of the weights of the poses of T.
4. The reverse pool is the same K poses with the chosen one replaced by the pose D has; W_r is
   the sum of the weights of its poses of S. Z_T(y) is the sum of exp(beta s_M) over the waters
   of T in the proposed configuration y, the moved water among them.
5. The move is accepted with probability min(1, r),
   ln r = ln(g_r / g_f) + ln Z_S(x) - ln Z_T(y) + beta (s_C - s_D) + ln W_f - ln W_r
   - beta [dU - (u_C - u_D)], with dU = U(y) - U(x) the change of the potential energy of the
   simulation and g_f, g_r the probabilities of the two directions.
6. An accepted water keeps the velocity of its centre of mass, and its velocities relative to it
   are turned with the molecule. A move that is not accepted changes nothing.

Where cuFEP follows the paper and where it differs:

| point | the paper | cuFEP |
|---|---|---|
| the energies u of the trial poses and the scores s | computed by a CUDA kernel: the direct-space and reciprocal-space PME interaction of every pose, so that dU - (u_C - u_D) is a correction of the finite mesh only | OpenMM gives no energy per molecule. u is a stand-in: Lennard-Jones and reaction-field Coulomb (dielectric 78.5) within the cutoff of the run, the ligand with its charges times `lambda_electrostatics` and its soft-core at `lambda_sterics`, without REST2; s is the same within 0.5 nm. Both are computed by kernels on the CUDA device of the Context (CuPy, double precision, the atoms found through a grid of cells); on a platform without CUDA the same function is computed by NumPy on the processor, which is the reference of the tests and much slower. What the stand-in misses is in dU - (u_C - u_D), which uses the potential energy of the Context in the state the replica is simulated in (its lambda values and its REST2 scale). The ratio above is a Metropolis-Hastings ratio for any u and s, so the moves keep the distribution of that state exactly; a poor stand-in costs acceptance, not correctness |
| the anchors | uniform in the whole cell ("draws K independent candidate anchors uniformly from the complete periodic box"), so that h is constant, cancels, and no volume appears; K = 16384 | the same by default. With `inner_fraction` that share of the anchors is drawn uniformly in the sphere and the rest uniformly outside it; h then differs between the two regions and stays in the weights (the paper's w_i = exp(-beta u_i) / h_0(i) before h_0 is cancelled). In a cell much larger than the sphere few uniform anchors fall into it (0.3 % for 4k) |
| the point of a water that is put on an anchor, and that decides its region | not said | the oxygen, for both |
| the probabilities of the two directions | g_f and g_r, not given | 1/2 each |
| the cell | orthorhombic (a limit of the kernel) | any cell |
| when a move is attempted | every 1000 MD steps (4 ps), in the protein-ligand leg | in the complex leg, one for every `interval_ps` (4 ps) of the MD of a replica: after the MD of the iteration in which a multiple of the interval was passed (an iteration is 2.5 ps), in the equilibration and the production of the replica exchange, and in the windows of the precalculation of adaptive lambda scheduling |
| which waters | all relocatable waters have "identical, lambda-invariant nonbonded parameters" and no interaction within themselves; for a charged ligand the alchemical waters are left out | every water (residue HOH); the same two conditions are checked, and a system that does not meet them is refused. cuFEP has no charged ligands |
| a water with a virtual site (a 4-site model such as OPC) | the protocol of the paper uses TIP3P, three sites | all sites of the water are turned and shifted as one body, so the virtual site is where the System puts it for the new places of the three atoms; it has no mass, so it does not count in the centre of mass and gets no velocity |

What is recorded: the samples of the leg hold, per iteration and lambda window, the moves
attempted and accepted in each direction (`relocations`), and the result of the leg gives their
sums over the production by window (`relocation_mc`), as numbers without a verdict. The settings
of the moves are among the conditions of the samples of the complex leg: a leg sampled with them
is not continued without them or with other ones.

## Grand canonical moves of water

The water relocation moves keep the number of waters of the system: a water that enters the place
of the ligand is taken from the rest of the cell. With `sampling.gcmc.enabled: true` the complex
leg instead makes grand canonical Monte Carlo moves of water in a sphere around the ligand: water
molecules are inserted into the sphere and deleted from it, in equilibrium with a reservoir of
bulk water. The default is off. The method is that of grand (Samways, Bruce Macdonald and Essex,
J. Chem. Inf. Model. 2020, 60, 4436) with the theory of Ross et al. (J. Chem. Theory Comput. 2017,
13, 6373); the sentences it rests on are quoted in `docs/configuration.md` (`sampling.gcmc`).

- **Ghost waters.** The System keeps its particles. A water that is deleted stays in it as a
  ghost, a molecule without nonbonded interactions, and an insertion puts a ghost at a place in
  the sphere and gives it its interactions. The ghosts of the start are waters of the prepared
  system, those whose oxygen is farthest from the ligand (`ghost_waters`); they are switched off
  from the pre-equilibration on. Switching is done in the Context while it runs (charge and
  epsilon of every site of the water in the nonbonded force and in the soft-core force of the
  ligand); a Context switched so gives the energies and forces of a System built so, on both
  backends (`spikes/m33_ghost_waters_on_the_backends.py`, the tests). With no ghost left an
  insertion cannot be tried, and the leg stops with that reason (as grand does); it is not taken
  for a failure of a worker or of the backend. A ghost is still a molecule for the Monte Carlo
  barostat of the pre-equilibration, which counts it in the ideal-gas term of a volume move: 72
  ghosts in 810 nm3 are 3.7 bar, by which the pressure of the waters that interact is lower than
  the one recorded.
- **The move.** With probability 1/2 an insertion, else a deletion. An insertion puts the oxygen
  of a ghost at a point drawn uniformly in the sphere, the molecule turned at random; a deletion
  takes one of the N waters that interact and have their oxygen in the sphere. An inserted water
  keeps the velocity of its centre of mass, and its velocities relative to it are turned with the
  molecule, as in the water relocation moves (one function for both): they stay those of a rigid
  body, with the kinetic energy they had. (Until 2026-11-28 they were left as they were, with
  components along the bonds of the turned molecule, which the constraints of the next MD step
  took out. The samples of a leg record the rule of the mover, `velocities_of_an_inserted_water`
  in their settings; samples of the earlier rule record none, are analysed as they are, and are
  not continued by this version, so that no estimate is made of both.) The acceptance is

      P(insert) = min[1, exp(B) exp(-beta dE) / (N + 1)]
      P(delete) = min[1, N exp(-B) exp(-beta dE)]
      B = beta mu'_sol + ln(V_sphere / V°)

  with mu'_sol the excess chemical potential of water in bulk water and V° its standard volume.
  The sphere is centred on the geometric centre of the heavy atoms of the ligand and follows it.
- **Batches and two tests.** A single insertion into liquid water is accepted about once in ten
  thousand trials. One move is therefore `attempts` trials, evaluated in batches of `batch_size`
  from one configuration with an approximate potential (the stand-in energy of the water
  relocation moves, on the GPU of the Context): the first trial of a batch that is accepted is
  tested again with the potential energy of the simulation, min(1, exp(-beta [(U(j) - U(i)) -
  (U'(j) - U'(i))])), and applied if it passes; the rest of the batch is thrown away and the count
  of trials goes on from that trial. Each trial is so one Metropolis step with the full potential,
  and the approximate potential decides only how many trials pass.
- **Constant volume.** The grand canonical ensemble has a fixed volume, so the alchemical run of
  the leg (the precalculation of the lambda windows and the replica exchange) has no barostat.
  The pre-equilibration and the MD for the restraint keep theirs; the cell is then fixed at the
  cell of the last configuration of the pre-equilibration (grand's examples go on from the last frame
  of an MD at constant pressure). Until 2026-11-28 it was the mean cell of the second half, every
  molecule shifted whole into it; in a bilayer whose cell still drifted the molecules then overlapped.
  A leg that has run under that rule stays in the cell it has run in (that of the checkpoint
  of its replica exchange, else of the start configurations of its windows), so that it is
  continued with the System its samples record.
  The solvent leg is as without the moves, at constant pressure. The free energy of the complex
  leg is then one at constant volume and constant chemical potential of the water in the sphere,
  that of the solvent leg one at constant pressure; the result records the pressure at which the
  cell of the complex leg was equilibrated as the pressure of that leg.
- **Replica exchange and MBAR.** Every replica carries the list of its waters that interact
  through the exchange, and its worker gives that list to the Context before the MD. The
  energies of a configuration in all windows are taken in one Context with its own waters, so
  the term of the chemical potential, -mu N, is the same in every window: it cancels in the
  exchange between windows and in MBAR, whose formulas are those of a leg without the moves.
  REST2, adaptive lambda scheduling (whose windows make the moves too) and the checks of the
  analysis are unchanged. The tail correction of the ligand leaves out the waters that are
  switched off in each configuration.

What is recorded: the samples of the leg hold, per iteration and lambda window, the trials of an
insertion and of a deletion, how many were made, and the number of waters in the sphere after the
move (`gcmc`); the checkpoint holds the waters of every replica; the result of the leg gives the
sums over the production by window and the mean, lowest and highest number of waters in the
sphere (`gcmc`), as numbers without a verdict. The settings of the moves, with the two numbers of
the water as they were used, are among the conditions of the samples of the complex leg.

Not in this version: nonequilibrium insertions (GCNCMC), moves in the whole cell, moves in the
solvent leg, the moves together with the water relocation moves.

## Checks of the analysis

### Free energy and uncertainty of a leg

The production that is analysed begins after the equilibration of the run
(`sampling.equilibration_ns` per lambda window). With `analysis.detect_equilibration: true` it
begins later: automatic equilibration detection (Chodera 2016, as openmmtools applies it to
replica exchange) takes the effective energy of every iteration of the production, the sum over
the replicas of the reduced potential of each in the state it was simulated in, and among 100
time origins spread evenly over it the one from which the series has the most uncorrelated
samples; the iterations before that origin are left out as well. The free energy, every part of
the analysis below and the numbers of the structure are then of the remaining iterations, and
the result records what was left out (`equilibration_detected`: iterations, the statistical
inefficiency and the number of uncorrelated samples from there on). Only the start is taken
from the detection: the samples are thinned as described next, with or without it. The default
is off (`docs/configuration.md` gives the source and what the detection does on the stored
benchmark runs).

MBAR assumes uncorrelated samples. The samples of every lambda window are therefore thinned
by the statistical inefficiency g of that window: the window keeps every ceil(g)-th
iteration. g is taken from the energy difference between the window and a neighbouring one,
over the configurations the window holds from iteration to iteration. For a window with two
neighbours both differences are examined and the larger g is used; the result records both
values and which neighbour gave it.

The free energy of the leg that is reported is that of decoupling: MBAR on the samples that
are kept, minus the correction for the dispersion tail ("Long-range dispersion") of the same
iterations. Everything that is said about the leg is said about this quantity, and every part
of the production is estimated in the same way, from its own iterations: the two halves, the
points of the forward and reverse series, the consecutive blocks, and every estimate of the
jackknife below. A standard error or a check of a number is therefore always one of that
number, including the correlation between MBAR and the correction. (The comparison of BAR with
MBAR is the exception: both are estimators of the MBAR part on the same samples, and they are
compared without the correction. The standard error of the sum of BAR between neighbouring
windows comes from the same time blocks as that of MBAR, the largest of the numbers of blocks
that can be used: two neighbouring pairs share a window, so the errors of the pairs are not
independent and their sum in quadrature is no error of the sum. A pair counts only if its two
windows determine their difference by the rules below.)

The whole production and every part of it are used only if their free energy is determined
(the whole production by the same rules as every part; analysis version 7):

- every window keeps at least four samples in it;
- the MBAR equations of its samples have a solution;
- its samples determine the solution of its MBAR equations. Without samples that connect two
  groups of windows the equations hold for any difference of their free energies, and a solver
  that starts from a value returns it. The Hessian of the equations, which the Newton step of
  the solver uses, is H = diag(N) - P P^T, with P_in the probability that sample n is of window
  i: the Laplacian of a graph whose edge between two windows weighs the samples they share. The
  free energies are unique if this graph is connected, that is if H with window 0 fixed has no
  zero eigenvalue. With its smallest eigenvalue lambda, the solver, which stops where every
  |gradient_i| / N_i is below 1e-8, may stop up to 1e-8 |N| / lambda away from the solution,
  and the statistical standard error of that combination of free energies is about
  1 / sqrt(lambda) (Shirts and Chodera 2008). A part is determined if the first is at most a
  hundredth of the second and at most 0.01 kT, finer than the 0.01 kcal/mol to which free
  energies are reported: the samples, not the start of the solver, fix its free energies.
  (With the relative bound alone, two windows that shared 1e-9 of their samples had halves with
  errors of thousands of kT that were the start of the solver, the whole, and agreed.) A
  connected part of low precision is determined, however long its
  chain of windows: 24 windows that overlap by 0.25, or 34 windows of one Hamiltonian, each
  pair sharing 1/34, are. This does not depend on `analysis.min_overlap`, and a part whose
  windows are not connected is found whatever it is;
- the correction has at least four observations in it (tail energies for the correction by
  reweighting, volumes of the coupled state for the isotropic one), all of them finite (and the
  volumes positive). For the correction by reweighting the result also gives the effective
  number of its observations, 1 / sum w^2 of their normalised exponential weights, and the
  largest weight, so that a correction resting on a few configurations shows; no level is
  applied to them.

A part that is not determined is left out with the reason: points and blocks are listed in
`parts_not_determined`, a half has no value, and the other blocks keep their places.
A determined part whose neighbouring windows overlap among its samples by less than
`analysis.min_overlap` (0.03, the level of Klimovich et al. below which the estimate "will
likely be substantially incorrect") is used, and shown with its error and a warning: it is
listed in `parts_below_level` (for the halves in `halves_warning`); these are notes on the
figure, not verdicts on the leg. The overlap of the whole production is a check of its own (`NOT RELIABLE`);
since every split of a chain of windows into two groups crosses a pair of neighbours, a leg
whose windows are not connected always fails it. A production without the four observations of
the correction gives the leg no free energy.

This is the procedure of Klimovich, Shirts and Mobley 2015 (section 2.5: the series may be
dU/dlambda or "any of the potential energy differences") and of Mey et al. 2020 (section 8.2)
for simulations whose states have different correlation times, and it is what alchemlyb does
(`decorrelate_u_nk`, method `dE`, uniform intervals), with one difference: alchemlyb takes the
difference to the next window only. MBAR uses the energies of a configuration in both
neighbours, and a degree of freedom that is slow can change the difference to one neighbour
and leave the other alone. In a model with three windows where only the difference to the
previous window is slow, the standard error from the next window alone was a third of the
scatter of the estimate between independent runs (0.05 against 0.16 kT); with both neighbours
the two agree (0.06 against 0.07 kT). On 86 stored TYK2 legs the slowest window was found
through the previous window in 23.

Of replica-exchange runs Mey et al. say that "the replicas should undergo equivalent random
walks in alchemical space, and the replicas can be subsampled with the same g". cuFEP's
replicas do not undergo such walks: with the 15 to 19 windows that adaptive lambda scheduling
chose for a complex leg in those runs, the replicas made no round trip between the ends in 51
of 53 legs of 5 ns on TYK2, and one in the other two; with the 33 to 35 windows of the default
start since 2026-11-22, none in two of three legs and one in the third. One g for all windows, from the reduced potential
summed over the replicas (the series of openmmtools 0.26, which OpenFE uses), read about 2
iterations there, where the slowest window of a complex leg has 23 to 412 (median 76; an
iteration is 2.5 ps). cuFEP used that g at first; the standard errors of the binding free
energy were then about half of what they are with one g per window.

The standard error of a leg. MBAR's own standard error takes the samples it is given as
independent. Thinning every window by its own g does not make them so in replica exchange: a
configuration that changes slowly travels from window to window, each window then looks
decorrelated on its own, and the samples of different windows are still correlated with each
other. Grossfield et al. 2018 describe the ensemble of a replica-exchange simulation as
"constructed from" correlated trajectories that return many times (section 8), and Rizzi et
al. 2020 found the MBAR error after an analysis of the autocorrelation at the low end of the
spread between independent replica-exchange runs. cuFEP therefore takes the standard error
from blocks of time that are the same for all windows:

- the production is divided into B consecutive blocks of the same length, the free energy is
  estimated B times, each time without the samples of one block, and the standard error is
  sqrt((B - 1) / B * sum over the blocks of (dG without the block - their mean)^2), the
  delete-one-block jackknife. A block of time holds what all windows sampled in it, so the
  correlation between the windows stays inside the blocks. "For correlated data, blocks that
  are uncorrelated are the unit that is resampled", and with few of them the jackknife is the
  method of choice (Grossfield et al., section 7.6);
- it is computed for B = 40, 20, 10 and 5 (`analysis.error_blocks`), the four values are stored
  (`delta_g_error_blocks`), and the largest is the standard error that is reported, unless the
  analytical one is larger. The analytical one combines the standard errors MBAR and the
  correction give for independent samples; both come from the same configurations of the
  coupled state, so their correlation is taken from the jackknife replicates of the most blocks
  that can be used (`correction_correlation`), and where none can, their sum, an upper bound of
  the error of their difference, is taken and named ("analytical upper bound"). Five blocks give
  an error with four degrees of freedom,
  which scatters by about a third; on 86 stored legs the error from blocks was below the
  analytical one in 12, and the analytical one is already at the low end of the spread
  between runs (Rizzi et al.). Which of the two is reported is recorded
  (`delta_g_error_from`). Blocks
  have to be longer than the correlation time, and the standard error stops growing with
  their length once they are (Grossfield et al., section 7.3.2); the correlation time between
  the windows is not known beforehand, so the longest blocks that still give an estimate are
  included. The four numbers of blocks are cuFEP's choice; none of the sources gives numbers,
  and none describes this procedure for replica exchange as a whole;
- a number of blocks is not used if a block is shorter than the interval at which the slowest
  window keeps its samples, or if the estimate without one of its blocks is not determined
  (see above). Where no number can be used (a very short leg) the standard error is the
  analytical one, and the leg says so with the reason;
- the standard error MBAR gives is kept next to it (`delta_g_mbar_error`, `dg_bind_mbar_error`,
  column `mbar_error` of `summary.csv`). Every part of the production (each half, each point of
  the forward and reverse series, each consecutive block) gets its standard error in the same
  way from five blocks within it (`analysis.part_error_blocks`).

How well it does was measured on a model: eight harmonic windows of the same curvature whose
end states have the same free energy, a configuration that keeps 0.97 of its distance from the
centre of its window from one iteration to the next, and exchange between the replicas (the
model of a review of this code). Over 48 independent runs of 600 iterations:

| | standard error of MBAR / scatter of the runs | reported standard error / scatter of the runs |
|---|---|---|
| the model | 0.28 | 0.83 |
| the same without correlation | 0.93 | 1.07 |
| three windows without exchange, one of them slow (24 runs) | 0.95 | 1.00 |

So the reported standard error is still about a sixth too small where the slow motion has a
statistical inefficiency of a ninth of the run (66 of 600 iterations), and it is not too
small where there is no correlation. (From 48 runs the scatter itself is known to about a
tenth.) One trajectory does not give more: its longest blocks are then only two statistical
inefficiencies long.

Two overlap matrices are computed, both in the normalisation in which every window holds the
same number of samples:

- from all iterations, thinned by the one g of the summed potential, the same number in every
  window. This is the matrix of the figure;
- from the samples the free energy is estimated from. pymbar's matrix is O_ij = N_j sum_n
  W_ni W_nj with the MBAR weights W and the numbers of samples N; its rows sum to 1. With the
  same N in every window it is symmetric. With different N it measures the numbers as well: a
  window that keeps 30 samples next to one that keeps 3000 reads about 0.003 in one direction
  however well the two overlap, and on 18 stored legs the smaller direction of a pair was
  below 0.03 in 15, of which 13 have a lowest overlap of 0.038 to 0.105 with equal numbers.
  The numbers of samples have a check of their own, so cuFEP evaluates the matrix with every
  window's samples weighing the
  same in total: O_ij = sum_n w_n P_ni P_nj, where P_ni = exp(f_i - u_i(x_n)) / sum_k
  exp(f_k - u_k(x_n)) and w_n = 1 / N_s for a sample of window s. For equal numbers this is
  pymbar's matrix exactly. pymbar's own matrix of these samples is stored in the result too
  (`overlap_used_weighted_by_number`); no level is applied to it.

### The checks

After the production the analysis of each leg makes four checks. They change nothing in what
is run. What they find is written to the result line, to `result.json` (`quality`), to
`summary.csv` and to the benchmark report. The report compares every ligand with a result
with experiment and with the predictions published with the set, NOT RELIABLE ones included,
also in two groups apart, and names the pair of windows below the level
(`overlap_below_level` in `summary.json`). Every level is one that the literature gives:
Klimovich, Shirts and Mobley, "Guidelines for the analysis of free energy calculations", J.
Comput. Aided Mol. Des. 2015, 29, 397, and Mey et al., "Best Practices for Alchemical Free
Energy Calculations", LiveCoMS 2020.

| check | level | message | source |
|---|---|---|---|
| overlap | every pair of neighbouring lambda windows overlaps by at least `analysis.min_overlap` (0.03) in the production, in both matrices above; the message says which one is below | `NOT RELIABLE` | Klimovich et al., section 3.5: "values as low as 0.03 [seem] to be tolerable to yield a reliable free energy estimate (as long as the resulting error is sufficiently low)". "Anything below that number should serve as a caution sign ...; in this case, not only will the estimated variance increase but the free energy estimate itself will likely be substantially incorrect" |
| independent samples | the lambda window with the fewest has at least `analysis.min_effective_samples` (50): as estimated (iterations of the production divided by g of the window) and as the number of samples MBAR gets from it (every ceil(g)-th iteration) | few independent samples | Mey et al., section 8.2: "it is recommended that the effective number of uncorrelated samples Neff >= ~50 if the BAR or MBAR estimators ... are used" |

The overlap comes first. The overlap matrix is "a quantitative estimator of the phase-space
overlap" (Klimovich et al., section 3.4), "trustworthy results should in general have at
least a tridiagonal overlap matrix" (section 3.5), and the other checks presuppose it:
without overlap the estimate and its standard error mean nothing. `NOT RELIABLE` is
therefore the only message that says a result cannot be used.

The message about the samples says that more sampling is advisable, with the number and the
level; it does not say that a result is wrong.

These two are the only checks. cuFEP has no criteria of its own (the owner, 2026-11-20). The
forward and reverse series, the two halves of the production, the consecutive blocks and BAR
between neighbouring windows are computed and shown (`analysis.png`, `result.json`) as numbers
without a verdict, as OpenFE shows its forward and reverse convergence. Earlier versions also
judged the two halves of a leg against 1 kT, BAR against MBAR against 1 kcal/mol, and the
samples after a reduction by the ratio of two standard errors: the 1 kT of Mey et al. is the
caption of a figure about one step of a relative perturbation, and for a whole leg the
difference of its halves is already about 2 kT from statistical noise alone when its standard
error is 0.6 kcal/mol; the comparison with BAR and the reduction were cuFEP's own.

A value that cannot be estimated (a half of a production that is too short, a BAR estimate
for windows without overlap) is `null` in the result files, with the reason in a text field
next to it; the files hold no NaN or Infinity and are read by any JSON parser. A leg whose
analysis gives no free energy at all is a failed leg.

What no check of one run can show. "It is difficult to establish with certainty that good
sampling has been achieved, but it is not difficult to rule out high-quality sampling"
(Grossfield et al. 2018): the checks can rule convergence out, they cannot prove it. A slow
change that occurs once in a run, or not at all, leaves no trace in the statistical
inefficiency. "The variance in free energy afforded by any estimator will always be an
underestimate of the true uncertainty" (Mey et al., section 8.4), and Rizzi et al. 2020 found
the MBAR error from an analysis of the autocorrelation at the low end of the spread between
independent replica-exchange runs. The guides' way to measure the precision of a protocol is
"two or three runs of the same perturbation" (Mey et al.): `sampling.repeats`.

### What the checks say about TYK2 and A2A

86 legs of earlier TYK2 runs with 5 ns of production per window (most with 19 and 21
windows), read again with version 6 of this analysis (about 20 s per leg, eight at a time).
Read again with version 7, these 86 legs and the 34 legs of the runs of 2026-11-19 give the
same free energies, standard errors, halves, overlaps, numbers of samples and verdicts as with
version 6 (every window of every leg keeps at least four samples in the whole production); the
standard error of the sum of BAR, now from time blocks, is 0.10-1.22 kcal/mol (median 0.31),
1.39 times the analytical error of MBAR in the median, for 119 of the 120 legs (one leg is too
short for blocks):

| | complex legs (43) | solvent legs (43) |
|---|---|---|
| overlap below 0.03 | 1 (among all iterations) | 0 |
| statistical inefficiency of the slowest window, in iterations of 2.5 ps | 23-412, median 76 | 1-281, median 25 |
| samples MBAR gets from that window | 5-87, median 26; fewer than 50 in 37 legs | 7-1000, median 77; fewer than 50 in 16 legs |
| standard error of the leg, kcal/mol (from time blocks in 39 and 34 legs, the analytical one in the others) | 0.24-1.10, median 0.49 | 0.11-0.44, median 0.19 |
| standard error MBAR gives, kcal/mol | 0.17-0.62, median 0.33 | 0.11-0.44, median 0.13 |
| ratio of the two | 0.28-3.03, median 1.49 | 0.80-2.89, median 1.37 |
| what the standard error leaves of the samples | 1-48, median 12; fewer than 50 in all 43 legs | 4-696, median 33; fewer than 50 in 25 legs |
| halves more than 1 kT apart | 24 legs; in 4 of them by more than 2 standard errors | 8 legs; in 4 of them by more than 2 standard errors |
| halves not compared; compared below the level of overlap | 1; 3 | 0; 1 |
| BAR and MBAR more than 1 kcal/mol apart | 0 | 0 |

The overlap holds in all legs but one. What limits these legs is the number of independent
samples: in 5 ns the slowest window of a complex leg gives MBAR about 26 samples where 50 are
recommended, and the standard error from time blocks is 1.5 times the one MBAR gives for
them, which leaves about 12. At that number the halves of most complex legs are more than 1
kT apart, all but 4 of them by no more than twice the standard error of the difference. In
the solvent leg the slowest window is the first one, the fully interacting ligand in water,
in 39 of the 43 legs. In 34 further legs of 5 and 10 ns (17 complex, 17 solvent) the medians
are 0.48 and 0.27 kcal/mol from time blocks against 0.36 and 0.20 from MBAR, and no complex
leg and 3 solvent legs reach 50.

The ratio of the two standard errors has a statistical error of its own. In model runs
without any correlation its median was 1.07 to 1.12, and it was above 1.41, which halves the
count of the samples, in 15 of 144 runs. In 3 complex and 9 solvent legs of the 86 it is
below 1, so the standard error from time blocks is the smaller one there; in the leg with
0.28, the slowest window kept 10 samples and only 5 blocks could be used. With that few
samples neither standard error is known well.

One run against several. The same complex leg of five TYK2 ligands, run three times with
different seeds, differs with a standard deviation of 0.7 to 2.9 kcal/mol (pooled 1.8), the
binding free energy with 0.3 to 2.9 (pooled 1.6). The standard error of such a complex leg
from one run is 0.49 kcal/mol from time blocks (root mean square over the 15 legs; 0.36 from
MBAR), that of a binding free energy 0.2 to 0.8 (median 0.5). The standard error of one run
does not contain the slow changes described under "What windows do not cure": changes that
do not occur within the run leave no trace in its blocks either.

The standard errors in the table below are the ones MBAR gives for the kept samples; the
table was made before the standard error from time blocks. The three continued legs in it were
continued by a version that started the random streams of a resumed leg from their beginning
(see "Running again in the same output directory"): their second 5 ns were driven by the same
thermal noise as the first, so they are no evidence of what a longer run gives.

Runs that were continued, and one with more windows:

| leg | production | free energy, kcal/mol | samples of the slowest window | halves apart |
|---|---|---|---|---|
| TYK2 ejm_31, complex, 16 windows | 5 ns | 37.05 ± 0.27 | 36 | 1.8 kT (2.0 standard errors) |
| the same leg continued | 10 ns | 36.46 ± 0.23 | 30 | 0.7 kT (0.9) |
| TYK2 jmc_23, complex, 15 windows | 5 ns | 37.78 ± 0.62 | 32 | 1.9 kT (0.8) |
| the same leg continued | 10 ns | 36.41 ± 0.51 | 38 | 6.2 kT (3.4) |
| A2A 4e, complex, 15 windows | 5 ns | 38.20 ± 0.63 | 9 | 3.1 kT (1.3) |
| the same leg continued | 10 ns | 36.74 ± 0.44 | 32 | 1.4 kT (0.9) |
| TYK2 ejm_31, complex, vanish stage started with 13 windows: 19 windows, lowest overlap 0.145 | 5 ns | 36.86 ± 0.26 | 38 | 2.2 kT (2.4) |

These are observations of single continuations. In the ejm_31 leg the difference of the
halves went from 1.8 to 0.7 kT while the samples of its slowest window went from 36 to 30
(a slower window came into view: the one at which the discharge stage begins, with a
statistical inefficiency of 132 iterations); in the jmc_23 leg the difference rose from 1.9
to 6.2 kT. The two halves of 5 ns and the two halves of 10 ns are the same trajectory cut at
other places, so these numbers cannot tell a slow fluctuation from a change in how often slow
conformations are visited; that takes independent runs and the conformations they visit.
More lambda windows raise the overlap; they do not add independent samples. In the A2A leg a
window near the end of the vanish stage is correlated over 126 to 214 iterations (0.3 to 0.5
ns), so 5 ns give 9 independent samples and 10 ns give 32; the bilayer of that system had
also not settled when the alchemical run began.

Why the default production is 10 ns per window. It is the default of OpenFE's ABFE protocol. Of 55 TYK2 complex legs with 5 ns, 11 had 50 independent samples in their
slowest window; the statistical inefficiencies of that window (quartiles 52, 68 and 116
iterations) put 33 of them there with 10 ns, if they stay as they are, and 46 with 20 ns. Of
54 solvent legs, 33 had 50 with 5 ns. A longer run can also show a slower change, as the
ejm_31 leg above did, so 10 ns do not guarantee the 50. These counts are of the samples MBAR
gets. By what the standard error from time blocks leaves of them, no complex leg of 5 ns
reaches 50 (table above), and the complex leg of ejm_31 run for 10 ns in one piece counts as
8 (22 samples kept, ratio 1.6).

`analysis.png` in each repeat directory shows, for both legs, the forward and reverse series
(the free energy from growing parts of the production, counted from its start and from its
end, at up to `analysis.convergence_points` points, each drawn at the fraction of the
production it really uses), the two halves, the estimates from `analysis.convergence_blocks`
consecutive blocks that do not overlap, the band of the standard error and the overlap
matrix. The number of round trips of the replicas, the exchange between neighbouring windows
and the statistical inefficiency and the samples of every window are in `result.json`; no
level is applied to round trips and exchange. The acceptance of exchange
(`swap_acceptance_neighbour`) is accepted / proposed swaps between the replicas in each pair of
neighbouring windows over the production, as the samples record them per iteration; samples of
earlier versions do not, and have none. `neighbour_transition_fraction` is the fraction of
iterations after which a replica had moved to a neighbouring window, which earlier versions
reported as the acceptance; it misses moves that are undone within one iteration of mixing.

### When a leg carries a message

cuFEP does not run anything again by itself. The user runs the leg concerned again; the other legs and ligands keep their results and the binding free energy is put
together again.

- `NOT RELIABLE`: more lambda windows in the stage that the message names
  (`lambda_schedule.adaptive.initial_windows`, a different `overlap_threshold`, or
  `lambda_schedule.windows` with `mode: fixed`), then `cufep run CONFIG --ligand NAME --leg
  complex --reschedule`. The leg starts anew from its pre-equilibration and restraint with
  windows determined again; the windows, samples and result it had are moved to
  `previous_<n>/` in the leg directory.
- `NOT RELIABLE` "among the samples the free energy is estimated from": the windows overlap
  when all iterations are taken, but not in the samples that are left after every window was
  thinned by its correlation. A longer production gives the slow window more samples; more
  lambda windows in that stage raise the overlap.
- Few independent samples: a longer production. `cufep run CONFIG --ligand NAME --leg complex --production-ns 20` (or a
  larger `sampling.production_ns`) continues the replica-exchange run of that leg from its
  last iteration; the analysis then covers the whole production. The number is the length of
  the production in total, not what is added: a leg that has 10 ns runs 10 ns more.

A run restricted with `--ligand` or `--leg` writes its timing to `timing_rerun.json` and
leaves `timing.json` of the whole run in place.

### Running again in the same output directory

A leg that runs again goes on from what it has, and the legs of a ligand are put together
again from what the directory holds. So that this cannot give a number from parts that do not
belong together:

*A leg and its own files.*

- What a leg gave before is set aside as `result.previous.json` when it runs again, for
  instance its result at 10 ns when it is continued to 20 ns. If the run fails, the leg is
  `FAILED`: it has no result, the ligand has no binding free energy from it, and the command
  ends with an error. The earlier result is a record and is never used in its place.
- A leg writes `started.json` when it begins and removes it with its result or its error. A leg
  that holds only that file was interrupted (the process was killed, the machine stopped,
  Ctrl-C) and is `FAILED` until it has run to its end. For a job process that ended without a
  result the scheduler writes the error itself, with the exit code. Result files are written in
  one step; one that cannot be read is a failed leg.
- A leg gets a result only if it has reached the length that was asked for, with every
  iteration stored once. A run that was interrupted before its first checkpoint starts again
  from iteration 0 with an empty storage. Where the production begins is recorded before the
  first iteration, so that an interrupted leg is not analysed with its equilibration (`cufep
  analyze` says "no production yet" for a leg that has not reached it).
- Samples are continued only under the conditions they were made under: the lambda states,
  temperature, pressure and barostat, friction, time step, length of an iteration, backend,
  precision, the System with its restraint and REST2 region (a checksum of its XML), the tail
  correction (its two radii, the numbers of atoms and a checksum of their parameters), and
  the settings behind them in the words of the configuration (soft-core, REST2, tail). A leg
  that has samples keeps their backend. The XML of a System names the version of OpenMM that
  wrote it, so samples are also not continued after OpenMM has been upgraded.
- A leg that is continued gets random numbers of its own. The stream of every worker (thermal
  noise of the integrator, and that of the barostat) is derived from the seed of the leg, the
  number of the start, the worker and the purpose; the number of the start is kept in the
  checkpoint and raised each time the leg is resumed, and a worker that is started again
  within a run gets a new number. Exchange decisions depend on the seed and the iteration and
  never repeated. Versions before this one derived the streams from the seed and the worker
  alone, so a resumed leg repeated the thermal noise of its start: results of legs that those
  versions continued should not be relied on.
- Stored samples are never changed. The iterations are kept in memory; at a checkpoint they go
  to a new part, `hrex/segment_<n>.h5`, which is written whole under another name, made
  durable and renamed. Only then is `hrex/manifest.json`, which names the parts in order with
  their numbers of iterations, tail energies and frames, and holds the attributes of the
  samples, replaced in one step (written under another name, made durable, renamed). Every
  reader goes by the manifest. Whenever a write fails or the writer is killed, the manifest
  names complete parts only: a reader sees one state of the run, nothing is stored twice, and
  the parts the last checkpoint rests on are as they were. (Earlier versions added the rows to
  one file. HDF5 rewrites a compressed chunk of frames when rows are added to it, so the frames
  of the last checkpoint were lost when the writer was killed in the middle; review of PR 1.)
  A truncation, as when a leg is resumed from its checkpoint, replaces the manifest alone. What
  is stored is what the manifest names, whatever ends a write, an error or an interrupt: rows
  that it names are never kept in memory to be stored again. When the sampler opens a leg it
  removes what the manifest does not name (parts a truncation dropped, what a failed write
  left); the manifest counts the parts made, so a name that a manifest held is never given
  again, and a reader that finds a part gone reads the new manifest.
- No part is opened for writing once it exists, so a reader and the sampler never meet in a
  file. `cufep analyze` can be used on a leg that is being sampled; it sees the iterations up
  to the last checkpoint, and no checkpoint waits for it. If the samples cannot be stored (a
  full disk), the error is raised with the last checkpoint and its samples kept, and the leg is
  resumed by running it again on the same backend.
- The checkpoint names the iteration and the number of stored rows; records (tail energies,
  frames) carry the row of their iteration. A checkpoint of an earlier version does not record
  how many of its iterations were stored, and iterations that were run without being stored (a
  warm-up) cannot be told from stored ones, whatever its records show. A leg is therefore not
  continued from such a checkpoint: the run is refused before anything is written, with what to
  do (`--reschedule` starts the leg anew and keeps its samples in `previous_<n>/`). A leg that
  holds all its iterations is analysed as it is.
- Workers are started by one function for the replica exchange, the lambda precalculation and a
  change of GPUs: if one does not start, every worker that was started is stopped before the
  error is raised, and the workers and the GPUs the sampler names agree.
- A leg is repeated on stock OpenMM (`backend.fallback`) only after a failure of the backend:
  a worker that died, hung, did not start, or failed twice from the same configurations, or an
  error of OpenMM. Not after a failure to store the samples, and not if the leg had samples
  from an earlier run (a leg that is continued): its samples stay in `hrex/` and the leg fails
  with what to do. What a failed cuPotential run left is kept in
  `hrex_failed_cupotential/`, `hrex_failed_cupotential_2/`, ...
- The number of MD steps of an iteration and the numbers of iterations of the equilibration
  and the production follow one rule for the run and for every check of stored samples
  against a run: 2.5 ps at 3 fs are 833 steps (2.499 ps), and 10 ns are 4000 iterations.
- The prepared systems, the pre-equilibration, the trajectories from which a restraint is
  chosen and the restraint are used again only if the
  input files and the settings that went into them are the same (a trajectory also only if the
  pre-equilibration it started from is the one that is there). Only the lambda windows of a
  leg can be determined again in place, with `--reschedule`. With `membrane.complex_leg: water`
  the trajectories start from the pre-equilibration of the complex in the bilayer
  (`restraint_equil.npz`), and the restraint is held to two files: to that one, and to the
  pre-equilibration of the complex in water from which the alchemical run starts
  (`equil.npz`), at which its anchors were held to the rules. A directory prepared with one
  value of the setting is refused by a run with the other before anything is written.
- A lambda schedule from a file (`lambda_schedule.file`) must be one of the same leg and of the
  REST2 scale of the run: its stages, its two end states and its scale are checked before the
  leg takes it over, and before the run writes anything. Give a directory as written by `cufep
  benchmark freeze`, or the file of one leg together with `--leg`.
- A leg whose archive (`--reschedule`) was begun and not completed is not run as it is: the
  run is refused until `--reschedule` is given again, which completes the archive.
- How the bilayer changed and how far the ligand was from its input pose are taken from the
  files of the leg whenever its result is made, as numbers. The notes with a verdict on them
  that earlier versions wrote into those files and into their results (whether the bilayer had
  settled, whether the ligand had left its pose or the pocket) are removed; a result of
  analysis version 9 or earlier is made again from its samples for that. The lambda windows
  and samples that `--reschedule` sets aside in `previous_<n>/` take copies of the
  pre-equilibration and the restraint with them; an archive of an earlier version is read with
  those of the leg directory above it, which the leg used again.

*The legs of a ligand.*

- A complex and a solvent leg, and the repeats of a ligand, are combined only if they were
  made under the same temperature and pressure, from the same prepared systems, and with the
  same correction for the dispersion tail; with the correction by reweighting also with the
  same outer radius (`analysis.lrc_cutoff_nm`), because the tail energies are recorded to
  that radius during the run and cannot be turned into those of another. If they were not,
  the ligand gets no binding free energy, the result names what differs, and the command
  ends with an error. Backend, time step, number of lambda windows, lengths and REST2 may
  differ between the legs. A run that would sample a leg under another temperature, pressure
  or outer radius than a finished leg it does not sample is refused before it samples
  anything, and so is a run whose analysis the samples of such a leg cannot give: a tail
  correction for samples made without its data, or a correction by reweighting for a leg with
  no tail energies in its production. What an analysis needs from samples is checked in one
  place for the run, for the legs it does not sample and for `cufep analyze`.
- A leg that a run does not sample (`--leg`, `--ligand`) keeps the result of an earlier run.
  If that result is of another analysis (another version of the method, other settings of
  the `analysis` section, or other levels for the notes on bilayer and pose), it is made again
  from the samples of the leg before the legs are combined; the result it replaces is kept as
  `result.replaced.json`. If the samples are no longer there, the leg is combined as it is,
  and the result names it (`old_analysis`) with the checks its result does not hold.
- `cufep analyze LEG_DIRECTORY` analyses a leg with the settings of its own run and with the
  temperature and iteration length of its samples, so it says what the run said. The settings
  are those the result of the leg records; for a leg without a result (running, or
  interrupted) those its run stored with the samples before its first iteration; else those
  of the result the leg had before. The output names which, and the two radii of the tail
  correction that were used. `--config` gives other settings; an outer radius other than the
  one the samples recorded their tail energies to is refused. It only prints.

*The record of the directory.*

- Everything that can refuse a run is checked for all its ligands and legs before anything is
  written. A run that is refused leaves `resolved_config.yaml` and the results as they were and
  writes its own configuration to `resolved_config.refused.yaml`.
- If a run is accepted and its configuration differs from the one of the earlier run in the
  directory in more than resources and output, the earlier record is kept as
  `resolved_config.previous_<n>.yaml`.

*Output directories of earlier versions.* A configuration file of an earlier version is read
if the settings it holds still exist or held their defaults; a removed setting that the user
had changed stops the reading with the settings that replaced it. Files that an earlier
version wrote without a record of what they were made from are treated as follows:

| file | what happens |
|---|---|
| samples (`hrex/samples.h5`) without recorded conditions | read and analysed, never added to, and not written to. That version stored neither the soft-core nor pressure and friction, and it rewrote temperature, backend and states whenever it opened the file, so nothing on the disk shows what the samples were made with. They are held to what their file does hold and to what their version could do: a run with another temperature, other lambda states or another length of an iteration is refused, so is one that asks for more production than they have, and so is one that asks for a treatment that version did not have (`alchemy.annihilate_electrostatics: true`, `alchemy.softcore_b: 2`). A leg that has all its iterations is analysed as it is: the release of the restraint is computed at the temperature of the samples, and the pressure, which they do not record, is named as not compared when the legs are combined. To sample further, the leg is started anew with `--reschedule` |
| samples in one file (`hrex/samples.h5`, without `manifest.json`) | read as that version stored them: up to its stored counts (`committed_counts`), or, for the versions before them, up to the iterations that all their row datasets hold. A run that continues them never writes to that file: it adds parts and a manifest that names the file, with those counts, as its first part |
| a checkpoint without the number of its stored rows | not continued from (see "A leg and its own files" above); the run is refused before anything is written |
| samples that record their conditions with a checksum of the System and without the settings behind it (one version) | before a run, the System is built again from the files of the leg and all recorded conditions are compared, so that a run with another soft-core, restraint or REST2 region is refused before anything is written. If a file needed for that is missing, the run is refused |
| prepared systems, pre-equilibration, restraint without a record | refused where what the file does hold contradicts the run (bilayer or not, barostat, box shape, ligand; length of the MD, a larger time step than configured; restraint selection and force constants). Otherwise used, and the leg and the ligand result name them (`origin_not_recorded`) |
| prepared systems, pre-equilibration, restraint of an earlier version, with a record | refused, whatever the record says, where the file holds less than it was made for: a box lower than OpenMM's rule for the cutoff, a pre-equilibration of whole 20 ps frame intervals shorter than asked, a manual restraint with the configured force constants under `force_constants: auto` |
| a result without the version of its analysis | made again from the samples, see above |
| `summary.json` of an earlier version, read by `cufep benchmark report` | the verdicts of checks that are no longer made (halves, BAR against MBAR, NOT CONVERGED) are not shown; the result has the flag that the checks of this version were not made, which are made when the legs are analysed again |
| an interrupted run that did not record where its production begins | `cufep analyze` asks for the boundary: `--equilibration-ns` (a length of 0 or more) or `--equilibration-iterations` (exactly), not both |

## Long-range dispersion

During sampling the soft-core ligand-environment interaction is truncated at the cutoff.
The missing energy is evaluated explicitly on coupled-state configurations out to 2 nm
(plus a homogeneous estimate beyond) and its free energy is added at the coupled end
state of each leg. An isotropic tail correction would assume the same atom density
around the ligand in the protein and in water. With `analysis.lrc: isotropic` the homogeneous
tail is C / V in every configuration, and the free energy of adding it under constant pressure is
-kT ln < exp(-C / (kT V)) > over the volumes of the coupled state, the same estimate as for the
tail energies by reweighting (analysis version 8; earlier versions took C / <V>).

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
  holds one worker. The rest is divided among the jobs that run the trajectories from which a
  restraint is chosen (`restraints.selection: interactions`, one worker per trajectory), that
  are in the lambda precalculation
  (its new windows of an iteration run at the same time) or sampling, in proportion to the
  work they still have to do and never more than one worker per window or replica. When a job starts sampling or ends, the shares are recomputed, and jobs start
  or stop workers at their next iteration. With one ligand the solvent leg therefore uses
  the free GPUs while the complex leg is still in pre-equilibration, and the complex leg
  takes all of them afterwards.
- A job starts only when its first worker and the first worker of every running job fit on
  the GPUs: a worker of a job with oversubscription factor f takes 1/f of a GPU, and no GPU
  is ever given more. Without MPS (factor 1) every GPU runs one worker, so one GPU runs one job
  at a time.
- The GPUs are CUDA device numbers as `CUDA_VISIBLE_DEVICES` leaves them (`resources.gpus`,
  `--gpus`); cuFEP sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` so that a device number names the same
  GPU as nvidia-smi does, whose load decides which GPUs are free and is reported.
- If the scheduler fails or is ended by Ctrl-C, SIGTERM or a hangup (these reach it alone:
  every job runs in a process group of its own), it stops every job, which stops its workers
  and closes its files, waits for them (a job that has not ended after 120 s is killed with its
  workers; a further Ctrl-C meanwhile is noted and does not cut this short), and only then
  gives up the lock of the output directory. A scheduler that is killed (SIGKILL) cannot do
  this: its jobs get a signal from the kernel when it ends (Linux `PR_SET_PDEATHSIG`) and stop
  in order, workers likewise end with their job, and every job holds a lock of its own leg
  while it lives, so a new run in the directory is refused until the last job has ended.
- With more jobs than can run at once, the others wait; freed capacity goes to waiting
  jobs first. The same code runs on one GPU.

`timing.json` in the output directory lists the time per phase and job, and the GPU
utilization during the run.

## How cuFEP compares

The absolute binding protocol of OpenFE, from its documentation and source code (see
"Sources"), beside cuFEP's:

| | OpenFE | cuFEP |
|---|---|---|
| lambda windows are given as | lambda values | a number or lambda values, per leg and stage |
| default windows | complex 30, solvent 14 | adaptive; fixed mode: complex 30, solvent 14 |
| time per lambda window | 1 ns equilibration + 10 ns | 1 ns equilibration + 10 ns |
| runs per calculation | 3 repeats | 1; `sampling.repeats` for more |
| reported uncertainty | standard deviation over repeats | standard error from blocks of time common to all lambda windows (delete-one-block jackknife of MBAR minus the tail correction, on samples decorrelated window by window), not below the analytical one; standard deviation over repeats if more than one |
| checks after the run | plots of the MBAR overlap matrix, the replica exchange matrix, the forward and reverse convergence and the replica states; no verdict | overlap below 0.03 and fewer than 50 independent samples per window, the levels of the literature; the series, halves and BAR shown without a verdict; messages in the result |
| a calculation that is not good enough | left to the user | the user runs the leg concerned again, longer or with more lambda windows |
| replica exchange | Hamiltonian, 2.5 ps per iteration | Hamiltonian, 2.5 ps per iteration; REST2 on the ligand and the residues within 0.4 nm of it (`sampling.rest2`) |
| MD before the alchemical run | 0.25 + 0.5 + 5 ns | 0.25 + 0.5 + 5 ns; a complex leg with the default `restraints.selection: interactions`: 0.25 + 0.5 ns with the heavy atoms of protein and ligand held and no free MD (a bilayer: 0.25 + 1.0 + 10 ns, held), from the end of which the alchemical run and the four trajectories of 3 ns for the restraint start |
| restraint | Boresch, atoms chosen automatically, fixed force constants | from the interactions of unrestrained MD of the complex by default (`restraints.selection: interactions`); as OpenFE with `auto` |
| water sampling | none | none by default; in the complex leg, grand canonical Monte Carlo of water in a sphere around the ligand with `sampling.gcmc` ("Grand canonical moves of water"), or water relocation moves between that sphere and the bulk with `sampling.relocation_mc` (RelocationMC of Ding et al.; "Water relocation moves") |
| GPUs | one GPU per run | replicas spread over all GPUs, several per GPU under MPS |
| membrane | prepared by the user | switched on in the configuration; oriented, built and hydrated by cuFEP. `membrane.complex_leg: water` (cuFEP's own, off by default) keeps the bilayer for the MD of the restraint and makes the alchemical run of the complex leg in water |

Further differences:

| | OpenFE | cuFEP |
|---|---|---|
| compute backend | stock OpenMM | cuPotential plugin for the nonbonded force and integrator; stock OpenMM as fallback |
| standard state correction | analytical | numerical, analytical as a check |
| ligand-environment dispersion tail | isotropic | explicit, by reweighting the coupled state |
| Lennard-Jones between atoms of the ligand | cut off as in the original System | no cutoff (hundredths of a kcal/mol; "Interactions inside the ligand") |
| electrostatics between atoms of the ligand | annihilated: switched off with the charges (lambda^2; 1-4 pairs lambda) | kept at every lambda window by default; annihilated by OpenFE's rule with `alchemy.annihilate_electrostatics: true` |
| soft-core | alpha 0.5, powers a 1, b 1, c 6 | the same by default; `alchemy.softcore_b: 2` for the squared (1 - lambda) |
| decorrelation of the samples | one statistical inefficiency for all windows, from the reduced potential summed over the replicas (openmmtools) | one per lambda window, the larger of those of its energy differences to the two neighbouring windows (alchemlyb's default takes the next window only) |

Sources: Midgley et al. 2025; Ding et al., arXiv:2603.22274 v3 (the water relocation moves);
OpenFE 1.13 source and documentation (the plots are those of `multistate_analysis.py`); SOMD2 documentation (oversubscription factor); Mey et
al., "Best Practices for Alchemical Free Energy Calculations", LiveCoMS 2020; Grossfield et
al., "Best Practices for Quantification of Uncertainty and Sampling Quality in Molecular
Simulations", LiveCoMS 2018; Rizzi et al., "The SAMPL6 SAMPLing challenge: assessing the
reliability and efficiency of binding free energy calculations", J. Comput. Aided Mol. Des.
2020, 34, 601; openmmtools 0.26 source (`multistateanalyzer.py`); alchemlyb documentation
(`decorrelate_u_nk`).

## Limits of version 0.1

Neutral ligands only. No cofactors yet. No symmetry correction (the ligand's graph
symmetry number is reported); a small symmetric ligand that turns over in its pocket leaves
the unrestrained window without overlap, which the result line reports. Symmetric POPC-type
bilayers only. Grand canonical water sampling only as an option of the complex leg, in a sphere
around the ligand (`sampling.gcmc`, off by default). REST2 covers the ligand and the residues
within 0.4 nm of it by default; a slow rearrangement further away, such as a side chain
behind the first shell, is sampled faster only if its residues are named
(`sampling.rest2.residues`).
