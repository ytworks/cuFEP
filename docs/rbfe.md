# Relative binding free energies (`cufep rbfe`)

`cufep rbfe` computes the relative binding free energy (ddG) of every edge of a network of ligands and, from
the network, the binding free energy (dG) of every ligand. It shares with the absolute binding free energies
of `cufep run` the backends (the cuPotential plugin by default, stock OpenMM as the fallback), the
preparation of the protein, the bilayer and the water, the pre-equilibration, Hamiltonian replica exchange
with REST2, adaptive lambda scheduling, the GPU scheduler with NVIDIA MPS, and the analysis of a leg (MBAR
and its checks, `docs/protocol.md`, "Checks of the analysis"). Its YAML is its own.

The two are two protocols of one engine of a leg (`src/cufep/engine/`): a leg goes from one end state to the
other through lambda windows, and the engine runs it whatever its states are (the pre-equilibration, the
backend, the windows and the start configurations of their replicas, the replica exchange, the fallback to stock
OpenMM, the analysis of the path). The protocol of the ABFE decouples a ligand (stages restrain, discharge,
vanish); the protocol of a relative leg transforms ligand A into B through a hybrid topology (stage morph). The
result of a leg gives the free energy along its path as `delta_g` (kJ/mol), with the checks of the analysis and
the hybrid topology. The sections that mean the same in both YAMLs are one schema (`CommonConfig`).

```bash
cufep rbfe plan config.yaml      # ligands, edges and atom mappings, without simulating
cufep rbfe run config.yaml       # run (or resume) every edge
cufep rbfe analyze runs/rbfe     # the ddG and dG from the results the legs have; writes nothing
```

`cufep rbfe run` ends with exit code 1 when a leg of this run failed or has no free energy (windows that its
samples do not connect), 0 otherwise (also when some edges have no
ddG yet because only some legs were run, `--leg`). `--edge`, `--leg` and `--repeat` restrict a run to those legs;
`--reschedule` starts the legs that run anew (their windows, samples and result are set aside in `previous_<n>/`).
A run in a directory that holds results is refused before it writes anything if a prepared edge of the network (one
it runs, or one whose results it combines) was made from other inputs or settings (protein, ligands, force
field, solvation, bilayer, cutoff, ionic strength, pH ...), by an earlier rule of the preparation, or if its record
cannot be read; the refusal names every such edge and one way to prepare them all again (the analysis leaves them
out, with the same way). These commands name the run directory (`-o`, which need not be `output.dir` of the
configuration) and quote edge names for the shell (names of ligands may hold spaces).

## Inputs

- `system.protein`: the protein PDB, prepared as for the ABFE (`docs/protocol.md`, "Inputs").
- `system.ligands`: one SDF with every ligand of the network, each with explicit hydrogens and its pose in
  the pocket, all in one frame (as the public benchmark gives them). The atom mapping is made from these
  poses.
- Experimental values (optional): `system.experiment` (a CSV with the columns `ligand` and `dg`, kcal/mol)
  or the SDF tags `r_exp_dg` or `exp_dg`. They set the one constant of the dG that the edges leave, and are
  given next to the results.
- States of one compound (optional): `system.states`, a CSV with the columns `ligand` and `compound`, and
  `form`, `solution_fraction` and `symmetry_number` ("States of a compound").
- Ligands of any net charge (the box is neutralised with counterions, `thermodynamics.ionic_strength_M` of salt).
  An edge that changes the net charge by one charge has an alchemical water ("Edges that change the net charge");
  a larger change is refused before anything is written.

## The calculation of an edge

An edge A -> B has two legs, the complex and the ligand in solvent, each a transformation of A into B along
one coordinate lambda from 0 (A) to 1 (B): ddG = dG(complex) - dG(solvent), with the standard error
sqrt(se_complex^2 + se_solvent^2). The legs of all repeats of an edge must have been made under the same
conditions (temperature, pressure, prepared systems, correction of the dispersion tail: the check of the ABFE),
and each must have its standard error; otherwise the edge has no ddG, says why, and the network is made from the
other edges. The end states A and B are each built in a force field that holds its own ligand alone (two ligands
of one bonded graph in one force field would share the template of one, with its charges).

### Edges that change the net charge

The co-alchemical water of Chen et al. (J. Chem. Theory Comput. 2018, 14, 6346; its full text could not be
obtained), with the rule of OpenFE's relative protocol (`explicit_charge_correction`, from perses'
`transform_waters_into_ions`, read in its source; `src/cufep/rbfe/charge.py`): with dq = q(A) - q(B) = +1 or -1,
one water of end state A, the one of the lowest residue index among those with no atom within
`alchemy.alchemical_water_cutoff_nm` (0.8 nm) of the solutes (the protein, the ligand, lipids), is in end state B an
ion of the sign of dq: its oxygen takes the charge, sigma and epsilon of the most abundant monovalent ion of that
sign in the system of the complex leg (Na+ or Cl- of the force field if there is none), the same ion in both legs
so that its hydration cancels in the ddG (an edge whose legs give the ion another charge, sigma or epsilon is refused
and left out of the analysis; where each leg took it from is not compared), its hydrogens charge 0.
Both ends then have
one total charge. In the hybrid the water's parameters go from A to B as a core atom's do: its charges with
`lambda_electrostatics_core` (offsets in the NonbondedForce, which the cuPotential plugin carries), the Lennard-Jones
of each of its pairs interpolated linearly with `lambda_sterics_core` (in the force of the ligand with its
environment; its Lennard-Jones beyond the cutoff is not corrected, as the ligand's). The water keeps its geometry and
masses; it is not one of the waters of the grand canonical moves (its residue is named ALW in the topology of the
hybrid). Both legs of the edge have one. The lambda windows are those of adaptive lambda scheduling, as for every
edge. A change of more than one charge is refused (OpenFE: "Absolute charge changes greater than 1 are currently
not supported").

### Atom mapping

Kartograf (Ries et al., J. Chem. Theory Comput. 2024) maps the atoms of the two poses: atoms at most 0.95 A
apart, hydrogens only onto hydrogens, whole rings only and no ring that changes, no bond broken; its defaults,
which are those of OpenFE's relative protocol. Then OpenFE's rule from perses: where a constrained bond
between two mapped atoms changes its length, or is constrained at one end only, its hydrogen leaves the
mapping (with `map_hydrogens_on_hydrogens_only: false`, the atom mapped onto a hydrogen).

The hybrid keeps the unmapped hydrogens of both ligands on a mapped atom, and the integrator of the cuPotential plugin
constrains three hydrogens of one length on an atom at most (SHAKE: "only SETTLE and SHAKE constraints are
implemented"; OpenMM's own integrators take the rest with CCMA). Where Kartograf's mapping would put more on one
atom, it is changed before the systems are built (cuFEP's own choice, made of the two published steps below; OpenFE
and perses, whose integrators have CCMA, need neither):

- a mapped pair of heavy atoms that both carry hydrogens constrained to other lengths in the end states (the O of a
  hydroxyl onto the C of a methyl in TYK2, a CH2 onto an NH; a carbon of four bonds onto one of three, C-H 1.0940
  and 1.0859 A in OpenFF 2.2.1): such hydrogens cannot be mapped onto each other (their constrained bond would
  change). Kartograf maps again without that pair, passed to it as a filter of its own kind, as its own
  `filter_element_changes` ("Forces a mapping to exclude any alchemical element changes in the core") leaves out
  every pair of other elements, and keeps whole rings and the largest connected part as after its filters (the ethyl
  beyond that CH2 leaves the mapping too; an NH of a ring onto a CH2 takes the ring out);
- a mapped heavy atom on which the hybrid would carry more than three hydrogens (a methyl group turned between the
  poses, its hydrogens farther apart than 0.95 A): its unmapped hydrogens of A and B are mapped onto each other, the
  closest pair first, as LOMAP maps the hydrogens of its mapped heavy atoms (lomap2, `MCS.all_atom_match_list`: "In
  3D mode, ensure that maps happen to the closest atom in 3D coordinates").

The other edges keep Kartograf's mapping, which `prep.json` records as `mapping_kartograf` beside the mapping of
each leg. A mapping only chooses the path between the two end states: the end states and the treatment of the
atoms unique to one ligand are those of every edge, and what changes is which atoms are unique (fewer in the first
case, more in the second). An edge whose hybrid would still have such hydrogens on the plugin is refused before
anything of it is written. In TYK2 these were 5 of the 22 edges, in A2A 1 of 37; before, their legs ran on stock
OpenMM. A run on the plugin checks the edges prepared before this (their `prep.json` holds a mapping of such a
hybrid): one none of whose legs has a result is prepared again, what its directory held kept in
`edges/<edge>.earlier_mapping_<n>/`; one with results is refused before the run writes anything, with the way to
prepare it again (its results, computed on stock OpenMM, stay as they are for the analysis).

A custom core (`mapping.core_smarts`, a SMARTS pattern; none by default) restricts the mapping to the atoms in a
match of the pattern in both ligands: the other atoms are unique to each ligand and change with the soft-core, as a
group that is "decoupled in the intermediate lambda windows" to sample it better (the custom cores of Ross et al.
2023, Supplementary section "OX2"). The network of LOMAP is planned from the mappings so restricted. A match of
the pattern is a set of atoms (the orders of the atoms of a symmetric pattern are one match), and a pattern with more
than 1000 matches in a ligand, the most that are searched, is refused. A mapping in which a group of atoms unique to
one ligand is bonded to two or more mapped atoms is refused (its bonds and angles would hold the mapped atoms at both
ends of the transformation): the mapping of each leg as the hybrid is made of it, after the rule of the constrained
bonds, before anything of the edge is written; LOMAP plans no edge with one, and `cufep rbfe plan` shows none (with
the pairs of a hydrogen and a heavy atom taken out as that rule takes them where the bonds of hydrogens are
constrained); an edge prepared before this check is checked when it is taken again and in the analysis, which leaves
it out. The record of a prepared edge (`prep/prep.json`) is checked before it is read: what it was made from (the
inputs and settings every preparation records), both legs, each with the numbers of atoms of its end states and a
mapping that is not empty, one to one, of ASCII atom numbers of the two ligands, and, where the net charge changes,
the alchemical water of both legs with its atoms and its ion (none where it does not); a record of another form
leaves its edge out, with the reason, and the analysis of the other edges goes on. The legs of an edge are taken only
with the record they were made from (the digest each leg records, `prepared_systems_sha256`). The ligands of the
edge are those of its records (`ligand_a.json`, `ligand_b.json`), else those of the input of the run that the record
was made from (its digests), for every reader (the analysis, the checks before a run, the hot atoms of REST2): the
run writes the record of a ligand that is missing again, that one alone, with its partial charges; the analysis
reads the SDF of the run. A custom core mapped by an earlier rule of cuFEP is prepared again. An edge whose record of
its preparation is lost
while its legs hold results is refused: it is prepared anew only when its directory is moved away.

### Hybrid topology

The rules of the HybridTopologyFactory of OpenFE's relative protocol (openfe 1.13, from perses), with
`turn_off_core_unique_exceptions` (its default since 1.11), quoted in `src/cufep/rbfe/hybrid.py`:

- mapped (core) atoms take the parameters of A at lambda 0 and of B at 1: charges, Lennard-Jones (sigma and
  epsilon of every pair interpolated linearly between its values in A and in B), and bonds, angles and
  torsions of core atoms;
- atoms unique to A lose their charges in the first half of lambda and their Lennard-Jones in the second,
  atoms unique to B gain their Lennard-Jones in the first half and their charges in the second (OpenFE's
  default lambda functions); the Lennard-Jones of a unique atom goes with the soft-core of Gapsys, Seeliger
  and de Groot (J. Chem. Theory Comput. 2012, 8, 2373), alpha 0.85; charges have no soft-core;
- a unique atom keeps all its bonded terms at full strength at both ends (a dummy), in both legs alike; the
  atoms unique to A and those unique to B never interact; where a mapped atom carries unique atoms of both
  ligands, the dummy atoms there have the terms of the next section instead (cuFEP's departure from OpenFE);
- a pair of core atoms that is an exception of the force field (1-2, 1-3 or 1-4) at one end only, which happens
  only where the atoms between them change (a linker of another length, a ring opened or closed), has its
  charge product and Lennard-Jones interpolated linearly between the exception at that end and the normal pair
  at the other, as OpenFE's hybrid;
- a bonded term that a System has twice counts twice (OpenFE keeps one of them); the bonds, angles and torsions
  of the environment are those of A, and an edge whose B has other ones is refused.

How the energy is carried (cuFEP's own; the energies are those of the rules): the NonbondedForce, which the
cuPotential plugin replaces, holds only the charges of the ligand, as particle parameter offsets under the
three electrostatic lambdas and REST2 (`spikes/m36`: the plugin gives the differences between states within
0.037 kT of a System with the charges written in). The Lennard-Jones of the ligand with its surroundings is a
CustomNonbondedForce on the stock kernels, as in the ABFE, and every pair inside the ligand is computed
explicitly, without cutoff, as the ABFE computes them (`docs/protocol.md`, "Interactions inside the ligand").

### Dummy atoms where both ligands have unmapped atoms on one atom

The problem. In ejm_42~ejm_54 of TYK2 the carbonyl carbon C17 is mapped and carries the unmapped ethyl of
ejm_42 (C19) and the unmapped NH-ethyl of ejm_54 (N29). With every term of a dummy kept, the dummy C19 keeps at
lambda 1 its angle to O18 and its dihedrals X-N16-C17-C19 across the urea C-N bond of ejm_54: when ejm_54 turns
about that bond, the dummy turns with it against its dihedrals. The torsional barrier of that bond rose from
53.6 kJ/mol (ejm_54 alone) to 88.4, and the solvent leg (5 ns) never reached the conformer of ejm_54 with the
hydrogen bond N29-H28...N13 (1.8 to 2.2 A) that the leg of the earlier mapping, and the ABFE of ejm_54, reached
(`spikes/out/m44`, not tracked). Such an atom is common: 57 of them in the prepared edges of TYK2 and A2A (a ring
C-H against C-F, C-Cl or C-O, CH3 against CF3, the amide carbonyl, sp3 carbons with another substituent).

The rules, from M. Fleck, M. Wieder and S. Boresch, "Dummy atoms in alchemical free energy calculations",
J. Chem. Theory Comput. 17, 4403-4419 (2021), quoted in `src/cufep/rbfe/junction.py`. At the end where a ligand
is absent, its unmapped atoms bonded to the mapped atom X (through one atom each, the dummy bridge atom D) are
anchored to the mapped neighbours of X alone (the unmapped atoms of the other ligand have no term with them, sec.
2.4), and the number of those neighbours is the type of the junction:

| mapped neighbours of X | junction (Fleck 2021) | terms of the dummy atoms with mapped atoms other than X |
|---|---|---|
| 1 | terminal (sec. 2.3.1) | every angle kept; of the dihedrals Z-P-X-D, those from one atom Z kept, the others deleted |
| 2 | dual, bond, angle and dihedral (sec. 2.3.2, TOL2PYR-2) | the angle P-X-D kept and the other deleted; one dihedral D-X-P-Z with periodicity 1 and 100 kcal/mol, its minimum where the dummy is, every other dihedral D-X-P-Z deleted; dihedrals D'-D-X-P of the deeper dummy atoms to that P only; terms with dummy atoms of two groups at X deleted; where no anchor has a mapped neighbour other than X, the two angles P-X-D at 90 degrees with 100 kcal/(mol rad^2) instead |
| 3 | nonplanar triple (sec. 2.3.3) | the three angles P-X-D with 3.55 kcal/(mol rad^2) (the value of its calculations) and equilibrium values adapted to the ligand of that end; every dihedral D-X-P-Z deleted; D'-D-X-P to one P only |

The force constants are CHARMM's, whose angle energy is K (theta - theta0)^2: in OpenMM's (k/2)(theta - theta0)^2
they are k = 2K (its dihedral energy is OpenMM's). With them the dummy atoms have three nonredundant terms each ("The derivation of eq 2 relies
crucially on using exactly three, nonredundant degrees of freedom (three bonded energy terms) per dummy atom",
sec. 2.1.1) and add the same factor to the free energy of both legs. Checked
(`tests/unit/test_rbfe_hybrid.py`): with the dummy atoms at their minimum, what the hybrid adds to the ligand of
an end is the same wherever that ligand's atoms are and at every angle of the bonds of X; the earlier hybrid
added from 24.97 to 59.74 kJ/mol over the angles of the amide C-N bond of N-methylpropanamide into
1,3-dimethylurea, and 4.6 to 39.4 kJ/mol on ejm_42~ejm_54.

cuFEP's own choices, where the source is silent: of the two anchorings the source gives a dual junction, the
bond, angle and dihedral, which leaves the dummy where it is (the two angles at 90 degrees move a group out of
the plane of X along lambda, which "may reduce phase-space overlap and necessitate additional λ-states", sec.
2.1.2: on 99afe76 the windows of short legs of toluene into phenol no longer overlapped); its P and Z are the
first, by the index of the hybrid, with a Z bonded to more than one atom ("problems may arise if the 'anchor'
of the dihedral angle term is itself a rotamer, such as a methyl group"); improper torsions with D and a mapped
atom other than X are deleted at a dual and a triple junction (terms beyond the three); the one P or Z of a
triple and a terminal junction that keeps its dihedrals is the atom of the lowest index; the adapted angles of a triple junction are those of the direction of D that fits the
three angles of the force field best, with X and its neighbours where the other ligand has them (Fleck 2021
minimises the molecule with the dummy at 0.05 kcal/(mol rad^2)); along lambda each changed term goes linearly
from the force field's term, where its ligand exists, to the dummy's in a narrow range about the middle of the
leg: those of A as x goes from 0.4 to 0.5 (`lambda_junction_delete`), those of B from 0.5 to 0.6
(`lambda_junction_insert`). From 0 to 0.4 and from 0.6 to 1 each ligand then turns about the bonds of X as it would
alone, with the dummy atoms of the other on their own terms, next to the hot windows of REST2, and a conformer
reached in a hot window is carried to the end by the replica exchange. Changed with the steric lambdas (99afe76,
8e2a945), the two forms of the terms of a group, which anchor it to different neighbours of X, were both partly on in
most windows: in ejm_42~ejm_54 the urea C-N bond of ejm_54 could not turn in any window but the last (a torsional
barrier of 220 to 640 kJ/mol in the windows 1-3 and 5-7, the group turned with the bond), and its solvent leg of 5 ns
never reached the conformer with the hydrogen bond N29-H28...N13; with the junction lambdas the barrier divided by
the REST2 scale is 44 to 50 kJ/mol in the windows from 0.62 to 0.88, against 53.6 at lambda 1. A mapped atom with unmapped atoms of one ligand only keeps OpenFE's rule, and an edge
without such an atom has the System of the earlier version. Legs of an earlier version of the hybrid
(`run.hybrid_version` below 3) at an edge with such an atom are left out of the analysis, and a run refuses to
continue them, with the command that computes them anew.

### Lambda windows

Adaptive lambda scheduling by the rule of Midgley et al. (J. Chem. Inf. Model. 2025, 65, 512), which is about
relative transformations: a precalculation starts from 9 evenly spaced windows of 101 candidates and adds a
window where neighbouring windows overlap by 0.04 or less (the rule of `docs/protocol.md`, "Adaptive lambda
scheduling"). Each leg has its own (`lambda_schedule.adaptive.legs: each`): 50 ps in each new window in the
solvent leg (the paper's), 500 ps after 500 ps in the complex leg (the values of the ABFE), the windows of an
iteration at the same time on the GPUs of the leg. The paper decides the windows of both legs from the ligand
in solution (`legs: solvent`); in the TYK2 example that left a pair of windows of a complex leg at an overlap of
0.013. A schedule is used only with the REST2 scale it was made with: a run compares the scale of every schedule its
legs would use, those a complex leg takes from the solvent leg included, before it writes anything. A complex leg
takes the windows of its solvent leg again after `--reschedule` of the complex leg alone: windows of the solvent leg
of another scale are made usable by setting that scale (a refusal writes it with every digit it needs), or by
choosing them anew first (`--leg solvent --reschedule`). With `--reschedule` of both legs the complex leg takes only
the windows its solvent leg publishes in that run (`windows_of_the_run.json` beside them), not those it set aside. `mode: fixed` uses the windows of `windows` (11 evenly spaced, OpenFE's
number). Windows that a leg does not choose itself all start from the end of its pre-equilibration, minimised
in their own state (OpenFE minimises every replica, 5000 steps) and 10 ps of MD in it, all at the same time on
the GPUs of the leg; the equilibration of the replica exchange then relaxes them further. (At lambda 0 the atoms
unique to B do not interact, so water can move where they are; a window in which they interact must not start
from that without a minimisation: cuFEP's first version did, and those windows went to NaN.)

### Sampling

The pre-equilibration of the hybrid System at lambda 0 (ligand A, the unique atoms of B dummies): in the
solvent leg NVT, NPT and 0.5 ns of MD; in the complex leg the solute held in its pose through NVT and NPT (and
through `membrane.equilibration_ns` of a new bilayer), without free MD, as OpenFE starts its relative legs from
the minimised input pose (the settings of `sampling.pre_equilibration` of the ABFE). Then
Hamiltonian replica exchange with REST2 on the whole hybrid ligand and the pocket (the settings of
`sampling.rest2` of the ABFE): 1 ns of equilibration and 5 ns of production per window, the defaults of
OpenFE's relative protocol. One calculation per edge (`sampling.repeats: 1`).

`sampling.rest2.region: perturbed` (a choice; the default is `pocket`) puts only the atoms around the perturbation
of the hybrid ligand into the hot region, in both legs, and no protein residue (`residues` may add some): the default
hot atoms of the relative calculations of the public benchmark, by the rule its help states ("The perturbed atoms are
grouped into sets in which the atoms are contiguously connected ... The process stops when the next addition would
exceed the maximum number of heavy atoms, which is 24 if the hot region does not contain a ring, or 30 if the hot
region contains a ring"), quoted in full in `src/cufep/rbfe/hot_region.py` with cuFEP's choices where it is silent (the
perturbed atoms are the unique atoms and the mapped atoms whose element changes; a rotatable bond is a single bond
outside a ring between two heavy atoms with another heavy neighbour each, as RDKit counts them; hydrogens follow their
heavy atom). The ligand is then hot only in part: a charge of a hot atom carries sqrt(s) and one of a cold atom does
not (the NonbondedForce, which the plugin carries, has electrostatic lambdas of each kind), and every pair inside the
ligand and with the environment carries the factor of its hot atoms (s for two, sqrt(s) for one), every proper
torsion of four hot atoms s, as REST2 scales them (Wang, Friesner and Berne 2011). A bond of two mapped atoms is
read in the end state that has it (a bond that closes a ring in B is a bond to a ring). The charges of an alchemical
water
are cold whatever the region. The energies of every state equal those of a System with the scaling written into its
parameters (`tests/unit/test_rbfe_rest2_region.py`), on both backends (`tests/gpu`).

Grand canonical moves of water in the complex leg (`sampling.gcmc`, on by default here and off in the ABFE): the
relative calculations of the public benchmark of Ross et al. (Commun. Chem. 2023) ran "All complex leg simulations
... in the muVT ensemble whereby water molecules were sampled with grand canonical Monte Carlo; all solvent
simulations were run in NPT". The moves are those of the ABFE (`docs/protocol.md`, "Grand canonical moves of
water"; `src/cufep/sampling/gcmc.py`), with their settings: insertions and deletions of water in a sphere around the
ligand, the waters farthest from the ligand ghosts from the pre-equilibration on (which has its barostat), and the
replica exchange without a barostat in the mean cell of the pre-equilibration (in a complex leg without free MD
its last cell). For a hybrid (cuFEP's own): the radius of the sphere is from the heavy atoms of
both end states, so that A -> B and B -> A have one radius, and the centre of the sphere of a move is the centre of
the heavy atoms of the hybrid ligand (the atoms heavy in either end state, by the elements of both: an atom of A
mapped onto a hydrogen of B is one of them both ways) in the configuration of that move, the same atoms both ways: one configuration has one sphere (the two hybrids start from other configurations,
the mapped atoms where A or B has them), each molecule taken whole along its bonds (those of the hybrid System, which its topology
does not all hold) and the two end states put together by their centres, so that a ligand longer than half the cell
has the sphere of the molecule whatever atom comes first; the ghosts of the start are the waters farthest from the
centre of the hybrid with each mapped atom midway between its positions in the two end states, the same waters both
ways (the hybrid holds the mapped atoms where A has them; earlier versions counted from that centre, and a leg whose
pre-equilibration or samples they made keeps those ghosts: where the record of the pre-equilibration names no
centre and has them, or without a record where the leg goes on from its samples; a leg that goes on from its
samples has the ghosts whose System has the checksum its samples recorded, whatever its record), checked against the
sphere around the centre they were counted from (the one the record names, else every centre whose rule gives
them, of which one suffices: the centre of the hybrid and the heavy atoms of A, the rule of the full TYK2 run of
08d58d5; the midway positions and the atoms heavy in either end, this version); ghosts chosen anew, where the leg has
none of these, are checked around the centre of this version alone (legs of other commits of the pull request that
brought these moves to the relative legs are not taken up); the ghost
waters a pre-equilibration was made with are recorded beside it (`equil_ghosts.json`, with the centre: a set of
numbers of the waters of the leg, each once); a record that is not that is a
pre-equilibration whose ghosts are not known, and it is read only by a leg that would start from it; a
leg that would start its replica exchange from a pre-equilibration of other
ghosts or without that record (a relative leg does not take one on trust), or from start configurations in another
cell than the pre-equilibration fixes, is refused before the run writes anything, with the command for that leg:
`--reschedule` sets such a pre-equilibration aside with the windows and samples and computes the leg anew; a leg that
goes on from its samples, or is analysed again from complete ones, does not start from its pre-equilibration and is
not refused for it; a checkpoint whose replicas are in another cell than the pre-equilibration fixes is not
continued, and only start configurations the leg uses are compared (an archive that `--reschedule` began and did not
complete is completed by the next `--reschedule`, with what it set aside, and refuses a run without it); a water
switched off has no charge and no Lennard-Jones in the NonbondedForce
and no Lennard-Jones with the ligand at either end state; in the stand-in energy with which the trials are screened the
ligand interacts with the parameters of both ends at full strength in every window, and the test with the full
potential of the window that every trial that passes is given corrects it (what the stand-in is changes how many
trials pass, not what is sampled). The solvent leg runs at constant pressure. With a water model for which cuFEP
has no excess chemical potential and standard volume (`docs/configuration.md`, `sampling.gcmc`) a relative
configuration is refused when it is read unless it gives those numbers or `sampling.gcmc.enabled: false`.

### Membrane proteins

`membrane.enabled` builds the complex leg in a POPC bilayer (Lipid21) as the ABFE does, with the membrane
barostat; the solvent leg is in water. Ligand B is moved with the rigid motion that the building of the
system gave ligand A. With a water of more than three sites the bilayer is built with the 3-site water of the
lipid patch and given the water of the run afterwards, as in the ABFE.

### Running the edges

On GPUs the legs of an edge start as soon as the edge is prepared, while the next edges are prepared (a complex
leg in a bilayer takes about two minutes to prepare). The record of the run (`resolved_config.yaml`,
`network.json`) is written when the first edge is prepared, before any job starts: a run whose first edge cannot be
prepared stops with nothing written, as one that is refused; an edge whose preparation stops after jobs of other
edges have started is a failure of the run (`failed_in_this_run`, exit code 1), and the other edges go on. Without
a GPU every edge is prepared before the legs run one after another.

## The network

- Edges: `network.edges`, or the LOMAP network (Liu et al., J. Comput. Aided Mol. Des. 2013; lomap2's
  `generate_lomap_network` with its default score and Kartograf, as the OpenFE benchmark of Baumann et al.
  2026 planned its networks). lomap2's rule (`GraphGen`): every pair is scored; the pairs of a score below
  1 - `distance_cutoff` (0.6) are left out; from the rest, edges are taken away from the lowest score up as long
  as every ligand stays in a cycle, no edge becomes the only link between two parts and no path between two ligands
  gets longer than `max_path_length` (6) edges; the parts that this leaves are then joined by the pairs of the
  highest score between them, a second one between each pair of parts for a cycle, whatever their score (of the 22
  edges of the TYK2 set, 8 join parts so, with scores of 0.25 to 0.37).
- Cycle closure: the sum of the ddG around every cycle of a cycle basis, with its standard error, as
  numbers without a verdict.
- The ddG that the network gives every edge (`network_ddg` in `summary.json` and `edges.csv`): dG(B) - dG(A) of the
  estimate below, with its standard error from the covariance of the estimate, what the cycle closure correction of
  Wang et al. (J. Chem. Theory Comput. 2013) gives an edge; an edge whose legs gave no ddG gets it too.
- The edges combined into cycle closures and dG must have been made under the same conditions (temperature,
  pressure, tail correction) from the same preparation besides their ligands (protein, force field, solvation,
  bilayer, pH ...), and a ligand of one name must be one molecule in all of them; otherwise each edge gives its
  ddG and the run says what differs, without cycle closures and without dG. `edges.csv` gives the lowest overlap
  of all iterations and that among the samples of the estimate, the two checks behind NOT RELIABLE.
- A leg records the version of its hybrid System (`run.hybrid_version`). Version 1, every leg whose record has
  neither this key nor `grand_canonical`, multiplied the term of A alone of the proper torsions of the core by the
  REST2 factor: a leg of it in which REST2 heated a window (`run.rest2.scale_of_window`), on an edge whose ligand B
  has terms in those torsions, is left out of the ddG as a missing leg; so is one for which this cannot be told (its
  hot region was `perturbed`, which version 1 chose by an earlier rule, or the prepared systems of its edge cannot
  be read). Its RESULT line says why and how to compute it again (`--edge ... --repeat ... --leg ... --reschedule`,
  that leg alone), and a note of the run names the edges. A run that continues a leg from its samples whose result
  is of version 1, or that has no result yet, first builds the System of this version and compares it with the
  checksum the samples recorded: the same (a leg without REST2) goes on, another is refused before the run writes,
  with that command. A leg computed from other prepared systems than the record of its edge (by its result or, stopped
  before its result, by the marker of its start or its error, which name them from this version on) is refused with
  or without `--reschedule`, which would keep the pre-equilibration made from them: the preparation they were made
  from is restored, or the edge is computed in a new `output.dir`. So is a leg that would start from a
  pre-equilibration whose positions are not of as many atoms as the System of the leg (a leg of main, which names no
  prepared systems, beside the preparation of another receptor), and `--reschedule` is told only where the
  pre-equilibration it keeps fits that System. `cufep benchmark report` takes the legs of a run directory
  as `cufep rbfe analyze` takes them, whatever results they have at the moment, not an earlier `summary.json`.
- dG of every ligand, from the edges alone: the maximum likelihood estimate of Xu (J. Chem. Inf. Model.
  2019, 59, 4720), by cinnabar's solver (OpenFE's `openfe gather --report dg`), edges weighted by the inverse
  of their variance. The edges fix the differences between the ligands; the one constant they leave is set
  as relative calculations set it:
  - with experimental values: the mean of the estimates of the ligands that have one is the mean of those
    values (the predictions of the public benchmark of Ross et al., Commun. Chem. 2023, and of the OpenFE
    benchmark of Baumann et al. 2026 are referenced so). The standard error of a ligand is that of its
    estimate against that mean;
  - without any: the mean of all estimates is 0 (cinnabar's output), dG relative to the mean of the network;
    the result says so.

## States of a compound

A compound may enter the network as several ligands, its states: binding modes or rotamers that do not interconvert
in the complex but do in solution, and protonation or tautomeric forms (`system.states`; without it every ligand is
a compound of its own). Each state is a node of the network and gets its dG from the edges; the dG of the compound
combines them (`cufep.rbfe.network.compound_dg`):

    dG(compound) = -kT ln sum_i P_i exp(-dG_i / kT) - kT ln s

- P_i, the population of the form of state i in solution (`solution_fraction`, 1 for a compound of one form): the
  combination of protonation states of Azimi et al. (J. Comput. Aided Mol. Des. 2022, eq 10, "The populations of
  each protonation state of the ligands and the corresponding standard binding free energies are combined"; for an
  acid of pKa at the pH of the assay P(A-) = alpha / (1 + alpha) and P(HA) = 1 / (1 + alpha), alpha = 10^(pH - pKa),
  eqs 12 to 14); the pKa correction of de Oliveira et al. (J. Chem. Theory Comput. 2019, 15, 424), whose full text
  could not be obtained, treats the same equilibrium. States of one form share its population; the populations of
  the forms of a compound add up to 1.
- Binding modes with P = 1: Ross et al. (Commun. Chem. 2023), Supplementary Equation 2, valid "as long as the binding
  conformations do not interconvert in the complex simulations but can readily interconvert in the solvent
  simulations".
- s, the symmetry number of the compound (`symmetry_number`, 1 by default): a group with rotational symmetry that
  turns freely in solution but not in the complex; Ross et al., Supplementary section 1.2.1: "cases like this need
  to have their binding free energy lowered by kT ln(2)".

The one constant the edges leave is set for the compounds (the mean over the compounds with an experimental value
is the mean of those values; the value of a compound is that of its states, which must agree), and the states get
the same constant. Standard errors from the covariance of the estimate, the states weighted by their share of the
sum. `compounds.csv` gives every compound with its states and their weights. An edge between states of two compounds
has the experimental ddG of those compounds, named so (`exp_ddg_of` in `summary.json` and `edges.csv`, "experiment of
the compounds" in its RESULT line): it is not what the edge computes; an edge between states of one compound has
none. In `ligands.csv` a state has the experimental dG of its compound, named so (`exp_dg_of`: "the compound 4j"),
one measurement for all its states. A run that takes only some states of a compound of the file combines those, and
says so. Which states a compound has, their populations and its symmetry number are inputs: cuFEP does not find
them. The relative benchmark of A2A takes them from the published map
(`benchmarks/a2a/rbfe_states.csv`, made by `cufep.benchmarks.curate` from the unmodified inputs: the 29 records, the
solvent pKa of Supplementary Table 17 at the pH of the assay, 7.5, and the symmetry numbers of Supplementary Table 16).

## Not in this version

- An edge that changes the net charge by more than one charge (refused before anything is written).
- A correction of the Lennard-Jones of the ligand with its surroundings beyond the cutoff: none, as OpenFE's
  relative protocol (`use_dispersion_correction: False` for the hybrid); the NonbondedForce of the
  environment keeps its own long-range correction.
- Water relocation moves (`sampling.relocation_mc`) and `membrane.complex_leg: water`, which are of the ABFE.

## All settings and their defaults

The sections `membrane`, `forcefield`, `backend`, `thermodynamics`, `dynamics`, `solvation` and `resources`
are those of the ABFE (`docs/configuration.md`); `sampling` and `analysis` too, with the defaults below.

```yaml
network:
  edges: null
  distance_cutoff: 0.4
  max_path_length: 6
  require_cycle_covering: true
mapping:
  atom_max_distance_A: 0.95
  map_hydrogens_on_hydrogens_only: true
  map_exact_ring_matches_only: true
  allow_partial_fused_rings: true
  allow_bond_breaks: false
  core_smarts: null
alchemy:
  softcore_alpha: 0.85
  alchemical_water_cutoff_nm: 0.8
lambda_schedule:
  mode: adaptive
  windows: 11
  adaptive:
    overlap_threshold: 0.04
    overlap_matrix: stage
    candidates: 101
    initial_windows: 9
    window_ps:
      complex: 500.0
      solvent: 50.0
    equilibration_ps:
      complex: 500.0
      solvent: 0.0
    sample_interval_ps: 0.5
    start_configurations: walk
    walk_ps: 5.0
    max_windows: 100
    legs: each
sampling:
  production_ns: 5.0
  equilibration_ns: 1.0
  gcmc:
    enabled: true
    interval_ps: 4.0
    attempts: 2500
    batch_size: 1000
    radius_margin_nm: 0.3
    ghost_waters: auto
    excess_chemical_potential_kcal_per_mol: auto
    standard_volume_nm3: auto
analysis:
  lrc: none
output:
  dir: runs/cufep_rbfe
```

## Basis of the recommended values

| settings | recommended value | basis |
|---|---|---|
| `mapping.core_smarts` | none | a custom core is a choice for a series, not a default: Kartograf's mapping as it is |
| `mapping.*` | Kartograf's defaults | Kartograf (Ries et al. 2024); the defaults of OpenFE's relative protocol (since openfe 1.7) |
| `alchemy.softcore_alpha` | 0.85 | Gapsys et al. 2012; OpenFE's `softcore_alpha` |
| `alchemy.alchemical_water_cutoff_nm` | 0.8 | OpenFE's `explicit_charge_correction_cutoff` |
| `network.*` | LOMAP: pairs of a score below 1 - 0.4 left out (but those that join the parts), every ligand in a cycle, paths of at most 6 edges | lomap2's defaults of `generate_lomap_network`, with which the OpenFE benchmark (Baumann et al. 2026) planned its networks |
| `lambda_schedule.adaptive.*` | 9 windows, 0.04, 101 candidates, the matrix of the stage | Midgley et al. 2025: the values and the rule of the paper, which is about relative transformations |
| `lambda_schedule.adaptive.legs` | each | the owner's choice of 2026-11-25: the paper's (`solvent`, the windows of the ligand in solution for both legs) left a pair of windows of the ejm_31 -> ejm_49 complex leg of TYK2 at an overlap of 0.013 (where a precalculation of the solvent leg of another run had added a window); each leg its own precalculation, as the ABFE |
| `lambda_schedule.adaptive.window_ps`, `equilibration_ps` | solvent 50 ps and 0; complex 500 ps after 500 ps | the solvent leg: the paper's; the complex leg: the values of the ABFE, measured on TYK2 (`docs/protocol.md`, "What was measured on TYK2") |
| `lambda_schedule.windows` | 11 | OpenFE's relative protocol (`lambda_windows`) |
| `sampling.production_ns`, `equilibration_ns` | 5 ns, 1 ns per window | OpenFE's relative protocol |
| `sampling.gcmc.enabled` | true (the complex leg) | the owner's decision (2026-11-26); the relative calculations of the public benchmark (Ross et al. 2023) ran their complex legs in the muVT ensemble with grand canonical moves of water. The other settings of the moves are those of the ABFE (`docs/configuration.md`) |
| `sampling.rest2.region: perturbed` | a choice, not the default | the default hot atoms of the relative calculations of the public benchmark (the rule of the help of its program, `src/cufep/rbfe/hot_region.py`); the default stays the region of the ABFE, the owner's choice of 2026-11-20 |
| `sampling.rest2` | the ABFE's: 1.5 in the middle of a leg, the ligand and the pocket | REST2 (Wang, Friesner and Berne 2011) for relative binding free energies: Wang, Berne and Friesner, PNAS 2012, 109, 1937; the scale and the region are those of the ABFE (`docs/configuration.md`) |
| `analysis.lrc` | none | OpenFE's `use_dispersion_correction: False` for the hybrid |
| the constant of the dG (no setting) | the mean of the experimental values, else the mean of the network | the dG of the public benchmark (Ross et al. 2023) and of the OpenFE benchmark (Baumann et al. 2026) are referenced to the mean of the experimental values; cinnabar gives estimates of mean 0 |
