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
reads the SDF of the run. A custom core of version 2 of its rule (whose search could stop before a match) is taken where Kartograf, run
again with the settings of its record, gives the mapping it recorded. An edge whose record of its preparation is lost
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
  atoms unique to A and those unique to B never interact;
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

### Lambda windows

Adaptive lambda scheduling by the rule of Midgley et al. (J. Chem. Inf. Model. 2025, 65, 512), which is about
relative transformations: a precalculation starts from 9 evenly spaced windows of 101 candidates and adds a
window where neighbouring windows overlap by 0.04 or less (the rule of `docs/protocol.md`, "Adaptive lambda
scheduling"). Each leg has its own (`lambda_schedule.adaptive.legs: each`): 50 ps in each new window in the
solvent leg (the paper's), 500 ps after 500 ps in the complex leg (the values of the ABFE), the windows of an
iteration at the same time on the GPUs of the leg. The paper decides the windows of both legs from the ligand
in solution (`legs: solvent`); in the TYK2 example that left a pair of windows of a complex leg at an overlap of
0.013. `mode: fixed` uses the windows of `windows` (11 evenly spaced, OpenFE's
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
read in the end state that has it (a bond that closes a ring in B is a bond to a ring); version 1 of the rule read it
in A, and a leg whose hot atoms the two versions choose otherwise records the version with its settings. Samples
of such a leg that record no version are continued where the System they record is that of the hot atoms of this
rule (versions 48eea8c to e0f9743 chose them and recorded no version), and refused before a run writes anything
where it is that of version 1; their free energy, whose end states are not scaled, stays in the analysis. The charges of an alchemical water
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
them, of which one suffices: the centre of the hybrid and the heavy atoms of A, up to e7af433; the midway positions
and the heavy atoms of A, e0f9743 to 1124b1e, whose records name "midway"; the midway positions and the atoms heavy
in either end, this version); ghosts chosen anew, where the leg has none of these, are checked around the centre of
this version alone. Where the edge maps a hydrogen of A onto a heavy atom of B, the atoms of the sphere of the moves
differ from those of earlier versions: the complex leg records them with its settings (`sphere_atoms`), and samples
of such a leg that record nothing are not continued (refused before a run writes anything); the ghost
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
  that leg alone), and a note of the run names the edges. `cufep benchmark report` takes the legs of a run directory
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
