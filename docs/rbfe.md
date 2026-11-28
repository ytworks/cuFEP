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
cufep rbfe analyze runs/rbfe --write   # ... and writes edges.csv, ligands.csv and summary.json again, under the
                                       # lock of the directory; refused where the summary holds the ddG of an
                                       # edge, or repeats for which it refused the edge one, and the directory
                                       # not every file that was made from (both legs of each such repeat, the
                                       # record of the prepared edge, the record of the samples where the
                                       # summary has the water moves of a leg from it), or where the summary
                                       # cannot be read and the legs do not give the values of the two tables.
                                       # A leg that a run began and did not finish has no result, and its
                                       # repeat is left out, as by a run
```

`cufep rbfe run` ends with exit code 1 when a leg of this run failed or has no free energy (windows that its
samples do not connect), 0 otherwise (also when some edges have no
ddG yet because only some legs were run, `--leg`). A run in a directory that holds results is refused before it
writes anything if a prepared edge of the network (one it runs, or one whose results it combines) was made
from other inputs or settings (protein, ligands, force
field, solvation, bilayer, cutoff, ionic strength, pH ...), or if a leg it runs has a pre-equilibration, lambda
windows or samples it cannot go on from: made with other settings (the grand canonical moves of water and their
values of the water among them), by a mover of another rule (samples of versions before 2026-11-28, whose moves
left the velocities of an inserted water as they were, are analysed as they are and not extended), or of another
System than this version builds (its checksum; the hybrid System of every leg with samples is built for that
before the run starts, several legs at a time). A leg stopped at the checkpoint of the start of its replica
exchange, before its first sample, is held to the conditions recorded with that checkpoint in the same way where
its start configurations have to be made again. A run is also refused if results that it would combine at its end
without sampling them (those of the edges or legs it does not run) are not what the directory made them from any
more, so that they would be combined without the comparisons made before: the record of the samples from which
the summary has the grand canonical moves of water of a complex leg (a result of an earlier version has them
there) was set aside; the result of a leg that the summary read is gone (not where a run began the leg again and
did not finish it: its `started.json` or `error.json` is there); or an edge has results and no record of its
prepared systems. Each such leg is held to the summary by itself, whatever the other leg of its repeat has now,
and only the repeats that the run combines (`sampling.repeats`); every result the run will combine must be one
that can be read. Where the summary of the directory cannot be
read, `edges.csv` is the record of the edges the run does not run: their legs must give the ddG it holds. The
directory, its record of the configuration and the results of its legs then stay as they were.

## Inputs

- `system.protein`: the protein PDB, prepared as for the ABFE (`docs/protocol.md`, "Inputs").
- `system.ligands`: one SDF with every ligand of the network, each with explicit hydrogens and its pose in
  the pocket, all in one frame (as the public benchmark gives them). The atom mapping is made from these
  poses.
- Experimental values (optional): `system.experiment` (a CSV with the columns `ligand` and `dg`, kcal/mol)
  or the SDF tags `r_exp_dg` or `exp_dg`. They set the one constant of the dG that the edges leave, and are
  given next to the results.
- Neutral ligands; an edge that changes the net charge is refused before anything is written (this version
  has no treatment of a charge change).

## The calculation of an edge

An edge A -> B has two legs, the complex and the ligand in solvent, each a transformation of A into B along
one coordinate lambda from 0 (A) to 1 (B): ddG = dG(complex) - dG(solvent), with the standard error
sqrt(se_complex^2 + se_solvent^2). The legs of all repeats of an edge must have been made under the same
conditions (temperature, pressure, prepared systems, correction of the dispersion tail: the check of the ABFE),
and each must have its standard error; otherwise the edge has no ddG, says why, and the network is made from the
other edges. The end states A and B are each built in a force field that holds its own ligand alone (two ligands
of one bonded graph in one force field would share the template of one, with its charges).

### Atom mapping

Kartograf (Ries et al., J. Chem. Theory Comput. 2024) maps the atoms of the two poses: atoms at most 0.95 A
apart, hydrogens only onto hydrogens, whole rings only and no ring that changes, no bond broken; its defaults,
which are those of OpenFE's relative protocol. Then OpenFE's rule from perses: where a constrained bond
between two mapped atoms changes its length, or is constrained at one end only, its hydrogen leaves the
mapping.

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

Grand canonical moves of water (`sampling.gcmc`, on by default in the complex leg; the owner's decision of
2026-11-28): the moves of the ABFE (`docs/protocol.md`, `src/cufep/sampling/gcmc.py`), water inserted into and
deleted from a sphere around the heavy atoms of the hybrid ligand (those of A and of B, so that the sphere is one
whichever way the edge is run), in equilibrium with bulk water. The waters farthest from the ligand are ghosts from
the start; the pre-equilibration, with its barostat, settles the cell without them, and the windows and the replica
exchange keep the cell of its last configuration, without a barostat. A ghost has no charge and no Lennard-Jones in
the NonbondedForce and in the Lennard-Jones of the hybrid ligand with its environment. The trials are first
evaluated with a stand-in energy in which each atom of the hybrid ligand has its charge and Lennard-Jones at the
lambda of the window; the one that passes is tested with the full potential of the replica. The solvent leg has no
moves. The windows a leg does not choose itself make the moves in the MD of their start too, on workers and in the
process of the leg alike. `auto` gives the excess chemical potential and the standard volume of the water only at
the conditions they were determined at (298.15 K, 1 bar, a cutoff of 0.9 nm; `docs/configuration.md`): a run under
others gives its own numbers or switches the moves off. A directory of a version before the moves, run again with
its YAML, is refused before anything is written (the pre-equilibration of its complex legs was made without
ghosts), and says that `sampling.gcmc.enabled: false` continues it as it was made.

### Membrane proteins

`membrane.enabled` builds the complex leg in a POPC bilayer (Lipid21) as the ABFE does, with the membrane
barostat; the solvent leg is in water. Ligand B is moved with the rigid motion that the building of the
system gave ligand A. With a water of more than three sites the bilayer is built with the 3-site water of the
lipid patch and given the water of the run afterwards, as in the ABFE.

## The network

- Edges: `network.edges`, or the LOMAP network (Liu et al., J. Comput. Aided Mol. Des. 2013; lomap2's
  `generate_lomap_network` with its default score and Kartograf, as the OpenFE benchmark of Baumann et al.
  2026 planned its networks): edges of score 0.4 or more, every ligand in a cycle.
- Cycle closure: the sum of the ddG around every cycle of a cycle basis, with its standard error, as
  numbers without a verdict.
- The edges combined into cycle closures and dG must have been made under the same conditions (temperature,
  pressure, tail correction, and the grand canonical moves of water of their complex legs: with or without them,
  the same excess chemical potential and standard volume of water, as the samples record them) from the same
  preparation besides their ligands (protein, force field, solvation, bilayer, pH ...), and a ligand of one name
  must be one molecule in all of them; otherwise each edge gives its ddG and the run says what differs, without
  cycle closures and without dG. The repeats of an edge whose complex legs were sampled with other moves give no
  ddG. Whether a complex leg made the moves is known of every result (a result of an earlier version with the
  counts of the moves made them); where the values of the water are not recorded (such a result without its
  samples), the edge is compared by the moves alone and the summary names it and its repeats. The result of a
  complex leg, and with it every repeat and edge of the summary (`water_moves`), also records the rule of the mover
  for the velocities of an inserted water (`velocities_of_an_inserted_water`; none for samples of a version
  before 2026-11-28, `water_rule_not_recorded`; the rule of a result is read from the record of its samples
  alone, so that a result whose HDF5 parts were set aside is still combined). Edges of rules that are known to
  differ are combined into one network, and a note of the summary names the edges of each rule (what the earlier
  rule does to a ddG has not been measured). An edge without a recorded rule is named as one of which it cannot
  be told whether it was sampled under the rule of the edges that record one, which the note names too; a network
  none of whose edges records a rule has a note that says so.
- `edges.csv` gives the lowest overlap of all iterations and that among the samples of the estimate, the two
  checks behind NOT RELIABLE, the window with the fewest independent samples of each leg and the legs below the
  level of the analysis (`few_independent_samples`; Mey et al. 2020), and the LOMAP score of the mapping.
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

## Benchmarks

`cufep benchmark run tyk2|a2a --protocol rbfe --tier smoke|full` runs a built-in benchmark set (`benchmarks/`) by
relative calculations: the settings of the set's relative example (`examples/rbfe_tyk2`,
`examples/rbfe_a2a_membrane`), the protein and every compound of the set (TYK2 16, A2A 17 in the bilayer) on the
LOMAP network above (the edges of the example are not used), and the experimental values of the set's table;
`smoke` runs the first edge of the example with shortened sampling. `cufep benchmark plan NAME` shows the network
before anything runs: for TYK2 the 22 edges with which the OpenFE benchmark (IndustryBenchmarks2024, Baumann et
al.) computed this set, in the same directions. `cufep benchmark report NAME DIR --protocol rbfe` compares a run
with experiment and the published results (`rbfe_report.md`, `rbfe_report_ligands.csv`, `rbfe_report_edges.csv`,
two figures): the dG of the compounds and the ddG of the edges of every method on the same compounds and edges,
with 95 % bootstrap intervals (for the dG, each method centred on the compounds compared, in every resample), and
for every edge its LOMAP score, its windows, lowest overlap, fewest independent samples, the backend of its legs
and its flags (NOT RELIABLE, few independent samples). The report is made from `summary.json` of the run, so it
can be made again from the files that are kept of a run (`benchmarks/results/`); of a summary of an earlier
version the legs with few independent samples are those the analysis noted in the repeats of each ddG (or, without
its notes, the numbers of the legs against the level of the run), and "not recorded" is said of a leg for which
the summary gives neither. A
directory that is not a run of the set named is refused before anything is written (`benchmarks/README.md`).

## Not in this version

- An edge that changes the net charge (refused before anything is written).
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
alchemy:
  softcore_alpha: 0.85
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
| `mapping.*` | Kartograf's defaults | Kartograf (Ries et al. 2024); the defaults of OpenFE's relative protocol (since openfe 1.7) |
| `alchemy.softcore_alpha` | 0.85 | Gapsys et al. 2012; OpenFE's `softcore_alpha` |
| `network.*` | LOMAP, score 0.4, every ligand in a cycle, paths of at most 6 edges | lomap2's defaults of `generate_lomap_network`, with which the OpenFE benchmark (Baumann et al. 2026) planned its networks |
| `lambda_schedule.adaptive.*` | 9 windows, 0.04, 101 candidates, the matrix of the stage | Midgley et al. 2025: the values and the rule of the paper, which is about relative transformations |
| `lambda_schedule.adaptive.legs` | each | the owner's choice of 2026-11-25: the paper's (`solvent`, the windows of the ligand in solution for both legs) left a pair of windows of the ejm_31 -> ejm_49 complex leg of TYK2 at an overlap of 0.013 (where a precalculation of the solvent leg of another run had added a window); each leg its own precalculation, as the ABFE |
| `lambda_schedule.adaptive.window_ps`, `equilibration_ps` | solvent 50 ps and 0; complex 500 ps after 500 ps | the solvent leg: the paper's; the complex leg: the values of the ABFE, measured on TYK2 (`docs/protocol.md`, "What was measured on TYK2") |
| `lambda_schedule.windows` | 11 | OpenFE's relative protocol (`lambda_windows`) |
| `sampling.production_ns`, `equilibration_ns` | 5 ns, 1 ns per window | OpenFE's relative protocol |
| `sampling.gcmc.enabled` | true (the complex leg) | the owner's decision of 2026-11-28; the moves and their other settings are those of the ABFE, with their basis (`docs/configuration.md`), where they are off by default |
| `sampling.rest2` | the ABFE's: 1.5 in the middle of a leg, the ligand and the pocket | REST2 (Wang, Friesner and Berne 2011) for relative binding free energies: Wang, Berne and Friesner, PNAS 2012, 109, 1937; the scale and the region are those of the ABFE (`docs/configuration.md`) |
| `analysis.lrc` | none | OpenFE's `use_dispersion_correction: False` for the hybrid |
| the constant of the dG (no setting) | the mean of the experimental values, else the mean of the network | the dG of the public benchmark (Ross et al. 2023) and of the OpenFE benchmark (Baumann et al. 2026) are referenced to the mean of the experimental values; cinnabar gives estimates of mean 0 |
