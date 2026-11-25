# Relative binding free energies (`cufep rbfe`)

`cufep rbfe` computes the relative binding free energy (ddG) of every edge of a network of ligands and, from
the network, the binding free energy (dG) of every ligand. It shares with the absolute binding free energies
of `cufep run` the backends (the cuPotential plugin by default, stock OpenMM as the fallback), the
preparation of the protein, the bilayer and the water, the pre-equilibration, Hamiltonian replica exchange
with REST2, adaptive lambda scheduling, the GPU scheduler with NVIDIA MPS, and the analysis of a leg (MBAR
and its checks, `docs/protocol.md`, "Checks of the analysis"). Its YAML is its own.

```bash
cufep rbfe plan config.yaml      # ligands, edges and atom mappings, without simulating
cufep rbfe run config.yaml       # run (or resume) every edge
cufep rbfe analyze runs/rbfe     # the ddG and dG from the results the legs have; writes nothing
```

## Inputs

- `system.protein`: the protein PDB, prepared as for the ABFE (`docs/protocol.md`, "Inputs").
- `system.ligands`: one SDF with every ligand of the network, each with explicit hydrogens and its pose in
  the pocket, all in one frame (as the public benchmark gives them). The atom mapping is made from these
  poses.
- Experimental values (optional): `system.experiment` (a CSV with the columns `ligand` and `dg`, kcal/mol)
  or the SDF tags `r_exp_dg` or `exp_dg`. They are used for the reference of the dG and for comparison.
- Neutral ligands; an edge that changes the net charge is refused before anything is written (this version
  has no treatment of a charge change).

## The calculation of an edge

An edge A -> B has two legs, the complex and the ligand in solvent, each a transformation of A into B along
one coordinate lambda from 0 (A) to 1 (B): ddG = dG(complex) - dG(solvent), with the standard error
sqrt(se_complex^2 + se_solvent^2).

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
  atoms unique to A and those unique to B never interact.

How the energy is carried (cuFEP's own; the energies are those of the rules): the NonbondedForce, which the
cuPotential plugin replaces, holds only the charges of the ligand, as particle parameter offsets under the
three electrostatic lambdas and REST2 (`spikes/m36`: the plugin gives the differences between states within
0.037 kT of a System with the charges written in). The Lennard-Jones of the ligand with its surroundings is a
CustomNonbondedForce on the stock kernels, as in the ABFE, and every pair inside the ligand is computed
explicitly, without cutoff, as the ABFE computes them (`docs/protocol.md`, "Interactions inside the ligand").

### Lambda windows

Adaptive lambda scheduling as Midgley et al. (J. Chem. Inf. Model. 2025, 65, 512) set it for relative
transformations: a precalculation of the ligand in solvent starts from 9 evenly spaced windows of 101
candidates, runs 50 ps in each new window, and adds a window where neighbouring windows overlap by 0.04 or
less (the rule of `docs/protocol.md`, "Adaptive lambda scheduling"); the windows it ends with are those of
both legs. The complex leg waits for them after its pre-equilibration and starts every window from a walk
through them (`fixed_walk_ps` in each). `lambda_schedule.adaptive.legs: each` gives each leg a precalculation
of its own; `mode: fixed` uses the windows of `windows` (11 evenly spaced, OpenFE's number).

### Sampling

The pre-equilibration of the hybrid System at lambda 0 (ligand A, the unique atoms of B dummies), then
Hamiltonian replica exchange with REST2 on the whole hybrid ligand and the pocket (the settings of
`sampling.rest2` of the ABFE): 1 ns of equilibration and 5 ns of production per window, the defaults of
OpenFE's relative protocol. One calculation per edge (`sampling.repeats: 1`).

### Membrane proteins

`membrane.enabled` builds the complex leg in a POPC bilayer (Lipid21) as the ABFE does, with the membrane
barostat; the solvent leg is in water. Ligand B is moved with the rigid motion that the building of the
system gave ligand A.

## The network

- Edges: `network.edges`, or the LOMAP network (Liu et al., J. Comput. Aided Mol. Des. 2013; lomap2's
  `generate_lomap_network` with its default score and Kartograf, as the OpenFE benchmark of Baumann et al.
  2026 planned its networks): edges of score 0.4 or more, every ligand in a cycle.
- Cycle closure: the sum of the ddG around every cycle of a cycle basis, with its standard error, as
  numbers without a verdict.
- dG of every ligand: the maximum likelihood estimate of Xu (J. Chem. Inf. Model. 2019, 59, 4720), by
  cinnabar's solver (OpenFE's `openfe gather --report dg`), edges weighted by the inverse of their variance.
  The estimate fixes differences only; its reference (`dg.references`):
  - `experiment` (default): the estimates shifted so that their mean over the ligands with an experimental
    value equals the mean of those values; the predictions of the public benchmark (Ross et al., Commun.
    Chem. 2023) and of the OpenFE benchmark are referenced so. The standard error is that of a ligand
    against that mean.
  - `abfe`: the ABFE of the ligands `dg.abfe_ligands`, computed by `cufep run` in the same run (in
    `<output>/abfe`, with the settings of `dg.abfe` over its defaults), enter the same likelihood as values of
    their nodes (cinnabar's node data, Xu 2019): the dG are then absolute without any experimental value.
  Both references can be asked for at once.

## Not in this version

- An edge that changes the net charge (refused before anything is written).
- A correction of the Lennard-Jones of the ligand with its surroundings beyond the cutoff: none, as OpenFE's
  relative protocol (`use_dispersion_correction: False` for the hybrid); the NonbondedForce of the
  environment keeps its own long-range correction.
- Water moves (`sampling.relocation_mc`, `sampling.gcmc`) and `membrane.complex_leg: water`, which are of
  the ABFE.

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
  fixed_walk_ps: 50.0
  adaptive:
    overlap_threshold: 0.04
    overlap_matrix: stage
    candidates: 101
    initial_windows: 9
    window_ps: 50.0
    equilibration_ps: 0.0
    sample_interval_ps: 0.5
    start_configurations: walk
    walk_ps: 5.0
    max_windows: 100
    legs: solvent
dg:
  references:
  - experiment
  abfe_ligands: []
  abfe: {}
sampling:
  production_ns: 5.0
  equilibration_ns: 1.0
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
| `lambda_schedule.adaptive.*` | 9 windows, 0.04, 101 candidates, 50 ps, the matrix of the stage, the solvent leg decides | Midgley et al. 2025: the values and the rule of the paper, which is about relative transformations; `legs: each` is cuFEP's own (as the ABFE) |
| `lambda_schedule.windows` | 11 | OpenFE's relative protocol (`lambda_windows`) |
| `lambda_schedule.fixed_walk_ps` | 50 ps | as `lambda_schedule.fixed_walk_ps` of the ABFE |
| `sampling.production_ns`, `equilibration_ns` | 5 ns, 1 ns per window | OpenFE's relative protocol |
| `sampling.rest2` | the ABFE's: 1.5 in the middle of a leg, the ligand and the pocket | REST2 (Wang, Friesner and Berne 2011) for relative binding free energies: Wang, Berne and Friesner, PNAS 2012, 109, 1937; the scale and the region are those of the ABFE (`docs/configuration.md`) |
| `analysis.lrc` | none | OpenFE's `use_dispersion_correction: False` for the hybrid |
| `dg.references` | experiment | the dG of the public benchmark (Ross et al. 2023) and of the OpenFE benchmark (Baumann et al. 2026) are referenced to the mean of the experimental values; `abfe` is cuFEP's own |
