# The cuFEP protocol

cuFEP computes absolute binding free energies by double decoupling with Boresch
restraints. The choices follow OpenFE's `AbsoluteBindingProtocol` unless noted; the
differences are listed at the end.

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

Membrane proteins are detected from the structure (a slab of exposed hydrophobic
residues crossed by helices or a barrel). The complex is oriented by a consensus of three
methods, embedded in a POPC bilayer with `Modeller.addMembrane`, and cavities inside the
protein at the depth of the bilayer are filled with water, because the builder adds bulk
water only above and below the lipids.

## Thermodynamic cycle

```
complex leg:  restraint on  ->  ligand charges off  ->  ligand Lennard-Jones off (soft-core)
solvent leg:                    ligand charges off  ->  ligand Lennard-Jones off (soft-core)

dG_bind = dG(solvent leg) - dG(complex leg) - dG(release)
```

`dG(release)` is the free energy of releasing the restraint on the decoupled ligand into
the standard-state volume. It is computed by numerical integration of the restraint's
partition function; Boresch's analytical formula is reported next to it as a check.

The ligand's own nonbonded interactions are kept at every lambda value, so the decoupled
ligand is the same in both legs and no vacuum leg is needed.

## Steps of one leg

1. Minimise and run plain MD of the fully interacting system on stock OpenMM
   (complex: 0.25 ns NVT, 0.5 ns NPT, 5 ns NPT; a bilayer system: restrained NVT/NPT,
   then at least 10 ns unrestrained, extended until the box area and the number of waters
   around the ligand stop drifting).
2. Complex only: choose the Boresch restraint from that trajectory. Ligand anchor: three
   bonded atoms of an aromatic ring near the centre of the molecule. Protein anchor: three
   bonded backbone atoms in stable secondary structure, RMSF below 0.1 nm, 0.5-1.5 nm
   from the ligand anchor. Candidates with a restrained angle near 0 or 180 degrees are
   rejected and the one with the smallest fluctuations is taken. Force constants:
   10 kcal/mol/A^2 and 80 kcal/mol/rad^2. If the ligand pose is not stable in this
   trajectory the run stops.
3. Find the lambda schedule (see below).
4. Hamiltonian replica exchange, one replica per state, 2.5 ps per iteration: 1 ns
   equilibration, then production.
5. MBAR on decorrelated samples.

Every repeat is a completely independent run of steps 1-5 with a different seed. The
reported uncertainty is the standard deviation over repeats (three by default).

## Adaptive lambda schedule

The schedule is found per leg, without energy derivatives. Starting from the coupled
state, a short simulation is run in the current state and the next state is placed where
the standard deviation of the energy difference to it reaches 1 kT; the interval is also
checked from the other side and split if it is too wide. After the equilibration phase
the MBAR overlap of every neighbouring pair is checked and a state is inserted where it
is below 0.03. The final schedule is written to `schedule.yaml`; samples collected
before the schedule was final are not used. A later run reuses the file unless
`--reschedule` is given.

## Production length

Production is checked only at pre-declared lengths (5 ns and 10 ns per state by default).
A leg stops at the first length where each state has at least 50 effective samples, the
replicas have made at least two round trips, and no neighbouring overlap is below 0.03.
The free energy estimate itself never decides when to stop, and all repeats of a leg are
extended together.

## Long-range dispersion

During sampling the soft-core ligand-environment interaction is truncated at the cutoff.
The missing energy is evaluated explicitly on coupled-state configurations out to 2 nm
(plus a homogeneous estimate beyond) and its free energy is added at the coupled end
state of each leg. An isotropic tail correction would assume the same atom density
around the ligand in the protein and in water.

## Differences from OpenFE's protocol

| | OpenFE | cuFEP |
|---|---|---|
| compute backend | stock OpenMM | cuPotential plugin for the nonbonded force and integrator; stock OpenMM as fallback |
| lambda schedule | fixed (30 complex / 14 solvent states) | adaptive, about 30 states per leg for a drug-like ligand |
| restraint reference values | last frame of the pre-equilibration | circular mean over the trajectory |
| standard-state correction | analytical | numerical, analytical as a check |
| ligand-environment dispersion tail | isotropic | explicit, by reweighting the coupled state |
| production length | fixed (10 ns per state) | 5 ns, extended to 10 ns if the stated criteria are not met |
| membranes | prepared by the user | detected, oriented, built and hydrated automatically |

## Limits of version 0.1

Neutral ligands only. No cofactors yet. No symmetry correction (the ligand's graph
symmetry number is reported). Symmetric POPC-type bilayers only.
