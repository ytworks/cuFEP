# Configuration reference

A run is described by one YAML file. The principles:

- every setting can be changed in the YAML;
- every setting has a default, and the default is the recommended value (the basis of each
  one is given below);
- `auto` is used only where the value has to be computed from the input: restraint atoms
  (`restraints.selection: auto`; `interactions` computes them from an MD of the complex), GPUs,
  box shape, membrane orientation and the oversubscription factor. Everything else is a plain
  value.

Only `system.protein` and `system.ligand` are required. `cufep init --full` writes a file
with every setting, `cufep plan config.yaml` shows what a run would use, including the
lambda windows, and every run writes the settings it used to `resolved_config.yaml`.

## All settings and their defaults

A unit test keeps this block identical to the defaults in the code.

```yaml
version: 1
system:
  protein: protein.pdb            # required
  ligand: ligand.sdf              # required; several molecules = one calculation each
  cofactors: []
  ligand_names: null              # run only these records of the SDF
membrane:
  enabled: false                  # true builds a lipid bilayer; cuFEP never decides this itself
  orientation: auto               # auto | input | transfer | slab | memembed
  lipid: POPC
  padding_nm: 1.5
  equilibration_ns: 10.0          # of a bilayer that was just built; with restraints.selection: interactions the solute is held through it
forcefield:
  protein: amber14/protein.ff14SB.xml
  water: amber14/tip3p.xml        # tip3p, tip3pfb, opc3, spce, tip4pew, tip4pfb or opc of amber14/; a bilayer needs a 3-site one
  lipid: amber19/lipid21.xml
  ligand: openff-2.2.1
  ligand_charges: am1bcc          # am1bcc | am1bcc-nagl
backend:
  name: cupotential               # cupotential | openmm
  fallback: openmm                # openmm | none
  precision: mixed                # mixed | single | double (the plugin has no double: it runs in mixed then)
  platform: CUDA
thermodynamics:
  temperature_K: 298.15
  pressure_bar: 1.0
  ionic_strength_M: 0.15
  ph: 7.4
dynamics:
  timestep_fs: 4.0
  constraints: hbonds             # hbonds | none
  hmr_mass_amu: 3.0
  friction_per_ps: 1.0
  cutoff_nm: 0.9
  barostat_frequency: 25          # MD steps between barostat moves; no stretch of MD of one replica or window may be shorter
solvation:
  box_shape: auto                 # auto | cube | dodecahedron
  padding_complex_nm: 1.0
  padding_solvent_nm: 1.5
restraints:
  type: boresch
  selection: interactions         # interactions (from MD of the complex: the section below) | auto (the search of OpenFE) | manual (give host_atoms and guest_atoms)
  force_constants: openfe         # openfe (the two values below) | auto (from the fluctuations) | custom
  k_r_kcal_per_A2: 10.0
  k_angle_kcal_per_rad2: 80.0
  host_atoms: null
  guest_atoms: null
  rmsf_cutoff_nm: 0.1             # these four: the search of selection auto
  host_min_distance_nm: 0.5
  host_max_distance_nm: 1.5       # also, for every selection: the binding site in which ligand_rmsd_from_input_A is measured
  dssp_filter: true
  interactions:                   # selection interactions only
    trajectories: 4               # unrestrained NPT runs of the complex, each from the pose that was given (held while the solvent settles)
    length_ns: 3.0
    discard_ns: 1.0               # the first part of every trajectory that is not analysed
    frame_interval_ps: 10.0
    min_frame_fraction: 0.4       # an interaction is kept, and the anchor angles are in range, in this fraction of the frames
    angle_range_deg: [45.0, 135.0]
    stable_structure_fraction: 0.5      # a residue of stable secondary structure: in more than this fraction of the frames
    second_interaction_ratio: 0.6       # two interactions of a residue: the second in more than this fraction of the first's frames
    fallback_frame_fraction: 0.15       # without anchors from interactions: the backbone nearest to the ligand
    fallback_cutoff_nm: 1.0
alchemy:
  softcore_alpha: 0.5
lambda_schedule:
  mode: adaptive                  # adaptive | fixed; one value, or one per leg: {complex: adaptive, solvent: fixed}
  windows:                        # per stage: a number of lambda windows, or the lambda values
    complex:
      restrain: 6
      discharge: 11
      vanish: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    solvent:
      discharge: 5
      vanish: [0.0, 0.12, 0.24, 0.36, 0.48, 0.6, 0.7, 0.77, 0.85, 1.0]
  adaptive:
    overlap_threshold:            # one per leg, or one value; not given: 0.18 where the leg reads stage, 0.3 where pair
      complex: 0.3
      solvent: 0.18
    overlap_matrix:               # stage | pair: the overlap matrix the threshold is compared with; one per leg, or one value
      complex: pair
      solvent: stage
    candidates: 101               # one value, or one per leg
    initial_windows:              # per stage: a number of evenly spaced windows, or the lambda values (candidates)
      complex:                    # the recommended windows of mode fixed (the defaults of lambda_schedule.windows.complex)
        restrain: 6
        discharge: 11
        vanish: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
      solvent:
        discharge: 5
        vanish: 9
    window_ps:                    # MD per new window of the precalculation that is recorded; a number = both legs
      complex: 500.0
      solvent: 50.0
    equilibration_ps:             # MD in a new window before that
      complex: 500.0
      solvent: 0.0
    sample_interval_ps: 0.5
    walk_ps: 5.0
    max_windows:
      complex: 100
      solvent: 100
  fixed_walk_ps: 50.0             # fixed mode: MD in each window that gives the replicas their start configurations
  file: null                      # a schedule.yaml of an earlier run of the same leg (or a directory of them), used as it is
sampling:
  pre_equilibration:
    complex: {nvt_ns: 0.25, npt_ns: 0.5, md_ns: 5.0}    # npt_ns: twice as long for a bilayer; md_ns: selection auto and manual (interactions: no free MD here, its trajectories are that MD)
    solvent: {nvt_ns: 0.1, npt_ns: 0.2, md_ns: 0.5}
  exchange: hrex                  # hrex | none
  time_per_iteration_ps: 2.5
  equilibration_ns: 1.0           # a number, or {complex: ..., solvent: ...}
  production_ns: 10.0             # a number, or {complex: ..., solvent: ...}
  repeats: 1
  seed: 20261118
  checkpoint_interval_ps: 250.0
  rest2:
    scale: 1.5                    # temperature of the hot region / temperature of the run, mid-leg; 1.0 = off
    region: pocket                # pocket (also protein residues near the ligand) | ligand
    pocket_cutoff_nm: 0.4
    residues: []                  # protein residues added to the hot region, as in the input PDB: ["960", "A:981", "B:898A"]
analysis:
  estimator: mbar
  lrc: reweight                   # reweight | isotropic | none
  lrc_cutoff_nm: 2.0              # outer radius of the explicit tail; larger than dynamics.cutoff_nm
  min_overlap: 0.03               # NOT RELIABLE below this overlap of two neighbouring lambda windows
  convergence_points: 10          # points of the forward and reverse series
  convergence_blocks: 5           # consecutive blocks of the production with an estimate of their own
  min_effective_samples: 50.0     # message if the lambda window with the fewest uncorrelated samples has fewer
  error_blocks: [40, 20, 10, 5]   # numbers of time blocks for the standard error of a leg; the largest error is reported
  part_error_blocks: 5            # time blocks for the standard error of each part of the production (halves, series, blocks)
resources:
  gpus: auto                      # auto | a list of CUDA device numbers (as CUDA_VISIBLE_DEVICES leaves them)
  oversubscription_factor: auto   # auto | 1 | 2 | 4 | ...
  cpu_threads_per_worker: 2
output:
  dir: runs/cufep
```

## Lambda windows

A leg passes through stages: the complex leg through `restrain`, `discharge` and `vanish`,
the solvent leg through `discharge` and `vanish`. Within a stage lambda runs from 0 at its
start to 1 at its end, as OpenFE's `lambda_restraints`, `lambda_elec` and `lambda_vdw`:

| stage | lambda 0 | lambda 1 |
|---|---|---|
| restrain | no Boresch restraint | full restraint |
| discharge | ligand charges on | ligand charges off |
| vanish | ligand Lennard-Jones on | ligand Lennard-Jones off (soft-core) |

The window at the end of a stage is the window at the start of the next one.

`lambda_schedule.mode: fixed` uses `lambda_schedule.windows`. Each stage is given either as
the lambda values, an increasing list from 0.0 to 1.0, or as a number of windows. A number
re-samples the recommended values of that stage, so the relative spacing is kept: evenly
spaced stages stay evenly spaced, and the vanish stage stays denser towards its end.

```yaml
lambda_schedule:
  mode: fixed
  windows:
    complex: {restrain: 4, discharge: 8, vanish: 20}
    solvent: {discharge: 5, vanish: [0.0, 0.2, 0.4, 0.6, 0.75, 0.9, 1.0]}
```

`lambda_schedule.mode: adaptive`, the default, lets a short precalculation decide the
windows; `lambda_schedule.adaptive` holds its parameters. The rule is described in
[protocol.md](protocol.md#adaptive-lambda-scheduling). The windows it chooses are written to
`schedule.yaml` in the leg directory; copying them into `lambda_schedule.windows` reproduces
the schedule in fixed mode.

The precalculation starts from `lambda_schedule.adaptive.initial_windows`. Each stage is given
either as a number of evenly spaced windows or as its lambda values, an increasing list from
0.0 to 1.0. The values need not be evenly spaced, and each must be one of the `candidates`
(a multiple of 0.01 with 101 candidates). The schedule never has fewer windows than this
start. Where the rule adds a window it spaces the windows of that bad section evenly again, so
values of the start inside a bad section move; those outside every bad section stay as given.
A start the precalculation could not use is refused when the configuration is read: a value
that is no candidate, a number of windows that does not fit on the candidates, more windows
than `max_windows`.

By default the solvent leg starts from 5 and 9 evenly spaced windows and the complex leg from
the recommended windows of mode fixed, the defaults of `lambda_schedule.windows.complex`: 6
evenly spaced windows for restrain, 11 for discharge and the 15 values of the vanish stage, 30
windows in all. The start is a setting of its own: it does not follow a change of
`lambda_schedule.windows`. Its vanish stage is given as values, multiples of 0.05: with a number
of `candidates` on which they do not lie (they lie on a number one above a multiple of 20, as
21, 101 or 201), or with a `max_windows.complex` below 30, the start of the complex leg has to
be set as well; the refusal says that it is the default start.

```yaml
lambda_schedule:
  adaptive:
    initial_windows:
      complex: {restrain: 3, discharge: 5, vanish: [0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 1.0]}
```

`lambda_schedule.adaptive.overlap_matrix` says which overlap the rule compares with
`overlap_threshold`. `stage`, the default of the solvent leg, is the element of the overlap
matrix of all windows of the stage, as in the paper. `pair`, the default of the complex leg, is
the element of the overlap matrix of the two neighbouring windows alone: it lies between 0 and
0.5 and rises as the two windows get closer, whatever other windows the stage has. A threshold means different spacings under the two: the
table of the basis below gives both for harmonic windows. The analysis of a result always
checks the matrix of all windows of the leg (`analysis.min_overlap`), in which windows that are
much closer than they need to be read low: the result of a start with very many windows can
fall below that level and carry its verdict (protocol.md, "Why the threshold cannot go much
higher").

The complex leg and the solvent leg can run different window settings. `initial_windows`,
`max_windows`, `windows` and the two lengths of the precalculation have a value for each leg.
`mode`, `overlap_threshold`, `overlap_matrix` and `candidates` are one value for both legs, or
one for each leg (both legs must then be named). The defaults of `overlap_threshold` and
`overlap_matrix` are one for each leg, and the two belong together, since a threshold means
another spacing under each matrix:

- A leg whose `overlap_threshold` is not given has the threshold of the matrix it reads: 0.18
  for `stage`, 0.30 for `pair`. `overlap_matrix: stage` alone therefore gives both legs the rule
  and the threshold that were the default until 2026-11-22 (the complex leg then started from
  `initial_windows.complex: {restrain: 3, discharge: 5, vanish: 9}`).
- A threshold that is given is compared with the matrix of its leg. Without `overlap_matrix`
  that is the default of the leg: the pair alone in the complex leg. Files written before
  `overlap_matrix` existed meant the matrix of the stage in both legs, so `cufep` says in a note
  what such a file is read as; `overlap_matrix: stage` beside the threshold gives the earlier
  meaning.

```yaml
lambda_schedule:
  mode: {complex: adaptive, solvent: fixed}     # the solvent leg uses lambda_schedule.windows.solvent
  adaptive:
    overlap_threshold: {complex: 0.25, solvent: 0.18}
    candidates: 201                             # both legs
```

A leg keeps the `schedule.yaml` it has: after a change of these settings the precalculation of
a leg that was already scheduled runs again only with `cufep run --reschedule`.

## Basis of the recommended values

| settings | recommended value | basis |
|---|---|---|
| force fields, AM1-BCC charges, 4 fs with hydrogen mass repartitioning, 0.9 nm cutoff, 298.15 K, 1 bar, 0.15 M | as listed | defaults of OpenFE's ABFE protocol |
| `sampling.pre_equilibration`, `equilibration_ns`, `time_per_iteration_ps`, `exchange` | 0.25 + 0.5 + 5 ns; 1 ns; 2.5 ps; replica exchange | defaults of OpenFE's ABFE protocol. With 1 ns of MD instead of 5 ns (as SOMD2 uses) TYK2 ejm_31 gave -11.2 ± 1.5 kcal/mol over three repeats against -10.6 ± 1.9, the same within the spread; 5 ns is kept |
| `sampling.production_ns` | 10 ns per lambda window | default of OpenFE's ABFE protocol. A3FE and Boltz-ABFE use 5 ns. With 5 ns the lambda window with the fewest independent samples of a TYK2 complex leg held about 29 (11 of 55 legs reached the 50 that Mey et al. 2020 recommend for MBAR); see protocol.md, "Checks of the analysis". The two examples and the benchmark sets set 5 ns (the owner's choice, 2026-11-20); a set runs with the settings of its example (`src/cufep/benchmarks/registry.yaml`) |
| `sampling.repeats` | 1 | the requirement of the project: the accuracy of a binding free energy comes from one calculation per compound. OpenFE's default is 3; more than one repeat makes the reported uncertainty the standard deviation over repeats |
| `restraints` search settings and force constants, `alchemy.softcore_alpha` | as listed | defaults of OpenFE's ABFE protocol. `force_constants: auto` takes kT / sigma^2 of each restrained coordinate instead, as A3FE and SOMD2 do: over the frames the restraint is made from, which are those of the pre-equilibration for `selection: auto` and `manual` and the analysed frames of the MD for the restraint for `selection: interactions`. Anchors are held to OpenFE's check_angle_not_flat once, with the force constants the restraint gets (10 kT is 44 degrees from a straight line at 20 kcal/mol/rad^2, 22 at 80, 11 at 320): of every frame, of the configuration the run starts from, of the prepared pose of given anchors and of the reference angles of the restraint themselves. A candidate that breaks it is passed over, and given anchors that do are refused. The six coordinates of a restraint are measured as its force measures them: every displacement between two of its six atoms by OpenMM's periodic rule. On TYK2 ejm_31 (three repeats each) the two gave the same binding free energy within the spread of the repeats (-10.6 ± 1.9 and -10.8 ± 1.2 kcal/mol), so the fixed values of OpenFE stay the default |
| anchors of the Boresch restraint | OpenFE's rules | Two of the conditions of OpenFE's `EvaluateBoreschAtoms`: theta_A and theta_B stay, in every frame, where their restraint energy is at least 10 kT from that at 0 and at 180 degrees (`check_angle_not_flat`, 22 degrees with the default force constant), and anchors whose H2-H1-H0-G0-G1-G2 have three in a row on a straight line (|cos| of their bond vectors above 0.9, `is_collinear` from Yank) in every frame are refused. Rules of cuFEP's own of earlier versions (the same margin for the two other angles; 30 degrees in the last frame) are gone (owner, 2026-11-20). OpenFE's limits on the dihedrals and on the variances were never part of cuFEP |
| `restraints.selection` | interactions | the owner's choice (2026-11-21): the anchors of the restraint are chosen from the interactions between protein and ligand in unrestrained MD of the complex (the rows below). `auto` is the search of OpenFE's protocol, the default until then: the benchmark results of commit 3b45bfe and the example results recorded before 2026-11-21 were made with it. A restraint of one selection is not used by a run of another |
| `restraints.selection: interactions`: the MD (`restraints.interactions.trajectories`, `length_ns`, `discard_ns`, `frame_interval_ps`) | 4 trajectories of 3 ns, the first 1 ns left out, a frame every 10 ps | The method of Ding, Liu, Wang, Xia, Wang and Yan, "Development and large-scale benchmarks of a protein-ligand absolute binding free energy toolkit", arXiv:2603.22274 v3. Sec. 2: "We run an unrestrained molecular dynamics (MD) simulation in the isothermal-isobaric (NPT) ensemble and use the resulting trajectory to determine the anchor atoms and their reference geometries." Sec. 2.4: "Boresch restraint anchors are selected from four concurrent NPT trajectories at the target temperature. Each trajectory is 3 ns long, with frames saved every 10 ps. The first 1 ns of each trajectory is discarded, and the remaining frames are combined for anchor selection." As in the paper, which minimises and starts them at once, the trajectories start from the pose that was given; cuFEP's own is what lies between: the heavy atoms of protein and ligand are held by position restraints while the solvent and the box settle (0.25 ns NVT and 0.5 ns NPT; with a bilayer that cuFEP has just built 0.25 ns NVT, 1.0 ns NPT and then `membrane.equilibration_ns`, 10 ns, all of it with these atoms held), and no free MD precedes the trajectories (the owner's decision of 2026-11-21: the first version of this selection started them from the end of 5 ns, or for a bilayer 10 ns, of free MD, and two runs of the A2A example then chose their restraints from two different states of the ligand, ALA 59 and ASN 253). The trajectories differ by their random numbers only; they are a plain MD of the leg and run by the settings of the YAML that every plain MD of the leg has (`backend`, `thermodynamics`, `dynamics`, `resources`, the seed), on the backend of the leg with its fallback; the frames of the first part are left out by their time, which at 4 fs is 100 of the 300 frames of a trajectory. The alchemical run starts from the end of the pre-equilibration as with every selection, which here is the pose that was given (the source starts it from the minimised input too) |
| `restraints.interactions.min_frame_fraction`, `angle_range_deg`, `fallback_frame_fraction`, `fallback_cutoff_nm` (what decides) | 0.40; 45 to 135 degrees; 0.15; 1.0 nm | Sec. 2.1 of the same paper: "Candidate interactions are grouped by ligand anchor atom, protein residue, and interaction type, and retained if they occur in at least 40% of the analyzed frames." "Candidates are retained only if the L1-L2-L3 and P1-L1-L2 angles each lie within 45 to 135 [degrees] in at least 40% of all analyzed frames. These qualifying frames are counted within the merged interaction frames for the candidate." "If no interaction candidates survive filtering, or if they yield no valid anchor set, the algorithm falls back to a distance-based method. [...] For each frame, the closest pair of a selected ligand atom and a protein backbone atom is identified using minimum-image distances, with a 1.0 nm cutoff. [...] Candidate residues must account for at least 15% of the analyzed frames. Anchor triples undergo the same geometric filtering, using a 15% frame threshold." "Frame-count thresholds are rounded down to integers in both selection methods. If the distance-based method also yields no valid anchor set, the calculation stops with an error." So does cuFEP, before anything is sampled; a ligand without three bonded heavy atoms that can be guest anchors is refused before any MD, because neither method can give it anchors. The atoms: "Protein anchors are drawn from the N, C-alpha, and C backbone atoms of a single candidate residue, with all six orderings evaluated. The first ligand anchor, L1, is selected from the interacting atom or atoms within three bonds of it. Eligible L1 atoms are heavy atoms with at least two heavy-atom neighbors, excluding sulfonyl sulfur and oxygen atoms and carboxyl oxygen atoms. L2 is a nonterminal heavy atom bonded to L1, and L3 is a distinct nonterminal heavy atom bonded to L2." A threshold is the fraction of the frames that are analysed, rounded down, with the fraction taken as the shortest decimal that is the number the configuration holds (0.29 of 800 frames is 232; the source multiplies two binary numbers, which gives 231 there). The distance method looks for the nearest pair among every atom named N, CA or C of the protein, those of caps included, as the source does: a frame whose nearest atom belongs to a residue without the three anchors is a frame of no residue. A configuration in which one of the two thresholds would be no frame (fewer analysed frames than one over the fraction) is refused before anything is written: every interaction that occurs at all would be kept. The paper only rounds down; the refusal is the owner's decision (2026-11-21) |
| the ranking of the candidates (`stable_structure_fraction`, `second_interaction_ratio`, and constants of `cufep.alchemy.interaction_anchors`) | 0.5; 0.6; the constants of the source | The paper gives the order of what counts and two numbers: "We prioritize protein residues that maintain stable secondary structures, specifically alpha-helices, 3-10-helices, pi-helices, extended strands, or beta-bridges, in more than 50% of trajectory frames." "Candidates are ranked by the frequency with which the P1-L1-L2 and L1-P1-P2 angles lie in this range, with penalties for larger mean deviations from 90 [degrees]. The score also favors protein residues with stable secondary structure. A further bonus is applied to residues with at least two retained ionic, cation-pi, hydrogen-bond, or halogen-bond interactions. This bonus requires the second-most-frequent interaction to occur more than 60% as often as the most frequent interaction. Interaction priorities decrease from ionic interactions to cation-pi interactions, hydrogen bonds, halogen bonds, and aromatic stacking interactions. Ties are resolved by sorting based on atom indices." It gives no formula. The numbers are those of the published source of the paper (github.com/ByteDance-Seed/felis at commit d6dff91, `felis/protocols/boresch/anchor_scoring.py`, `anchor_constants.py`, `anchor_ligand.py`, `anchor_expand.py`, `anchor_distance.py`; no code of it is in cuFEP, `docs/LICENSING.md`): the count of frames of each of the two angles, multiplied by 1 while the two mean deviations from 90 degrees add up to 40 degrees or less and by 0.7 from 60 degrees on (a straight line between), raised by 3 levels for a stable residue, in 50 levels of max(10, frames // 50) frames below the largest count; plus the weight of the interaction (ionic 0.9, cation-pi 0.7, hydrogen bond 0.6, halogen bond 0.4, stacking 0.1) and 7 for the bonus; compared in hundredths. The distance method ranks by the two counts together in steps of 2% of the frames, then by the deviations. L1 is the first suitable atom in the order: the interacting atom, then the atoms one, two and three bonds away, those equally far in the order in which a breadth-first walk reaches them (the atoms bonded to an atom by ascending number; the source walks through a set there, whose order is not defined); the atoms that are never L1 are those of the patterns `[#16](=O)=O`, `C(=O)[O!R]` and `C(=O)[O-]` (the second also covers the oxygens of an ester outside a ring); "nonterminal" is an atom with more than one bonded atom, hydrogens counted. These constants order candidates and are no settings; they decide nothing by themselves. cuFEP's own, where the source leaves the order open: among the atoms equally far from an interacting atom, the walk goes through the atoms bonded to each atom by ascending number (the source walks a set there), so that the first suitable one is the first the walk reaches, not the one of the lowest number; and equal scores are ordered by the atoms of the anchors in both methods |
| the interactions (ProLIF) and the secondary structure of `restraints.selection: interactions` | ProLIF 2.2.3; DSSP of MDTraj | "ProLIF is used to characterize specific protein-ligand interactions." As in the source (`felis/external/prolif_tools.py`): the ten interactions HBDonor, HBAcceptor, Anionic, Cationic, XBAcceptor, XBDonor, CationPi, PiCation, EdgeToFace and FaceToFace, every occurrence, for the residues within 6.0 A, with ProLIF's own geometric parameters; an interaction belongs to the first ligand atom ProLIF names for it; the ligand with the chemistry of the SDF and RDKit's aromaticity. Differences, cuFEP's own: the ProLIF of the image is the unpatched release pinned in `docker/environment.yml` (the source patches 2.0.3 so that a charged atom bonded to one of the opposite charge is no ion; 2.2.3 has that in its own patterns, and no longer takes a backbone nitrogen for a hydrogen-bond acceptor); residues are told apart by their index in the topology (the source keys them by name and number, which two chains can share); the angles are measured with the ligand in the periodic image next to the protein, that of its nearest protein atom (the source measures them as the coordinates were written), and the distance method measures to the nearest image in any cell; the five secondary structures of the paper are counted by the DSSP of MDTraj (the source uses the DSSP of MDAnalysis, which tells helix, strand and loop apart only) |
| reference values and force constants of a restraint from interactions | the centre of the fullest of 20 bins; `restraints.force_constants` | The paper says that the trajectory determines "the anchor atoms and their reference geometries" and not how. The source (`filter_mask_histogram_plot.py`, `find_ideal_geometry.py`, `main_boresch_restraints.py`): over the frames in which both anchor angles lie within 45 to 135 degrees, a histogram of 20 bins per coordinate (0 to 180 degrees for the angles, -180 to 180 for the dihedral angles, the range of the data for the distance) and the centre of its fullest bin, rounded to 0.1 degree and 0.001 nm. cuFEP's own: the anchors that are chosen also pass the two rules of OpenFE that every restraint of cuFEP passes (the row above), over the analysed frames and at the configuration the alchemical run starts from; a candidate that does not is passed over and the result says how many were. The force constants and the form of the restraint stay those of `restraints.force_constants` (10 kcal/mol/A^2, 80 kcal/mol/rad^2, harmonic in all six coordinates). The paper: "For the harmonic distance and angle terms, the force constants are 2 kcal/mol/A^2 and 80 kcal/mol/rad^2, respectively. For the cosine-based dihedral term, the force constant is 80 kcal/mol." The distance constant of the paper is set with `force_constants: custom` and `k_r_kcal_per_A2: 2` Where the source is silent, cuFEP's own: a candidate without one frame in which both anchor angles are in range has no reference values and is passed over for the next one, as a candidate that breaks one of OpenFE's two rules is; the configuration the alchemical run starts from is held to the rule of every frame (the flat angle) and must define the dihedral angles, and the rule of a trajectory (not on a line in every frame) is asked of the analysed frames. The force constants stay those of `restraints.force_constants` for this selection too, 10 kcal/mol/A^2 for the distance where the source uses 2 (the owner's decision of 2026-11-21, on these numbers from the free frames of four runs of the examples: the distance fluctuates by 0.18 to 0.23 A, against a width of 0.24 A of the restraint with 10 and 0.54 A with 2; turning on the restraint in the bound state costs 1.5 to 2.5 kcal/mol, of which the distance term is 0.12 to 0.21 with 10 and 0.03 to 0.06 with 2; the angles and dihedral angles, 80 in both, are the rest) |
| distance of the ligand from its input pose | a number, no level | reported with the result of the complex leg (`ligand_rmsd_from_input_A`), as OpenFE reports the ligand RMSD without a limit. It is measured for every `restraints.selection` over the free MD that selection has (the second half of the frames of the pre-equilibration for `auto` and `manual`; for `interactions`, whose pre-equilibration holds the pose, the analysed frames of the four trajectories, of which at most 200 are taken, evenly spaced (800 analysed frames give 200; fewer than 200 are all taken), and all of those counted), superposed on the backbone atoms (N, CA, C) within `restraints.host_max_distance_nm` + 0.5 nm of the centre of the ligand's heavy atoms in the first frame (the binding site of the search of `selection: auto`), or on all C-alpha atoms if those are fewer than 12, so that setting changes this number under every selection. `restraints.pose_rmsd_warning_nm` of earlier versions (a warning above 0.3 nm, a level of cuFEP's own) is read with a notice if it holds 0.3 and refused otherwise |
| `sampling.rest2.scale` | 1.5 (447 K at 298 K) | A torsion barrier E is crossed at a rate proportional to exp(-E / (kT x scale)). With 1.5 a barrier of 6 kcal/mol, which takes nanoseconds to cross at 298 K, is crossed about 30 times as often, while an amide bond (16 kcal/mol or more) stays 18 kT high. The scale must stay below the point where amide bonds isomerise: with 2.5 (745 K) the solvent legs of five TYK2 ligands ended with a cis amide in 6 to 15 of their 21 windows, none without REST2, and three of the five binding free energies moved 4-5 kcal/mol away from experiment (`spikes/m10_amide_isomers.py`). SOMD2's own default is 1.0 (off) |
| `sampling.rest2.residues` | none | named as in the input PDB: residue number, with the chain of the input and the insertion code where needed (`"B:898A"`). The preparation gives every peptide segment a chain of its own (A, B, ... in order) and records which chain of the input it came from (`input_chains` in `prep/prep.json`); a number that two chains of the input have must be given with its chain. Prepared systems of an earlier version have no such record and are refused when residues are named |
| `sampling.rest2.region` | pocket | the owner's choice (2026-11-20), for both examples too. The hot region of the complex leg holds the ligand and the protein residues with a heavy atom within `pocket_cutoff_nm` (0.4 nm, cuFEP's own choice) of a heavy atom of the ligand in the prepared structure; the default of SOMD2 is the ligand alone (`ligand`). One calculation of each example with it (ALS threshold 0.18, 5 ns of production per window) gave -8.94 ± 0.77 kcal/mol for TYK2 ejm_31 (experiment -9.54) and -11.95 ± 0.73 for A2A 4e (experiment -12.07); one run each, and runs of a complex leg differ among themselves by more than these errors (protocol.md, "REST2"): the full benchmark runs with the same settings gave -9.84 ± 0.33 and -18.91 ± 0.52 (`benchmarks/results/`) |
| `lambda_schedule.windows` | complex 30 windows, solvent 14 | default lambda values of OpenFE's ABFE protocol, which uses the same soft-core form and replica exchange |
| `lambda_schedule.adaptive.overlap_threshold`, `candidates` | 0.30 in the complex leg, 0.18 in the solvent leg; 101 | The method and `candidates`: Midgley et al., J. Chem. Inf. Model. 2025, 65, 512, whose threshold is 0.04. 0.18 of the matrix of the stage is the owner's choice (2026-11-20) for both legs, and the default of the solvent leg: with 0.04 the production of the TYK2 and A2A complex legs had neighbouring windows below the overlap of 0.03 of Klimovich et al. (0.024 at vanish 0.75-0.88, which the precalculation had read as 0.096 and 0.073); with 0.10 to 0.18 the lowest overlap in their production was 0.065 to 0.147. The threshold cannot go much higher: an element of the overlap matrix is shared among all windows that overlap, so windows that are close have elements near 1 / (their number) and no added window can raise them. With 0.20 TYK2 legs added windows until `max_windows`, 48 and 32 then (see protocol.md). The complex leg: 0.30 of the overlap of a pair alone (`overlap_matrix`), the owner's choice (2026-11-22) together with the start of that leg (`initial_windows`, where the runs are given); 0.18 of the matrix of the stage until then. In the three runs of the TYK2 example with it the rule added 3 to 5 windows to the 30 of the start and ended with every pair above the threshold (lowest overlap of a pair per stage 0.305 to 0.418). A leg whose threshold is not given has the threshold of the matrix it reads, 0.18 for `stage` and 0.30 for `pair`: the two numbers are the owner's choices above, and that a leg which is only told another matrix takes the number of that matrix is cuFEP's own (review of 2026-11-22: in the matrix of a stage 26, 26 and 27 of the 29 pairs of the default start of the complex leg read at or below 0.30 in the three runs of the TYK2 example, against 6 by the pair alone; only pairs at an end of a stage read above it, the largest 0.333 among the 784 pairs recorded in the nine runs. From that start, harmonic windows went to `max_windows` with 0.30 on the matrix of the stage: `test_a_leg_that_is_only_told_its_overlap_matrix_gets_the_threshold_of_that_matrix`). The solvent leg with `pair` and 0.30 has not been run. Each of the two settings is one value for both legs or one per leg |
| `lambda_schedule.adaptive.overlap_matrix` | pair in the complex leg, stage in the solvent leg | `stage`: the paper's rule, which reads the overlap matrix of the current windows. `pair`: the overlap matrix of the two neighbouring windows alone, cuFEP's own and the owner's decision (2026-11-21), for a start with more windows and a threshold above what the elements of the matrix of a stage reach between windows that are close (a pair at an end of a stage can read higher). For unit-width harmonic windows d apart the element in the matrix of the stage is d exp(-d^2/4) / (2 sqrt(pi)) once windows are close: at most 0.242, at d = sqrt(2), and 0.22, 0.13, 0.07 at d = 1, 0.5, 0.25. That of the pair alone is 0.22, 0.40, 0.47, 0.49 at d = 2, 1, 0.5, 0.25, and 0.5 where the windows meet (measured: `test_the_overlap_of_a_pair_alone_rises_as_windows_get_closer_and_the_matrix_of_the_stage_falls`). With 21 such windows 1.5 apart and a threshold of 0.33 the rule ends with every pair above the threshold when it reads the pair, and adds windows until `max_windows` when it reads the stage (`test_many_initial_windows_and_a_raised_threshold_converge_by_the_overlap_of_a_pair_alone`). The complex leg reads `pair` by default since 2026-11-22, the owner's choice with the start of that leg (`initial_windows`); `stage` in both legs until then. The solvent leg reads `stage`, as the paper |
| `lambda_schedule.adaptive.window_ps`, `equilibration_ps` | solvent leg: 50 ps, no equilibration; complex leg: 500 ps after 500 ps of equilibration | Solvent leg: the paper's value, set there for a ligand in solution. Complex leg: cuFEP's own values. The pocket adapts to a partly decoupled ligand over hundreds of picoseconds, and a 50 ps precalculation read the overlap of the affected windows several times too high (protocol.md, "What was measured on TYK2") |
| `lambda_schedule.adaptive.initial_windows` | solvent leg: discharge 5, vanish 9, evenly spaced; complex leg: the recommended windows of mode fixed (restrain 6, discharge 11, the 15 default values of `windows.complex.vanish`) | Solvent leg: 9 is the paper's number for one lambda coordinate; the discharge energy is quadratic in lambda and starts with fewer (cuFEP's own choice). Complex leg: the owner's choice (2026-11-22) from the runs of plan `docs/plans/2026-11-21-als-initial-window-design.md`; restrain 3, discharge 5, vanish 9 until then. The TYK2 example (ejm_31, 5 ns of production per window, one calculation per run, the same file and seed for the runs of a start) was run three times with each of three starts of the complex leg, all with the overlap of a pair alone and the threshold 0.30, the solvent leg as it was. Binding free energies in kcal/mol, each ± its standard error (experiment: -9.54). A, this start (30 windows, the recommended values of OpenFE's ABFE protocol): -9.36 ± 0.56, -9.63 ± 0.37, -9.33 ± 0.51; standard deviation of the three 0.17. B (30 windows, denser where 37 stored complex legs sample slowly, the 33 of the two full benchmark runs and the four runs of ejm_31 named below; half of the vanish windows at lambda 0.7 to 0.9): -10.01 ± 0.34, -9.64 ± 0.52, -11.71 ± 0.53; 1.11. C (59 windows placed as B): -11.43 ± 0.46, -6.41 ± 0.59, -7.64 ± 0.43; 2.62. The start until then, with the threshold 0.18 of the matrix of the stage, four runs: -12.16 ± 0.32, -12.23 ± 0.64, -8.84 ± 0.91, -8.33 ± 0.74; 2.09. Three runs give a standard deviation only roughly, and this is one ligand of one target: the A2A example and the benchmark sets have not been run with this start (not run). The runs are in the plan; protocol.md ("The starts of the complex leg that were compared") gives their windows and checks. A stage can also be given the lambda values of its start, which need not be evenly spaced: cuFEP's own, the owner's order of 2026-11-21 (the paper starts from evenly spaced windows) |
| `lambda_schedule.adaptive.sample_interval_ps`, `max_windows` | 0.5 ps; 100 in both legs | The paper gives neither. The interval is cuFEP's own choice. The limit is the owner's choice (2026-11-21; 48 and 32 until then). With the threshold 0.18 of the matrix of the stage no leg of the two full benchmark runs came near it: 17 to 30 windows in the complex legs, 16 to 21 in the solvent legs; with the defaults of the complex leg since 2026-11-22 that leg of the TYK2 example had 33 to 35 windows (three runs). A leg whose precalculation does not reach the threshold grows until the limit, or until the windows it would refine are one candidate apart, and ends there with a warning |
| `lambda_schedule.fixed_walk_ps` | 50 ps | cuFEP's own choice: in fixed mode the replicas get their start configurations from a walk through the windows with this much MD in each |
| `analysis.min_overlap` | 0.03 | Klimovich, Shirts and Mobley 2015, section 3.5: "values as low as 0.03 [seem] to be tolerable to yield a reliable free energy estimate"; below it the estimate "will likely be substantially incorrect". Also the level of Mey et al. 2020. The first condition for a usable result |
| `analysis.min_effective_samples` | 50 | Mey et al., Best Practices for Alchemical Free Energy Calculations, 2020, section 8.2: "it is recommended that the effective number of uncorrelated samples Neff >= ~50 if the BAR or MBAR estimators ... are used". Counted for each lambda window from the statistical inefficiency of its energy difference to the next window (Klimovich, Shirts and Mobley 2015, section 2.5; alchemlyb's default) |
| (no other level) | | The two settings above are the only levels of a verdict on the analysis, both from the literature; cuFEP has no criteria of its own (owner, 2026-11-20). The forward and reverse series, the two halves, the consecutive blocks and BAR are shown as numbers without a verdict, as OpenFE shows its forward and reverse series. `analysis.halves_tolerance_kT` and `analysis.estimators_tolerance_kcal_mol` of earlier versions are read with a notice if they hold their old default and refused otherwise |
| `analysis.convergence_points`, `convergence_blocks` | 10, 5 | cuFEP's own choice for the number of points of the forward and reverse series and for the number of consecutive blocks |
| `analysis.error_blocks`, `part_error_blocks` | 40, 20, 10, 5; 5 | cuFEP's own choice. The standard error of a leg is that of a delete-one-block jackknife over blocks of time that are the same for all lambda windows, computed for each of these numbers of blocks, of the free energy as it is reported (MBAR minus the tail correction); the largest is reported, unless the analytical one is larger. `part_error_blocks` is the number of blocks within each half, each point of the forward and reverse series and each consecutive block. Grossfield et al. 2018 (sections 7.3.2 and 7.6) ask for blocks longer than the correlation time, for a standard error that no longer grows with their length, and name the jackknife for few blocks; they give no numbers. 5 is the fewest blocks that still give an error, 40 the most that leaves a block of a leg of 10 ns (4000 iterations) 100 iterations long. On a model with slow configurations that travel between the windows the largest of the four was 0.83 of the scatter between independent runs, the error of MBAR 0.28 (protocol.md, "Free energy and uncertainty of a leg") |
| `analysis.lrc` | reweight | see protocol.md, long-range dispersion |
| `analysis.lrc_cutoff_nm` | 2.0 nm | cuFEP's own choice. The explicit sum runs from `dynamics.cutoff_nm` to this radius (within 0.49 of the shortest height of the cell) and a homogeneous tail is added beyond it; a radius at or inside the cutoff is refused, because the interactions between it and the cutoff are already in the MD |
| `backend.precision` in the plain MD of a leg | the precision asked | the owner's decision (2026-11-21): the pre-equilibration and the MD from which a restraint is chosen run in the precision of the YAML, as the alchemical run does, single too. Earlier versions ran them in mixed precision where single was asked, a rule of their own without a source; a pre-equilibration they made with `backend.precision: single` is not used again. The plugin has no double precision: where double is asked it runs in mixed precision, plain MD and alchemical run alike |
| `dynamics.barostat_frequency` | 25 steps | default of OpenMM and of OpenFE's ABFE protocol. The replicas that share a GPU take turns in one OpenMM Context, and the Monte Carlo barostat counts the steps of the Context, not of a replica (OpenMM 8.4). Every stretch of MD that one configuration gets in one go (an iteration of a replica, `sampling.time_per_iteration_ps`; a window of the precalculation, which starts from a configuration of its own; not the walks through the windows, which are one trajectory) must therefore have at least this many steps, or some replicas would never get a move of the box. A configuration that breaks this is refused before anything is written. The default iteration has 625 steps (2.5 ps at 4 fs) and gives every replica 25 moves |
| `forcefield.water` | amber14/tip3p.xml | default of OpenFE's ABFE protocol. The water box is built with the model whose number of sites matches the file (measured with OpenMM 8.4: tip3p, tip3pfb, opc3 and spce with 3 sites, tip4pew, tip4pfb and opc with 4); other files are refused before anything is prepared. The bilayer is built with 3-site waters, so `membrane.enabled` needs a 3-site model |
| `resources.oversubscription_factor` | from the size of the system: 16 up to 10,000 atoms, 8 up to 60,000, 4 up to 200,000, else 2 | measured total throughput per GPU under NVIDIA MPS |
| `membrane.equilibration_ns` | 10 ns | cuFEP's own choice for a freshly built bilayer |
| the bilayer at the end of its equilibration | numbers, no level | the relative change of the mean area in the membrane plane and the change of the mean number of water molecules within 0.6 nm of the ligand between the last two quarters of `membrane.equilibration_ns`, reported with the result of the complex leg (`bilayer`); the run is never made longer. `membrane.area_tolerance` and `pocket_water_tolerance` of earlier versions (a warning above 1 % and 1.5 waters, levels of cuFEP's own) are read with a notice if they hold those values and refused otherwise |
