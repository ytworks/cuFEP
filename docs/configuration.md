# Configuration reference

A run is described by one YAML file. The principles:

- every setting can be changed in the YAML;
- every setting has a default, and the default is the recommended value (the basis of each
  one is given below);
- `auto` is used only where the value has to be computed from the input: restraint atoms,
  GPUs, box shape, membrane orientation and the oversubscription factor. Everything else is
  a plain value.

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
  equilibration_ns: 10.0
  area_tolerance: 0.01            # warning if the area still changes by more than this fraction at the end of the equilibration
  pocket_water_tolerance: 1.5     # ... or the number of waters near the ligand by more than this
forcefield:
  protein: amber14/protein.ff14SB.xml
  water: amber14/tip3p.xml
  lipid: amber19/lipid21.xml
  ligand: openff-2.2.1
  ligand_charges: am1bcc          # am1bcc | am1bcc-nagl
backend:
  name: cupotential               # cupotential | openmm
  fallback: openmm                # openmm | none
  precision: mixed                # mixed | single | double
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
solvation:
  box_shape: auto                 # auto | cube | dodecahedron
  padding_complex_nm: 1.0
  padding_solvent_nm: 1.5
restraints:
  type: boresch
  selection: auto                 # auto | manual (give host_atoms and guest_atoms)
  force_constants: openfe         # openfe (the two values below) | auto (from the fluctuations) | custom
  k_r_kcal_per_A2: 10.0
  k_angle_kcal_per_rad2: 80.0
  host_atoms: null
  guest_atoms: null
  rmsf_cutoff_nm: 0.1
  host_min_distance_nm: 0.5
  host_max_distance_nm: 1.5
  dssp_filter: true
  pose_rmsd_warning_nm: 0.3       # warn when the ligand ends up further than this from the pose that was given
alchemy:
  softcore_alpha: 0.5
lambda_schedule:
  mode: adaptive                  # adaptive | fixed
  windows:                        # per stage: a number of lambda windows, or the lambda values
    complex:
      restrain: 6
      discharge: 11
      vanish: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    solvent:
      discharge: 5
      vanish: [0.0, 0.12, 0.24, 0.36, 0.48, 0.6, 0.7, 0.77, 0.85, 1.0]
  adaptive:
    overlap_threshold: 0.04
    candidates: 101
    initial_windows:
      complex:
        restrain: 3
        discharge: 5
        vanish: 9
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
      complex: 48
      solvent: 32
  fixed_walk_ps: 50.0             # fixed mode: MD in each window that gives the replicas their start configurations
  file: null                      # a schedule.yaml of an earlier run of the same leg (or a directory of them), used as it is
sampling:
  pre_equilibration:
    complex: {nvt_ns: 0.25, npt_ns: 0.5, md_ns: 5.0}
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
    region: ligand                # ligand | pocket (also protein residues near the ligand)
    pocket_cutoff_nm: 0.4
    residues: []                  # protein residues added to the hot region, e.g. ["960", "A:981"]
analysis:
  estimator: mbar
  lrc: reweight                   # reweight | isotropic | none
  lrc_cutoff_nm: 2.0
  min_overlap: 0.03               # NOT RELIABLE below this overlap of two neighbouring lambda windows
  convergence_points: 10          # points of the forward and reverse series
  convergence_blocks: 5           # consecutive blocks of the production with an estimate of their own
  min_effective_samples: 50.0     # message if the lambda window with the fewest independent samples has fewer
  halves_tolerance_kT: 1.0        # message if the two halves of the production differ by more than this many kT
  estimators_tolerance_kcal_mol: 1.0 # message if BAR and MBAR differ by more
  error_blocks: [40, 20, 10, 5]   # numbers of time blocks for the standard error of a leg; the largest error is reported
  halves_error_blocks: 5          # time blocks for the standard error of each half of the production
resources:
  gpus: auto                      # auto | a list of device indices
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

## Basis of the recommended values

| settings | recommended value | basis |
|---|---|---|
| force fields, AM1-BCC charges, 4 fs with hydrogen mass repartitioning, 0.9 nm cutoff, 298.15 K, 1 bar, 0.15 M | as listed | defaults of OpenFE's ABFE protocol |
| `sampling.pre_equilibration`, `equilibration_ns`, `time_per_iteration_ps`, `exchange` | 0.25 + 0.5 + 5 ns; 1 ns; 2.5 ps; replica exchange | defaults of OpenFE's ABFE protocol. With 1 ns of MD instead of 5 ns (as FEP+ and SOMD2 use) TYK2 ejm_31 gave -11.2 ± 1.5 kcal/mol over three repeats against -10.6 ± 1.9, the same within the spread; 5 ns is kept |
| `sampling.production_ns` | 10 ns per lambda window | default of OpenFE's ABFE protocol. Schrödinger's absolute binding FEP+, A3FE and Boltz-ABFE use 5 ns. With 5 ns the lambda window with the fewest independent samples of a TYK2 complex leg held about 29 (11 of 55 legs reached the 50 that Mey et al. 2020 recommend for MBAR); see protocol.md, "Checks of the analysis" |
| `sampling.repeats` | 1 | FEP+ and Flare run a calculation once. OpenFE's default is 3; more than one repeat makes the reported uncertainty the standard deviation over repeats |
| `restraints` search settings and force constants, `alchemy.softcore_alpha` | as listed | defaults of OpenFE's ABFE protocol. `force_constants: auto` takes kT / sigma^2 of each restrained coordinate in the pre-equilibration instead, as A3FE and SOMD2 do. On TYK2 ejm_31 (three repeats each) the two gave the same binding free energy within the spread of the repeats (-10.6 ± 1.9 and -10.8 ± 1.2 kcal/mol), so the fixed values of OpenFE stay the default |
| `restraints.pose_rmsd_warning_nm` | 0.3 nm | cuFEP's own choice: 2 A is the usual limit for calling two poses the same, and thermal motion adds to it. OpenFE reports the ligand RMSD without a limit; like OpenFE, cuFEP does not stop |
| `sampling.rest2.scale` | 1.5 (447 K at 298 K) | A torsion barrier E is crossed at a rate proportional to exp(-E / (kT x scale)). With 1.5 a barrier of 6 kcal/mol, which takes nanoseconds to cross at 298 K, is crossed about 30 times as often, while an amide bond (16 kcal/mol or more) stays 18 kT high. The scale must stay below the point where amide bonds isomerise: with 2.5 (745 K, inside the 670-1000 K that FEP+ uses for its hot region) the solvent legs of five TYK2 ligands ended with a cis amide in 6 to 15 of their 21 windows, none without REST2, and three of the five binding free energies moved 4-5 kcal/mol away from experiment (`spikes/m10_amide_isomers.py`). SOMD2's own default is 1.0 (off) |
| `sampling.rest2.region` | ligand | default hot region of FEP+ and of SOMD2. `pocket` adds the protein residues within `pocket_cutoff_nm` (0.4 nm, cuFEP's own choice) of the ligand, as FEP+ does on request |
| `lambda_schedule.windows` | complex 30 windows, solvent 14 | default lambda values of OpenFE's ABFE protocol, which uses the same soft-core form and replica exchange |
| `lambda_schedule.adaptive.overlap_threshold`, `candidates` | 0.04, 101 | Midgley et al., J. Chem. Inf. Model. 2025, 65, 512. The threshold has to stay low: an element of the overlap matrix is shared among all windows that overlap, so windows that are close have elements near 1 / (their number) and no added window can raise them. On the TYK2 complex leg a threshold of 0.20 added windows until `max_windows` (see protocol.md) |
| `lambda_schedule.adaptive.window_ps`, `equilibration_ps` | solvent leg: 50 ps, no equilibration; complex leg: 500 ps after 500 ps of equilibration | Solvent leg: the paper's value, set there for a ligand in solution. Complex leg: cuFEP's own values. The pocket adapts to a partly decoupled ligand over hundreds of picoseconds, and a 50 ps precalculation read the overlap of the affected windows several times too high (protocol.md, "What was measured on TYK2") |
| `lambda_schedule.adaptive.initial_windows` | restrain 3, discharge 5, vanish 9 in both legs | 9 is the paper's number for one lambda coordinate; the restrain and discharge energies are linear and quadratic in lambda and start with fewer (cuFEP's own choice for these two) |
| `lambda_schedule.adaptive.sample_interval_ps`, `max_windows` | 0.5 ps; 48 and 32 | cuFEP's own choice; the paper does not give them |
| `lambda_schedule.fixed_walk_ps` | 50 ps | cuFEP's own choice: in fixed mode the replicas get their start configurations from a walk through the windows with this much MD in each |
| `analysis.min_overlap` | 0.03 | Klimovich, Shirts and Mobley 2015, section 3.5: "values as low as 0.03 [seem] to be tolerable to yield a reliable free energy estimate"; below it the estimate "will likely be substantially incorrect". Also the warning level of Flare and of Mey et al. 2020. The first condition for a usable result |
| `analysis.min_effective_samples` | 50 | Mey et al., Best Practices for Alchemical Free Energy Calculations, 2020, section 8.2: "it is recommended that the effective number of uncorrelated samples Neff >= ~50 if the BAR or MBAR estimators ... are used". Counted for each lambda window from the statistical inefficiency of its energy difference to the next window (Klimovich, Shirts and Mobley 2015, section 2.5; alchemlyb's default) |
| `analysis.halves_tolerance_kT` | 1 | Mey et al. 2020, figure 9: the difference between the forward and the reverse free energy "at the midpoint of the simulation time gives an indication of the overall convergence of the simulation, with differences over 1 kBT indicating poor convergence". The source applies it to each step of a relative perturbation; cuFEP applies it to a leg |
| `analysis.estimators_tolerance_kcal_mol` | 1 | Mey et al. 2020, section 8.4: "differences between methods that are larger than 1 kcal/mol ... would indicate poor convergence". The guide compares TI with BAR-type estimators; cuFEP compares BAR between neighbouring windows with MBAR |
| `analysis.convergence_points`, `convergence_blocks` | 10, 5 | cuFEP's own choice for the number of points of the forward and reverse series and for the number of consecutive blocks |
| `analysis.error_blocks`, `halves_error_blocks` | 40, 20, 10, 5; 5 | cuFEP's own choice. The standard error of a leg is that of a delete-one-block jackknife over blocks of time that are the same for all lambda windows, computed for each of these numbers of blocks; the largest is reported. Grossfield et al. 2018 (sections 7.3.2 and 7.6) ask for blocks longer than the correlation time, for a standard error that no longer grows with their length, and name the jackknife for few blocks; they give no numbers. 5 is the fewest blocks that still give an error, 40 the most that leaves a block of a leg of 10 ns (4000 iterations) 100 iterations long. On a model with slow configurations that travel between the windows the largest of the four was 0.83 of the scatter between independent runs, the error of MBAR 0.28 (protocol.md, "Free energy and uncertainty of a leg") |
| `analysis.lrc` | reweight | see protocol.md, long-range dispersion |
| `resources.oversubscription_factor` | from the size of the system: 16 up to 10,000 atoms, 8 up to 60,000, 4 up to 200,000, else 2 | measured total throughput per GPU under NVIDIA MPS |
| `membrane.equilibration_ns` | 10 ns | cuFEP's own choice for a freshly built bilayer |
| `membrane.area_tolerance`, `pocket_water_tolerance` | 0.01; 1.5 | cuFEP's own choice: the mean area in the membrane plane and the mean number of water molecules within 0.6 nm of the ligand are compared between the last two quarters of the equilibration. A larger change is reported with the result; the run is not made longer |
