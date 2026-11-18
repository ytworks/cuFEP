# M0 report: risk spikes

Date: 2026-11-18. Hardware: 8 x RTX PRO 6000 Blackwell Server Edition, driver 595.99.02.
Software: image `cufep:dev` (CUDA 13.4.3, OpenMM 8.4, cuPotential OpenMM plugin 0.3.5).
Scripts are in `spikes/`; raw logs stay local (`spikes/out/`, not committed).

Timings were measured to choose defaults, but they are not reported here: the NVIDIA
licence of the cuPotential runtime restricts publishing performance results.

## Results

| # | Question | Result |
|---|---|---|
| 1 | Do the CUDA platform and the plugin run in the container on the R595 driver? | Yes, through the CUDA forward-compatibility package in the base image. |
| 2 | Does the alchemical System give the same energies on both backends? | Yes. On the TYK2 complex (cubic and dodecahedral boxes, 12 lambda states, frames from the coupled, half-decoupled and decoupled states): relative energy difference below 3e-5, relative force difference below 1e-3. |
| 2b | Are between-state energy differences accurate enough for MBAR? | Against a tight Ewald reference (tolerance 1e-6, double precision) the plugin's differences deviate by at most 0.03 kT, stock PME at the same nominal tolerance (5e-4) by up to 0.07 kT. The plugin is at least as accurate as stock. |
| 2c | Can samples from the two backends share one MBAR data set? | No. Their between-state differences disagree by up to 0.08 kT (the plan's limit was 0.02 kT). A leg is always sampled by one backend. |
| 3 | Is the fully decoupled state stable on the plugin? | Yes, 1 ns NPT without incident. |
| 4 | Live charge offsets, plugin integrator with a stock restraint force, one Context time-sliced over replicas? | All work. |
| 5 | Cost of parameter updates and energy evaluation | Measured. Led to the force-group scheme below. |
| 6 | Several worker processes on several GPUs in one container; dodecahedral box | Eight workers ran concurrently on eight GPUs with both backends, next to another project's MPS daemon. The dodecahedral box passes the equivalence test. |
| 7 | Semi-isotropic pressure coupling for bilayers | The stock `MonteCarloMembraneBarostat` runs with the plugin force and either integrator. Over five runs of 20-40 ns each on a 128-lipid POPC patch, area per lipid is 61.0 ± 0.5 A^2 with the plugin (force and integrator) against 60.9 ± 0.3 A^2 on stock OpenMM (standard error over runs); volume and potential energy agree within the same statistics. |

## Decisions taken from the spikes

- **Force layout: split.** The plugin replaces only the `NonbondedForce` (PME, charges
  scaled through particle offsets). The soft-core LJ `CustomNonbondedForce` (ligand x
  environment interaction group) and the restraint stay on stock kernels in the same System.
- **State energies through force groups.** The NonbondedForce energy is an exact
  quadratic in `lambda_electrostatics`, so three evaluations give it for every state
  (interpolation error 1e-4 kT on the plugin); the soft-core group is evaluated once per
  distinct `lambda_sterics`; the restraint is linear. This removes most parameter updates,
  which are the expensive path on the plugin, and makes the lambda search nearly free.
  The full energy matrix is evaluated every 2.5 ps iteration, as in OpenFE.
- **Membrane pressure coupling: rung 1 of the plan's ladder** (stock membrane barostat,
  plugin force, plugin integrator).
- **Fallback is per leg.** If the plugin fails, the whole leg is repeated on stock OpenMM;
  samples are never mixed (result 2c).
- **Worker processes, shared replica state in memory-mapped files.** `/dev/shm` is used
  when it is large enough; Docker's default 64 MB is not, so run the container with
  `--shm-size=16g` (or `--ipc=host`). Otherwise the arrays go to the run directory.

## Things that differ from the plan

- The plugin is not on conda-forge. The image installs it from the conda channel shipped
  with NVIDIA's GitHub release, pinned by SHA256.
- The ligand-only leg uses a cubic box when the plugin is the backend. The plugin needs
  every box height to be at least three cutoffs, and for a small solute a cube is then the
  smallest valid box. The complex keeps the dodecahedron.
- The lambda search checks the width of each interval in both directions. With the
  forward estimate alone, one interval near the decoupled end had an MBAR overlap of only
  0.04; with the reverse check the overlaps are uniform (0.14-0.30) at about 30 states.

## Open observations

- The POPC patch settles at about 61 A^2 per lipid with a 0.9 nm cutoff, on both backends;
  experiment is about 64 A^2. A run with a 1.0 nm cutoff, the value Lipid21 was validated
  with, is in progress to decide the default cutoff for membrane systems.
- Three independent orientation methods (hydrophobic slab, transfer-energy model, memembed)
  agree with each other within about 7 degrees for the A2A receptor but differ from OPM by
  12-16 degrees. For the other OPM structures tried, the transfer-energy model is within
  2 degrees of OPM. The membrane milestone uses a consensus of the three methods.
