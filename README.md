# cuFEP

Absolute binding free energies (ABFE) on OpenMM, with NVIDIA
[cuPotential](https://github.com/NVIDIA/cuPotential) as the default compute backend and
stock OpenMM as a second backend that is always available.

You give cuFEP a protein PDB and a ligand SDF with the ligand placed in the pocket.
Every other setting has a recommended default and can be changed in YAML
([docs/configuration.md](docs/configuration.md)).

- Soluble and membrane proteins. With `membrane: {enabled: true}` the complex is oriented
  and embedded in a lipid bilayer.
- Double decoupling with automatically chosen Boresch restraints, Hamiltonian replica
  exchange and MBAR. Defaults and terms follow established FEP software;
  [docs/protocol.md](docs/protocol.md) compares cuFEP with FEP+, Flare and OpenFE.
- Lambda windows: adaptive lambda scheduling by default (the method of Cresset Flare), or a
  fixed number or list of lambda values per leg and stage.
- One GPU or many: the replicas of a calculation are spread over all free GPUs, several per
  GPU through NVIDIA MPS.
- Benchmarks against experiment and FEP+ for a kinase (TYK2) and a GPCR in a bilayer (A2A).

**Status:** work in progress. The plan and its milestones are in [docs/PLAN.md](docs/PLAN.md),
the results of the first risk checks in [docs/M0_REPORT.md](docs/M0_REPORT.md).
Version 0.1 handles neutral ligands only and has no cofactor support yet.

## Quick start

Requirements: Linux, Docker with the NVIDIA Container Toolkit, an NVIDIA GPU. The image is
built for CUDA 13.4; professional and data-centre GPUs work with driver branch R535 or
newer through the forward-compatibility package in the image, GeForce cards need R615+.

```bash
# build the image (downloads NVIDIA's cuPotential runtime; see docs/LICENSING.md)
docker build -f docker/Dockerfile -t cufep:latest .

# a shell alias for the examples below
cufep() { docker run --rm --gpus all --shm-size=16g -u "$(id -u):$(id -g)" \
                     -v "$PWD":/work cufep:latest "$@"; }

cufep check-env                                   # software, GPUs, both backends
cufep init protein.pdb ligand.sdf -o config.yaml  # write a two-line configuration
cufep plan config.yaml                            # what will be done, without simulating
cufep run config.yaml                             # run or resume
```

`--shm-size=16g` (or `--ipc=host`) matters: replica coordinates are shared between
worker processes in memory, and Docker's default of 64 MB is too small. Without it cuFEP
falls back to files in the output directory and says so.

`cufep run` prints one line per ligand, for example
`RESULT ejm_31: dG_bind = ... ± ... kcal/mol (MBAR uncertainty, 1 repeat)`, and writes:

| file | content |
|---|---|
| `resolved_config.yaml`, `auto_report.md` | every setting that was used, and why the computed ones were chosen |
| `protein_prepared.pdb` | the protein as simulated |
| `summary.csv`, `summary.json` | binding free energies, uncertainties, warnings |
| `timing.json` | time per phase and job, GPU utilization |
| `<ligand>/repeat_<n>/<leg>/` | restraint, lambda windows (`schedule.yaml`), samples, per-leg result and log |

One calculation per ligand is run by default, as in FEP+ and Flare. `sampling.repeats`
runs it several times independently and reports the spread.

## Configuration

The smallest configuration:

```yaml
system:
  protein: protein.pdb
  ligand: ligand.sdf        # several molecules in the SDF = one calculation each
```

`cufep init --full` writes every setting with its default, and
[docs/configuration.md](docs/configuration.md) explains them. The ones most often changed:

```yaml
membrane: {enabled: false}           # true builds a POPC bilayer around the protein
backend: {name: cupotential}         # or openmm (stock); falls back to openmm if needed
sampling:
  production_ns: 5                   # per lambda window; or {complex: 10, solvent: 5}
  repeats: 1                         # independent repeats of the whole calculation
lambda_schedule:
  mode: adaptive                     # adaptive lambda scheduling; or fixed:
  windows:                           # a number of windows or the lambda values, per stage
    complex: {restrain: 6, discharge: 11, vanish: 15}
    solvent: {discharge: 5, vanish: [0.0, 0.12, 0.24, 0.36, 0.48, 0.6, 0.7, 0.77, 0.85, 1.0]}
resources:
  gpus: auto                         # or a list of device indices, e.g. [0, 1]
  oversubscription_factor: auto      # workers that share a GPU at the same time (NVIDIA MPS)
output: {dir: runs/my_ligand}
```

`cufep plan config.yaml` prints the lambda windows that a run would use.

## Examples

| directory | system |
|---|---|
| [examples/kinase_tyk2](examples/kinase_tyk2) | TYK2 kinase with ligand ejm_31 (soluble) |
| [examples/membrane_a2a](examples/membrane_a2a) | adenosine A2A receptor with compound 4e, in a POPC bilayer (`membrane: {enabled: true}`) |

```bash
cufep run examples/kinase_tyk2/config.yaml
cufep run examples/membrane_a2a/config.yaml
```

## Benchmarks

```bash
cufep benchmark list
cufep benchmark run tyk2 --tier smoke     # one ligand, minutes: checks the pipeline
cufep benchmark run tyk2 --tier subset    # five ligands
cufep benchmark run a2a --tier full       # every ligand
cufep benchmark report tyk2 runs/benchmark_tyk2_subset
```

One calculation per ligand is run, as FEP+ does; `--repeats` runs more. The report compares
cuFEP with experiment and with the FEP+ values of the Schrödinger public benchmark. FEP+ values come from relative calculations shifted to the experimental
mean, so the report separates raw errors (cuFEP only) from the like-for-like statistics.

## Development

```bash
spikes/run.sh -m pytest tests/unit -q     # unit tests on the CPU, inside the image
```

Unit tests run on stock OpenMM without a GPU and are run by CI. Checks that need a GPU
and the plugin are the scripts in `spikes/`.

## Licence

cuFEP is Apache-2.0. The cuPotential runtime that the image downloads is proprietary
NVIDIA software; read [docs/LICENSING.md](docs/LICENSING.md) before publishing an image.
Benchmark inputs come from the Schrödinger public binding free energy benchmark (MIT).
