# cuFEP

Absolute binding free energies (ABFE) on OpenMM, with NVIDIA
[cuPotential](https://github.com/NVIDIA/cuPotential) as the default compute backend and
stock OpenMM as a second backend that is always available.

You give cuFEP a protein PDB and a ligand SDF with the ligand placed in the pocket.
Everything else has a default or is chosen automatically, and can be set in YAML.

- Soluble and membrane proteins. With `membrane: {enabled: true}` the complex is oriented
  and embedded in a lipid bilayer.
- Double decoupling with automatically chosen Boresch restraints, Hamiltonian replica
  exchange and MBAR; the protocol follows OpenFE's ABFE protocol
  ([docs/protocol.md](docs/protocol.md) lists the differences).
- Adaptive lambda windows by default.
- One GPU or many: all free GPUs are used by default.
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
`RESULT ejm_31: dG_bind = ... ± ... kcal/mol (3 repeat(s))`, and writes:

| file | content |
|---|---|
| `resolved_config.yaml`, `auto_report.md` | every setting that was used, and why the automatic ones were chosen |
| `protein_prepared.pdb` | the protein as simulated |
| `summary.csv`, `summary.json` | binding free energies, spread over repeats, warnings |
| `<ligand>/repeat_<n>/<leg>/` | restraint, lambda schedule, samples, per-leg result and log |

## Configuration

The smallest configuration:

```yaml
system:
  protein: protein.pdb
  ligand: ligand.sdf        # several molecules in the SDF = one calculation each
```

`cufep init --full` writes every setting with its default. The ones most often changed:

```yaml
membrane: {enabled: false}           # true builds a POPC bilayer around the protein
backend: {name: cupotential}         # or openmm (stock); falls back to openmm automatically
sampling: {repeats: 3, production_ns: [5, 10]}
lambda_schedule: {mode: adaptive}    # or fixed, with file: schedule.yaml from an earlier run
resources: {gpus: auto}              # or a list of device indices, e.g. [0, 1]
output: {dir: runs/my_ligand}
```

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
cufep benchmark run tyk2 --tier subset    # five ligands, three repeats
cufep benchmark run a2a --tier full       # every ligand, three repeats
cufep benchmark report tyk2 runs/benchmark_tyk2_subset
```

The report compares cuFEP with experiment and with the FEP+ values of the Schrödinger
public benchmark. FEP+ values come from relative calculations shifted to the experimental
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
