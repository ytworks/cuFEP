# cuFEP

[![CI](https://github.com/ytworks/cuFEP/actions/workflows/ci.yml/badge.svg)](https://github.com/ytworks/cuFEP/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![OpenMM 8.4](https://img.shields.io/badge/OpenMM-8.4-blue.svg)

**English** | [日本語](README.ja.md)

cuFEP computes **absolute binding free energies (ABFE)** of small molecules to proteins with
[OpenMM](https://openmm.org). You give it a protein PDB and a ligand SDF with the ligand placed in
the pocket; cuFEP prepares the system, chooses the restraint and the lambda windows, runs
Hamiltonian replica exchange with REST2 on all GPUs of the machine, and reports the binding free
energy with its standard error and the checks of the alchemical best-practice literature.
NVIDIA's [cuPotential](https://github.com/NVIDIA/cuPotential) plugin for OpenMM is the default
compute backend; stock OpenMM is the second backend and is always available.

> **Status: version 0.1, under development.** Neutral ligands only, no cofactors yet. The
> interfaces and the defaults may still change. See [PROGRESS.md](PROGRESS.md) for the current
> state and [Limitations](#limitations).

- [Features](#features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Outputs](#outputs)
- [Reading the results](#reading-the-results)
- [Configuration](#configuration)
- [Examples](#examples)
- [Benchmarks](#benchmarks)
- [Limitations](#limitations)
- [Documentation](#documentation)
- [Development](#development)
- [References](#references)
- [License](#license)

## Features

- **Two input files.** A protein PDB and a ligand SDF; an SDF with several molecules gives one
  calculation per molecule. Every other setting has a recommended default, documented with its
  basis, and can be changed in YAML.
- **Soluble and membrane proteins.** With `membrane: {enabled: true}` the complex is oriented and
  embedded in a POPC bilayer. cuFEP never decides this by itself.
- **An established protocol.** Double decoupling with Boresch restraints chosen automatically
  from the interactions between protein and ligand in MD of the complex,
  Hamiltonian replica exchange with REST2 on the ligand and its pocket, and MBAR. Defaults and
  terms follow the literature and published open methods, and the documentation gives the source
  of each default.
- **Lambda windows.** Adaptive lambda scheduling by default (Midgley et al. 2025), or a fixed
  number or list of lambda values per leg and stage.
- **Checks from the literature only.** The overlap of neighbouring lambda windows (at least 0.03)
  and the number of uncorrelated samples per window (at least 50). Everything else (forward and
  reverse series, halves, blocks, BAR) is reported as numbers without a verdict.
- **All GPUs.** The replicas of a calculation are spread over all free GPUs in proportion to the
  work, several per GPU through NVIDIA MPS. One GPU works too.
- **Resumable.** A run can be stopped and continued; a single leg can be run again, longer or
  with more lambda windows, while the rest keeps its results.
- **Benchmarks** on a kinase (TYK2) and a GPCR in a bilayer (A2A) from the Schrödinger public
  binding free energy benchmark, against experiment and the FEP+ predictions published with it.

## How it works

For each ligand cuFEP computes the binding free energy from two legs: the ligand decoupled
from the protein (complex leg) and from water (solvent leg).

1. **Preparation.** The protein is prepared from the PDB as it is: hydrogens in the file are
   kept, missing heavy atoms rebuilt, chain breaks capped, missing loops never modelled. Ligand
   parameters come from OpenFF 2.2.1 with AM1-BCC charges. The system is solvated in water, or
   embedded in a bilayer ([protocol: Inputs](docs/protocol.md#inputs),
   [Systems](docs/protocol.md#systems)).
2. **Pre-equilibration** of the fully interacting system (minimisation, NVT, NPT, then 5 ns of
   MD in the complex leg).
3. **Boresch restraint.** Four unrestrained trajectories of the complex (3 ns each) are analysed
   with ProLIF, and the anchors are chosen from the interactions between protein and ligand
   that last (Ding et al. 2026). With `restraints.selection: auto` they are chosen from the
   trajectory of step 2 with the rules of OpenFE instead
   ([Steps of one leg](docs/protocol.md#steps-of-one-leg)).
4. **Lambda windows** chosen by adaptive lambda scheduling: a short precalculation inserts
   windows until neighbouring windows overlap enough
   ([Adaptive lambda scheduling](docs/protocol.md#adaptive-lambda-scheduling)).
5. **Hamiltonian replica exchange** with one replica per lambda window, REST2 on the ligand and
   the residues within 0.4 nm of it ([REST2](docs/protocol.md#rest2)); 1 ns of equilibration
   and 10 ns of production per window.
6. **MBAR** on samples decorrelated window by window, with a standard error from blocks of time,
   and the checks of the literature ([Checks of the analysis](docs/protocol.md#checks-of-the-analysis)).

## Requirements

- Linux with Docker and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).
- An NVIDIA GPU. The image is built for CUDA 13.4. It is tested on NVIDIA RTX PRO 6000 Blackwell
  Server Edition GPUs with driver 595.99.02, which runs CUDA 13.4 through the forward-compatibility
  package of the image; which other drivers and GPUs work follows NVIDIA's
  [CUDA compatibility rules](https://docs.nvidia.com/deploy/cuda-compatibility/latest/forward-compatibility.html).
- Disk space for the image and the runs.

Everything else (OpenMM 8.4, the cuPotential plugin, OpenFF, AmberTools, pymbar, PDBFixer,
RDKit) is in the Docker image.

## Installation

Build the image from this repository:

```bash
git clone https://github.com/ytworks/cuFEP.git
cd cuFEP
docker build -f docker/Dockerfile -t cufep:latest .
```

The build downloads NVIDIA's cuPotential runtime, which is proprietary NVIDIA software; read
[docs/LICENSING.md](docs/LICENSING.md) before publishing an image that contains it. An image with
stock OpenMM only, without NVIDIA proprietary software, is built with
`--build-arg WITH_CUPOTENTIAL=0`:

```bash
docker build -f docker/Dockerfile --build-arg WITH_CUPOTENTIAL=0 -t cufep:stock .
```

No pre-built image is published yet.

## Quick start

```bash
# a shell function for the commands below
cufep() { docker run --rm --gpus all --shm-size=16g -u "$(id -u):$(id -g)" \
                     -v "$PWD":/work cufep:latest "$@"; }

cufep check-env                                   # software, GPUs, both backends
cufep init protein.pdb ligand.sdf -o config.yaml  # write a minimal configuration
cufep plan config.yaml                            # what will be done, without simulating
cufep run config.yaml                             # run or resume
```

`--shm-size=16g` (or `--ipc=host`) matters: replica coordinates are shared between worker
processes in memory, and Docker's default of 64 MB is too small. Without it cuFEP falls back to
files in the output directory and says so.

`cufep run` prints one line per ligand after the time of day, for example

```text
RESULT ejm_31: dG_bind = ... ± ... kcal/mol (standard error from time blocks, 1 repeat)
```

## Outputs

| file | content |
|---|---|
| `resolved_config.yaml`, `auto_report.md` | every setting that was used, and why the computed ones were chosen |
| `protein_prepared.pdb` | the protein as simulated |
| `summary.csv`, `summary.json` | binding free energies, uncertainties, messages |
| `timing.json` | time per phase and job, GPU utilisation |
| `<ligand>/repeat_<n>/<leg>/` | restraint, lambda windows (`schedule.yaml`), samples, result (`result.json`) and log of the leg; in the complex leg also the trajectories the restraint was chosen from (`restraint_md/`) and their interactions (`interactions.json`, `interactions.csv`, `interactions.png`) |
| `<ligand>/repeat_<n>/analysis.png` | forward and reverse series, halves and blocks of the production, and the overlap matrix of both legs |

One calculation per ligand is run by default. Its uncertainty is the
standard error of the free energy as it is reported (MBAR minus the tail correction), from blocks
of the production that are the same for all lambda windows, and never below the analytical one;
the standard error MBAR gives is kept next to it (`mbar_error`). `sampling.repeats` runs the
calculation several times independently and reports the spread.

## Reading the results

The analysis checks every leg ([protocol: Checks of the analysis](docs/protocol.md#checks-of-the-analysis)).
The levels are those of the literature (Klimovich, Shirts and Mobley 2015; Mey et al. 2020).
Nothing is run again automatically: you run the leg concerned again, and the rest of the
calculation keeps its results.

| message | meaning | what to do |
|---|---|---|
| `NOT RELIABLE (overlap)` | two neighbouring lambda windows overlap by less than 0.03 in the production, among all iterations or among the samples the free energy is estimated from. Without overlap the free energy cannot be used | give that stage more lambda windows and start the leg anew; if only the samples of the estimate are below the level, a longer production also helps |
| `few independent samples` | a lambda window gives MBAR fewer than 50 uncorrelated samples (Mey et al. 2020); the free energy and its uncertainty rest on few samples | run the leg longer; it continues from where it stopped |
| `FAILED` | a leg did not finish, was interrupted, or its samples determine no free energy; the ligand has no binding free energy from that leg | read the message, then run the leg again |
| `LEGS MADE UNDER DIFFERENT CONDITIONS` | the complex and the solvent leg (or two repeats) differ in temperature, pressure, prepared systems or tail correction; the ligand gets no binding free energy | run the legs that differ again under the conditions of the others |
| `old analysis` | a leg that this run did not sample has a result of an earlier version of the analysis and its samples are gone | run that leg again |
| `the standard error of this leg is the analytical one ...` | the leg is too short for blocks of time, so its standard error takes the samples as independent | run the leg longer |
| `parts of the production without an estimate` | some points of the forward and reverse series or some blocks have no determined free energy, for example too few samples in that part; they are left out of `analysis.png` | nothing |
| `parts of the production whose lambda windows overlap below the level of a reliable estimate` | some points of the series, halves or blocks rest on an overlap below 0.03; their estimates are shown in `analysis.png` with this note | nothing |
| `checks of this version not made` | `cufep benchmark report` read a summary written by an earlier version of cuFEP | analyse the legs again |
| `origin of files not recorded` | the leg used files of an earlier version of cuFEP that do not say what they were made from | a new output directory, if that matters |

These are the flags and the notes of the analysis. Only `NOT RELIABLE` and `few independent
samples` judge the analysis, by the two checks of the literature; the others say that something is
missing, cannot be compared or rests on less than the whole production. Other warnings of a run
(for example a fall-back to stock OpenMM, an incomplete repeat, the graph symmetry of the ligand,
what the preparation changed) are printed under the result line and kept in `summary.json`.
Other quantities are reported as numbers: how the bilayer changed at
the end of its equilibration (`bilayer`), how far the ligand was from its input pose
(`ligand_rmsd_from_input_A`) and how far it moved in the coupled state (`structure`), the
forward and reverse series, the halves, the blocks and BAR. No check can show that one run is
converged: a slow change that does not occur during the run is invisible in it.

### Running a leg again

```bash
# longer: the complex leg of one ligand goes on from 10 to 20 ns per lambda window
# (--production-ns is the length in total, not what is added)
cufep run config.yaml --ligand ejm_31 --leg complex --production-ns 20

# more lambda windows: change lambda_schedule in config.yaml (for example
# adaptive.initial_windows.complex.vanish: 13), then start that leg anew
cufep run config.yaml --ligand ejm_31 --leg complex --reschedule

# look at a leg while it is being sampled (up to its last checkpoint)
cufep analyze runs/my_ligand/ejm_31/repeat_0/complex
```

With `--reschedule` the windows and samples the leg had are moved to `previous_<n>/` in its
directory; its pre-equilibration and restraint are used again. A leg that runs again keeps what
it gave before as `result.previous.json`. It continues only under the conditions and from the
input its files were made with; a run with another temperature, another input file or other
settings of the preparation stops and asks for a new `output.dir`
([protocol: Running again in the same output directory](docs/protocol.md#running-again-in-the-same-output-directory)).

## Configuration

The smallest configuration:

```yaml
system:
  protein: protein.pdb
  ligand: ligand.sdf        # several molecules in the SDF = one calculation each
```

`cufep init --full` writes every setting with its default, and
[docs/configuration.md](docs/configuration.md) explains each one with the basis of its default.
The ones most often changed:

```yaml
membrane: {enabled: false}           # true builds a POPC bilayer around the protein
backend: {name: cupotential}         # or openmm (stock); falls back to openmm if needed
sampling:
  production_ns: 10                  # per lambda window; or {complex: 20, solvent: 10}
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

`cufep plan config.yaml` prints the settings a run would use: in fixed mode its lambda windows, in
adaptive mode the rule and the windows the precalculation starts from (the windows themselves are
chosen when the run starts). Every run writes the settings it used to `resolved_config.yaml`.

## Examples

| directory | system | experiment (kcal/mol) |
|---|---|---|
| [examples/kinase_tyk2](examples/kinase_tyk2) | TYK2 kinase (PDB 4GIH) with ligand ejm_31, in water | -9.54 |
| [examples/membrane_a2a](examples/membrane_a2a) | adenosine A2A receptor (PDB 5OLZ) with compound 4e, in a POPC bilayer | -12.07 |

```bash
cufep run examples/kinase_tyk2/config.yaml
cufep run examples/membrane_a2a/config.yaml

# a quick check of the pipeline: one ligand with very short sampling
cufep benchmark run tyk2 --tier smoke
```

Both examples set 5 ns of production per lambda window (the default is 10 ns); every other
setting is the default. The ligand of each example is in its benchmark set, and the full benchmark
runs gave -9.84 ± 0.33 kcal/mol for TYK2 ejm_31 and -18.91 ± 0.52 kcal/mol for A2A 4e (standard
errors of the one calculation; [benchmarks/results](benchmarks/results/README.md)). One calculation
of each example with the same settings on an earlier commit gave -8.94 ± 0.77 and -11.95 ± 0.73
([docs/configuration.md](docs/configuration.md), `sampling.rest2.region`): runs of a complex leg
differ among themselves by more than the standard error of one run
([protocol: REST2](docs/protocol.md#rest2)).

## Benchmarks

```bash
cufep benchmark list
cufep benchmark run tyk2 --tier smoke     # one ligand, very short: checks the pipeline
cufep benchmark run tyk2 --tier subset    # five ligands
cufep benchmark run a2a --tier full       # every ligand
cufep benchmark report tyk2 runs/benchmark_tyk2_subset
```

| set | target | ligands |
|---|---|---|
| `tyk2` | TYK2 kinase (PDB 4GIH) | 16 |
| `a2a` | adenosine A2A receptor (PDB 5OLZ), in a bilayer | 17 compounds (29 states) |

A set runs with the settings of its example (`examples/kinase_tyk2/config.yaml`,
`examples/membrane_a2a/config.yaml`: 5 ns of production per lambda window, every other setting
the default); only the input files, the choice of ligands and the output directory differ (the
smoke tier also shortens the sampling), and the example's ligand is one of the set's. One calculation per ligand is run; `--repeats` runs more. The inputs come from the
[Schrödinger public binding free energy benchmark](https://github.com/schrodinger/public_binding_free_energy_benchmark)
(Ross et al. 2023, MIT licence; [benchmarks/README.md](benchmarks/README.md)). The report
compares cuFEP with experiment and with the FEP+ predictions published with the set. The FEP+
values come from relative calculations shifted to the experimental mean, so the report separates
the raw errors (cuFEP only) from the like-for-like statistics. Every ligand with a result is
compared, NOT RELIABLE ones included; the report also compares the NOT RELIABLE ligands and the
others apart and says where each NOT RELIABLE leg is below the level of overlap.

The full sets were run once, with the settings of the examples: TYK2 16 of 16 ligands, A2A 17 of
17 compounds, no result NOT RELIABLE. The RMSE after removing the mean error is 1.74 kcal/mol for
TYK2 and 2.26 for A2A (FEP+ 0.47 and 1.08), Kendall tau 0.12 and 0.38 (FEP+ 0.81 and 0.59), and
the mean signed error -0.62 and -4.12. The results are in
[benchmarks/results](benchmarks/results/README.md); the comparison with experiment, FEP+ and
another ABFE method is on the accuracy page of the wiki
([docs/wiki/Accuracy.md](docs/wiki/Accuracy.md), Japanese [docs/wiki/Accuracy-ja.md](docs/wiki/Accuracy-ja.md)).

## Limitations

Version 0.1:

- neutral ligands only (a ligand with a net charge stops with a message); no cofactors;
- no symmetry correction: the graph symmetry number of the ligand is reported, and a small
  symmetric ligand that turns over in its pocket leaves the unrestrained window without overlap,
  which the result reports;
- symmetric POPC-type bilayers only; no grand canonical water sampling;
- REST2 covers the ligand and the residues within 0.4 nm of it; a slow rearrangement further away
  is sampled faster only if its residues are named (`sampling.rest2.residues`).

See [protocol: Limits of version 0.1](docs/protocol.md#limits-of-version-01).

## Documentation

| document | content |
|---|---|
| [docs/protocol.md](docs/protocol.md) | the method: systems, thermodynamic cycle, lambda windows, REST2, analysis and its checks, GPUs, comparison with OpenFE's protocol |
| [docs/configuration.md](docs/configuration.md) | every setting, its default and the basis of the default |
| [docs/LICENSING.md](docs/LICENSING.md) | licences of cuFEP, of the cuPotential runtime and of the benchmark inputs |
| [docs/M0_REPORT.md](docs/M0_REPORT.md) | the first risk checks of the plugin and the platform |
| [docs/plans/](docs/plans/README.md) | implementation plans (in Japanese) |
| [PROGRESS.md](PROGRESS.md) | the current state of the project |
| [examples/README.md](examples/README.md), [benchmarks/README.md](benchmarks/README.md) | the examples and the benchmark sets |

## Development

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Development runs in the same
Docker image, through `scripts/in_docker.sh`:

```bash
scripts/in_docker.sh python -m pytest -q                     # unit tests on the CPU
scripts/in_docker.sh --gpus 0 python -m pytest tests/gpu -q  # GPU tests on GPU 0
```

The rules of the project for people and coding agents alike are in [CLAUDE.md](CLAUDE.md) (read
by Claude Code; [AGENTS.md](AGENTS.md) points other agents to it) and in
[.claude/rules/](.claude/rules/fep-invariants.md).

## References

cuFEP implements published methods. If you use it, please cite the methods your results rest
on:

- Boresch, Tettinger, Leitgeb and Karplus, "Absolute binding free energies: a quantitative
  approach for their calculation", J. Phys. Chem. B 2003, 107, 9535 (Boresch restraints).
- Wang, Friesner and Berne, "Replica exchange with solute scaling: a more efficient version of
  replica exchange with solute tempering (REST2)", J. Phys. Chem. B 2011, 115, 9431.
- Shirts and Chodera, "Statistically optimal analysis of samples from multiple equilibrium
  states", J. Chem. Phys. 2008, 129, 124105 (MBAR).
- Ding, Liu, Wang, Xia, Wang and Yan, "Development and large-scale benchmarks of a
  protein-ligand absolute binding free energy toolkit", arXiv:2603.22274 (the anchors of the
  restraint from the interactions of an MD).
- Midgley, Bariami, Habgood and Mackey, "Adaptive Lambda Scheduling: A Method for Computational
  Efficiency in Free Energy Perturbation Simulations", J. Chem. Inf. Model. 2025, 65, 512.
- Klimovich, Shirts and Mobley, "Guidelines for the analysis of free energy calculations",
  J. Comput. Aided Mol. Des. 2015, 29, 397.
- Mey et al., "Best Practices for Alchemical Free Energy Calculations", Living J. Comput. Mol.
  Sci. 2020, 2, 18378.
- Eastman et al., "OpenMM 8: Molecular Dynamics Simulation with Machine Learning Potentials",
  J. Phys. Chem. B 2024, 128, 109.
- Ross et al., "The maximal and current accuracy of rigorous protein-ligand binding free energy
  calculations", Commun. Chem. 2023, 6, 222 (the benchmark sets).

## License

cuFEP is released under the [Apache License 2.0](LICENSE). The cuPotential runtime that the
image downloads is proprietary NVIDIA software under the NVIDIA Software License Agreement; it
also restricts publishing performance results of NVIDIA products, so the documentation of cuFEP
contains no timings of the plugin. The benchmark inputs come from the Schrödinger public binding
free energy benchmark (MIT licence, `benchmarks/THIRD_PARTY/`). The other software in the image
keeps its own licence. Details: [docs/LICENSING.md](docs/LICENSING.md).
