# Licensing notes

## cuFEP

cuFEP itself is released under the Apache License 2.0 (`LICENSE`).

## NVIDIA cuPotential

cuFEP uses NVIDIA's cuPotential OpenMM plugin as its default compute backend.

- The plugin's source is Apache-2.0. The cuPotential runtime library (`libcupo`) is
  proprietary and governed by the **NVIDIA Software License Agreement**
  (`LICENSE-cupo.txt` in the cuPotential repository).
- cuFEP does not contain or redistribute the runtime. `docker/Dockerfile` downloads
  NVIDIA's release channel from GitHub when *you* build the image and copies the NVIDIA
  licence texts to `/licenses` inside it.
- If you push an image that contains the runtime to a registry, you are redistributing it.
  The agreement allows distribution only as part of an application with material
  additional functionality and under its other conditions. Read it before publishing.
  An image built with `--build-arg WITH_CUPOTENTIAL=0` contains no NVIDIA proprietary
  software and has no such restriction.
- The agreement restricts disclosing benchmark or performance results of NVIDIA
  products without NVIDIA's written permission. cuFEP's reports and documentation
  therefore contain accuracy results only, and no timings of the plugin.

This is a summary for orientation, not legal advice.

## Benchmark inputs

`benchmarks/*/inputs/` contains unmodified files from the Schrödinger
[public binding free energy benchmark](https://github.com/schrodinger/public_binding_free_energy_benchmark)
(MIT licence, copyright 2023 Gregory A. Ross; see `benchmarks/THIRD_PARTY/`).
`benchmarks/*/ligands.sdf` and `ligands.csv` are derived from them by
`cufep/benchmarks/curate.py`. `examples/` reuses the same structures.

## Other software in the image

OpenMM (MIT and LGPL parts), OpenFF Toolkit and force fields (MIT / CC-BY), AmberTools
(mixed open-source licences), pymbar (MIT), MDTraj (LGPL), PDBFixer (MIT), RDKit (BSD),
ProLIF (Apache-2.0) and MDAnalysis (LGPL-3.0-or-later), through which ProLIF reads the frames
of a trajectory. ProLIF brings gemmi (MPL-2.0), pandas, networkx, dill, multiprocess and psutil
(BSD-3-Clause) and tqdm (MPL-2.0 and MIT). The licences are those that the packages of the
image state (`docker/environment.yml`: ProLIF 2.2.3, MDAnalysis 2.11).
