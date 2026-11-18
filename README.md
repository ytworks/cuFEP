# cuFEP

Absolute binding free energy (ABFE) calculations on OpenMM, with NVIDIA
[cuPotential](https://github.com/NVIDIA/cuPotential) as the default compute backend
and stock OpenMM as a second, always-available backend.

**Status: under construction.** The implementation plan and its milestones are in
[docs/PLAN.md](docs/PLAN.md).

Goals:

- Inputs are a protein PDB and a ligand SDF (the ligand already placed in the pocket).
- Everything else has working defaults or is chosen automatically, and is set through YAML.
- Soluble and membrane proteins (lipid bilayer built automatically).
- Adaptive lambda windows by default, Hamiltonian replica exchange, MBAR.
- Runs on one GPU or many.
- An official Docker image, and reproducible benchmarks against experiment and FEP+.

## Docker

```bash
docker build -f docker/Dockerfile -t cufep:latest .
docker run --rm --gpus all cufep:latest --help
```

The image downloads NVIDIA's cuPotential runtime at build time; it is governed by the
NVIDIA Software License Agreement (copied to `/licenses` in the image). Build with
`--build-arg WITH_CUPOTENTIAL=0` for a stock-OpenMM-only image.

## Licence

cuFEP is Apache-2.0. Benchmark inputs under `benchmarks/` come from the Schrödinger
public binding free energy benchmark (MIT); see `benchmarks/THIRD_PARTY/`.
