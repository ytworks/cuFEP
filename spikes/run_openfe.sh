#!/bin/bash
# Run a script inside the OpenFE reference image with the working tree mounted.
#   GPUS='"device=3"' spikes/run_openfe.sh spikes/m4_openfe_reference.py --protein ... --ligand ... --out ...
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
exec docker run --rm -i --gpus "${GPUS:-all}" --shm-size=8g --ulimit core=0 \
  -u "$(id -u):$(id -g)" -v "$repo":/work -w /work \
  -e CUDA_MPS_PIPE_DIRECTORY=/tmp/cufep-no-mps -e OPENMM_CPU_THREADS=4 \
  --entrypoint python cufep-openfe-ref:latest "$@"
