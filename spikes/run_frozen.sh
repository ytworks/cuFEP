#!/bin/bash
# Run the `cufep` command from a built image (code frozen at build time), with the working
# tree mounted only for inputs and outputs. Use this for long production runs so that later
# edits to the source tree cannot change a running calculation.
#   IMAGE=cufep:<sha> spikes/run_frozen.sh run config.yaml --gpus 3,4,5 -o runs/x
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
exec docker run --rm -i --gpus all --shm-size="${SHM_SIZE:-16g}" --ulimit core=0 \
  -u "$(id -u):$(id -g)" -v "$repo":/work -w /work \
  -e CUDA_MPS_PIPE_DIRECTORY=/tmp/cufep-no-mps \
  "${IMAGE:-cufep:latest}" "$@"
