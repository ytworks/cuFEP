#!/bin/bash
# Run the `cufep` command from a built image (code frozen at build time), with the working
# tree mounted only for inputs and outputs. Use this for long production runs so that later
# edits to the source tree cannot change a running calculation.
#   IMAGE=cufep:<sha> spikes/run_frozen.sh run config.yaml --gpus 3,4,5 -o runs/x
# The container gets all GPUs of the host, or those of $GPUS (host numbers; inside it they are
# numbered from 0 again). Two runs at the same time each take GPUs of their own this way, with
# an MPS daemon of their own that sees no GPU of the other:
#   GPUS=0,1,2,3 IMAGE=cufep:<sha> spikes/run_frozen.sh run a.yaml -o runs/a
#   GPUS=4,5,6,7 IMAGE=cufep:<sha> spikes/run_frozen.sh run b.yaml -o runs/b
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
gpus=all
if [[ -n "${GPUS:-}" ]]; then
  [[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "run_frozen.sh: GPUS takes host GPU numbers such as 0,1 (got $GPUS)" >&2; exit 2; }
  gpus="\"device=$GPUS\""
fi
exec docker run --rm -i --gpus "$gpus" --shm-size="${SHM_SIZE:-16g}" --ulimit core=0 \
  -u "$(id -u):$(id -g)" -v "$repo":/work -w /work \
  -e CUDA_MPS_PIPE_DIRECTORY=/tmp/cufep-no-mps \
  "${IMAGE:-cufep:latest}" "$@"
