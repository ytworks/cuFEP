#!/bin/bash
# Run a Python script inside the image with a private NVIDIA MPS daemon, so that several
# worker processes can use the same GPU at the same time.
#   GPUS='"device=1"' spikes/m3_speed_mps.sh spikes/m3_speed.py iter ... --devices 0,0,0,0 --mps
# Environment: GPUS, IMAGE, MPS_THREAD_PCT (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for the clients)
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
exec docker run --rm -i --gpus "${GPUS:-all}" --shm-size="${SHM_SIZE:-16g}" --ulimit core=0 \
  -u "$(id -u):$(id -g)" \
  -v "$repo":/work -w /work \
  -e PYTHONPATH=/work/src -e CUFEP_DATA=/work \
  -e MPS_THREAD_PCT="${MPS_THREAD_PCT:-}" \
  --entrypoint bash "${IMAGE:-cufep:dev}" -c '
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/cufep-mps CUDA_MPS_LOG_DIRECTORY=/tmp/cufep-mps-log
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d || { echo "could not start the MPS daemon" >&2; exit 3; }
    if [ -n "$MPS_THREAD_PCT" ]; then export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$MPS_THREAD_PCT"; fi
    python "$@"; rc=$?
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1
    exit $rc' bash "$@"
