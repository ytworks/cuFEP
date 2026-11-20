#!/bin/bash
# Run a script inside the cufep:dev image with the working tree mounted.
#   spikes/run.sh spikes/m0_01_plugin_waterbox.py [args...]
# Environment: GPUS (docker --gpus value, default "all"), IMAGE (default cufep:dev).
# The container is set up by scripts/in_docker.sh; GPUS is passed to docker as it is and the
# script runs in /work, whatever directory it is called from, as before; an image that does not
# exist is not built or pulled (scripts/in_docker.sh --build builds it).
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
exec "$repo/scripts/in_docker.sh" --docker-gpus "${GPUS:-all}" --workdir /work \
    --image "${IMAGE:-cufep:dev}" python "$@"
