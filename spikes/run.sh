#!/bin/bash
# Run a script inside the cufep:dev image with the working tree mounted.
#   spikes/run.sh spikes/m0_01_plugin_waterbox.py [args...]
# Environment: GPUS (host GPU numbers, `all` or a docker --gpus value, default "all"),
# IMAGE (default cufep:dev). The container is set up by scripts/in_docker.sh.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
exec "$repo/scripts/in_docker.sh" --gpus "${GPUS:-all}" --image "${IMAGE:-cufep:dev}" python "$@"
