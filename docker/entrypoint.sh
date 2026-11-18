#!/bin/bash
# cuFEP container entrypoint.
#   docker run IMAGE run config.yaml        -> cufep run config.yaml
#   docker run IMAGE python script.py       -> python script.py   (also bash, pytest, pip)
set -e

case "${1:-}" in
  python|python3|bash|sh|pytest|pip|nvidia-smi)
    exec "$@"
    ;;
  *)
    exec cufep "$@"
    ;;
esac
