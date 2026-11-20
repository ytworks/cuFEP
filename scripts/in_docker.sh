#!/usr/bin/env bash
# Run a command inside the official cuFEP image with this repository mounted at /work.
# Everything except CI is executed this way (.claude/rules/docker-execution.md).
#
#   scripts/in_docker.sh python -m pytest -q
#   scripts/in_docker.sh --gpus 0 python -m pytest tests/gpu -q
#   scripts/in_docker.sh --gpus 0,1,2,3 --detach --name tyk2 cufep run examples/kinase_tyk2/config.yaml
#
# Options (before the command):
#   --gpus LIST      host GPU numbers (comma separated) or `all`; no GPU when omitted. Inside the
#                    container the GPUs passed are numbered from 0 again. Any other value is
#                    passed to `docker run --gpus` as it is (e.g. '"device=1"').
#   --detach         start in the background; follow with `docker logs -f NAME`. The container is
#                    kept after it ends (logs, exit status).
#   --name NAME      container name. A finished container of that name that this script started
#                    from this repository is replaced, so the same command continues a stopped
#                    run; any other container of that name is left alone (error).
#   --mount SRC:DST  additional bind mount (repeatable)
#   --image IMAGE    image to run (default: $CUFEP_IMAGE, else cufep:dev)
#   --build          build the image from docker/Dockerfile first
#
# The mounted source is the one that runs (PYTHONPATH=/work/src, CUFEP_DATA=/work), as the
# calling user, so files written under the repository belong to that user. A run whose result
# matters uses an image built from a commit instead (spikes/run_frozen.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${CUFEP_IMAGE:-cufep:dev}"
GPUS=""
DETACH=0
BUILD=0
NAME=""
MOUNTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --detach) DETACH=1; shift ;;
        --name) NAME="$2"; shift 2 ;;
        --mount) MOUNTS+=(-v "$2"); shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --build) BUILD=1; shift ;;
        --) shift; break ;;
        -*) echo "in_docker.sh: unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

if [[ $BUILD -eq 1 ]] || ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    docker build -f "$ROOT/docker/Dockerfile" -t "$IMAGE" "$ROOT" >&2
fi
if [[ $# -eq 0 ]]; then
    echo "in_docker.sh: no command given" >&2
    exit 2
fi

# Start in the directory that corresponds to the caller's, when that is inside the repository.
WORKDIR=/work
case "$PWD/" in
    "$ROOT"/*) WORKDIR="/work${PWD#"$ROOT"}" ;;
esac

ARGS=(--user "$(id -u):$(id -g)" -v "$ROOT":/work -w "$WORKDIR"
      --shm-size="${SHM_SIZE:-16g}" --ulimit core=0
      -e PYTHONPATH=/work/src
      -e CUFEP_DATA=/work
      -e CUDA_MPS_PIPE_DIRECTORY=/tmp/cufep-no-mps
      -e OPENMM_CPU_THREADS="${OPENMM_CPU_THREADS:-4}")
if [[ "$GPUS" == all ]]; then
    ARGS+=(--gpus all)
elif [[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    ARGS+=(--gpus "\"device=$GPUS\"")
elif [[ -n "$GPUS" ]]; then
    ARGS+=(--gpus "$GPUS")
fi
if [[ -n "$NAME" ]]; then
    FOUND="$(docker container inspect \
        --format '{{.State.Status}} {{index .Config.Labels "cufep.repository"}}' "$NAME" 2> /dev/null || true)"
    if [[ "$FOUND" == "exited $ROOT" ]]; then
        docker rm "$NAME" > /dev/null
    elif [[ -n "$FOUND" ]]; then
        echo "in_docker.sh: a container named $NAME exists (${FOUND%% *}); stop it or use another name" >&2
        exit 2
    fi
    ARGS+=(--name "$NAME" --label "cufep.repository=$ROOT")
fi
if [[ $DETACH -eq 1 ]]; then
    ARGS+=(--detach)
else
    ARGS+=(--rm --interactive)
    if [[ -t 0 && -t 1 ]]; then ARGS+=(--tty); fi
fi

# The command is run as it is (`cufep ...`, `python ...`, `bash -c ...`), found on the PATH of
# the image, instead of through the image's entrypoint, which prefixes `cufep` to most commands.
exec docker run "${ARGS[@]}" ${MOUNTS[@]+"${MOUNTS[@]}"} --entrypoint /usr/bin/env "$IMAGE" "$@"
