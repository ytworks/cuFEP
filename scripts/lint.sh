#!/usr/bin/env bash
# Run ruff, pinned here, from its official image on this repository: the same version locally and
# in CI (job `harness`). The rules are in pyproject.toml.
#
#   scripts/lint.sh               # ruff check .
#   scripts/lint.sh --fix         # further options go to `ruff check`
#   scripts/lint.sh src/cufep     # paths narrow the check (default: the whole repository)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUFF_IMAGE="ghcr.io/astral-sh/ruff:0.17.5"

PATHS=(.)
for arg in "$@"; do
    if [[ "$arg" != -* ]]; then PATHS=(); break; fi
done
exec docker run --rm --user "$(id -u):$(id -g)" -v "$ROOT":/io -w /io "$RUFF_IMAGE" check --no-cache "$@" \
    ${PATHS[@]+"${PATHS[@]}"}
