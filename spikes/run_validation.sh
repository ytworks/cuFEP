#!/bin/bash
# Protocol validation (plan step 9): TYK2 ejm_31 and T4 lysozyme / toluene, three repeats each,
# with adaptive lambda scheduling and with the recommended fixed windows. Runs one after another
# from a frozen image on the given GPUs.
#   IMAGE=cufep:<sha> spikes/run_validation.sh 1,2,3,4,5 [names...]
set -uo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
gpus="${1:?comma separated GPU indices}"; shift
names=("$@"); [ ${#names[@]} -eq 0 ] && names=(tyk2_adaptive tyk2_fixed t4l_adaptive t4l_fixed)
for name in "${names[@]}"; do
  cfg="spikes/out/validation/$name.yaml"
  echo "$(date -u +%H:%M:%S) start $name"
  IMAGE="${IMAGE:-cufep:latest}" "$repo/spikes/run_frozen.sh" run "$cfg" --gpus "$gpus" > "$repo/runs/val_$name.log" 2>&1
  echo "$(date -u +%H:%M:%S) end $name (exit $?)"
done
