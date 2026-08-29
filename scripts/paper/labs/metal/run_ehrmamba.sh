#!/usr/bin/env bash
# Paper cell: Labs x EHRMamba, seed ${SEED:-1}.
# No text encoder in this task, so --freeze-encoder and
# --max-frozen-text-cache are deliberately not passed: they would do
# nothing, and a flag that does nothing does not belong in a config.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../../common.sh"

launch "paper_labs_ehrmamba_seed${SEED}" \
  --task labs --model ehrmamba \
  --ehr-root "$EHR_ROOT" \
  --cache-dir "${CACHE_DIR:-$HOME/pyhealth_cache/paper_labs}" \
  "${PROTOCOL_FLAGS[@]}" \
  --num-layers 2 --mamba-state-size 16 --mamba-conv-kernel 4
