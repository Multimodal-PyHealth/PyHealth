#!/usr/bin/env bash
# Paper cell: Labs x Bottleneck Transformer, seed ${SEED:-1}.
# No text encoder in this task, so --freeze-encoder and
# --max-frozen-text-cache are deliberately not passed: they would do
# nothing, and a flag that does nothing does not belong in a config.
# Clip norm and Adam epsilon come from PROTOCOL_FLAGS like every other
# backbone. The runner used to hardcode 0.5 / 1e-6 for this model alone.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../../common.sh"

launch "paper_labs_bottleneck_transformer_seed${SEED}" \
  --task labs --model bottleneck_transformer \
  --ehr-root "$EHR_ROOT" \
  --cache-dir "${CACHE_DIR:-$HOME/pyhealth_cache/paper_labs}" \
  "${PROTOCOL_FLAGS[@]}" \
  --heads 4 --num-layers 2 --bottlenecks-n 4 --fusion-startidx 1
