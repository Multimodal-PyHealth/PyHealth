#!/usr/bin/env bash
# Paper cell: Labs x JambaEHR, seed ${SEED:-1}.
# No text encoder in this task, so --freeze-encoder and
# --max-frozen-text-cache are deliberately not passed: they would do
# nothing, and a flag that does nothing does not belong in a config.
# Depth is --jamba-transformer-layers / --jamba-mamba-layers. The spec also
# lists --num-layers=2 for JambaEHR, but the model never reads it, so it is
# not passed.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../../common.sh"

launch "paper_labs_jambaehr_seed${SEED}" \
  --task labs --model jambaehr \
  --ehr-root "$EHR_ROOT" \
  --cache-dir "${CACHE_DIR:-$HOME/pyhealth_cache/paper_labs}" \
  "${PROTOCOL_FLAGS[@]}" \
  --heads 4 --jamba-transformer-layers 2 --jamba-mamba-layers 6
