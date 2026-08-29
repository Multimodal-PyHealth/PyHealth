#!/usr/bin/env bash
# Paper cell: Labs x MLP, seed ${SEED:-1}.
# No text encoder in this task, so --freeze-encoder and
# --max-frozen-text-cache are deliberately not passed: they would do
# nothing, and a flag that does nothing does not belong in a config.
# MLP has no dropout parameter, so --dropout is inert here; the runner logs
# that and records it in run_config as inert_flags.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../../common.sh"

launch "paper_labs_mlp_seed${SEED}" \
  --task labs --model mlp \
  --ehr-root "$EHR_ROOT" \
  --cache-dir "${CACHE_DIR:-$HOME/pyhealth_cache/paper_labs}" \
  "${PROTOCOL_FLAGS[@]}"
