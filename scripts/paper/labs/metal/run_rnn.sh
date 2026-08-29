#!/usr/bin/env bash
# Paper cell: Labs x RNN, seed ${SEED:-1}.
# No text encoder in this task, so --freeze-encoder and
# --max-frozen-text-cache are deliberately not passed: they would do
# nothing, and a flag that does nothing does not belong in a config.
# GRU, one layer, unidirectional. The spec does not pin RNN depth; it is
# pinned here so the config, not a library default, decides it.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../../common.sh"

launch "paper_labs_rnn_seed${SEED}" \
  --task labs --model rnn \
  --ehr-root "$EHR_ROOT" \
  --cache-dir "${CACHE_DIR:-$HOME/pyhealth_cache/paper_labs}" \
  "${PROTOCOL_FLAGS[@]}" \
  --rnn-type GRU --rnn-layers 1
