#!/usr/bin/env bash
# Paper cell: Labs + Notes + CXR x Bottleneck Transformer, seed ${SEED:-1}.
# Same note subsetting as labs_notes, plus CXR: patch tokens from the
# vision encoder are mean-pooled to one vector per image event, inside
# UnifiedMultimodalEmbeddingModel.
# Clip norm and Adam epsilon come from PROTOCOL_FLAGS like every other
# backbone. The runner used to hardcode 0.5 / 1e-6 for this model alone.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../../common.sh"
: "${CXR_ROOT:?set CXR_ROOT to the MIMIC-CXR root}"

launch "paper_labs_notes_cxr_bottleneck_transformer_seed${SEED}" \
  --task notes_labs_cxr --model bottleneck_transformer \
  --ehr-root "$EHR_ROOT" \
  --note-root "$NOTE_ROOT" \
  --cxr-root "$CXR_ROOT" \
  --cxr-variant sunlab \
  --cache-dir "${CACHE_DIR:-$HOME/pyhealth_cache/paper_notes_labs_cxr}" \
  "${PROTOCOL_FLAGS[@]}" \
  --freeze-encoder --max-frozen-text-cache "$FROZEN_TEXT_CACHE" \
  --heads 4 --num-layers 2 --bottlenecks-n 4 --fusion-startidx 1
