#!/usr/bin/env bash
# Paper cell: Labs + Notes + CXR x JambaEHR, seed ${SEED:-1}.
# Same note subsetting as labs_notes, plus CXR: patch tokens from the
# vision encoder are mean-pooled to one vector per image event, inside
# UnifiedMultimodalEmbeddingModel.
# Depth is --jamba-transformer-layers / --jamba-mamba-layers. The spec also
# lists --num-layers=2 for JambaEHR, but the model never reads it, so it is
# not passed.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../../common.sh"
: "${CXR_ROOT:?set CXR_ROOT to the MIMIC-CXR root}"

launch "paper_labs_notes_cxr_jambaehr_seed${SEED}_${IMAGE_BACKBONE}" \
  --task notes_labs_cxr --model jambaehr \
  --ehr-root "$EHR_ROOT" \
  --note-root "$NOTE_ROOT" \
  --cxr-root "$CXR_ROOT" \
  --cxr-variant sunlab \
  --image-backbone "$IMAGE_BACKBONE" \
  --cache-dir "${CACHE_DIR:-$HOME/pyhealth_cache/paper_notes_labs_cxr}" \
  "${PROTOCOL_FLAGS[@]}" \
  --freeze-encoder --max-frozen-text-cache "$FROZEN_TEXT_CACHE" \
  --heads 4 --jamba-transformer-layers 2 --jamba-mamba-layers 6
