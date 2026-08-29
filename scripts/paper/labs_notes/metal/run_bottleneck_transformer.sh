#!/usr/bin/env bash
# Paper cell: Labs + Notes x Bottleneck Transformer, seed ${SEED:-1}.
# Notes are the subsetted clinical headers — radiology: indication and
# impression; discharge: chief complaint. That subsetting lives in
# NotesLabsMIMIC4, not here. Bio_ClinicalBERT stays frozen and its
# [CLS] vectors are cached, capped at $FROZEN_TEXT_CACHE entries.
# Clip norm and Adam epsilon come from PROTOCOL_FLAGS like every other
# backbone. The runner used to hardcode 0.5 / 1e-6 for this model alone.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../../common.sh"

launch "paper_labs_notes_bottleneck_transformer_seed${SEED}" \
  --task notes_labs --model bottleneck_transformer \
  --ehr-root "$EHR_ROOT" \
  --note-root "$NOTE_ROOT" \
  --cache-dir "${CACHE_DIR:-$HOME/pyhealth_cache/paper_notes_labs}" \
  "${PROTOCOL_FLAGS[@]}" \
  --freeze-encoder --max-frozen-text-cache "$FROZEN_TEXT_CACHE" \
  --heads 4 --num-layers 2 --bottlenecks-n 4 --fusion-startidx 1
