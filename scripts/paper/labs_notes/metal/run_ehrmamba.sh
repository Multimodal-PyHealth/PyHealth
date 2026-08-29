#!/usr/bin/env bash
# Paper cell: Labs + Notes x EHRMamba, seed ${SEED:-1}.
# Notes are the subsetted clinical headers — radiology: indication and
# impression; discharge: chief complaint. That subsetting lives in
# NotesLabsMIMIC4, not here. Bio_ClinicalBERT stays frozen and its
# [CLS] vectors are cached, capped at $FROZEN_TEXT_CACHE entries.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../../common.sh"

launch "paper_labs_notes_ehrmamba_seed${SEED}" \
  --task notes_labs --model ehrmamba \
  --ehr-root "$EHR_ROOT" \
  --note-root "$NOTE_ROOT" \
  --cache-dir "${CACHE_DIR:-$HOME/pyhealth_cache/paper_notes_labs}" \
  "${PROTOCOL_FLAGS[@]}" \
  --freeze-encoder --max-frozen-text-cache "$FROZEN_TEXT_CACHE" \
  --num-layers 2 --mamba-state-size 16 --mamba-conv-kernel 4
