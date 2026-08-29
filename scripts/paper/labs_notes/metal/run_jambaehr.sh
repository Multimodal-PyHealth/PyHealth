#!/usr/bin/env bash
# Paper cell: Labs + Notes x JambaEHR, seed ${SEED:-1}.
# Notes are the subsetted clinical headers — radiology: indication and
# impression; discharge: chief complaint. That subsetting lives in
# NotesLabsMIMIC4, not here. Bio_ClinicalBERT stays frozen and its
# [CLS] vectors are cached, capped at $FROZEN_TEXT_CACHE entries.
# Depth is --jamba-transformer-layers / --jamba-mamba-layers. The spec also
# lists --num-layers=2 for JambaEHR, but the model never reads it, so it is
# not passed.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../../common.sh"

launch "paper_labs_notes_jambaehr_seed${SEED}" \
  --task notes_labs --model jambaehr \
  --ehr-root "$EHR_ROOT" \
  --note-root "$NOTE_ROOT" \
  --cache-dir "${CACHE_DIR:-$HOME/pyhealth_cache/paper_notes_labs}" \
  "${PROTOCOL_FLAGS[@]}" \
  --freeze-encoder --max-frozen-text-cache "$FROZEN_TEXT_CACHE" \
  --heads 4 --jamba-transformer-layers 2 --jamba-mamba-layers 6
