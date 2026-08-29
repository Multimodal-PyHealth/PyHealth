#!/usr/bin/env bash
# Paper cell: Labs + Notes x RNN, seed ${SEED:-1}.
# Notes are the subsetted clinical headers — radiology: indication and
# impression; discharge: chief complaint. That subsetting lives in
# NotesLabsMIMIC4, not here. Bio_ClinicalBERT stays frozen and its
# [CLS] vectors are cached, capped at $FROZEN_TEXT_CACHE entries.
# GRU, one layer, unidirectional. The spec does not pin RNN depth; it is
# pinned here so the config, not a library default, decides it.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../../common.sh"

launch "paper_labs_notes_rnn_seed${SEED}" \
  --task notes_labs --model rnn \
  --ehr-root "$EHR_ROOT" \
  --note-root "$NOTE_ROOT" \
  --cache-dir "${CACHE_DIR:-$HOME/pyhealth_cache/paper_notes_labs}" \
  "${PROTOCOL_FLAGS[@]}" \
  --freeze-encoder --max-frozen-text-cache "$FROZEN_TEXT_CACHE" \
  --rnn-type GRU --rnn-layers 1
