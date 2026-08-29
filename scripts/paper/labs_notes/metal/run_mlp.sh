#!/usr/bin/env bash
# Paper cell: Labs + Notes x MLP, seed ${SEED:-1}.
# Notes are the subsetted clinical headers — radiology: indication and
# impression; discharge: chief complaint. That subsetting lives in
# NotesLabsMIMIC4, not here. Bio_ClinicalBERT stays frozen and its
# [CLS] vectors are cached, capped at $FROZEN_TEXT_CACHE entries.
# MLP has no dropout parameter, so --dropout is inert here; the runner logs
# that and records it in run_config as inert_flags.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../../common.sh"

launch "paper_labs_notes_mlp_seed${SEED}" \
  --task notes_labs --model mlp \
  --ehr-root "$EHR_ROOT" \
  --note-root "$NOTE_ROOT" \
  --cache-dir "${CACHE_DIR:-$HOME/pyhealth_cache/paper_notes_labs}" \
  "${PROTOCOL_FLAGS[@]}" \
  --freeze-encoder --max-frozen-text-cache "$FROZEN_TEXT_CACHE"
