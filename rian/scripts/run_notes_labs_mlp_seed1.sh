#!/usr/bin/env bash
# Paper cell: notes_labs x MLP x seed 1 (cache-1M compute-matched rerun of the original 200k-cache cell).
# Full stay, empty-sequence missingness, frozen BERT, bf16 AMP, 1e6-entry [CLS] cache.
# Tree = 8d4a4c9 + the frozen-cache-cap commit only (results-identical, faster epochs).
set -euo pipefail
TREE="${TREE:-/home/rianatri/ml4h-tranche1-cache1m}"
GPU="${GPU:-0}"
EHR_ROOT="${EHR_ROOT:-/shared/rsaas/physionet.org/files/mimiciv/2.2}"
NOTE_ROOT="${NOTE_ROOT:-/shared/rsaas/physionet.org/files/mimic-note}"
CACHE_DIR="${CACHE_DIR:-/home/rianatri/pyhealth_cache/tranche1_v4_notes_labs}"
NAME=notes_labs_mlp_seed1_cache1m
cd "$TREE"; mkdir -p logs output
PYTHONPATH="$TREE" CUDA_VISIBLE_DEVICES="$GPU" nohup python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \
  --task notes_labs --model mlp \
  --ehr-root "$EHR_ROOT" --note-root "$NOTE_ROOT" --cache-dir "$CACHE_DIR" \
  --output-dir "$TREE/output/tranche1_$NAME" \
  --embedding-dim 128 --hidden-dim 128 --batch-size 32 --lr 1e-4 --dropout 0.1 \
  --epochs 50 --patience 5 --seed 1 --use-amp --amp-dtype bf16 --freeze-encoder \
  --max-frozen-text-cache 1000000 \
  > "logs/$NAME.out" 2>&1 &
echo $! > "logs/$NAME.pid"
echo "launched $NAME pid $(cat logs/$NAME.pid) on GPU $GPU"
