#!/usr/bin/env bash
# Paper cell: notes_labs x MLP x seed 1 — BERT UNFROZEN (trained end-to-end), no [CLS] cache.
# Identical to run_notes_labs_mlp_seed1.sh except --freeze-encoder is dropped.
# The cache flag is kept for config parity (inert when the encoder trains). --text-grad-checkpoint-rows
# bounds activation memory (per-layer checkpointing + 256-row chunks); the math is unchanged.
# Tree = cache1m + the gradient-checkpointing commit.
set -euo pipefail
TREE="${TREE:-/home/rianatri/ml4h-tranche1-unfrozen}"
GPU="${GPU:-0}"
EHR_ROOT="${EHR_ROOT:-/shared/rsaas/physionet.org/files/mimiciv/2.2}"
NOTE_ROOT="${NOTE_ROOT:-/shared/rsaas/physionet.org/files/mimic-note}"
CACHE_DIR="${CACHE_DIR:-/home/rianatri/pyhealth_cache/tranche1_v4_notes_labs}"
NAME=notes_labs_mlp_seed1_unfrozen
cd "$TREE"; mkdir -p logs output
PYTHONPATH="$TREE" CUDA_VISIBLE_DEVICES="$GPU" nohup python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \
  --task notes_labs --model mlp \
  --ehr-root "$EHR_ROOT" --note-root "$NOTE_ROOT" --cache-dir "$CACHE_DIR" \
  --output-dir "$TREE/output/tranche1_$NAME" \
  --embedding-dim 128 --hidden-dim 128 --batch-size 32 --lr 1e-4 --dropout 0.1 \
  --epochs 50 --patience 5 --seed 1 --use-amp --amp-dtype bf16 \
  --max-frozen-text-cache 1000000 --text-grad-checkpoint-rows 256 \
  > "logs/$NAME.out" 2>&1 &
echo $! > "logs/$NAME.pid"
echo "launched $NAME pid $(cat logs/$NAME.pid) on GPU $GPU"
