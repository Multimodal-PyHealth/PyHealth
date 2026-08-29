#!/usr/bin/env bash
# notes_labs x Transformer (2 layers, 4 heads) x seed 1 — frozen BERT + 1e6-entry [CLS] cache.
# Full stay, empty-sequence missingness, bf16 AMP, batch 32, lr 1e-4, dropout 0.1, dim 128/128.
# Default: backgrounds python with nohup (bare metal). FOREGROUND=1 execs it (Slurm/Condor wrappers).
# GPU="" leaves CUDA_VISIBLE_DEVICES to the scheduler.
set -euo pipefail
TREE="${TREE:-/home/rianatri/ml4h-tranche1-cache1m}"
GPU="${GPU-0}"
EHR_ROOT="${EHR_ROOT:-/shared/rsaas/physionet.org/files/mimiciv/2.2}"
NOTE_ROOT="${NOTE_ROOT:-/shared/rsaas/physionet.org/files/mimic-note}"
CACHE_DIR="${CACHE_DIR:-/home/rianatri/pyhealth_cache/tranche1_v4_notes_labs}"
NAME=notes_labs_transformer_seed1_cache1m
cd "$TREE"; mkdir -p logs output
CMD=(python examples/mortality_prediction/unified_embedding_e2e_mimic4.py
  --task notes_labs --model transformer
  --ehr-root "$EHR_ROOT" --note-root "$NOTE_ROOT" --cache-dir "$CACHE_DIR"
  --output-dir "$TREE/output/tranche1_$NAME"
  --embedding-dim 128 --hidden-dim 128 --batch-size 32 --lr 1e-4 --dropout 0.1
  --epochs 50 --patience 5 --seed 1 --use-amp --amp-dtype bf16 --freeze-encoder
  --max-frozen-text-cache 1000000
  --heads 4 --num-layers 2)
export PYTHONPATH="$TREE"
[[ -n "$GPU" ]] && export CUDA_VISIBLE_DEVICES="$GPU"
if [[ "${FOREGROUND:-0}" == "1" ]]; then exec "${CMD[@]}"; fi
nohup "${CMD[@]}" > "logs/$NAME.out" 2>&1 &
echo $! > "logs/$NAME.pid"
echo "launched $NAME pid $(cat logs/$NAME.pid) on GPU ${GPU:-scheduler}"
