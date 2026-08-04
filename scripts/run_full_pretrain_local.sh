#!/bin/bash
# Bare-metal full pretraining driver: ONE 4-GPU DDP job at a time (50% of an
# 8-GPU box). Runs a list of arch:method PAIRS sequentially on the 4 GPUs in
# GPUS. Resumable: skips runs whose last.ckpt already exists. Split a combo's 9
# encoders across two boxes by giving each a different PAIRS subset.
#   GPUS=0,1,2,3 COMBO=notes_labs PAIRS="transformer:mae jamba:vjepa" bash run_full_pretrain_local.sh
set -uo pipefail
source /home/rianatri/miniconda3/etc/profile.d/conda.sh && conda activate pyhealth2
cd /home/rianatri/Multimodal-PyHealth-ssl
export PYTHONPATH=/home/rianatri/Multimodal-PyHealth-ssl TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOM (as in the slurm runner)
B=/shared/rsaas/rianatri/ssl
export HF_HOME=$B/huggingface TMPDIR=$B/tmp
export WANDB_PROJECT=pyhealth-multimodal WANDB_ENTITY=pyhealth-multimodal WANDB_DIR=$B/wandb

COMBO=${COMBO:-notes_labs}
BP_DIR=${BP_DIR:-$B/optuna_pretrain}
GPUS=${GPUS:-0,1,2,3}
NGPU=$(echo "$GPUS" | tr "," "\n" | grep -c .)
# XTRA: extra pretrain_ssl.py flags appended last, overriding tuned values
# (e.g. XTRA="--batch-size 32 --grad-accumulation-steps 2" to fit a 48GB GPU at
# the same effective batch size). Must stay last: --extra is argparse REMAINDER.
case "$COMBO" in
  notes_labs) TASK=notes_labs; EXTRA="" ;;
  notes_labs_vitals) TASK=notes_labs; EXTRA="--include-vitals" ;;
  *) TASK=$COMBO; EXTRA="" ;;
esac
OUT=$B/pretrain_full/$COMBO; mkdir -p "$OUT" "$B/logs"
EHR=/shared/rsaas/physionet.org/files/mimiciv/2.2
NOTE=/shared/rsaas/physionet.org/files/mimic-note

PAIRS_STR=${PAIRS:-"transformer:mae transformer:simmim transformer:vjepa jamba:mae jamba:simmim jamba:vjepa mamba:mae mamba:simmim mamba:vjepa"}
read -r -a PAIRS_ARR <<< "$PAIRS_STR"
for pair in "${PAIRS_ARR[@]}"; do
  arch=${pair%:*}; method=${pair#*:}
  bp=$BP_DIR/best_params_pt_${arch}_${method}_${TASK}.json
  [ -f "$bp" ] || { echo "skip (no best_params): $arch $method"; continue; }
  _mh="$OUT/${arch}_${method}_${TASK}_seed42/metrics_history.json"
  [ -f "$_mh" ] && [ "$(python -c "import json;print(len(json.load(open('$_mh'))))" 2>/dev/null)" = "50" ] && { echo "skip (done, 50ep): $arch $method"; continue; }
  echo "[$(date)] $arch/$method on GPUs $GPUS (ngpu=$NGPU)"
  CUDA_VISIBLE_DEVICES=$GPUS python scripts/run_full_pretrain.py \
    --best-params "$bp" --ehr-root "$EHR" --note-root "$NOTE" --cache-dir "$B/cache_notes_labs" \
    --output-dir "$OUT" --epochs 50 --num-workers "${NW:-8}" --nproc-per-node "$NGPU" $EXTRA \
    ${XTRA:+--extra $XTRA} \
    > "$B/logs/fullpt_${COMBO}_${arch}_${method}.out" 2>&1
done
echo "[$(date)] driver done (GPUS=$GPUS)."
