#!/bin/bash
#SBATCH --job-name=ssl_full_pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --array=0-2
#SBATCH --output=/scratch/rianatri/ssl/logs/full_pt_%x_%a_%j.out
# Full-scale (50-epoch) SSL pretraining at tuned HPs for one (combo, method),
# array over the 3 backbones. Partition/account/time set at submit time:
#   V-JEPA -> IllinoisComputes-GPU A100 (-A jimeng-ic, 2-4 day)
#   MAE/SimMIM -> eng-research-gpu A10 (-A jimeng-cs-eng) or scavenger
# FULL_COMBO in {labs_only, notes_only, notes_labs_vitals, notes_labs};
# FULL_METHOD in {mae, simmim, vjepa}.
set -eo pipefail
source /scratch/rianatri/Multimodal-PyHealth-ssl/scripts/slurm/_env_cc.sh
[ -x "${PYBIN:-}/python" ] || { echo "FATAL: env not set up"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOM

ARCHS=( transformer jamba mamba )
ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}
COMBO=${FULL_COMBO:?set FULL_COMBO}
METHOD=${FULL_METHOD:?set FULL_METHOD}
case "${COMBO}" in
  labs_only)         TASK=labs_only;  EXTRA="" ;;
  notes_only)        TASK=notes_only; EXTRA="" ;;
  notes_labs_vitals) TASK=notes_labs; EXTRA="--include-vitals" ;;
  *)                 TASK=notes_labs; EXTRA="" ;;
esac

BP=${SSLBASE}/optuna_pretrain/${COMBO}/best_params_pt_${ARCH}_${METHOD}_${TASK}.json
[ -f "${BP}" ] || { echo "FATAL: best_params not found: ${BP}"; exit 1; }
OUT=${SSLBASE}/pretrain_full/${COMBO}
mkdir -p "${OUT}"
# resume-friendly: skip if a completed (50-epoch) encoder already exists
DONE="${OUT}/${ARCH}_${METHOD}_${TASK}_seed42/metrics_history.json"
if [ -f "${DONE}" ] && [ "$(${PYBIN}/python -c "import json;print(len(json.load(open('${DONE}'))))" 2>/dev/null)" = "50" ]; then
    echo "skip (already 50 epochs): ${ARCH} ${METHOD} ${COMBO}"; exit 0
fi
# GPUs allocated to this task -> DDP world size.
NGPU=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr "," "\n" | grep -c .)
echo "[$(date)] full pretrain combo=${COMBO} method=${METHOD} arch=${ARCH} on $(hostname) ngpu=${NGPU}"
${PYBIN}/python scripts/run_full_pretrain.py \
    --best-params "${BP}" --ehr-root "${EHR_ROOT}" --note-root "${NOTE_ROOT}" \
    --cache-dir "${CACHE_DIR}" --output-dir "${OUT}" --epochs 50 --num-workers 8 \
    --nproc-per-node "${NGPU}" ${EXTRA}
echo "[$(date)] full pretrain combo=${COMBO} method=${METHOD} arch=${ARCH} done."
