#!/bin/bash
# Optuna e2e hyperparameter search, ONE study per architecture, all at the
# standardized 128-dim/2-layer/4-head size.  Tunes training HPs (lr, weight_decay,
# dropout, batch_size, max_grad_norm, pos_weight, sampling_strategy) on a 5%
# patient subset of the FULL cached dataset, with N independent train/val
# subsplits per trial (objective = mean val PR-AUC) for robust selection.  Each
# study is capped at ~8h wall (OPTUNA_TIMEOUT); pruning + timeout govern how many
# trials fit.  Studies are independent (one SQLite db each), run OPTUNA_NGPU at a
# time across the free GPUs on the node.
set -euo pipefail

source /home/rianatri/miniconda3/etc/profile.d/conda.sh
conda activate pyhealth2
cd /home/rianatri/Multimodal-PyHealth-ssl

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/rianatri/Multimodal-PyHealth-ssl:${PYTHONPATH:-}

BASE=/shared/rsaas/rianatri/ssl
export HF_HOME="${BASE}/huggingface"
export TMPDIR="${BASE}/tmp"
mkdir -p "${HF_HOME}" "${TMPDIR}"

# W&B (Joshua's project); sunlab has internet -> online. Needs `wandb login` once.
export WANDB_PROJECT=pyhealth-multimodal
export WANDB_ENTITY=pyhealth-multimodal
export WANDB_DIR="${BASE}/wandb"; mkdir -p "${WANDB_DIR}"

EHR_ROOT=/shared/rsaas/physionet.org/files/mimiciv/2.2
NOTE_ROOT=/shared/rsaas/physionet.org/files/mimic-note

SUBSET_FRAC=${OPTUNA_SUBSET_FRAC:-0.05}   # 5% of patients, from the full cache
NSUBSPLITS=${OPTUNA_NSUBSPLITS:-2}        # train/val resplits per trial, averaged
VAL_FRAC=${OPTUNA_VAL_FRAC:-0.15}
EPOCHS=${OPTUNA_EPOCHS:-4}                 # epochs per subsplit
# 12h wall/arch.  Cheaper per-trial (2x4=8 model-trainings vs the old 3x6=18)
# buys ~25-30 completed trials so TPE clears its 5-trial startup and actually
# optimizes, instead of the ~5-8 pure-random trials the old 8h/3x6 config got.
TIMEOUT=${OPTUNA_TIMEOUT:-43200}          # ~12h wall per arch (the real governor)
NTRIALS=${OPTUNA_NTRIALS:-80}             # upper cap; timeout usually hits first
NGPU=${OPTUNA_NGPU:-4}
ARCHS_STR=${OPTUNA_ARCHS:-"mlp rnn transformer bottleneck_transformer ehrmamba jambaehr"}
read -r -a ARCHS <<< "${ARCHS_STR}"
# Modality combo (task) selection. TAG isolates the per-combo output dir so the
# e2e_<arch>.db files never collide across combos. EXTRA carries extra task flags
# (e.g. --include-vitals for the notes_labs+vitals combo).
TASK=${OPTUNA_TASK:-notes_labs}
TAG=${OPTUNA_TAG:-${TASK}}
EXTRA=${OPTUNA_EXTRA:-}

CACHE_DIR="${BASE}/cache_notes_labs"      # shared cache root (per-task subdirs by hash)
OUT="${BASE}/optuna/${TAG}"
LOG="${BASE}/logs"
mkdir -p "${OUT}" "${LOG}"

COMMON=(--ehr-root "${EHR_ROOT}" --note-root "${NOTE_ROOT}" --cache-dir "${CACHE_DIR}"
        --task "${TASK}" --freeze-encoder --num-workers 8
        --subset-frac "${SUBSET_FRAC}" --n-subsplits "${NSUBSPLITS}"
        --subsplit-val-frac "${VAL_FRAC}" --output-dir "${OUT}")
[ -n "${EXTRA}" ] && COMMON+=(${EXTRA})   # e.g. --include-vitals

# --- Step 1: load the full cached dataset once (serial, CPU) to confirm the
# cache hits before the parallel GPU studies each load it. ---
echo "[$(date)] Optuna cache check (full dataset, 5% pool) -> ${CACHE_DIR}"
python scripts/optuna_e2e.py "${COMMON[@]}" --model mlp --n-trials 0 --device cpu \
    > "${LOG}/optuna_cachewarm_${TAG}.log" 2>&1
echo "[$(date)] Optuna cache ready."

# --- Step 2: per-architecture studies, NGPU at a time, ~8h each. ---
run_study() {
    local gpu=$1 arch=$2
    echo "[$(date)] study ${arch} on GPU ${gpu} (frac=${SUBSET_FRAC}, subsplits=${NSUBSPLITS}, epochs=${EPOCHS}, timeout=${TIMEOUT}s)"
    CUDA_VISIBLE_DEVICES="${gpu}" python scripts/optuna_e2e.py "${COMMON[@]}" \
        --model "${arch}" --n-trials "${NTRIALS}" --epochs-per-trial "${EPOCHS}" \
        --timeout "${TIMEOUT}" --tune-arch-specific \
        --storage "sqlite:///${OUT}/e2e_${arch}.db" --study-name "e2e_${arch}_${TAG}" \
        > "${LOG}/optuna_${TAG}_${arch}.log" 2>&1 &
}

i=0
for arch in "${ARCHS[@]}"; do
    run_study $(( i % NGPU )) "${arch}"
    i=$(( i + 1 ))
    if (( i % NGPU == 0 )); then wait; fi   # drain a full wave before the next
done
wait

echo "[$(date)] All Optuna studies finished. Best params in ${OUT}/best_params_*.json"
