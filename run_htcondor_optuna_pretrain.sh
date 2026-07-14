#!/bin/bash
# Optuna SSL-PRETRAINING hyperparameter search, ONE study per (arch x method),
# all at the standardized 128-dim/2-layer/4-head encoder size.  Tunes SSL/optim
# HPs (lr, weight_decay, batch_size, warmup, max_grad_norm) + method knobs
# (mask_ratio/strategy/norm for MAE/SimMIM; ema_decay/num_target_blocks for
# V-JEPA) + arch knobs (state_size/conv_kernel for mamba/jamba, jamba layer mix,
# use_rope for transformer).  Objective = held-out SSL val loss (MINIMIZE) with a
# collapse guard.  Studies are independent (one SQLite db each), run OPTUNA_NGPU
# at a time across the free GPUs on the node (sunlab-c02, 8x RTX 6000 Ada 48 GB).
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
NSUBSPLITS=${OPTUNA_NSUBSPLITS:-1}        # SSL val loss is smooth; 1 subsplit is plenty
VAL_FRAC=${OPTUNA_VAL_FRAC:-0.15}
EPOCHS=${OPTUNA_EPOCHS:-4}                 # epochs per trial (cheap: BERT-bound, 5% pool)
TIMEOUT=${OPTUNA_TIMEOUT:-28800}          # ~8h wall per study (n_trials usually hits first)
NTRIALS=${OPTUNA_NTRIALS:-60}             # upper cap
NGPU=${OPTUNA_NGPU:-5}                     # studies in flight at once
ARCHS_STR=${OPTUNA_ARCHS:-"transformer jamba mamba"}
METHODS_STR=${OPTUNA_METHODS:-"mae simmim vjepa"}
read -r -a ARCHS <<< "${ARCHS_STR}"
read -r -a METHODS <<< "${METHODS_STR}"
# Modality combo (task) selection. COMBO isolates the per-combo output dir + logs.
# EXTRA carries extra task flags (e.g. --include-vitals for notes_labs+vitals).
TASK=${OPTUNA_TASK:-notes_labs}
COMBO=${OPTUNA_TAG:-${TASK}}
EXTRA=${OPTUNA_EXTRA:-}

CACHE_DIR="${BASE}/cache_notes_labs"      # shared cache root (per-task subdirs by hash)
OUT="${OPTUNA_OUT:-${BASE}/optuna_pretrain/${COMBO}}"
LOG="${BASE}/logs"
mkdir -p "${OUT}" "${LOG}"

COMMON=(--ehr-root "${EHR_ROOT}" --note-root "${NOTE_ROOT}" --cache-dir "${CACHE_DIR}"
        --task "${TASK}" --freeze-encoder --num-workers 8
        --subset-frac "${SUBSET_FRAC}" --n-subsplits "${NSUBSPLITS}"
        --subsplit-val-frac "${VAL_FRAC}" --output-dir "${OUT}")
[ -n "${EXTRA}" ] && COMMON+=(${EXTRA})   # e.g. --include-vitals

# --- Step 1: prime the shared cache once (serial, CPU) before the GPU studies. ---
echo "[$(date)] optuna-pretrain cache check (combo=${COMBO}, ${SUBSET_FRAC} pool) -> ${CACHE_DIR}"
python scripts/optuna_pretrain.py "${COMMON[@]}" --arch transformer --method mae \
    --n-trials 0 --device cpu > "${LOG}/optuna_pt_cachewarm_${COMBO}.log" 2>&1
echo "[$(date)] cache ready."

# --- Step 2: per-(arch,method) studies, NGPU at a time. ---
run_study() {
    local gpu=$1 arch=$2 method=$3
    local tag="${arch}_${method}"
    echo "[$(date)] study ${COMBO}/${tag} on GPU ${gpu} (epochs=${EPOCHS}, timeout=${TIMEOUT}s, trials<=${NTRIALS})"
    CUDA_VISIBLE_DEVICES="${gpu}" python scripts/optuna_pretrain.py "${COMMON[@]}" \
        --arch "${arch}" --method "${method}" --n-trials "${NTRIALS}" \
        --epochs-per-trial "${EPOCHS}" --timeout "${TIMEOUT}" \
        --storage "sqlite:///${OUT}/pt_${tag}.db" --study-name "pt_${COMBO}_${tag}" \
        > "${LOG}/optuna_pt_${COMBO}_${tag}.log" 2>&1 &
}

i=0
for arch in "${ARCHS[@]}"; do
    for method in "${METHODS[@]}"; do
        run_study $(( i % NGPU )) "${arch}" "${method}"
        i=$(( i + 1 ))
        if (( i % NGPU == 0 )); then wait; fi   # drain a full wave before the next
    done
done
wait

echo "[$(date)] All optuna-pretrain studies finished. Best params in ${OUT}/best_params_pt_*.json"
