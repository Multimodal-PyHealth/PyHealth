#!/bin/bash
#SBATCH --job-name=ssl_pt_optuna
#SBATCH --partition=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-26%8
#SBATCH --output=/scratch/rianatri/ssl/logs/pt_optuna_%a_%j.out
# One SSL pretraining Optuna study per (combo x arch x method) for the 3 new
# combos = 27 array tasks, <=8 concurrent. SQLite storage is resumable, so a
# scavenger preemption just requeues and continues (load_if_exists).
set -eo pipefail
source /scratch/rianatri/Multimodal-PyHealth-ssl/scripts/slurm/_env_cc.sh
[ -x "${PYBIN:-}/python" ] || { echo "FATAL: campus env not set up (PYBIN=${PYBIN:-unset})"; exit 1; }

ID=${SLURM_ARRAY_TASK_ID}
NPER=$(( ${#ARCHS[@]} * ${#METHODS[@]} ))          # 9 per combo
IFS='|' read -r TAG TASK EXTRA <<< "${COMBOS[$(( ID / NPER ))]}"
REM=$(( ID % NPER ))
ARCH=${ARCHS[$(( REM / ${#METHODS[@]} ))]}
METHOD=${METHODS[$(( REM % ${#METHODS[@]} ))]}

OUT=${SSLBASE}/optuna_pretrain/${TAG}
mkdir -p "${OUT}"
echo "[$(date)] study combo=${TAG} arch=${ARCH} method=${METHOD} on $(hostname) gpu=${CUDA_VISIBLE_DEVICES}"

${PYBIN}/python scripts/optuna_pretrain.py \
    --ehr-root "${EHR_ROOT}" --note-root "${NOTE_ROOT}" --cache-dir "${CACHE_DIR}" \
    --task "${TASK}" ${EXTRA} --arch "${ARCH}" --method "${METHOD}" --freeze-encoder \
    --subset-frac 0.05 --n-subsplits 1 --epochs-per-trial 4 \
    --timeout 28800 --n-trials 60 --num-workers 8 --device cuda \
    --storage "sqlite:///${OUT}/pt_${ARCH}_${METHOD}.db" \
    --study-name "pt_${TAG}_${ARCH}_${METHOD}" --output-dir "${OUT}"
echo "[$(date)] study combo=${TAG} arch=${ARCH} method=${METHOD} done."
