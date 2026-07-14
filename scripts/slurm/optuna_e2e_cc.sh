#!/bin/bash
#SBATCH --job-name=ssl_e2e_optuna
#SBATCH --partition=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --array=0-5%6
#SBATCH --output=/scratch/rianatri/ssl/logs/e2e_optuna_%a_%j.out
# Downstream (e2e) Optuna, one study per arch, for a single modality combo.
# Combo via E2E_COMBO env (default labs_only). SQLite resumable across preemption.
set -eo pipefail
source /scratch/rianatri/Multimodal-PyHealth-ssl/scripts/slurm/_env_cc.sh
[ -x "${PYBIN:-}/python" ] || { echo "FATAL: campus env not set up"; exit 1; }

ARCHS=( mlp rnn transformer bottleneck_transformer ehrmamba jambaehr )
ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}
COMBO=${E2E_COMBO:-labs_only}
case "${COMBO}" in
  labs_only)          TASK=labs_only;   EXTRA="" ;;
  notes_only)         TASK=notes_only;  EXTRA="" ;;
  notes_labs_vitals)  TASK=notes_labs;  EXTRA="--include-vitals" ;;
  *)                  TASK=${COMBO};    EXTRA="" ;;
esac

OUT=${SSLBASE}/optuna_e2e/${COMBO}
mkdir -p "${OUT}"
echo "[$(date)] e2e optuna combo=${COMBO} arch=${ARCH} on $(hostname)"
${PYBIN}/python scripts/optuna_e2e.py \
    --ehr-root "${EHR_ROOT}" --note-root "${NOTE_ROOT}" --cache-dir "${CACHE_DIR}" \
    --task "${TASK}" ${EXTRA} --model "${ARCH}" --freeze-encoder \
    --subset-frac 0.05 --n-subsplits 2 --subsplit-val-frac 0.15 \
    --n-trials 80 --epochs-per-trial 4 --timeout 43200 --tune-arch-specific \
    --num-workers 8 --device cuda \
    --storage "sqlite:///${OUT}/e2e_${ARCH}.db" \
    --study-name "e2e_${COMBO}_${ARCH}" --output-dir "${OUT}"
echo "[$(date)] e2e optuna combo=${COMBO} arch=${ARCH} done."
