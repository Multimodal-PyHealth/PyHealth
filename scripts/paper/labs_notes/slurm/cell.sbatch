#!/usr/bin/env bash
# Slurm wrapper: runs one Labs + Notes cell in the foreground on the allocated GPU.
#
#   sbatch --export=ALL,CELL=run_rnn.sh,TREE=...,EHR_ROOT=...,NOTE_ROOT=...,CACHE_DIR=... \
#          scripts/paper/labs_notes/slurm/cell.sbatch
#
# Frozen text cells peak ~8 GB VRAM. Compute nodes have no internet, so the
# Bio_ClinicalBERT weights must already be in the HF cache and the task cache
# must be built before submitting.
#SBATCH -J paper-labs_notes
#SBATCH -A jimeng-cs-eng
#SBATCH -p eng-research-gpu
#SBATCH --gres=gpu:A10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH -o /u/rianatri/scratch/pyhealth/logs/%x-%j.out
set -euo pipefail
: "${CELL:?set CELL=<metal script name>}"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate "${CONDA_ENV:-pyhealth2}"
export FOREGROUND=1 GPU=""   # Slurm already scopes CUDA_VISIBLE_DEVICES
exec bash "$(dirname "$(readlink -f "$0")")/../metal/$CELL"
