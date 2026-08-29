#!/usr/bin/env bash
# Slurm wrapper: runs one metal cell script in the foreground on the allocated GPU.
#
#   sbatch --export=ALL,CELL=run_notes_labs_jambaehr_seed1.sh,TREE=/u/rianatri/scratch/pyhealth/ml4h-tranche1,\
#          EHR_ROOT=...,NOTE_ROOT=...,CACHE_DIR=... rian/scripts/slurm/cell.sbatch
#
# Frozen cells peak ~8 GB VRAM, unfrozen ~13 GB: an A10 (24 GB) is enough. Compute nodes have no
# internet, so the BERT weights must already be in the HF cache and the task cache must be built.
#SBATCH -J pyhealth-cell
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
