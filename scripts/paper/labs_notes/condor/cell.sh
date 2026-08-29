#!/usr/bin/env bash
# HTCondor executable: activates the env and runs one labs_notes metal cell in the
# foreground. usage: cell.sh <metal script name>   (submitted via cell.sub)
set -euo pipefail
CELL="${1:?usage: cell.sh <metal script name>}"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate "${CONDA_ENV:-pyhealth2}"
export FOREGROUND=1 GPU=""
exec bash "$(dirname "$(readlink -f "$0")")/../metal/$CELL"
