#!/usr/bin/env bash
# Rian's cells: seeds 1,3,5 on labs and labs_notes, plus all of labs_notes_cxr.
#
# Same protocol as will.sh; only the data roots and CPU tuning differ. These
# nodes are shared and run several cells at once, and torch defaults to 64
# intra-op + 128 inter-op threads with nothing pinning them: four unpinned
# cells put ~800 threads on 128 cores and epoch time went 191s -> 8600s with
# the GPUs at 0-1%. Keep THREADS x concurrent_cells under the node's cores.
#
# THREADS is the whole CPU story here: this runner's --num-workers feeds the
# dataset build only, and it has no dataloader-worker flag, so pinning the
# BLAS/OMP thread pools is what keeps concurrent cells off each other.
set -euo pipefail
EHR_ROOT="${EHR_ROOT:-/shared/rsaas/physionet.org/files/mimiciv/2.2}"
NOTE_ROOT="${NOTE_ROOT:-/shared/rsaas/physionet.org/files/mimic-note}"
CXR_ROOT="${CXR_ROOT:-/shared/rsaas/physionet.org/files/MIMIC-CXR}"
THREADS="${THREADS:-8}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
source "$(dirname "$(readlink -f "$0")")/common.sh"
launch --num-workers "${NUM_WORKERS:-8}" "$@"
