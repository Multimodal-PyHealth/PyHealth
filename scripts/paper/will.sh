#!/usr/bin/env bash
# Will's cells: seeds 2 and 4 on labs and labs_notes.
set -euo pipefail
EHR_ROOT="${EHR_ROOT:-/home/ubuntu/mimiciv-data/ehr}"
NOTE_ROOT="${NOTE_ROOT:-/home/ubuntu/mimiciv-data}"
CXR_ROOT="${CXR_ROOT:-/home/ubuntu/mimiciv-data/CXR-jpg}"
source "$(dirname "$(readlink -f "$0")")/common.sh"
launch --num-workers "${NUM_WORKERS:-4}"
