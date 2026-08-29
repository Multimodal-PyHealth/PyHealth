# Shared protocol for every paper cell. Sourced by scripts/paper/*/metal/*.sh.
#
# Nothing model- or modality-specific belongs here: those flags live in the cell
# script, which is the only place a paper run's configuration is written down.
#
# Every value the runner uses is passed explicitly, including ones that happen to
# match the runner's own defaults (--max-grad-norm, --adam-eps). A default that
# lives only in code is a default nobody reading a results table can see, and
# that is exactly how bottleneck_transformer ended up training under a different
# clip norm and Adam epsilon than the five backbones it was tabled against.
#
# Env overrides: SEED TREE EHR_ROOT NOTE_ROOT CXR_ROOT CACHE_DIR GPU
#                FROZEN_TEXT_CACHE FOREGROUND

SEED="${SEED:-1}"
TREE="${TREE:-$HOME/ml4h-paper}"
EHR_ROOT="${EHR_ROOT:-/shared/rsaas/physionet.org/files/mimiciv/2.2}"
NOTE_ROOT="${NOTE_ROOT:-/shared/rsaas/physionet.org/files/mimic-note}"
CXR_ROOT="${CXR_ROOT:-}"
FROZEN_TEXT_CACHE="${FROZEN_TEXT_CACHE:-1000000}"
GPU="${GPU-0}"

# Dataset/task build parallelism. Infrastructure, not protocol: it changes how
# long the first run takes to materialise the task cache, not what the cache
# contains. Raise it when building a cold cache (the CXR one is slow).
NUM_WORKERS="${NUM_WORKERS:-1}"

RUNNER=examples/mortality_prediction/unified_embedding_e2e_mimic4.py

# dim 128/128, dropout 0.1, batch 32, lr 1e-4, 50 epochs, patience 5, bf16 AMP —
# the Tranche 1 protocol, identical across all six backbones and all three
# modality sets. --hidden-dim is inert for transformer/bottleneck/ehrmamba/
# jambaehr (their width is --embedding-dim) and --dropout is inert for mlp
# (pyhealth.models.mlp.MLP has no dropout parameter); both are passed anyway
# because the spec lists them, and the runner now logs and records exactly which
# flags a given backbone ignored.
PROTOCOL_FLAGS=(
  --embedding-dim 128 --hidden-dim 128
  --dropout 0.1
  --batch-size 32 --lr 1e-4 --adam-eps 1e-8 --max-grad-norm 1.0
  --epochs 50 --patience 5
  --seed "$SEED"
  --use-amp --amp-dtype bf16
  --num-workers "$NUM_WORKERS"
)

# launch <run name> <runner flag>...
# Bare metal backgrounds python with nohup; FOREGROUND=1 execs it so Slurm and
# Condor can own the process. GPU="" leaves CUDA_VISIBLE_DEVICES to the scheduler.
launch() {
  local name="$1"
  shift
  cd "$TREE"
  mkdir -p logs output
  local cmd=(python "$RUNNER" "$@" --output-dir "$TREE/output/$name")

  export PYTHONPATH="$TREE"
  [[ -n "$GPU" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

  # Echo the resolved command so the log is self-describing; run_config.json in
  # the output dir records the parsed values, including the inert ones.
  printf 'cell %s\n' "$name"
  printf 'cmd '
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [[ "${FOREGROUND:-0}" == "1" ]]; then exec "${cmd[@]}"; fi
  nohup "${cmd[@]}" >"logs/$name.out" 2>&1 &
  echo $! >"logs/$name.pid"
  echo "launched $name pid $(cat "logs/$name.pid") on GPU ${GPU:-scheduler}"
}
