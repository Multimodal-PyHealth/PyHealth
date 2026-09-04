# Shared Tranche 1 protocol. Sourced by rian.sh / will.sh, which add only the
# data roots and the CPU tuning appropriate to their machine.
#
#   TASK=labs_notes MODEL=ehrmamba SEED=3 bash scripts/paper/rian.sh
#
# Env: TASK MODEL SEED TREE EHR_ROOT NOTE_ROOT CXR_ROOT CACHE_DIR OUT GPU

TASK="${TASK:?set TASK=labs|labs_notes|labs_notes_cxr}"
MODEL="${MODEL:?set MODEL=mlp|rnn|transformer|bottleneck_transformer|ehrmamba|jambaehr}"
SEED="${SEED:?set SEED}"
TREE="${TREE:-$HOME/PyHealth}"
CACHE_DIR="${CACHE_DIR:-$HOME/pyhealth_cache/$TASK}"
OUT="${OUT:-$TREE/output}"

case "$TASK" in
  labs)           TASK_FLAG=labs;           ROOTS=() ;;
  labs_notes)     TASK_FLAG=notes_labs;     ROOTS=(--note-root "$NOTE_ROOT") ;;
  labs_notes_cxr) TASK_FLAG=notes_labs_cxr; ROOTS=(--note-root "$NOTE_ROOT" --cxr-root "$CXR_ROOT" --cxr-variant sunlab) ;;
  *) echo "unknown TASK=$TASK" >&2; exit 2 ;;
esac

case "$MODEL" in
  mlp)                    ARCH=(--mlp-layers 2 --mlp-activation relu) ;;
  rnn)                    ARCH=(--rnn-type GRU --rnn-layers 1) ;;
  transformer)            ARCH=(--heads 4 --num-layers 2) ;;
  bottleneck_transformer) ARCH=(--heads 4 --num-layers 2 --bottlenecks-n 4 --fusion-startidx 1) ;;
  ehrmamba)               ARCH=(--num-layers 2 --mamba-state-size 16 --mamba-conv-kernel 4) ;;
  jambaehr)               ARCH=(--heads 4 --jamba-transformer-layers 2 --jamba-mamba-layers 6) ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 2 ;;
esac

# Identical for every cell: the Tranche 1 protocol.
PROTOCOL=(--embedding-dim 128 --hidden-dim 128 --dropout 0.1
          --batch-size 32 --lr 1e-4 --epochs 50 --patience 5
          --use-amp --amp-dtype bf16 --freeze-encoder)

launch () {  # any extra flags are passed through
  cd "$TREE"; export PYTHONPATH="$TREE"
  mkdir -p logs "$OUT"
  [[ -n "${GPU:-}" ]] && export CUDA_VISIBLE_DEVICES="$GPU"
  python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \
    --task "$TASK_FLAG" --model "$MODEL" --seed "$SEED" \
    --ehr-root "$EHR_ROOT" "${ROOTS[@]}" \
    --cache-dir "$CACHE_DIR" --output-dir "$OUT" \
    "${PROTOCOL[@]}" "${ARCH[@]}" "$@" \
    2>&1 | tee "logs/${TASK}_${MODEL}_seed${SEED}.out"
}
