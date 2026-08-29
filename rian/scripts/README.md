# rian/scripts — Tranche 1 cells (notes_labs, seed 1)

One script per cell in `metal/`; `slurm/` and `condor/` are thin wrappers that run the same script
in the foreground on a scheduler-assigned GPU. Flags live only in `metal/`.

```
bash  rian/scripts/metal/run_notes_labs_jambaehr_seed1.sh                       # bare metal, GPU=0 default
GPU=3 bash rian/scripts/metal/run_notes_labs_jambaehr_seed1.sh
sbatch --export=ALL,CELL=run_notes_labs_jambaehr_seed1.sh,TREE=...,EHR_ROOT=...,NOTE_ROOT=...,CACHE_DIR=... rian/scripts/slurm/cell.sbatch
condor_submit CELL=run_notes_labs_jambaehr_seed1.sh TREE=... rian/scripts/condor/cell.sub
```

Env overrides: `TREE`, `GPU` (`""` = leave to scheduler), `EHR_ROOT`, `NOTE_ROOT`, `CACHE_DIR`.
Logs → `$TREE/logs/<cell>.out`, outputs → `$TREE/output/tranche1_<cell>/`. `PYTHONPATH` is pinned to
`$TREE` (the `pyhealth2` env has an editable install that otherwise shadows the checkout).

**Protocol (all cells):** full stay, empty sequences for missing modalities, Bio_ClinicalBERT `[CLS]`
per note, bf16 AMP, batch 32, lr 1e-4, dropout 0.1, dim 128/128, 50 epochs, patience 5, seed 1,
split by patient (144,586 / 18,073 / 18,074; 852 test positives). Frozen cells cache `[CLS]` with a
1e6-entry cap. Test metrics = sklearn AP / ROC-AUC over `predictions_<model>.csv` from the best-val
checkpoint. Timings: one RTX A6000 per cell on sunlab-serv-03.

## Backbones — frozen BERT + 1M cache

| cell | test PR-AUC | test ROC-AUC | best val PR-AUC (ep) | epoch | total |
|---|---|---|---|---|---|
| MLP | 0.566 | 0.944 | 0.593 (16) | 191 s | 1.5 h |
| Bottleneck Transformer | 0.691 | 0.961 | 0.696 (8) | 224 s | 1.2 h |
| JambaEHR | 0.808 | 0.975 | 0.820 (1) | 372 s | 1.0 h |
| RNN | running | | | | |
| Transformer | running | | | | |
| EHRMamba | running | | | | |

## Cache cap sweep — MLP, frozen BERT

| cap | test PR-AUC | best val (ep) | epoch after warm-up | total |
|---|---|---|---|---|
| 200k | 0.571 | 0.600 (22) | 4984 s | 38.9 h |
| 500k | 0.575 | 0.593 (24) | 2249 s | 19.0 h |
| 1M | 0.566 | 0.593 (16) | 191 s | 1.5 h |

Same model (train loss agrees to 2e-4, val dips on the same epochs); the cap only changes compute.
200k/500k are slower than a plain re-encode because, once full, the overflow path encoded each miss
twice (batched, then one row at a time). Fixed: *Encode a frozen-cache miss once per forward, even
when the cache is full.* Both slow rows above ran on the pre-fix code.

## Frozen + cache vs BERT trained end-to-end

| backbone | frozen + cache | unfrozen | epoch (frozen → unfrozen) |
|---|---|---|---|
| JambaEHR | 0.808 | 0.736 | 372 s → 7671 s |
| Bottleneck Transformer | 0.691 | 0.603 | 224 s → 7617 s |
| MLP (val PR-AUC, epoch 13) | 0.585 | 0.492 | 191 s → 7575 s |

Unfrozen = same flags minus `--freeze-encoder`, plus `--text-grad-checkpoint-rows 256` (a trainable
BERT keeps every note row's activations; one fat batch OOMs 47 GB without it; math unchanged). The
deficit is set within epoch 0 and stays constant — parallel curves, not divergence — consistent with
lr 1e-4 on all of BERT with no warmup and low-SNR gradients (one label per stay, 4.7% positives).
Single seed; replicates across three backbones.
