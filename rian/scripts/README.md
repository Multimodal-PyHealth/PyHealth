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
| RNN (GRU, 1 layer) | **0.821** | **0.979** | 0.828 (5) | 241 s | 1.1 h |
| JambaEHR | 0.808 | 0.975 | 0.820 (1) | 372 s | 1.0 h |
| EHRMamba | 0.780 | 0.970 | 0.801 (2) | 246 s | 0.9 h |
| Transformer | 0.686 | 0.945 | 0.679 (7) | 216 s | 1.1 h |
| Bottleneck Transformer | 0.680 | 0.961 | 0.693 (9) | 224 s | 1.2 h |
| MLP | 0.566 | 0.944 | 0.593 (16) | 191 s | 1.5 h |

Bottleneck Transformer was rerun on fixed code (see below); the other five rows are the
original runs. Epoch = mean train epoch after the cold first one (~1400 s while the cache
fills). Single seed.
Recurrent/SSM backbones (RNN, Jamba, EHRMamba) lead attention backbones by ~0.1 PR-AUC and MLP by
~0.2 under full-stay collection, where end-of-stay recency is highly informative.

## Runner fixes and what they changed

The runner branched on `args.model` to give `bottleneck_transformer` `max_grad_norm=0.5`
and Adam `eps=1e-6` while the other five backbones got 1.0 and 1e-8, and `run_config.json`
stored the unset flag rather than the resolved value, so nothing on disk revealed it. The
six-backbone table above was therefore not compute-matched. Fixed in `scripts/paper` +
runner: optimizer settings are CLI-only, `resolved_adam_eps` is recorded, `--amp-dtype`
without `--use-amp` is now an error instead of a silent fp32 run, the leaky by-sample split
fallback raises instead of warning, and flags a backbone ignores are logged as `inert_flags`.

Rerun on the fixed runner, same warm task cache and split so only code changed:

| cell | test PR-AUC before | after | delta |
|---|---|---|---|
| MLP | 0.566212 | 0.566212 | **bit-identical** |
| RNN | 0.821216 | 0.821216 | **bit-identical** |
| Bottleneck Transformer | 0.690708 | 0.679547 | −0.0112 |

MLP and RNN reproducing to every decimal (identical val curves, identical stop epoch) is the
control: the fix is provably a no-op for backbones already on 1.0/1e-8, which is what makes
the Bottleneck delta attributable. The old hardcode was genuinely helping that cell by ~0.011
PR-AUC; the fix trades that for a table where all six share one optimizer config.

## Labs + Notes + CXR — do not read as a CXR result yet

| cell | test PR-AUC | test ROC-AUC | best val (ep) | epoch |
|---|---|---|---|---|
| RNN, notes+labs+CXR | 0.813 | 0.977 | 0.838 (10) | 760 s |

On the 18,056 test patients shared with the notes+labs split, +CXR scores 0.8131 against
0.8215 without it. That is **not** evidence that chest X-rays do not help, because:

- **86% of samples have no in-window image at all** (14.2% have ≥1; mean 0.79 images per
  sample, 5.56 among those that have any). Only 28% of test patients have a CXR anywhere in
  their history.
- **The image encoder is untrained.** With no `field_embeddings` passed, the unified model
  builds `Conv2d(3, 128, 16x16, stride 16)` at random init plus a global mean pool — one
  128-d vector per image from a single randomly-initialised patch projection.
  `VisionEmbeddingModel` supports ImageNet resnet18/50 but that path is never taken.
- **Missingness correlates with the label**: 39% of positives carry an image against 14%
  overall, so "was imaged" is itself a mortality signal independent of pixel content.
- Images are 256x256 greyscale on disk (original DICOMs are ~3056x2544), replicated to three
  identical RGB channels, resized to 224, and not normalised.

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
