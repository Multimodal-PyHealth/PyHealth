# rian/scripts — Tranche 1 paper cells (notes_labs, seed 1)

One launcher per cell. Each mirrors the original `notes_labs × MLP × seed 1` cell
(`run_config.json` of that run), differing only in `--model` and its own flags.

```
GPU=2 bash rian/scripts/run_notes_labs_jambaehr_seed1.sh
GPU=3 bash rian/scripts/run_notes_labs_bottleneck_transformer_seed1.sh
GPU=0 bash rian/scripts/run_notes_labs_mlp_seed1.sh
```

Env overrides: `TREE` (checkout to run from), `GPU`, `EHR_ROOT`, `NOTE_ROOT`, `CACHE_DIR`.
Logs go to `$TREE/logs/<name>.out`, outputs to `$TREE/output/tranche1_<name>/`.
`PYTHONPATH` is pinned to `$TREE` because the `pyhealth2` env carries an editable install
that otherwise shadows the checkout.

## Protocol (all cells)

Full stay (admit → discharge), no ICD, empty sequences for missing modalities, frozen
Bio_ClinicalBERT with a 1e6-entry `[CLS]` cache, bf16 AMP, batch 32, lr 1e-4, dropout 0.1,
embedding/hidden 128, 50 epochs, patience 5, seed 1, split by patient with seed 1.
Tree = PR #1185 tip `8d4a4c9` + the "frozen `[CLS]` cache cap to 1e6" follow-up commit only.
All cells hit the same task cache (`NotesLabsMIMIC4_c447f3bb…`): 144,586 / 18,073 / 18,074
patients, 852 test positives. Test metrics are sklearn `average_precision_score` /
`roc_auc_score` over `predictions_<model>.csv` from the best-val checkpoint.

## Results (sunlab-serv-03, one RTX A6000 per cell, 2026-08-27)

| cell | test PR-AUC | test ROC-AUC | best val PR-AUC (epoch) | stopped at | epoch after warm-up | total train |
|---|---|---|---|---|---|---|
| MLP, 1e6 cache | 0.5662 | 0.9437 | 0.5931 (16) | 21 | 191 s | 1.5 h |
| MLP, 200k cache (original cell) | 0.5705 | 0.9386 | 0.5996 (22) | 27 | 4984 s | 38.9 h |
| JambaEHR (2 transformer + 6 mamba) | 0.8081 | 0.9751 | 0.8203 (1) | 6 | 372 s | 1.0 h |
| BottleneckTransformer (n=4, fusion start 1) | 0.6907 | 0.9608 | 0.6958 (8) | 13 | 224 s | 1.2 h |

Notes:

- The two MLP rows are the compute-matched cache comparison: same host, same GPU,
  same split, same flags — only the cache cap differs. Epoch time after warm-up drops
  26×; metrics land within run-to-run noise (not bit-identical: early stopping fired at
  a different epoch).
- Single seed. JambaEHR peaks at epoch 1; treat the gap to the other backbones as
  provisional until multi-seed.
