# scripts/paper — Tranche 1 paper cells

Three modality sets x six backbones x one seed per person. One script per cell,
and the script is the whole configuration: every value the runner uses is passed
on the command line, including ones that match the runner's defaults.

```
scripts/paper/
  common.sh                 protocol shared by all 18 cells
  labs/            metal/ slurm/ condor/
  labs_notes/      metal/ slurm/ condor/
  labs_notes_cxr/  metal/ slurm/ condor/
```

`metal/` holds the flags. `slurm/` and `condor/` are thin wrappers that activate
the env and exec the same metal script in the foreground on a scheduler-assigned
GPU — they never redefine a flag.

```bash
bash scripts/paper/labs_notes/metal/run_rnn.sh                    # GPU 0
GPU=3 SEED=1 bash scripts/paper/labs_notes/metal/run_rnn.sh
sbatch --export=ALL,CELL=run_rnn.sh,TREE=...,EHR_ROOT=...,NOTE_ROOT=...,CACHE_DIR=... \
    scripts/paper/labs_notes/slurm/cell.sbatch
condor_submit CELL=run_rnn.sh TREE=... scripts/paper/labs_notes/condor/cell.sub
```

Env overrides: `SEED` `TREE` `EHR_ROOT` `NOTE_ROOT` `CXR_ROOT` `CACHE_DIR` `GPU`
`FROZEN_TEXT_CACHE` `FOREGROUND`. `SEED` defaults to 1 (ratri@ieee.org);
teammates run the same scripts with `SEED=2..5`. Logs land in `$TREE/logs/`,
outputs in `$TREE/output/paper_<category>_<model>_seed<N>/`, and `PYTHONPATH` is
pinned to `$TREE` because the `pyhealth2` env has an editable install that
otherwise shadows the checkout.

## Protocol

`--embedding-dim 128 --hidden-dim 128 --dropout 0.1 --batch-size 32 --lr 1e-4
--adam-eps 1e-8 --max-grad-norm 1.0 --epochs 50 --patience 5 --use-amp
--amp-dtype bf16`, identical for every backbone and every modality set. Split is
by patient 0.8/0.1/0.1 at the run seed; selection is best val PR-AUC with
patience 5, and that checkpoint is what gets evaluated on test.

Labs + Notes and Labs + Notes + CXR freeze Bio_ClinicalBERT and cache its `[CLS]`
vector per note, capped at 1e6 entries (`FROZEN_TEXT_CACHE`). The cap is a RAM
fuse, not a knob: 200k / 500k / 1e6 gave 0.571 / 0.575 / 0.566 test PR-AUC on the
MLP cell, i.e. seed noise, while epoch time went 4984 s -> 2249 s -> 191 s.

Notes are the subsetted clinical headers the spec asks for — radiology:
indication and impression; discharge: chief complaint — truncated at 512
Bio_ClinicalBERT tokens. That subsetting lives in `NotesLabsMIMIC4`, not in these
scripts. ICD codes stay off (`include_icd=False`): MIMIC-IV stamps them at
`dischtime`, which leaks the label.

**Observation window.** These tasks have no window API: labs, radiology and CXR
are collected through discharge of every admission up to and including the one
the patient died in. The model therefore reads data up to the moment of death,
which is why ROC-AUC sits at 0.94-0.98. It is the same for all six backbones so
the comparison is internally fair, but no cell here is an early-warning result.
`run_config.json` records this as `observation_window: full_stay_through_discharge`.

## Flags a backbone ignores

The runner logs these at startup and writes them to `run_config.json` as
`inert_flags`, so a config can never appear to set something that never reached
the model:

| backbone | ignores |
|---|---|
| MLP | `--dropout` (the model has no dropout parameter), all architecture flags |
| RNN | `--heads`, `--num-layers`, bottleneck / mamba / jamba flags |
| Transformer | `--hidden-dim`, rnn / bottleneck / mamba / jamba flags |
| Bottleneck Transformer | `--hidden-dim`, rnn / mamba / jamba flags |
| EHRMamba | `--hidden-dim`, `--heads`, rnn / bottleneck / jamba flags |
| JambaEHR | `--hidden-dim`, `--num-layers`, rnn / bottleneck flags |

`--hidden-dim 128` and `--dropout 0.1` are passed everywhere anyway because the
spec lists them for all six; the table above is how you tell which ones bit.
JambaEHR is the one place the spec's `--num-layers=2` is dropped rather than
passed — its depth is `--jamba-transformer-layers` plus `--jamba-mamba-layers`,
and passing a flag the model never reads is the thing this directory exists to
stop.

`labs` passes neither `--freeze-encoder` nor `--max-frozen-text-cache`: there is
no text encoder in that task.

## Status

Existing seed-1 numbers (`rian/scripts`, tree `ml4h-tranche1-cache1m`) cover
**Labs + Notes only**, and they predate two fixes in this branch: the runner
hardcoded `max_grad_norm=0.5` and Adam `eps=1e-6` for `bottleneck_transformer`
alone, so that cell was not compute-matched to the other five. Those six cells
need re-running here before they go in a table.

The task cache also does not carry over — `emitted_data_version` is 5 on this
branch against 4 for the published runs — so each category builds its own cache
under `$HOME/pyhealth_cache/paper_<category>/` on first use.
