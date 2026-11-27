# Unified-embedding experiments: A/B, sweeps, ablations

`unified_embedding_experiments.py` drives the existing runner
(`unified_embedding_e2e_mimic4.py::run`) one arm at a time. Each arm
gets its own directory under `--output-root`; together they write
`summary.csv` and `summary.md` (arm, the one differing knob, seconds
per phase, frozen-text cache counters, test scores).

Driver options come first, then `--`, then the runner arguments
(passed through to every arm unchanged):

```
# A/B: two arms, one flag differs (booleans need no --values)
python examples/mortality_prediction/unified_embedding_experiments.py \
  --output-root ./output/xp_ab \
  ab --flag cache-frozen-text -- \
  --ehr-root /data/mimic-iv/2.2 --note-root /data/mimic-iv/note \
  --task notes_labs --model transformer --epochs 3 --device cuda:0 --seed 2

# Sweep: N values of one flag (--flag window-hours is an alias)
python examples/mortality_prediction/unified_embedding_experiments.py \
  --output-root ./output/xp_win \
  sweep --flag window-hours --values 0 24 48 -- \
  --ehr-root /data/mimic-iv/2.2 --task labs --model mlp --epochs 1 --seed 2

# Ablate: reference plus one arm per dropped modality / switched-off component
python examples/mortality_prediction/unified_embedding_experiments.py \
  --output-root ./output/xp_abl \
  ablate --modalities notes labs --off cache-frozen-text -- \
  --ehr-root /data/mimic-iv/2.2 --note-root /data/mimic-iv/note \
  --task notes_labs --model transformer --epochs 3 --device cuda:0 --seed 2
```

`ablate --modalities` drops each modality at the model stage through the
runner's `--drop-modality`: same task, same samples, same split, same
seed. `labs` drops the lab values and their observation mask, `notes`
drops the admission-note field; the cohort keeps every patient either
way. Dropping everything is refused, as is dropping a modality the
task does not have.

Rules the driver enforces (it exits 2 instead of running): unknown
`--flag`; a flag inert for the chosen backbone or task (e.g.
`--hidden-dim` with `--model transformer`, the text-cache flags with
`--task labs` — backbone reachability comes from the runner's own
wiring map, so the two cannot disagree); arms that would differ in more
than the declared knob; duplicate values; `ablate` with nothing to drop
or nothing left after the drop; `ablate` with a passed-through
`--drop-modality` (the arms set it themselves, so a passed value would
be silently replaced). Sweeping `--seed` is allowed but
changes the split by design. Seed, split sizes, and sample identity are
checked across arms and reported in `summary.md` — identical when they
match, naming the arms that differ when they do not. The identity check
compares unique inference-split patient ids; train/val splits and labels
are not compared, and an empty predictions file reports not recorded.
Pass `--output-root`, not `--output-dir` (each arm gets its own
`--output-dir` underneath it).

## Equivalent Python API

```python
from examples.mortality_prediction import unified_embedding_experiments as xp

runner = xp._load_runner()
base = runner.parse_args(["--ehr-root", "/data/mimic-iv/2.2",
                          "--task", "labs", "--model", "mlp",
                          "--epochs", "1", "--seed", "2"])
arms = xp.build_arms("sweep", base, flag="window-hours",
                     values=["0", "24", "48"])
xp.main(["--output-root", "./output/xp_win", "sweep",
         "--flag", "window-hours", "--values", "0", "24", "48",
         "--", "--ehr-root", "/data/mimic-iv/2.2",
         "--task", "labs", "--model", "mlp",
         "--epochs", "1", "--seed", "2"])
```

## Small utilities you will actually use

- Cache counters: every `metrics.json` carries
  `frozen_text_cache: {hits, misses, size}`; the summary rolls them
  up per arm. Compare the on/off arms of
  `ab --flag cache-frozen-text` to see the timing effect.
  `UnifiedMultimodalEmbeddingModel.reset_frozen_text_cache_stats()`
  zeroes them mid-session.
- DataLoader workers: `--num-workers 1` (below 1 is refused when the
  runner builds the dataset itself: a zero-worker Dask cluster never
  computes, the local path divides by the worker count in
  `set_task`, and a negative count dies inside loader construction;
  a prebuilt sample dataset may still load with 0).
  `--no-dask-distributed` builds the cache with the local Dask
  scheduler instead of starting a cluster. `--pin-memory`,
  `--persistent-workers` (needs workers > 0), `--prefetch-factor`
  (needs workers > 0). On CPU, workers 1 with `OMP_NUM_THREADS`
  pinned is the stable setting.
- Cache dir: share one `--cache-dir` across arms of a sweep; the
  task cache key already includes task + window, so arms reuse each
  other's dataset builds. A killed run can leave a stale
  `tasks/<hash>/build.lock`; delete the task dir and rerun.

## Real output (public demo fixture, toy magnitudes, no scientific claim)

`sweep --flag window-hours --values 0 24` on
`test-resources/core/mimic4demo`, labs, mlp, 1 epoch, CPU:

```
arm,differs,seed,train_s,eval_s,infer_s,cache_hits,cache_misses,cache_size,score_accuracy,score_f1,score_loss,score_pr_auc,score_roc_auc
sweep_observation_window_hours_0.0,observation_window_hours=0.0,2,16.934,8.491,8.727,0,0,0,0.5,0.0,0.6831296682357788,1.0,1.0
sweep_observation_window_hours_24.0,observation_window_hours=24.0,2,16.913,8.432,8.743,0,0,0,0.5,0.0,0.6831296682357788,1.0,1.0
```

Both arms agree to all digits: the 13-patient demo stays are short
enough that a 24 h window never binds. On real MIMIC-IV the same
command separates the arms; treat every number above as a timing
smoke test, not a result.
