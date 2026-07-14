"""End-to-end Optuna hyperparameter optimization, one study per architecture.

Architecture *size* is fixed at the standardized 128-dim / 2-layer / 4-head
compute budget (see examples/.../unified_embedding_e2e_mimic4.py).  This script
tunes the *training* hyperparameters end-to-end (full dataset -> model -> train
-> validate), with the SAME standardized search space across every architecture
so the comparison is fair: each architecture gets equal tuning budget over the
same knobs.

For each architecture you run a separate study (study_name = e2e_<model>_<task>);
the best params are written to best_params_<model>.json and become the
"standardized hyperparameters" used for the final Table-2 comparison runs.

The expensive dataset/task build happens ONCE; each trial only rebuilds the
(small) model and trains for --epochs-per-trial epochs.  Trials report val
PR-AUC per epoch and are pruned by a MedianPruner.

Example
-------
    # Tune the transformer head on the notes_labs task, 40 trials, 8 epochs each:
    python scripts/optuna_e2e.py \
      --ehr-root /data/mimic-iv/2.2 --note-root /data/mimic-iv/note \
      --task notes_labs --model transformer --freeze-encoder \
      --n-trials 40 --epochs-per-trial 8 \
      --storage sqlite:///output/optuna/e2e.db \
      --output-dir output/optuna

Resume / parallelize: point multiple workers at the SAME --storage and
--study-name; Optuna coordinates trial assignment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import optuna

REPO_ROOT = Path(__file__).resolve().parents[1]
E2E_PATH = REPO_ROOT / "examples" / "mortality_prediction" / "unified_embedding_e2e_mimic4.py"


def _load_e2e_module():
    """Import the e2e runner as a module so we can reuse its build helpers."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("unified_e2e", E2E_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E2E = _load_e2e_module()

# Re-export the heavy lifting from the e2e runner (single source of truth).
from pyhealth.datasets import (  # noqa: E402
    get_dataloader,
    sample_balanced,
    sample_weighted,
    split_by_patient,
    split_by_sample,
)
from pyhealth.trainer import Trainer  # noqa: E402
from pyhealth.utils import set_seed  # noqa: E402

# Standardized compute budget — fixed for every architecture and every trial.
# 128-dim / 2-layer / 4-head (head_dim 32).  Bumped up from 64/1 because that
# was too narrow: V-JEPA pretraining drifted and most downstream heads sat near
# mortality prevalence.  VRAM is BERT-bound (~9-11 GB on 48 GB RTX 6000 Ada), so
# the larger backbone is effectively free.
STD_EMBEDDING_DIM = 128
STD_HIDDEN_DIM = 128
STD_NUM_LAYERS = 2
STD_HEADS = 4

MONITOR_METRIC = "pr_auc"


def _base_args(cli: argparse.Namespace) -> SimpleNamespace:
    """A namespace with every attribute the e2e build helpers read, at the
    standardized size.  Per-trial knobs are overwritten in _trial_args()."""
    return SimpleNamespace(
        # data / task
        ehr_root=cli.ehr_root,
        note_root=cli.note_root,
        cache_dir=cli.cache_dir,
        task=cli.task,
        dev=cli.dev,
        num_workers=cli.num_workers,
        observation_window_hours=cli.observation_window_hours,
        note_source=cli.note_source,
        note_extraction=cli.note_extraction,
        tokenizer_model=cli.tokenizer_model,
        icd_codes=cli.icd_codes,
        include_vitals=cli.include_vitals,
        # model (standardized size; non-size knobs filled per trial)
        model=cli.model,
        embedding_dim=STD_EMBEDDING_DIM,
        hidden_dim=STD_HIDDEN_DIM,
        num_layers=STD_NUM_LAYERS,
        heads=STD_HEADS,
        dropout=0.1,
        freeze_encoder=cli.freeze_encoder,
        text_finetune_mode="full",
        # rnn
        rnn_type="GRU",
        rnn_layers=1,
        bidirectional=False,
        # bottleneck
        bottlenecks_n=4,
        fusion_startidx=1,
        # mamba / jamba
        mamba_state_size=16,
        mamba_conv_kernel=4,
        jamba_transformer_layers=1,
        jamba_mamba_layers=1,
    )


def _patient_pool(dataset, frac, seed: int):
    """Sample a fraction of PATIENTS from the full dataset as the tuning pool.
    Patient-level so a patient never straddles the later train/val subsplits."""
    pats = list(dataset.patient_to_index.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    if frac and frac < 1.0:
        pats = pats[: max(2, int(len(pats) * frac))]
    return pats


def _subsplit(dataset, pool_patients, val_frac: float, seed: int):
    """One patient-level train/val split of the pool (indices into ``dataset``)."""
    rng = np.random.default_rng(seed)
    pats = list(pool_patients)
    rng.shuffle(pats)
    n_val = max(1, int(len(pats) * val_frac))
    val_pats, train_pats = pats[:n_val], pats[n_val:]
    tr = list(chain(*[dataset.patient_to_index[p] for p in train_pats]))
    va = list(chain(*[dataset.patient_to_index[p] for p in val_pats]))
    return dataset.subset(tr), dataset.subset(va)


def _suggest_hparams(trial: optuna.Trial, model: str, tune_arch: bool) -> Dict[str, Any]:
    """Standardized training-HP search space (identical across architectures),
    plus an optional small arch-specific structural extension."""
    hp: Dict[str, Any] = {
        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 5.0]),
        "pos_weight": trial.suggest_float("pos_weight", 1.0, 12.0),
        "sampling_strategy": trial.suggest_categorical(
            "sampling_strategy", ["none", "undersample", "weighted"]
        ),
    }
    if tune_arch:
        if model == "rnn":
            hp["rnn_type"] = trial.suggest_categorical("rnn_type", ["GRU", "LSTM"])
            hp["bidirectional"] = trial.suggest_categorical("bidirectional", [False, True])
        elif model == "bottleneck_transformer":
            hp["bottlenecks_n"] = trial.suggest_categorical("bottlenecks_n", [2, 4, 8])
        elif model in ("ehrmamba", "jambaehr"):
            hp["mamba_state_size"] = trial.suggest_categorical("mamba_state_size", [8, 16, 32])
            hp["mamba_conv_kernel"] = trial.suggest_categorical("mamba_conv_kernel", [2, 4])
    return hp


def _trial_args(base: SimpleNamespace, hp: Dict[str, Any]) -> SimpleNamespace:
    args = SimpleNamespace(**vars(base))
    args.dropout = hp["dropout"]
    for k in ("rnn_type", "bidirectional", "bottlenecks_n",
              "mamba_state_size", "mamba_conv_kernel"):
        if k in hp:
            setattr(args, k, hp[k])
    return args


def make_objective(cli, sample_dataset, pool_patients, label_key):
    """Each trial trains on ``--n-subsplits`` independent patient-level train/val
    splits of the tuning pool and returns the MEAN best val PR-AUC, so the chosen
    HPs are robust to a single lucky/unlucky split."""
    base = _base_args(cli)
    import torch

    def objective(trial: optuna.Trial) -> float:
        set_seed(cli.seed)
        hp = _suggest_hparams(trial, cli.model, cli.tune_arch_specific)
        args = _trial_args(base, hp)
        epochs = cli.epochs_per_trial

        fold_best = []
        # extra metrics (at each fold's best epoch) averaged across folds, for W&B.
        _EXTRA = ["val_roc_auc", "val_f1", "val_accuracy", "val_loss", "train_vram_peak_mb"]
        fold_extra = {k: [] for k in _EXTRA}
        for k in range(cli.n_subsplits):
            train_ds, val_ds = _subsplit(sample_dataset, pool_patients,
                                         cli.subsplit_val_frac, seed=cli.seed + 1000 + k)
            if hp["sampling_strategy"] == "undersample":
                train_ds = sample_balanced(train_ds, ratio=1.0, seed=cli.seed, label_key=label_key)
            elif hp["sampling_strategy"] == "weighted":
                train_ds = sample_weighted(train_ds, seed=cli.seed, label_key=label_key)

            model = E2E._build_model(args, sample_dataset)
            if cli.pretrained_ckpt:
                E2E._load_pretrained_weights(model, cli.pretrained_ckpt)
            model._pos_weight = torch.tensor([hp["pos_weight"]], dtype=torch.float32)

            train_loader = get_dataloader(train_ds, batch_size=hp["batch_size"], shuffle=True)
            val_loader = get_dataloader(val_ds, batch_size=hp["batch_size"], shuffle=False)
            trainer = Trainer(model=model, metrics=["pr_auc", "roc_auc", "f1", "accuracy"],
                              device=cli.device, enable_logging=False)

            best = {"v": 0.0, "rec": {}}

            def _prune_cb(epoch: int, record: Dict, _k=k):
                v = record.get(f"val_{MONITOR_METRIC}")
                if v is None:
                    return
                if v > best["v"] or not best["rec"]:
                    best["v"] = max(best["v"], v)
                    best["rec"] = dict(record)   # full record at this fold's best epoch
                # report on a global step (fold*epochs+epoch) so MedianPruner can
                # compare trials at aligned points across the whole subsplit budget.
                trial.report(v, _k * epochs + epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            try:
                trainer.train(
                    train_dataloader=train_loader, val_dataloader=val_loader,
                    epochs=epochs, optimizer_params={"lr": hp["lr"]},
                    weight_decay=hp["weight_decay"], max_grad_norm=hp["max_grad_norm"],
                    monitor=MONITOR_METRIC, load_best_model_at_last=False,
                    epoch_callback=_prune_cb,
                )
            except optuna.TrialPruned:
                raise
            except RuntimeError as e:
                # OOM or numerical blow-up: prune rather than fail the whole study.
                print(f"[trial {trial.number}] RuntimeError -> pruned: {e}")
                raise optuna.TrialPruned()

            fold_best.append(best["v"])
            trial.set_user_attr(f"fold{k}_pr_auc", round(best["v"], 4))
            for m in _EXTRA:
                if best["rec"].get(m) is not None:
                    fold_extra[m].append(best["rec"][m])

        # mean of each extra metric across folds -> user_attrs (logged to W&B per trial)
        for m, vals in fold_extra.items():
            if vals:
                trial.set_user_attr(m, round(sum(vals) / len(vals), 4))

        return sum(fold_best) / len(fold_best)

    return objective


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna e2e HPO, one study per architecture.")
    p.add_argument("--ehr-root", required=True)
    p.add_argument("--note-root", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output-dir", default="./output/optuna")
    p.add_argument("--task", default="notes_labs",
                   choices=["stagenet", "icd_labs", "clinical_notes_icd_labs", "notes_labs", "notes_only", "labs_only"])
    p.add_argument("--model", default="transformer",
                   choices=["mlp", "rnn", "transformer", "bottleneck_transformer",
                            "cross_attn_fusion", "simple_fusion", "ehrmamba", "jambaehr"])
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--epochs-per-trial", type=int, default=8)
    p.add_argument("--timeout", type=int, default=None, help="Study wall-clock cap (seconds).")
    p.add_argument("--subset-frac", type=float, default=None,
                   help="Fraction of patients to use as the tuning pool (e.g. 0.05 = 5%%). "
                        "Sampled from the full cached dataset. None = use all patients.")
    p.add_argument("--n-subsplits", type=int, default=1,
                   help="Independent train/val subsplits per trial; objective = mean val PR-AUC.")
    p.add_argument("--subsplit-val-frac", type=float, default=0.15,
                   help="Validation fraction within each patient-level subsplit.")
    p.add_argument("--storage", default=None,
                   help="Optuna storage URL, e.g. sqlite:///output/optuna/e2e.db. "
                        "Default None = in-memory (single process, no resume).")
    p.add_argument("--study-name", default=None,
                   help="Override study name. Default: e2e_<model>_<task>.")
    p.add_argument("--tune-arch-specific", action="store_true",
                   help="Also tune small arch-specific structural knobs (rnn_type, "
                        "bottlenecks_n, mamba_state_size, ...). Size stays fixed at 128/2.")
    p.add_argument("--pretrained-ckpt", default=None,
                   help="Optional SSL checkpoint to initialize each trial's model.")
    # task / data passthrough
    p.add_argument("--observation-window-hours", type=int, default=24)
    p.add_argument("--note-source", default="discharge", choices=["discharge", "radiology"])
    p.add_argument("--note-extraction", default="regex")
    p.add_argument("--tokenizer-model", default=None)
    p.add_argument("--icd-codes", action="store_true", default=False)
    p.add_argument("--include-vitals", action="store_true", default=False)
    p.add_argument("--freeze-encoder", action="store_true", default=False)
    p.add_argument("--dev", nargs="?", type=int, const=1000, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    cli = parse_args()
    out_dir = Path(cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- build dataset + task + splits ONCE (shared by every trial) ----
    set_seed(cli.seed)
    base = _base_args(cli)
    base_dataset = E2E._build_base_dataset(base)
    task = E2E._build_task(base)
    sample_dataset = base_dataset.set_task(task, num_workers=cli.num_workers)
    if len(sample_dataset) == 0:
        raise RuntimeError("Task produced zero samples. Check roots/tables.")
    label_key = list(sample_dataset.output_schema.keys())[0]
    pool = _patient_pool(sample_dataset, cli.subset_frac, seed=cli.seed)
    n_pool_samples = sum(len(sample_dataset.patient_to_index[p]) for p in pool)
    print(f"[optuna] dataset ready: {len(sample_dataset)} samples; tuning pool = "
          f"{len(pool)} patients (~{n_pool_samples} samples, frac={cli.subset_frac}); "
          f"{cli.n_subsplits} subsplit(s)/trial, val_frac={cli.subsplit_val_frac}; label={label_key}")

    # Cachewarm-only mode: build (and cache) the dataset, then exit. Lets a serial
    # step prime the shared cache before parallel per-arch GPU studies launch.
    if cli.n_trials <= 0:
        print("[optuna] n_trials<=0: dataset/cache built, exiting (cachewarm only).")
        return

    study_name = cli.study_name or f"e2e_{cli.model}_{cli.task}"
    # Per-arch sampler seed: give each architecture a DISTINCT random-search
    # sequence.  With one shared seed, archs whose search space has the same
    # shape (mlp/transformer, ehrmamba/jambaehr) drew identical trials and picked
    # identical "best" params.  n_startup_trials=5 (default 10) so TPE starts
    # modeling the objective after 5 trials instead of never — our per-arch
    # budget (~25-30 trials) barely cleared 10 before, leaving it pure random.
    sampler_seed = cli.seed + int.from_bytes(hashlib.md5(cli.model.encode()).digest()[:2], "big")
    study = optuna.create_study(
        study_name=study_name,
        storage=cli.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed, multivariate=True,
                                           n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2, n_startup_trials=5),
    )

    # Opt-in W&B: one run per study in the dedicated sweeps project. Metric names
    # match the full-run epoch labels (val_pr_auc, val_roc_auc, ...) so charts unify.
    from pyhealth import _wandb
    wrun = _wandb.init_run(
        config={"model": cli.model, "task": cli.task, "metric": MONITOR_METRIC,
                "subset_frac": cli.subset_frac, "epochs_per_trial": cli.epochs_per_trial},
        name=study_name, group=cli.task, job_type="optuna-e2e",
        project=_wandb.sweeps_project())

    def _wandb_cb(_study, trial):
        if trial.value is None:
            return
        extra = {m: trial.user_attrs[m] for m in
                 ("val_roc_auc", "val_f1", "val_accuracy", "val_loss", "train_vram_peak_mb")
                 if m in trial.user_attrs}
        _wandb.log(wrun, {"sweep/trial": trial.number, f"val_{MONITOR_METRIC}": trial.value,
                          f"best/val_{MONITOR_METRIC}": _study.best_value, **extra,
                          **{f"hp/{k}": v for k, v in trial.params.items() if isinstance(v, (int, float))}})

    objective = make_objective(cli, sample_dataset, pool, label_key)
    study.optimize(objective, n_trials=cli.n_trials, timeout=cli.timeout,
                   gc_after_trial=True, show_progress_bar=False, callbacks=[_wandb_cb])

    # ---- persist results ----
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("[optuna] WARNING: no trial completed (all pruned/failed). "
              "Nothing to write — increase epochs/trials or loosen pruning.")
        _wandb.finish(wrun)
        return

    best = {
        "study_name": study_name,
        "model": cli.model,
        "task": cli.task,
        "metric": MONITOR_METRIC,
        "n_trials": len(study.trials),
        "n_completed": len(completed),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "fixed": {"embedding_dim": STD_EMBEDDING_DIM, "hidden_dim": STD_HIDDEN_DIM,
                  "num_layers": STD_NUM_LAYERS, "heads": STD_HEADS},
        "search": {"subset_frac": cli.subset_frac, "n_subsplits": cli.n_subsplits,
                   "subsplit_val_frac": cli.subsplit_val_frac,
                   "epochs_per_trial": cli.epochs_per_trial, "metric": "mean_val_pr_auc"},
    }
    best_path = out_dir / f"best_params_{cli.model}_{cli.task}.json"
    with best_path.open("w") as f:
        json.dump(best, f, indent=2)
    try:
        study.trials_dataframe().to_csv(out_dir / f"trials_{cli.model}_{cli.task}.csv", index=False)
    except Exception as e:
        print(f"[optuna] could not write trials csv: {e}")

    _wandb.summary(wrun, {f"best/val_{MONITOR_METRIC}": study.best_value, "n_completed": len(completed),
                          **{f"best_hp/{k}": v for k, v in study.best_params.items()}})
    _wandb.finish(wrun)

    print(f"\n[optuna] BEST {MONITOR_METRIC}={study.best_value:.4f} for {cli.model}")
    print(f"[optuna] best_params: {json.dumps(study.best_params, indent=2)}")
    print(f"[optuna] wrote {best_path}")


if __name__ == "__main__":
    main()
