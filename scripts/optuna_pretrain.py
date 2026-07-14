"""Optuna hyperparameter optimization for SSL PRETRAINING, one study per
(architecture x method).

The encoder *size* is fixed at the standardized 128-dim / 2-layer / 4-head
budget (matching the downstream e2e backbone so encoders transfer 1:1).  Each
study tunes the pretraining/optimization hyperparameters end-to-end plus the
small arch-specific backbone knobs that the user cares about:

    common (all):   lr, weight_decay, batch_size, warmup_steps, max_grad_norm
    mae / simmim:   mask_ratio, mask_strategy, norm_pix_loss
    ijepa / vjepa:  ema_decay, num_target_blocks
    transformer:    use_rope
    mamba:          state_size, conv_kernel
    jamba:          state_size, conv_kernel, jamba_transformer_layers, jamba_mamba_layers

Objective = **held-out SSL validation loss (MINIMIZE)** on a patient-level
subsplit of a small patient pool.  SSL has no labels, so a lower reconstruction
/ latent-prediction loss on unseen patients is the proxy for a better encoder.

Collapse guard: a degenerate config can drive the loss to ~0 (representation
collapse), which a naive minimizer would *prefer*.  Any epoch whose val loss
drops below ``COLLAPSE_FLOOR`` prunes the trial and tags it ``collapsed`` so TPE
avoids that region instead of chasing it.

Example
-------
    python scripts/optuna_pretrain.py \
      --ehr-root /data/mimic-iv/2.2 --note-root /data/mimic-iv/note \
      --task notes_labs --arch mamba --method mae --freeze-encoder \
      --subset-frac 0.05 --n-trials 40 --epochs-per-trial 4 --timeout 43200 \
      --storage sqlite:///output/optuna_pretrain/pt_mamba_mae.db \
      --study-name pt_mamba_mae --output-dir output/optuna_pretrain
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
PRETRAIN_PATH = REPO_ROOT / "scripts" / "pretrain_ssl.py"


def _load_pretrain_module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("pretrain_ssl_mod", PRETRAIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PT = _load_pretrain_module()

from pyhealth.datasets import get_dataloader  # noqa: E402
from pyhealth.models.pretrain import MultimodalIJEPA  # noqa: E402
from pyhealth.models.pretrain.backbones import ARCH_CHOICES  # noqa: E402
from pyhealth.models.pretrain.trainer import PretrainTrainer  # noqa: E402
from pyhealth.utils import set_seed  # noqa: E402

# Standardized encoder size — fixed for every architecture and every trial.
STD_EMBEDDING_DIM = 128
STD_NUM_LAYERS = 2
STD_HEADS = 4

# Val loss below this is treated as representation collapse (degenerate 0-loss),
# not a genuinely good config.  SSL losses here are O(0.1-1); ~0 means collapse.
COLLAPSE_FLOOR = 1e-3


def _base_args(cli: argparse.Namespace) -> SimpleNamespace:
    """Namespace with every attribute pretrain_ssl._build_model reads, seeded
    from pretrain_ssl._DEFAULTS then pinned to the standardized size."""
    ns = SimpleNamespace(**dict(PT._DEFAULTS))
    ns.ehr_root = cli.ehr_root
    ns.note_root = cli.note_root
    ns.cache_dir = cli.cache_dir
    ns.task = cli.task
    ns.method = cli.method
    ns.arch = cli.arch
    ns.dev = cli.dev
    ns.num_workers = cli.num_workers
    ns.observation_window_hours = cli.observation_window_hours
    ns.note_source = cli.note_source
    ns.note_extraction = cli.note_extraction
    ns.tokenizer_model = cli.tokenizer_model
    ns.icd_codes = cli.icd_codes
    ns.include_vitals = cli.include_vitals
    ns.freeze_encoder = cli.freeze_encoder
    # standardized size
    ns.embedding_dim = STD_EMBEDDING_DIM
    ns.num_layers = STD_NUM_LAYERS
    ns.heads = STD_HEADS
    ns.decoder_dim = STD_EMBEDDING_DIM
    ns.predictor_dim = STD_EMBEDDING_DIM
    # per-modality mask ratios are argparse-only (not in _DEFAULTS); default off.
    ns.lab_mask_ratio = None
    ns.text_mask_ratio = None
    return ns


def _patient_pool(dataset, frac, seed: int):
    pats = list(dataset.patient_to_index.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    if frac and frac < 1.0:
        pats = pats[: max(2, int(len(pats) * frac))]
    return pats


def _subsplit(dataset, pool_patients, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    pats = list(pool_patients)
    rng.shuffle(pats)
    n_val = max(1, int(len(pats) * val_frac))
    val_pats, train_pats = pats[:n_val], pats[n_val:]
    tr = list(chain(*[dataset.patient_to_index[p] for p in train_pats]))
    va = list(chain(*[dataset.patient_to_index[p] for p in val_pats]))
    return dataset.subset(tr), dataset.subset(va)


def _suggest_hparams(trial: optuna.Trial, arch: str, method: str) -> Dict[str, Any]:
    """Standardized SSL/optim search space + method- and arch-specific knobs."""
    hp: Dict[str, Any] = {
        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "warmup_steps": trial.suggest_categorical("warmup_steps", [0, 500, 1000, 2000]),
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 5.0]),
    }
    if method in ("mae", "simmim"):
        hp["mask_ratio"] = trial.suggest_float("mask_ratio", 0.25, 0.75)
        hp["mask_strategy"] = trial.suggest_categorical("mask_strategy", ["random", "block"])
        hp["norm_pix_loss"] = trial.suggest_categorical("norm_pix_loss", [False, True])
    else:  # ijepa / vjepa
        hp["ema_decay"] = trial.suggest_categorical(
            "ema_decay", [0.99, 0.995, 0.996, 0.999, 0.9995]
        )
        hp["num_target_blocks"] = trial.suggest_categorical("num_target_blocks", [2, 4, 6])

    if arch == "transformer":
        hp["use_rope"] = trial.suggest_categorical("use_rope", [False, True])
    elif arch == "mamba":
        hp["state_size"] = trial.suggest_categorical("state_size", [8, 16, 32, 64])
        hp["conv_kernel"] = trial.suggest_categorical("conv_kernel", [2, 4])
    elif arch == "jamba":
        hp["state_size"] = trial.suggest_categorical("state_size", [8, 16, 32])
        hp["conv_kernel"] = trial.suggest_categorical("conv_kernel", [2, 4])
        hp["jamba_transformer_layers"] = trial.suggest_categorical("jamba_transformer_layers", [1, 2])
        hp["jamba_mamba_layers"] = trial.suggest_categorical("jamba_mamba_layers", [1, 2])
    return hp


def _trial_args(base: SimpleNamespace, hp: Dict[str, Any]) -> SimpleNamespace:
    args = SimpleNamespace(**vars(base))
    for k in ("mask_ratio", "mask_strategy", "norm_pix_loss", "ema_decay",
              "num_target_blocks", "use_rope", "state_size", "conv_kernel",
              "jamba_transformer_layers", "jamba_mamba_layers"):
        if k in hp:
            setattr(args, k, hp[k])
    return args


def make_objective(cli, sample_dataset, pool_patients):
    base = _base_args(cli)

    def objective(trial: optuna.Trial) -> float:
        set_seed(cli.seed)
        hp = _suggest_hparams(trial, cli.arch, cli.method)
        args = _trial_args(base, hp)
        epochs = cli.epochs_per_trial

        fold_best = []
        vram_peaks = []
        for k in range(cli.n_subsplits):
            train_ds, val_ds = _subsplit(sample_dataset, pool_patients,
                                         cli.subsplit_val_frac, seed=cli.seed + 1000 + k)
            model = PT._build_model(args, sample_dataset)
            ema_fn = None
            if cli.method in ("ijepa", "vjepa") and isinstance(model, MultimodalIJEPA):
                ema_fn = model.update_target_encoder

            train_loader = get_dataloader(train_ds, batch_size=hp["batch_size"], shuffle=True)
            val_loader = get_dataloader(val_ds, batch_size=hp["batch_size"], shuffle=False)
            trainer = PretrainTrainer(model=model, device=cli.device, enable_logging=False,
                                      ema_update_fn=ema_fn)

            best = {"v": float("inf"), "vram": 0.0}

            def _cb(epoch: int, record: Dict, _k=k):
                v = record.get("val_loss")
                if v is None:
                    return
                if v < COLLAPSE_FLOOR:
                    trial.set_user_attr("collapsed", True)
                    raise optuna.TrialPruned()
                best["v"] = min(best["v"], v)
                best["vram"] = max(best["vram"], record.get("train_vram_peak_mb", 0) or 0)
                trial.report(v, _k * epochs + epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            try:
                trainer.train(
                    train_dataloader=train_loader, val_dataloader=val_loader,
                    epochs=epochs, optimizer_params={"lr": hp["lr"]},
                    weight_decay=hp["weight_decay"], max_grad_norm=hp["max_grad_norm"],
                    scheduler="cosine", warmup_steps=hp["warmup_steps"],
                    save_every_n_epochs=10 ** 9, epoch_callback=_cb,
                )
            except optuna.TrialPruned:
                raise
            except RuntimeError as e:
                print(f"[trial {trial.number}] RuntimeError -> pruned: {e}")
                raise optuna.TrialPruned()

            fold_best.append(best["v"])
            if best["vram"]:
                vram_peaks.append(best["vram"])
            trial.set_user_attr(f"fold{k}_val_loss", round(best["v"], 5))

        if vram_peaks:
            trial.set_user_attr("train_vram_peak_mb", round(max(vram_peaks), 1))
        return sum(fold_best) / len(fold_best)

    return objective


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna SSL-pretraining HPO, one study per (arch x method).")
    p.add_argument("--ehr-root", required=True)
    p.add_argument("--note-root", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output-dir", default="./output/optuna_pretrain")
    p.add_argument("--task", default="notes_labs",
                   choices=["stagenet", "icd_labs", "clinical_notes_icd_labs", "notes_labs", "notes_only", "labs_only"])
    p.add_argument("--arch", default="transformer", choices=list(ARCH_CHOICES))
    p.add_argument("--method", default="mae", choices=["mae", "simmim", "ijepa", "vjepa"])
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--epochs-per-trial", type=int, default=4)
    p.add_argument("--timeout", type=int, default=None, help="Study wall-clock cap (seconds).")
    p.add_argument("--subset-frac", type=float, default=0.05,
                   help="Fraction of patients used as the tuning pool. None = all.")
    p.add_argument("--n-subsplits", type=int, default=1,
                   help="Independent train/val subsplits per trial; objective = mean val loss.")
    p.add_argument("--subsplit-val-frac", type=float, default=0.15)
    p.add_argument("--storage", default=None, help="Optuna storage URL (sqlite:///...). None = in-memory.")
    p.add_argument("--study-name", default=None, help="Default: pt_<arch>_<method>_<task>.")
    p.add_argument("--freeze-encoder", action="store_true", default=False)
    # task / data passthrough
    p.add_argument("--observation-window-hours", type=int, default=24)
    p.add_argument("--note-source", default="discharge", choices=["discharge", "radiology"])
    p.add_argument("--note-extraction", default="regex")
    p.add_argument("--tokenizer-model", default=None)
    p.add_argument("--icd-codes", action="store_true", default=False)
    p.add_argument("--include-vitals", action="store_true", default=False)
    p.add_argument("--dev", nargs="?", type=int, const=1000, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    cli = parse_args()
    out_dir = Path(cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cli.seed)
    base = _base_args(cli)
    base_dataset = PT._build_base_dataset(base)
    task = PT._build_task(base)
    sample_dataset = base_dataset.set_task(task, num_workers=cli.num_workers)
    if len(sample_dataset) == 0:
        raise RuntimeError("Task produced zero samples. Check roots/tables.")
    pool = _patient_pool(sample_dataset, cli.subset_frac, seed=cli.seed)
    n_pool_samples = sum(len(sample_dataset.patient_to_index[p]) for p in pool)
    print(f"[optuna-pt] dataset ready: {len(sample_dataset)} samples; pool = {len(pool)} patients "
          f"(~{n_pool_samples} samples, frac={cli.subset_frac}); arch={cli.arch} method={cli.method}; "
          f"{cli.n_subsplits} subsplit(s)/trial, val_frac={cli.subsplit_val_frac}")

    if cli.n_trials <= 0:
        print("[optuna-pt] n_trials<=0: dataset/cache built, exiting (cachewarm only).")
        return

    study_name = cli.study_name or f"pt_{cli.arch}_{cli.method}_{cli.task}"
    # Distinct sampler seed per (arch, method) so studies don't shadow each other.
    tag = f"{cli.arch}_{cli.method}".encode()
    sampler_seed = cli.seed + int.from_bytes(hashlib.md5(tag).digest()[:2], "big")
    study = optuna.create_study(
        study_name=study_name,
        storage=cli.storage,
        load_if_exists=True,
        direction="minimize",  # SSL val loss: lower is better
        sampler=optuna.samplers.TPESampler(seed=sampler_seed, multivariate=True,
                                           n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1, n_startup_trials=5),
    )

    # Opt-in W&B: one run per study in the dedicated sweeps project. Labels match
    # the full-run epoch labels (val_loss, train_vram_peak_mb) so charts unify.
    from pyhealth import _wandb
    wrun = _wandb.init_run(
        config={"arch": cli.arch, "method": cli.method, "task": cli.task,
                "subset_frac": cli.subset_frac, "epochs_per_trial": cli.epochs_per_trial},
        name=study_name, group=cli.task, job_type="optuna-pretrain",
        project=_wandb.sweeps_project())

    def _wandb_cb(_study, trial):
        if trial.value is None:
            return
        extra = {"train_vram_peak_mb": trial.user_attrs["train_vram_peak_mb"]} \
            if "train_vram_peak_mb" in trial.user_attrs else {}
        _wandb.log(wrun, {"sweep/trial": trial.number, "val_loss": trial.value,
                          "best/val_loss": _study.best_value, **extra,
                          **{f"hp/{k}": v for k, v in trial.params.items() if isinstance(v, (int, float))}})

    objective = make_objective(cli, sample_dataset, pool)
    study.optimize(objective, n_trials=cli.n_trials, timeout=cli.timeout,
                   gc_after_trial=True, show_progress_bar=False, callbacks=[_wandb_cb])

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("[optuna-pt] WARNING: no trial completed (all pruned/failed/collapsed). "
              "Nothing to write — loosen pruning or increase epochs.")
        _wandb.finish(wrun)
        return

    best = {
        "study_name": study_name,
        "arch": cli.arch,
        "method": cli.method,
        "task": cli.task,
        "objective": "mean_val_loss (minimize)",
        "n_trials": len(study.trials),
        "n_completed": len(completed),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "fixed": {"embedding_dim": STD_EMBEDDING_DIM, "num_layers": STD_NUM_LAYERS, "heads": STD_HEADS},
        "search": {"subset_frac": cli.subset_frac, "n_subsplits": cli.n_subsplits,
                   "subsplit_val_frac": cli.subsplit_val_frac,
                   "epochs_per_trial": cli.epochs_per_trial},
    }
    best_path = out_dir / f"best_params_pt_{cli.arch}_{cli.method}_{cli.task}.json"
    with best_path.open("w") as f:
        json.dump(best, f, indent=2)
    try:
        study.trials_dataframe().to_csv(
            out_dir / f"trials_pt_{cli.arch}_{cli.method}_{cli.task}.csv", index=False)
    except Exception as e:
        print(f"[optuna-pt] could not write trials csv: {e}")

    _wandb.summary(wrun, {"best/val_loss": study.best_value, "n_completed": len(completed),
                          **{f"best_hp/{k}": v for k, v in study.best_params.items()}})
    _wandb.finish(wrun)

    print(f"\n[optuna-pt] BEST val_loss={study.best_value:.4f} for {cli.arch}/{cli.method}")
    print(f"[optuna-pt] best_params: {json.dumps(study.best_params, indent=2)}")
    print(f"[optuna-pt] wrote {best_path}")


if __name__ == "__main__":
    main()
