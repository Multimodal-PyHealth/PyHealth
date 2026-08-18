"""Thin, opt-in Weights & Biases logging helper.

Everything here is a no-op unless ``WANDB_PROJECT`` is set in the environment
(and ``wandb`` is importable), so importing/using it never breaks a run that
doesn't want tracking. Entity/project/mode come from the standard W&B env vars:

    WANDB_PROJECT   e.g. pyhealth-multimodal   (required to enable logging)
    WANDB_ENTITY    e.g. pyhealth-multimodal   (team; optional)
    WANDB_MODE      online (default) | offline  (offline for no-internet nodes;
                                                 sync later with `wandb sync`)

Failures (no network, bad key, wandb missing) degrade to a warning + no-op —
tracking must never take down training.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional


def enabled() -> bool:
    return bool(os.environ.get("WANDB_PROJECT"))


def sweeps_project() -> Optional[str]:
    """Dedicated project for Optuna sweeps, kept separate from the real experiment
    runs so the main project isn't cluttered by per-trial study runs. Override with
    WANDB_PROJECT_SWEEPS; otherwise it's ``<WANDB_PROJECT>-sweeps``."""
    p = os.environ.get("WANDB_PROJECT")
    return os.environ.get("WANDB_PROJECT_SWEEPS") or (f"{p}-sweeps" if p else None)


def init_run(config: Optional[Dict[str, Any]] = None, name: Optional[str] = None,
             group: Optional[str] = None, job_type: Optional[str] = None,
             project: Optional[str] = None, tags: Optional[list] = None):
    """Start a W&B run, or return None if disabled/unavailable."""
    if not enabled():
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] WANDB_PROJECT set but wandb not installed — skipping tracking.")
        return None
    try:
        return wandb.init(
            project=project or os.environ["WANDB_PROJECT"],
            entity=os.environ.get("WANDB_ENTITY"),
            name=name,
            group=group,
            job_type=job_type,
            config=config or {},
            tags=[t for t in (tags or []) if t],
            reinit=True,
        )
    except Exception as e:  # network/auth/etc. — never fail the run over telemetry
        print(f"[wandb] init failed ({e}) — continuing without tracking.")
        return None


# Canonical metric namespaces so W&B groups panels into a few tidy sections
# (val/, test/, best/, loss/, sys/) instead of dozens of flat, redundant keys.
# Applied at log()/summary() so every caller stays consistent for free.  Only
# affects W&B display names; on-disk metrics_history.json keeps its raw keys.
_CANON = {
    "pr_auc": "val/pr_auc", "val_pr_auc": "val/pr_auc", "val_roc_auc": "val/roc_auc",
    "val_f1": "val/f1", "val_accuracy": "val/accuracy", "val_loss": "val/loss",
    "test_pr_auc": "test/pr_auc", "test_roc_auc": "test/roc_auc", "test_f1": "test/f1",
    "test_accuracy": "test/accuracy", "test_loss": "test/loss", "test_test_loss": "test/loss",
    "test_n": "test/n", "test_pos": "test/pos",
    "best_pr_auc": "best/pr_auc", "best_val_pr_auc": "best/pr_auc", "best_epoch": "best/epoch",
    "total": "loss/total", "train_loss": "loss/train",
    "train_vram_peak_mb": "sys/vram_peak_mb", "train_vram_allocated_mb": "sys/vram_allocated_mb",
    "learning_rate": "sys/lr", "epoch_time_s": "sys/epoch_time_s", "global_step": "sys/global_step",
}


def _canon_key(k: str) -> str:
    if k in _CANON:
        return _CANON[k]
    if k.startswith("modality_") or k.startswith("scale_"):  # per-modality / multi-scale SSL loss
        return "loss/" + k
    if k.startswith("train_loss_"):                          # per-component supervised loss
        return "loss/" + k[len("train_loss_"):]
    return k  # already-namespaced (hp/, arch/, sweep/, epoch, trial, ...) pass through


def _canon(record: Dict[str, Any]) -> Dict[str, Any]:
    return {_canon_key(k): v for k, v in record.items()}


def log(run, record: Dict[str, Any], step: Optional[int] = None) -> None:
    if run is None:
        return
    try:
        run.log(_canon(dict(record)), step=step)
    except Exception:
        pass


def summary(run, record: Dict[str, Any]) -> None:
    if run is None:
        return
    try:
        for k, v in _canon(record).items():
            run.summary[k] = v
    except Exception:
        pass


def finish(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass
