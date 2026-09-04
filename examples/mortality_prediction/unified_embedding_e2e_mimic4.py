"""End-to-end protocol runner for Unified Embedding on MIMIC-IV.

Trains and evaluates a unified-embedding model (RNN / Transformer /
BottleneckTransformer / EHRMamba / JambaEHR) on a MIMIC-IV mortality task,
then writes per-sample predictions to CSV.

Tasks
-----
--task labs (default)
    LabsMIMIC4: 10-dim lab vectors only.

--task notes_labs (recommended for multimodal)
    NotesLabsMIMIC4: notes + 10-dim lab vectors.

--task notes_labs_cxr
    NotesLabsCXRMIMIC4: notes + labs + chest-xray.

Example
-------
    python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
      --ehr-root /path/to/mimiciv/2.2 \\
      --task labs \\
      --model transformer \\
      --heads 4 --num-layers 2 \\
      --dev --device cpu \\
      --epochs 10 --batch-size 32 --lr 1e-3 \\
      --output-dir ./output/unified_e2e

    # EHRMamba on full dataset (no --dev):
    python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
      --ehr-root /data/mimic-iv/2.2 --note-root /data/mimic-iv/note \\
      --task notes_labs --model ehrmamba \\
      --embedding-dim 128 --num-layers 2 --seed 42

    # JambaEHR:
    python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
      --ehr-root /data/mimic-iv/2.2 --note-root /data/mimic-iv/note \\
      --task notes_labs --model jambaehr \\
      --embedding-dim 128 --jamba-transformer-layers 2 --jamba-mamba-layers 6
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient,
    split_by_sample,
)
from pyhealth.models import MLP, RNN, Transformer, UnifiedMultimodalEmbeddingModel
from pyhealth.models.bottleneck_transformer import BottleneckTransformer
from pyhealth.models.ehrmamba import EHRMamba
from pyhealth.models.jamba_ehr import JambaEHR
from pyhealth.tasks.multimodal_mimic4 import (
    LabsMIMIC4,
    NotesLabsCXRMIMIC4,
    NotesLabsMIMIC4,
)
from pyhealth.processors import fit_lab_standardizer
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed, write_run_config

logger = logging.getLogger(__name__)


class WandbLogger:

    def __init__(
        self,
        enabled: bool,
        project: str,
        entity: Optional[str],
        run_name: str,
        tags: list[str],
        config: Dict[str, Any],
        group: Optional[str] = None,
        job_type: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self._run = None
        if self.enabled:
            import wandb

            self._run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                tags=tags,
                config=config,
                group=group,
                job_type=job_type,
            )

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled:
            self._run.log(data, step=step)

    def finish(self) -> None:
        if self.enabled:
            self._run.finish()


def _build_base_dataset(args: argparse.Namespace) -> MIMIC4Dataset:
    ehr_tables = ["labevents"]
    note_tables = None
    cxr_kwargs = {}

    if args.task == "notes_labs":
        if not args.note_root:
            raise ValueError("--task notes_labs requires --note-root.")
        note_tables = ["discharge", "radiology"]

    if args.task == "notes_labs_cxr":
        if not args.note_root:
            raise ValueError("--task notes_labs_cxr requires --note-root.")
        if not args.cxr_root:
            raise ValueError("--task notes_labs_cxr requires --cxr-root.")
        note_tables = ["discharge", "radiology"]
        cxr_kwargs = dict(
            cxr_root=args.cxr_root,
            cxr_variant=args.cxr_variant,
            cxr_tables=["metadata", "negbio", "chexpert", "split"],
        )

    return MIMIC4Dataset(
        ehr_root=args.ehr_root,
        ehr_tables=ehr_tables,
        note_root=args.note_root if note_tables else None,
        note_tables=note_tables,
        cache_dir=args.cache_dir,
        dev=args.dev if args.dev else False,
        num_workers=args.num_workers,
        **cxr_kwargs,
    )


def _build_task(args: argparse.Namespace):
    if args.task == "notes_labs":
        return NotesLabsMIMIC4(
            window_hours=args.observation_window_hours,
        )
    if args.task == "notes_labs_cxr":
        return NotesLabsCXRMIMIC4(
            window_hours=args.observation_window_hours,
        )
    if args.task == "labs":
        return LabsMIMIC4(window_hours=args.observation_window_hours)
    raise ValueError(f"Unknown task: {args.task}")


def _split_dataset(
    dataset: Any, seed: int, allow_leaky_split: bool = False
) -> Tuple[Any, Any, Any]:
    """Split by patient. Refuse the leaky by-sample fallback unless asked.

    The fallback puts a patient with several admissions in both train and test,
    which inflates every metric, and it used to happen with no warning at all.
    A run that cannot be split correctly should stop rather than quietly produce
    numbers nobody can use.
    """
    train_ds, val_ds, test_ds = split_by_patient(dataset, [0.8, 0.1, 0.1], seed=seed)
    if len(train_ds) > 0 and len(test_ds) > 0:
        return train_ds, val_ds, test_ds

    if not allow_leaky_split:
        raise RuntimeError(
            f"split_by_patient produced an empty split (train={len(train_ds)}, "
            f"test={len(test_ds)}) on {len(dataset)} samples. The by-sample "
            "fallback would put the same patient in train and test, so this "
            "run is refused. Widen the cohort, or pass --allow-leaky-split if "
            "you are running a smoke test and know the metrics are garbage."
        )

    warnings.warn(
        "Falling back to split_by_sample at your request. The same patient may "
        "now appear in train and test, so these metrics are optimistic and not "
        "comparable to patient-split runs.",
        RuntimeWarning,
        stacklevel=2,
    )
    return split_by_sample(dataset, [0.8, 0.1, 0.1], seed=seed)


# Architecture flags each backbone actually consumes. Anything else the parser
# accepts is inert for that model and is warned about at startup, so a launcher
# can never appear to set something that never reached the model. --dropout is
# absent for mlp on purpose: pyhealth.models.mlp.MLP has no dropout parameter,
# and it takes **kwargs, so passing one is swallowed rather than rejected.
_ARCH_FLAGS_USED: Dict[str, Tuple[str, ...]] = {
    "mlp": ("embedding_dim", "hidden_dim", "mlp_layers", "mlp_activation"),
    "rnn": (
        "embedding_dim",
        "hidden_dim",
        "dropout",
        "rnn_type",
        "rnn_layers",
        "bidirectional",
    ),
    "transformer": ("embedding_dim", "dropout", "heads", "num_layers"),
    "bottleneck_transformer": (
        "embedding_dim",
        "dropout",
        "heads",
        "num_layers",
        "bottlenecks_n",
        "fusion_startidx",
    ),
    "ehrmamba": (
        "embedding_dim",
        "dropout",
        "num_layers",
        "mamba_state_size",
        "mamba_conv_kernel",
    ),
    "jambaehr": (
        "embedding_dim",
        "dropout",
        "heads",
        "mamba_state_size",
        "mamba_conv_kernel",
        "jamba_transformer_layers",
        "jamba_mamba_layers",
    ),
}

_ARCH_FLAGS_ALL: Tuple[str, ...] = tuple(
    sorted({flag for flags in _ARCH_FLAGS_USED.values() for flag in flags})
)


def _inert_arch_flags(model: str) -> list[str]:
    """Architecture flags this backbone ignores, as CLI spellings."""
    used = set(_ARCH_FLAGS_USED[model])
    return ["--" + f.replace("_", "-") for f in _ARCH_FLAGS_ALL if f not in used]


def _build_model(
    args: argparse.Namespace,
    sample_dataset: Any,
    numeric_standardizers: Optional[dict[str, Any]] = None,
):
    inert = _inert_arch_flags(args.model)
    if inert:
        logger.warning(
            "%s ignores these flags: %s. Do not read them as settings that "
            "took effect.",
            args.model,
            " ".join(inert),
        )

    unified = UnifiedMultimodalEmbeddingModel(
        processors=sample_dataset.input_processors,
        embedding_dim=args.embedding_dim,
        freeze_text_encoder=args.freeze_encoder,
        numeric_standardizers=numeric_standardizers,
        max_frozen_text_cache=args.max_frozen_text_cache,
        text_grad_checkpoint_rows=args.text_grad_checkpoint_rows,
    )

    if args.model == "mlp":
        return MLP(
            dataset=sample_dataset,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.mlp_layers,
            activation=args.mlp_activation,
            unified_embedding=unified,
        )
    if args.model == "rnn":
        return RNN(
            dataset=sample_dataset,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            unified_embedding=unified,
            rnn_type=args.rnn_type,
            num_layers=args.rnn_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
        )
    if args.model == "transformer":
        return Transformer(
            dataset=sample_dataset,
            embedding_dim=args.embedding_dim,
            heads=args.heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            unified_embedding=unified,
        )
    if args.model == "bottleneck_transformer":
        return BottleneckTransformer(
            dataset=sample_dataset,
            embedding_dim=args.embedding_dim,
            bottlenecks_n=args.bottlenecks_n,
            fusion_startidx=args.fusion_startidx,
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout,
            unified_embedding=unified,
        )
    if args.model == "ehrmamba":
        return EHRMamba(
            dataset=sample_dataset,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            state_size=args.mamba_state_size,
            conv_kernel=args.mamba_conv_kernel,
            dropout=args.dropout,
            unified_embedding=unified,
        )
    if args.model == "jambaehr":
        return JambaEHR(
            dataset=sample_dataset,
            embedding_dim=args.embedding_dim,
            num_transformer_layers=args.jamba_transformer_layers,
            num_mamba_layers=args.jamba_mamba_layers,
            heads=args.heads,
            dropout=args.dropout,
            state_size=args.mamba_state_size,
            conv_kernel=args.mamba_conv_kernel,
            unified_embedding=unified,
        )
    raise ValueError(f"Unknown model: {args.model}")


def _write_predictions(
    output_csv: Path,
    patient_ids: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    y_true_flat = y_true.reshape(-1).tolist()
    y_prob_flat = y_prob.reshape(-1).tolist()

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["patient_id", "y_true", "y_prob", "y_pred_threshold_0_5"],
        )
        writer.writeheader()
        for idx, prob in enumerate(y_prob_flat):
            writer.writerow(
                {
                    "patient_id": patient_ids[idx],
                    "y_true": int(y_true_flat[idx]),
                    "y_prob": float(prob),
                    "y_pred_threshold_0_5": int(float(prob) >= 0.5),
                }
            )


def run(args: argparse.Namespace) -> Path:
    set_seed(args.seed)

    base_dataset = _build_base_dataset(args)
    task = _build_task(args)
    sample_dataset = base_dataset.set_task(task, num_workers=args.num_workers)

    if len(sample_dataset) == 0:
        raise RuntimeError(
            "Task produced zero samples. Check roots/tables or adjust settings."
        )

    train_ds, val_ds, test_ds = _split_dataset(
        sample_dataset, seed=args.seed, allow_leaky_split=args.allow_leaky_split
    )

    # Lab z-scores, fit on the training split only. Missing values stay
    # missing; --no-lab-standardization runs raw labs as an ablation.
    numeric_standardizers: dict[str, Any] = {}
    if "labs" in sample_dataset.input_processors and not args.no_lab_standardization:
        numeric_standardizers["labs"] = fit_lab_standardizer(train_ds)

    model = _build_model(args, sample_dataset, numeric_standardizers)

    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = (
        get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
        if len(val_ds) > 0
        else None
    )
    test_loader = (
        get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)
        if len(test_ds) > 0
        else None
    )

    # The window belongs in the name: an observation-window arm is a different
    # experiment from the full-stay run at the same task/model/seed, and without
    # the suffix the two share an output directory and a W&B run name.
    window_suffix = (
        f"_w{int(args.observation_window_hours)}"
        if args.observation_window_hours
        else ""
    )
    exp_name = f"{args.task}_{args.model}_seed{args.seed}{window_suffix}"
    output_dir = Path(args.output_dir)

    wandb_logger = WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name or exp_name,
        tags=args.wandb_tags.split(",") if args.wandb_tags else [args.task, args.model],
        config=vars(args),
        # Group by arm and split by backbone so a many-cell sweep is navigable
        # instead of one flat list of runs.
        group=f"{args.task}{window_suffix}",
        job_type=args.model,
    )

    trainer = Trainer(
        model=model,
        metrics=["pr_auc", "roc_auc", "f1", "accuracy"],
        device=args.device,
        enable_logging=True,
        output_path=str(output_dir),
        exp_name=exp_name,
    )

    # Optimizer settings come from the CLI only. There used to be a per-model
    # branch here that gave bottleneck_transformer max_grad_norm=0.5 and Adam
    # eps=1e-6 while every other backbone got 1.0 and 1e-8, so a six-backbone
    # table that reads as compute-matched was not. If a model needs different
    # optimizer settings, the launcher has to say so.
    effective_lr = args.lr
    effective_max_grad_norm = args.max_grad_norm
    optimizer_params = {"lr": effective_lr, "eps": args.adam_eps}

    if args.epochs > 0 and len(train_ds) > 0:
        metrics_history = trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs,
            optimizer_params=optimizer_params,
            weight_decay=args.weight_decay,
            max_grad_norm=effective_max_grad_norm,
            monitor="pr_auc",
            load_best_model_at_last=True,
            patience=args.patience,
            use_amp=args.use_amp,
            amp_dtype=args.amp_dtype,
        )
        for epoch_record in metrics_history:
            wandb_logger.log(epoch_record, step=epoch_record["epoch"])

    # Test evaluation must not depend on the logger. This was gated on
    # wandb_logger.enabled, so a run without --wandb never computed test
    # metrics at all -- they were not merely unlogged, they were never
    # calculated.
    test_scores = None
    if test_loader is not None:
        test_scores = trainer.evaluate(test_loader)
        if wandb_logger.enabled:
            wandb_logger.log({f"test_{k}": v for k, v in test_scores.items()})

    if test_loader is not None:
        inference_loader, eval_split = test_loader, "test"
    elif val_loader is not None:
        inference_loader, eval_split = val_loader, "val"
        warnings.warn("No test split; predictions come from VAL.", RuntimeWarning)
    else:
        inference_loader, eval_split = train_loader, "train"
        warnings.warn(
            "No test or val split; predictions come from TRAIN and are held-in.",
            RuntimeWarning,
        )
    y_true, y_prob, _, patient_ids = trainer.inference(
        inference_loader, return_patient_ids=True
    )

    write_run_config(
        str(output_dir / exp_name),
        {
            **vars(args),
            "eval_split": eval_split,
            "lab_standardization": bool(numeric_standardizers),
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
        },
    )

    # metrics_history.json carries validation only, so without this the test
    # numbers that go in the paper live nowhere on disk -- only in stdout and
    # W&B, and are recoverable afterwards only by re-scoring predictions.
    if test_scores is not None:
        test_path = output_dir / exp_name / "test_metrics.json"
        with open(test_path, "w") as handle:
            json.dump({"eval_split": eval_split, **test_scores}, handle, indent=2)

    output_csv = output_dir / exp_name / f"predictions_{args.model}.csv"
    _write_predictions(output_csv, patient_ids, y_true, y_prob)

    wandb_logger.finish()

    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E2E unified embedding on MIMIC-IV with any of six sequence heads."
    )
    parser.add_argument("--ehr-root", type=str, required=True)
    parser.add_argument("--note-root", type=str, default=None)
    parser.add_argument("--cxr-root", type=str, default=None)
    parser.add_argument("--cxr-variant", type=str, default="sunlab", choices=["default", "sunlab"])
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./output/unified_e2e")

    parser.add_argument(
        "--task",
        type=str,
        choices=["labs", "notes_labs", "notes_labs_cxr"],
        default="labs",
        help=(
            "notes_labs: admission-context text (CC/HPI/PMH/MedsOnAdm) + labs. "
            "Recommended for multimodal. "
            "notes_labs_cxr: notes_labs plus in-window chest X-rays; requires "
            "--note-root and --cxr-root."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "rnn", "transformer", "bottleneck_transformer",
                 "ehrmamba", "jambaehr"],
        default="rnn",
    )

    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate, same for every model.",
    )
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=1e-8,
        help="Adam epsilon, same for every model (torch default).",
    )
    parser.add_argument(
        "--allow-leaky-split",
        action="store_true",
        default=False,
        help=(
            "Permit the by-sample split fallback when the patient split is "
            "empty. The same patient can then land in train and test. Smoke "
            "tests only — the metrics are not usable."
        ),
    )
    parser.add_argument(
        "--no-lab-standardization",
        action="store_true",
        default=False,
        help="Disable train-split lab z-scoring (raw-lab ablation).",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Enable automatic mixed precision training to reduce GPU memory usage.",
    )
    parser.add_argument(
        "--amp-dtype",
        "--amp_dtype",
        dest="amp_dtype",
        type=str,
        default=None,
        choices=["bf16", "fp16"],
        help=(
            "AMP dtype. Requires --use-amp; passing this alone is an error "
            "rather than a silently fp32 run. Defaults to bf16 with --use-amp."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--dev",
        nargs="?",
        type=int,
        const=1000,
        default=0,
        help=(
            "Dev mode: limit dataset to N patients for fast iteration. "
            "--dev (no value) defaults to 1000 patients. "
            "--dev 5000 limits to 5000. Omit for full dataset."
        ),
    )
    parser.add_argument(
        "--observation-window-hours",
        type=int,
        default=None,
        help=(
            "If set, collect labs/CXR/radiology only this many hours from each "
            "admission. Default: full stay (through discharge)."
        ),
    )
    parser.add_argument(
        "--max-frozen-text-cache",
        type=int,
        default=1_000_000,
        help=(
            "Max unique frozen [CLS] vectors on CPU. Default 1e6 (~3 GB "
            "fp32, ~8 GB with Python overhead). 0 means no cap. The cap is "
            "a RAM fuse, not what makes the cache fast: speedup needs "
            "cap >= unique notes. 200k is too small for full MIMIC."
        ),
    )
    parser.add_argument(
        "--text-grad-checkpoint-rows",
        type=int,
        default=0,
        help=(
            "Trainable text encoder only: enable gradient checkpointing and "
            "run note rows through BERT in chunks of this size. Bounds "
            "activation memory; the math is unchanged. 0 disables."
        ),
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        default=False,
        help=(
            "Freeze pretrained BERT text encoder weights and train only the "
            "downstream backbone (RNN/Transformer head + projection layer). "
        ),
    )
    parser.add_argument("--mlp-layers", type=int, default=2)
    parser.add_argument(
        "--mlp-activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "sigmoid", "leaky_relu", "elu"],
    )

    parser.add_argument("--rnn-type", type=str, default="GRU")
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)

    parser.add_argument("--bottlenecks-n", type=int, default=4)
    parser.add_argument("--fusion-startidx", type=int, default=1)

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm, same for every model.",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Log training/eval metrics to Weights & Biases.",
    )
    parser.add_argument("--wandb-project", type=str, default="pyhealth-mortality")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Defaults to '{model}_seed{seed}' if unset.",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        help="Comma-separated wandb tags, e.g. 'labs,rnn'. Defaults to '{task},{model}' if unset.",
    )

    parser.add_argument("--mamba-state-size", type=int, default=16,
                        help="SSM state size for EHRMamba and JambaEHR blocks.")
    parser.add_argument("--mamba-conv-kernel", type=int, default=4,
                        help="Causal conv kernel size for EHRMamba and JambaEHR blocks.")
    parser.add_argument("--jamba-transformer-layers", type=int, default=2,
                        help="Number of Transformer (attention) layers in JambaEHR.")
    parser.add_argument("--jamba-mamba-layers", type=int, default=6,
                        help="Number of Mamba (SSM) layers in JambaEHR.")

    args = parser.parse_args()

    # The Tranche 1 flag list says --amp_dtype "bf16" and never mentions
    # --use-amp, so following it literally used to give a silently fp32 run.
    if args.amp_dtype is not None and not args.use_amp:
        parser.error(
            f"--amp-dtype {args.amp_dtype} was passed without --use-amp, so "
            "mixed precision would be off and the run would be fp32 while the "
            "config claimed otherwise. Pass --use-amp, or drop --amp-dtype."
        )
    if args.amp_dtype is None:
        args.amp_dtype = "bf16"

    return args


if __name__ == "__main__":
    args = parse_args()
    output_csv_path = run(args)
    print(f"Saved predictions to: {output_csv_path}")