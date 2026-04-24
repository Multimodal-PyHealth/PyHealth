"""End-to-end protocol runner for Unified Embedding on MIMIC-IV + CXR.

Trains and evaluates a unified-embedding model (MLP / RNN / Transformer /
BottleneckTransformer / EHRMamba / JambaEHR) on MIMIC-IV mortality using
all four modalities: clinical notes, ICD codes, lab values, and chest X-rays.

This script is the CXR-extended version of unified_embedding_e2e_mimic4.py.
It adds --cxr-root, --cxr-variant, and the clinical_notes_icd_labs_cxr task
on top of the existing non-CXR tasks.

VRAM note: CXR embeddings push peak VRAM to ~40 GB. Use a small batch size
(2-4) and target the largest available GPU (A6000 / H100).

Example
-------
    # Quick sanity check (--dev + 1 epoch + small batch):
    python examples/mortality_prediction/unified_embedding_e2e_mimic4_cxr.py \\
      --ehr-root /shared/rsaas/physionet.org/files/mimiciv/2.2 \\
      --note-root /shared/rsaas/physionet.org/files/mimic-note \\
      --cxr-root /shared/rsaas/physionet.org/files/MIMIC-CXR \\
      --task clinical_notes_icd_labs_cxr \\
      --model mlp --quick-test

    # Smoke test (single forward + inference, no training):
    python examples/mortality_prediction/unified_embedding_e2e_mimic4_cxr.py \\
      --ehr-root /shared/rsaas/physionet.org/files/mimiciv/2.2 \\
      --note-root /shared/rsaas/physionet.org/files/mimic-note \\
      --cxr-root /shared/rsaas/physionet.org/files/MIMIC-CXR \\
      --task clinical_notes_icd_labs_cxr \\
      --model mlp --smoke-forward --dev

    # Full training with Transformer + CXR:
    python examples/mortality_prediction/unified_embedding_e2e_mimic4_cxr.py \\
      --ehr-root /shared/rsaas/physionet.org/files/mimiciv/2.2 \\
      --note-root /shared/rsaas/physionet.org/files/mimic-note \\
      --cxr-root /shared/rsaas/physionet.org/files/MIMIC-CXR \\
      --task clinical_notes_icd_labs_cxr \\
      --model transformer --heads 4 --num-layers 2 \\
      --epochs 10 --batch-size 4 --device cuda:0

    # JambaEHR + CXR:
    python examples/mortality_prediction/unified_embedding_e2e_mimic4_cxr.py \\
      --ehr-root /shared/rsaas/physionet.org/files/mimiciv/2.2 \\
      --note-root /shared/rsaas/physionet.org/files/mimic-note \\
      --cxr-root /shared/rsaas/physionet.org/files/MIMIC-CXR \\
      --task clinical_notes_icd_labs_cxr \\
      --model jambaehr --embedding-dim 128 \\
      --jamba-transformer-layers 2 --jamba-mamba-layers 6
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch

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
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4
from pyhealth.tasks.multimodal_mimic4 import (
    ClinicalNotesICDLabsMIMIC4,
    ClinicalNotesICDLabsCXRMIMIC4,
)
from pyhealth.trainer import Trainer


# ---------------------------------------------------------------------------
# 1. Dataset construction
# ---------------------------------------------------------------------------

def _build_base_dataset(args: argparse.Namespace) -> MIMIC4Dataset:
    """Load the MIMIC-IV base dataset with the appropriate tables.

    For non-CXR tasks this behaves identically to Rian's runner.
    For the CXR task it additionally passes cxr_root / cxr_variant /
    cxr_tables so the dataset loads chest X-ray metadata.
    """
    ehr_tables = ["diagnoses_icd", "procedures_icd", "labevents"]

    # Notes are needed for any task that includes clinical text.
    note_tables = None
    if args.task in ("clinical_notes_icd_labs", "clinical_notes_icd_labs_cxr"):
        if not args.note_root:
            raise ValueError(
                f"--task {args.task} requires --note-root."
            )
        note_tables = ["discharge", "radiology"]

    # CXR tables are only loaded for the CXR task.
    cxr_kwargs = {}
    if args.task == "clinical_notes_icd_labs_cxr":
        if not args.cxr_root:
            raise ValueError(
                "--task clinical_notes_icd_labs_cxr requires --cxr-root."
            )
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
        dev=args.dev,
        num_workers=args.num_workers,
        **cxr_kwargs,
    )


# ---------------------------------------------------------------------------
# 2. Task construction
# ---------------------------------------------------------------------------

def _build_task(args: argparse.Namespace):
    """Return the task object matching --task."""
    if args.task == "stagenet":
        return MortalityPredictionStageNetMIMIC4()
    if args.task == "clinical_notes_icd_labs":
        return ClinicalNotesICDLabsMIMIC4(window_hours=args.observation_window_hours)
    if args.task == "clinical_notes_icd_labs_cxr":
        return ClinicalNotesICDLabsCXRMIMIC4(window_hours=args.observation_window_hours)
    raise ValueError(f"Unknown task: {args.task}")


# ---------------------------------------------------------------------------
# 3. Train / val / test split
# ---------------------------------------------------------------------------

def _split_dataset(dataset: Any, seed: int) -> Tuple[Any, Any, Any]:
    """Split by patient first; fall back to split by sample if empty."""
    train_ds, val_ds, test_ds = split_by_patient(
        dataset, [0.8, 0.1, 0.1], seed=seed
    )
    if len(train_ds) == 0 or len(test_ds) == 0:
        train_ds, val_ds, test_ds = split_by_sample(
            dataset, [0.8, 0.1, 0.1], seed=seed
        )
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# 4. Model construction 
# ---------------------------------------------------------------------------

def _build_model(args: argparse.Namespace, sample_dataset: Any):
    """Instantiate UnifiedMultimodalEmbeddingModel + the chosen backbone."""
    unified = UnifiedMultimodalEmbeddingModel(
        processors=sample_dataset.input_processors,
        embedding_dim=args.embedding_dim,
    )

    if args.model == "mlp":
        return MLP(
            dataset=sample_dataset,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
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


# ---------------------------------------------------------------------------
# 5. Prediction CSV writer
# ---------------------------------------------------------------------------

def _write_predictions(
    output_csv: Path,
    patient_ids: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    """Write per-sample predictions to a CSV file."""
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


# ---------------------------------------------------------------------------
# 6. Main training + evaluation loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> Path:
    """Execute the full pipeline: load → task → split → train → evaluate."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    total_start = time.perf_counter()

    # Resolve CUDA device index for VRAM tracking.
    cuda_device_index = None
    if args.device and args.device.startswith("cuda"):
        device_index = torch.device(args.device).index
        cuda_device_index = 0 if device_index is None else device_index

    base_dataset = _build_base_dataset(args)
    task = _build_task(args)
    sample_dataset = base_dataset.set_task(task, num_workers=args.num_workers)

    if len(sample_dataset) == 0:
        raise RuntimeError(
            "Task produced zero samples. Check roots/tables or adjust settings."
        )

    print(f"Task sample count: {len(sample_dataset)}")

    # Print processor schemas so mismatches are caught early.
    print("Input processor schemas:")
    for key in sample_dataset.input_schema.keys():
        processor = sample_dataset.input_processors.get(key)
        if processor is None:
            print(f"  - {key}: <no processor>")
            continue
        print(f"  - {key}: {type(processor).__name__}, "
              f"schema={processor.schema()}")

    train_ds, val_ds, test_ds = _split_dataset(sample_dataset, seed=args.seed)
    model = _build_model(args, sample_dataset)

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

    print(f"Split sizes: train={len(train_ds)}, val={len(val_ds)}, "
          f"test={len(test_ds)}")

    # Debug batch diagnostics: print types, shapes, and schema mappings
    # for the first training batch so processor issues surface immediately.
    debug_batch = next(iter(train_loader))
    print("Batch field diagnostics (train batch 0):")
    for key in sample_dataset.input_schema.keys():
        processor = sample_dataset.input_processors.get(key)
        feature = debug_batch.get(key)
        schema = processor.schema() if processor is not None else ()
        print(f"  - {key}: type={type(feature).__name__}, schema={schema}")

        if isinstance(feature, tuple):
            for i, elem in enumerate(feature):
                shape = getattr(elem, "shape", None)
                print(f"      tuple[{i}] type={type(elem).__name__} "
                      f"shape={shape}")

        if processor is not None and isinstance(feature, tuple):
            for field_name in ("value", "time", "mask"):
                if field_name in schema:
                    idx = schema.index(field_name)
                    if idx < len(feature):
                        selected = feature[idx]
                        shape = getattr(selected, "shape", None)
                        print(f"      schema['{field_name}'] -> tuple[{idx}] "
                              f"type={type(selected).__name__} shape={shape}")

    exp_name = f"{args.model}_seed{args.seed}"
    output_dir = Path(args.output_dir)

    trainer = Trainer(
        model=model,
        metrics=["pr_auc", "roc_auc", "f1", "accuracy"],
        device=args.device,
        enable_logging=True,
        output_path=str(output_dir),
        exp_name=exp_name,
    )

    # Model-specific optimizer defaults (from Rian's runner).
    effective_lr = args.lr
    effective_max_grad_norm = args.max_grad_norm
    optimizer_params = {}

    if args.model == "bottleneck_transformer":
        if effective_lr is None:
            effective_lr = 1e-4
        if effective_max_grad_norm is None:
            effective_max_grad_norm = 0.5
        optimizer_params["eps"] = (
            args.adam_eps if args.adam_eps is not None else 1e-6
        )
    else:
        if effective_lr is None:
            effective_lr = 1e-3
        if args.adam_eps is not None:
            optimizer_params["eps"] = args.adam_eps

    optimizer_params["lr"] = effective_lr

    # --smoke-forward: skip training, only run inference to verify the
    # pipeline works end-to-end without waiting for a full epoch.
    peak_train_vram_mb = None
    train_runtime_sec = None

    if not args.smoke_forward and args.epochs > 0 and len(train_ds) > 0:
        if cuda_device_index is not None:
            torch.cuda.reset_peak_memory_stats(cuda_device_index)
            torch.cuda.synchronize(cuda_device_index)

        train_start = time.perf_counter()
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs,
            optimizer_params=optimizer_params,
            weight_decay=args.weight_decay,
            max_grad_norm=effective_max_grad_norm,
            monitor="pr_auc",
            load_best_model_at_last=True,
        )

        if cuda_device_index is not None:
            torch.cuda.synchronize(cuda_device_index)
            peak_train_bytes = torch.cuda.max_memory_allocated(cuda_device_index)
            peak_train_vram_mb = peak_train_bytes / (1024**2)

        train_runtime_sec = time.perf_counter() - train_start

    inference_loader = test_loader or val_loader or train_loader
    y_true, y_prob, _, patient_ids = trainer.inference(
        inference_loader, return_patient_ids=True
    )

    if cuda_device_index is not None:
        torch.cuda.synchronize(cuda_device_index)

    total_runtime_sec = time.perf_counter() - total_start

    # Benchmark summary — matches the Mamba CXR script's output format.
    print("Benchmark summary:")
    print(f"  total_runtime_sec:    {total_runtime_sec:.2f}")
    if train_runtime_sec is None:
        print("  training_runtime_sec: N/A (training skipped)")
        print("  peak_train_vram_mb:   N/A (training skipped)")
    else:
        print(f"  training_runtime_sec: {train_runtime_sec:.2f}")
        if peak_train_vram_mb is None:
            print("  peak_train_vram_mb:   N/A (non-CUDA device)")
        else:
            print(f"  peak_train_vram_mb:   {peak_train_vram_mb:.2f}")

    output_csv = output_dir / exp_name / f"predictions_{args.model}.csv"
    _write_predictions(output_csv, patient_ids, y_true, y_prob)
    return output_csv


# ---------------------------------------------------------------------------
# 7. CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E2E unified embedding on MIMIC-IV (+ CXR) with "
        "any of six backbone models."
    )

    # --- Data paths ---
    parser.add_argument("--ehr-root", type=str, required=True)
    parser.add_argument("--note-root", type=str, default=None)
    parser.add_argument(
        "--cxr-root", type=str, default=None,
        help="Root directory for MIMIC-CXR images. Required for "
        "clinical_notes_icd_labs_cxr task.",
    )
    parser.add_argument(
        "--cxr-variant", type=str, default="sunlab",
        choices=["default", "sunlab"],
        help="CXR directory layout variant.",
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./output/unified_e2e_cxr")

    # --- Task and model selection ---
    parser.add_argument(
        "--task", type=str, default="clinical_notes_icd_labs_cxr",
        choices=[
            "stagenet",
            "clinical_notes_icd_labs",
            "clinical_notes_icd_labs_cxr",
        ],
    )
    parser.add_argument(
        "--model", type=str, default="rnn",
        choices=[
            "mlp", "rnn", "transformer", "bottleneck_transformer",
            "ehrmamba", "jambaehr",
        ],
    )

    # --- Shared embedding / training hyperparameters ---
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size. Default is 4 (CXR is VRAM-heavy, ~40 GB peak).",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate. Default: 1e-3 (1e-4 for bottleneck_transformer).",
    )
    parser.add_argument(
        "--adam-eps", type=float, default=None,
        help="Adam epsilon. Default: 1e-8 (1e-6 for bottleneck_transformer).",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument(
        "--quick-test", action="store_true",
        help="Force --dev, 1 epoch, small batch for a quick sanity check.",
    )
    parser.add_argument(
        "--smoke-forward", action="store_true",
        help="Skip training; only run a single forward pass + inference "
        "to verify the pipeline end-to-end.",
    )

    # --- Task-specific ---
    parser.add_argument("--observation-window-hours", type=int, default=24)

    # --- RNN-specific ---
    parser.add_argument("--rnn-type", type=str, default="GRU")
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")

    # --- Transformer / BottleneckTransformer ---
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)

    # --- BottleneckTransformer-specific ---
    parser.add_argument("--bottlenecks-n", type=int, default=4)
    parser.add_argument("--fusion-startidx", type=int, default=1)

    # --- Training stability ---
    parser.add_argument(
        "--max-grad-norm", type=float, default=None,
        help="Gradient clip norm. Default: None (0.5 for bottleneck_transformer).",
    )

    # --- Mamba / JambaEHR-specific ---
    parser.add_argument("--mamba-state-size", type=int, default=16)
    parser.add_argument("--mamba-conv-kernel", type=int, default=4)
    parser.add_argument("--jamba-transformer-layers", type=int, default=2)
    parser.add_argument("--jamba-mamba-layers", type=int, default=6)

    args = parser.parse_args()

    if args.quick_test:
        args.dev = True
        args.epochs = 1
        args.batch_size = min(args.batch_size, 4)

    return args


if __name__ == "__main__":
    args = parse_args()
    output_csv_path = run(args)
    print(f"Saved predictions to: {output_csv_path}")