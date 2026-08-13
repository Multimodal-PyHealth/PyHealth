"""End-to-end protocol runner for Unified Embedding on MIMIC-IV.

Trains and evaluates a unified-embedding model (MLP / RNN / Transformer /
BottleneckTransformer / EHRMamba / JambaEHR) on a MIMIC-IV mortality task,
then writes per-sample predictions to CSV.

Tasks
-----
--task stagenet (default)
    MortalityPredictionStageNetMIMIC4: ICD codes + 10-dim lab vectors,
    patient-level samples aggregated across all admissions.

--task clinical_notes_icd_labs
    ClinicalNotesICDLabsMIMIC4: discharge/radiology notes + ICD + labs.
    Requires --note-root.  Legacy; ICD codes are discharge-coded (leakage).

--task notes_labs (recommended for multimodal)
    NotesLabsMIMIC4: admission-context note sections + labs, no ICD codes.
    Extracts Chief Complaint, HPI, PMH, Medications on Admission from the
    discharge note — text available at admission time, ~90%+ coverage.
    Requires --note-root.
    Add --freeze-encoder to freeze Bio_ClinicalBERT and train only the
    backbone; cuts BERT VRAM by ~50%, useful on smaller GPUs (≤24 GB).
    Add --icd-codes to include discharge-coded ICD codes (ablation only).

Example
-------
    python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
      --ehr-root /path/to/mimiciv/2.2 \\
      --task stagenet \\
      --model transformer \\
      --heads 4 --num-layers 2 \\
      --dev --device cpu \\
      --epochs 10 --batch-size 32 --lr 1e-3 \\
      --output-dir ./output/unified_e2e

    # EHRMamba on full dataset (no --dev):
    python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
      --ehr-root /data/mimic-iv/2.2 --note-root /data/mimic-iv/note \\
      --task clinical_notes_icd_labs --model ehrmamba \\
      --embedding-dim 128 --num-layers 2 --seed 42

    # JambaEHR:
    python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
      --ehr-root /data/mimic-iv/2.2 --note-root /data/mimic-iv/note \\
      --task clinical_notes_icd_labs --model jambaehr \\
      --embedding-dim 128 --jamba-transformer-layers 2 --jamba-mamba-layers 6
"""

from __future__ import annotations

import argparse
import inspect
import csv
import warnings
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch

from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    sample_balanced,
    sample_oversample,
    sample_weighted,
    split_by_patient,
    split_by_sample,
)
from pyhealth.models import MLP, RNN, Transformer, UnifiedMultimodalEmbeddingModel
from pyhealth.models.embedding import VisionEmbeddingModel
from pyhealth.models.bottleneck_transformer import BottleneckTransformer
from pyhealth.models.ehrmamba import EHRMamba
from pyhealth.models.jamba_ehr import JambaEHR
from pyhealth.processors import fit_lab_standardizer, lab_standardizer_fit_scope
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4
from pyhealth.tasks.multimodal_mimic4 import (
    ClinicalNotesICDLabsMIMIC4,
    ICDLabsMIMIC4,
    LabsOnlyMIMIC4,
    NotesLabsMIMIC4,
    CXRMultimodalMIMIC4,
)
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed, write_run_config


def _build_base_dataset(args: argparse.Namespace) -> MIMIC4Dataset:
    ehr_tables = ["diagnoses_icd", "procedures_icd", "labevents"]
    note_tables = None

    if args.task == "clinical_notes_icd_labs":
        if not args.note_root:
            raise ValueError("--task clinical_notes_icd_labs requires --note-root.")
        note_tables = ["discharge", "radiology"]

    if args.task == "icd_labs":
        ehr_tables = ["diagnoses_icd", "procedures_icd", "labevents"]

    if args.task in ("notes_labs", "notes_only"):
        if not args.note_root:
            raise ValueError(f"--task {args.task} requires --note-root.")
        note_tables = [getattr(args, "note_source", "discharge")]
        # Load ICD tables only when explicitly requested (they are discharge-coded).
        ehr_tables = (
            ["diagnoses_icd", "procedures_icd", "labevents"]
            if args.icd_codes
            else ["labevents"]
        )
        if args.include_vitals:
            if "chartevents" not in ehr_tables:
                ehr_tables.append("chartevents")

    if args.task == "labs_only":
        # Pure EHR baseline: only labevents, no notes, no ICD codes.
        ehr_tables = ["labevents"]
        note_tables = None

    cxr_tables = None
    if args.task in ("cxr_only", "cxr_labs", "cxr_notes_labs"):
        if not args.cxr_root:
            raise ValueError(f"--task {args.task} requires --cxr-root.")
        # ``metadata`` supplies image_path, StudyDate/StudyTime, and ViewPosition.
        cxr_tables = ["metadata"]
        ehr_tables = ["labevents"] if args.task != "cxr_only" else []
        if args.task == "cxr_notes_labs":
            if not args.note_root:
                raise ValueError("--task cxr_notes_labs requires --note-root.")
            note_tables = [getattr(args, "note_source", "discharge")]

    return MIMIC4Dataset(
        ehr_root=args.ehr_root,
        ehr_tables=ehr_tables,
        note_root=args.note_root if note_tables else None,
        note_tables=note_tables,
        cxr_root=args.cxr_root if cxr_tables else None,
        cxr_tables=cxr_tables,
        cxr_variant=args.cxr_variant,
        cache_dir=args.cache_dir,
        dev=args.dev if args.dev else False,
        num_workers=args.num_workers,
    )


def _build_task(args: argparse.Namespace):
    if args.task == "stagenet":
        return MortalityPredictionStageNetMIMIC4()
    if args.task == "icd_labs":
        return ICDLabsMIMIC4(window_hours=args.observation_window_hours)
    if args.task == "clinical_notes_icd_labs":
        return ClinicalNotesICDLabsMIMIC4(window_hours=args.observation_window_hours)
    if args.task in ("notes_labs", "notes_only"):
        # Pass only what this checkout's task class accepts. A flag the class
        # cannot honour must stop the run, not be dropped: a silently ignored
        # --discharge-note-policy would record one protocol in run_config.json
        # and execute another.
        wanted = dict(
            window_hours=args.observation_window_hours,
            include_icd=args.icd_codes,
            include_vitals=args.include_vitals,
            include_labs=(args.task != "notes_only"),
            note_extraction=getattr(args, "note_extraction", "regex"),
            note_source=getattr(args, "note_source", "discharge"),
            discharge_note_policy=getattr(
                args, "discharge_note_policy", "extraction"),
            text_normalize=getattr(args, "text_normalize", "none"),
        )
        accepted = set(
            inspect.signature(NotesLabsMIMIC4.__init__).parameters
        )
        defaults = {
            "include_icd": False,
            "include_vitals": False,
            "include_labs": True,
            "note_extraction": "regex",
            "note_source": "discharge",
            "discharge_note_policy": "extraction",
            "text_normalize": "none",
        }
        unsupported = [
            name
            for name, value in wanted.items()
            if name not in accepted and value != defaults.get(name)
        ]
        if unsupported:
            raise SystemExit(
                f"NotesLabsMIMIC4 in this checkout does not accept "
                f"{', '.join(sorted(unsupported))}. Either drop the flag or use "
                f"a checkout whose task class supports it."
            )
        task_kwargs = {k: v for k, v in wanted.items() if k in accepted}
        if args.task == "notes_only" and "include_labs" not in accepted:
            raise SystemExit(
                "--task notes_only needs a NotesLabsMIMIC4 that accepts "
                "include_labs; this checkout always emits labs."
            )
        task = NotesLabsMIMIC4(**task_kwargs)
        if args.tokenizer_model:
            schema_key = "admission_note_times"
            _, opts = task.input_schema[schema_key]
            task.input_schema[schema_key] = ("tuple_time_text", {**opts, "tokenizer_model": args.tokenizer_model})
            print(f"[tokenizer] Overriding tokenizer_model → {args.tokenizer_model}")
        # Note token budget. The processor default (128) truncates ~96% of
        # extracted discharge notes, so the encoder sees only their first ~20%.
        # input_schema is part of the task cache key, so each budget caches apart.
        _nml = getattr(args, "note_max_length", None)
        if _nml:
            # Fail loudly rather than let HF silently clamp to the encoder's
            # position-embedding limit (BERT/Bio_ClinicalBERT = 512).
            _tokmodel = args.tokenizer_model or task.input_schema[
                "admission_note_times"][1].get("tokenizer_model", "")
            if _nml > 512 and "longformer" not in _tokmodel.lower():
                raise SystemExit(
                    f"--note-max-length {_nml} exceeds the 512 position-embedding "
                    f"limit of {_tokmodel!r}. Use --tokenizer-model "
                    f"yikuan8/Clinical-Longformer for budgets >512."
                )
            schema_key = "admission_note_times"
            _, opts = task.input_schema[schema_key]
            task.input_schema[schema_key] = (
                "tuple_time_text", {**opts, "max_length": _nml}
            )
            print(f"[tokenizer] note max_length → {_nml}")
        return task
    if args.task == "labs_only":
        return LabsOnlyMIMIC4(window_hours=args.observation_window_hours)
    if args.task in ("cxr_only", "cxr_labs", "cxr_notes_labs"):
        return CXRMultimodalMIMIC4(
            window_hours=args.observation_window_hours,
            include_labs=args.task in ("cxr_labs", "cxr_notes_labs"),
            include_notes=args.task == "cxr_notes_labs",
            frontal_only=not args.cxr_all_views,
            image_size=args.cxr_image_size,
            max_images=args.cxr_max_images,
            note_source=args.note_source,
        )
    raise ValueError(f"Unknown task: {args.task}")


def _split_dataset(dataset: Any, seed: int) -> Tuple[Any, Any, Any, str]:
    """Split by patient, falling back to by-sample only if that yields nothing.

    The fallback is leaky: a patient with several admissions can then land in
    both train and test, which inflates the metrics. It only triggers on tiny
    cohorts, but it must not trigger silently, so the mode is returned and
    recorded alongside the run's results.
    """
    train_ds, val_ds, test_ds = split_by_patient(dataset, [0.8, 0.1, 0.1], seed=seed)
    if len(train_ds) == 0 or len(test_ds) == 0:
        warnings.warn(
            "split_by_patient produced an empty split, falling back to "
            "split_by_sample. The same patient may now appear in train and "
            "test, so these metrics are optimistic and not comparable to "
            "patient-split runs.",
            RuntimeWarning,
            stacklevel=2,
        )
        train_ds, val_ds, test_ds = split_by_sample(dataset, [0.8, 0.1, 0.1], seed=seed)
        return train_ds, val_ds, test_ds, "by_sample_fallback_leaky"
    return train_ds, val_ds, test_ds, "by_patient"


def _resolve_finetune_mode(args: argparse.Namespace) -> str:
    """--freeze-encoder is a back-compat alias for --text-finetune-mode frozen."""
    return "frozen" if args.freeze_encoder else args.text_finetune_mode


def _build_model(
    args: argparse.Namespace,
    sample_dataset: Any,
    numeric_standardizers: dict[str, Any] | None = None,
):
    finetune_mode = _resolve_finetune_mode(args)
    field_embeddings = None
    if "cxr" in sample_dataset.input_processors:
        # Reuse the repository's VisionEmbeddingModel in the unified image
        # branch; only the temporal alignment is supplied by unified.py.
        field_embeddings = {
            "cxr": VisionEmbeddingModel(
                dataset=sample_dataset,
                embedding_dim=args.embedding_dim,
                patch_size=16,
                backbone="patch",
                pretrained=False,
            )
        }
    unified = UnifiedMultimodalEmbeddingModel(
        processors=sample_dataset.input_processors,
        embedding_dim=args.embedding_dim,
        field_embeddings=field_embeddings,
        freeze_text_encoder=(finetune_mode == "frozen"),
        normalize_content=not getattr(args, "no_normalize_content", False),
        numeric_standardizers=numeric_standardizers,
        cache_frozen_text=not getattr(args, "no_text_cache", False),
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


def _load_pretrained_weights(model, ckpt_path: str) -> None:
    """Load SSL pretraining weights into a supervised model.

    Delegates to ``BaseModel.load_pretrained_state_dict``, which performs the
    architecture-specific key mapping (downstream models name the unified
    backbone ``_unified_backbone`` / ``_unified_jamba`` / ``_unified_blocks``)
    and REQUIRES full backbone coverage.

    The previous implementation here built ``model.state_dict()``, overwrote
    whichever checkpoint keys happened to map, and called ``strict=False``.
    Because the untouched tensors were already present, PyTorch reported no
    missing keys, so a jamba/mamba checkpoint that matched only ~6 of ~30
    backbone tensors trained on a largely RANDOM backbone while looking
    perfectly healthy. That silently corrupted real Table 2 cells.
    """
    print(f"[pretrain] Loading checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    try:
        stats = model.load_pretrained_state_dict(state)
    except ValueError as exc:
        # Surface this as a run-level failure naming the checkpoint. A partial
        # unified backbone must abort the job rather than train on random
        # weights, and the operator needs to know WHICH checkpoint was bad.
        raise RuntimeError(
            f"Refusing to train on a partial unified backbone from {ckpt_path}: {exc}"
        ) from exc
    print(
        "[pretrain] backbone {}/{} tensors, embedding {} matched; "
        "uninitialised: {}".format(
            stats["backbone_matched"], stats["backbone_target"],
            stats["embedding_matched"], stats["missing_keys"][:6],
        )
    )

def _note_availability_report(train_ds, sample_limit: int = 4000) -> dict:
    """Measure how often a note is actually present, and whether that leaks.

    A missing note is represented by a fixed placeholder embedding, so
    "has a real note" is trivially learnable. Measured on MIMIC-IV, mortality was
    5.67% where a note existed against 1.37% where it did not, a 4.1x gap: note
    AVAILABILITY carries outcome signal with no clinical content behind it. That
    confound has to be visible on every run rather than rediscovered, so it is
    measured on the TRAIN split (never test) and recorded in run_config.json.
    """
    import numpy as np

    total = len(train_ds)
    if total == 0:
        return {}
    # Stride across the split rather than taking a prefix: samples are grouped by
    # patient, so the first N are not representative of note availability.
    step = max(1, total // sample_limit)
    indices = range(0, total, step)
    present, labels = [], []
    for i in indices:
        try:
            sample = train_ds[i]
        except Exception:
            break
        field = sample.get("admission_note_times")
        if field is None:
            return {}
        try:
            mask = field["mask"] if isinstance(field, dict) else field[1]
            mask = torch.as_tensor(mask)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            # content tokens = attention mask less [CLS] and [SEP]
            content = int((mask.sum(dim=1) - 2).clamp(min=0).max())
        except Exception:
            return {}
        present.append(content > 5)
        labels.append(float(sample.get("mortality", 0)))
    if not present or all(present) or not any(present):
        return {"note_present_rate": float(np.mean(present)) if present else None,
                "n_inspected": len(present)}
    present = np.array(present); labels = np.array(labels)
    report = {
        "n_inspected": int(len(present)),
        "note_present_rate": float(present.mean()),
        "mortality_with_note": float(labels[present].mean()),
        "mortality_without_note": float(labels[~present].mean()),
    }
    ratio = (report["mortality_with_note"] /
             max(report["mortality_without_note"], 1e-9))
    report["mortality_ratio_present_vs_absent"] = float(ratio)
    print(
        f"[note-availability] {100*report['note_present_rate']:.1f}% of train "
        f"samples carry a real note; mortality {report['mortality_with_note']:.4f} "
        f"with vs {report['mortality_without_note']:.4f} without ({ratio:.1f}x)."
    )
    if ratio > 1.5 or ratio < 0.67:
        warnings.warn(
            f"Note availability is {ratio:.1f}x associated with the outcome. The "
            "missing-note placeholder is a constant embedding, so this is "
            "learnable signal with no clinical content. Report it, restrict to "
            "complete cases, or model missingness explicitly.",
            RuntimeWarning, stacklevel=2,
        )
    return report


def _compute_pos_weight(train_ds, label_key: str = "mortality") -> float:
    """Count pos/neg in train_ds and return n_neg/n_pos for BCE pos_weight."""
    n_pos = n_neg = 0
    for i in range(len(train_ds)):
        sample = train_ds[i]
        label = sample.get(label_key, 0)
        if hasattr(label, "__iter__"):
            label = next(iter(label))
        if float(label) > 0.5:
            n_pos += 1
        else:
            n_neg += 1
    if n_pos == 0:
        return 1.0
    # Cap at 10: n_neg/n_pos ≈ 37 on MIMIC-IV mortality is too extreme with
    # typical LRs and causes training oscillation. 10 still strongly corrects
    # for imbalance while keeping gradient magnitudes tractable.
    return min(10.0, n_neg / n_pos)


def run(args: argparse.Namespace) -> Path:
    set_seed(args.seed)

    base_dataset = _build_base_dataset(args)
    task = _build_task(args)
    sample_dataset = base_dataset.set_task(task, num_workers=args.num_workers)

    if len(sample_dataset) == 0:
        raise RuntimeError(
            "Task produced zero samples. Check roots/tables or adjust settings."
        )

    split_seed = getattr(args, "split_seed", None)
    split_seed_pinned = split_seed is not None
    if split_seed is None:
        split_seed = args.seed
        # Measured at full scale on labs_only: letting the split follow the seed
        # gives sd(PR-AUC) 0.0236, versus 0.0042 with the split pinned, so test-set
        # composition contributes ~31x the variance of initialisation. Cells
        # compared across different splits are unpaired and far less sensitive.
        warnings.warn(
            "--split-seed not set, so the patient split follows --seed and every "
            "seed draws a different test set. For an ablation, pin --split-seed "
            "across compared cells: it is roughly a 5x sensitivity gain for no "
            "extra compute.",
            RuntimeWarning, stacklevel=2,
        )
    print(f"[split] patient split seed={split_seed} "
          f"({'pinned' if split_seed_pinned else 'follows --seed'}); "
          f"training seed={args.seed}")
    train_ds, val_ds, test_ds, split_mode = _split_dataset(sample_dataset, seed=split_seed)
    if getattr(args, "pretrained_ckpt", None) and split_mode != "by_patient":
        raise ValueError(
            "Pretrained comparisons require a non-empty patient-level split. "
            "Refusing the sample-level fallback because the checkpoint's train "
            "statistics could then include a downstream test patient."
        )

    # Fit before any outcome-dependent resampling and exclusively on the train
    # subset.  ``SampleDataset.subset`` exposes only its selected indices here;
    # LabStandardizer iterates this object, never ``sample_dataset``.
    note_availability = _note_availability_report(train_ds)

    numeric_standardizers: dict[str, Any] = {}
    if "labs" in sample_dataset.input_processors and not getattr(
        args, "no_lab_standardization", False
    ):
        if "labs_mask" not in sample_dataset.input_processors:
            raise RuntimeError("Lab standardisation requires the labs_mask observation field.")
        lab_standardizer = fit_lab_standardizer(
            train_ds,
            value_field="labs",
            fit_scope=lab_standardizer_fit_scope(train_ds, value_field="labs"),
        )
        numeric_standardizers["labs"] = lab_standardizer
        print(
            "[lab-standardization] fitted on train split only: "
            f"counts={lab_standardizer.observed_count.tolist()} "
            f"mean={lab_standardizer.mean.tolist()} std={lab_standardizer.std.tolist()}"
        )
    elif "labs" in sample_dataset.input_processors:
        print("[lab-standardization] disabled; reproducing raw-lab baseline.")

    label_key = list(sample_dataset.output_schema.keys())[0]

    # Resolve effective sampling strategy.
    # --balanced-sampling / --balanced-ratio are legacy aliases for undersample.
    strategy = args.sampling_strategy
    if args.balanced_sampling and strategy == "none":
        strategy = "undersample"

    if strategy == "undersample":
        ratio = args.balanced_ratio
        print(f"[sampling] Undersampling negatives → pos:neg 1:{ratio}")
        train_ds = sample_balanced(train_ds, ratio=ratio, seed=args.seed, label_key=label_key)
        print(f"[sampling] Training size after undersample: {len(train_ds)}")

    elif strategy == "oversample":
        ratio = args.balanced_ratio
        print(f"[sampling] Oversampling positives → pos:neg 1:{ratio}")
        train_ds = sample_oversample(train_ds, ratio=ratio, seed=args.seed, label_key=label_key)
        print(f"[sampling] Training size after oversample: {len(train_ds)}")

    elif strategy == "weighted":
        print("[sampling] Weighted resampling (class-proportional, with replacement, no external sampler)")
        train_ds = sample_weighted(train_ds, seed=args.seed, label_key=label_key)
        print(f"[sampling] Training size after weighted resample: {len(train_ds)}")

    model = _build_model(args, sample_dataset, numeric_standardizers)

    # Load pretrained SSL weights if requested.
    if getattr(args, "pretrained_ckpt", None):
        _load_pretrained_weights(model, args.pretrained_ckpt)
    if args.numeric_input_stats_path:
        model.embedding_model.capture_numeric_encoder_input_stats(
            args.numeric_input_stats_path, field_name="labs"
        )
        # Use the same deterministic train-split batch as the pretraining
        # audit. This records the true projection input while avoiding a
        # misleading difference caused only by independent shuffling.
        audit_batch = next(iter(get_dataloader(
            train_ds, batch_size=args.batch_size, shuffle=False,
        )))
        with torch.no_grad():
            model(**audit_batch)
        if not Path(args.numeric_input_stats_path).is_file():
            raise RuntimeError("Numeric-input audit hook did not produce its artifact.")

    # Apply class-imbalance correction via BCE pos_weight.
    # pos_weight = n_neg / n_pos so the rare positive class gets proportionally
    # higher gradient signal, preventing all-negative collapse (F1=0).
    if args.pos_weight is not None:
        pw_value = args.pos_weight
    else:
        print(f"[pos_weight] Computing class balance from {len(train_ds)} training samples...")
        pw_value = _compute_pos_weight(train_ds, label_key=label_key)
    print(f"[pos_weight] Using pos_weight={pw_value:.2f} for binary BCE loss.")
    model._pos_weight = torch.tensor([pw_value], dtype=torch.float32)

    # DataLoader worker options arrive with the performance PR. Pass only what
    # this checkout's get_dataloader accepts, and stop if the caller asked for
    # one it cannot honour, so a requested option is never silently ignored.
    _wanted_loader = {
        "num_workers": args.loader_num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": (
            args.persistent_workers and args.loader_num_workers > 0
        ),
        "prefetch_factor": (
            args.prefetch_factor if args.loader_num_workers > 0 else None
        ),
    }
    _loader_accepts = set(inspect.signature(get_dataloader).parameters)
    _loader_defaults = {
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
    }
    _unsupported_loader = [
        name
        for name, value in _wanted_loader.items()
        if name not in _loader_accepts and value != _loader_defaults[name]
    ]
    if _unsupported_loader:
        raise SystemExit(
            f"get_dataloader in this checkout does not accept "
            f"{', '.join(sorted(_unsupported_loader))}. Drop the flag, or use a "
            f"checkout that includes the DataLoader worker options."
        )
    loader_kwargs = {
        k: v for k, v in _wanted_loader.items() if k in _loader_accepts
    }
    train_loader = get_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = (
        get_dataloader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        if len(val_ds) > 0
        else None
    )
    test_loader = (
        get_dataloader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        if len(test_ds) > 0
        else None
    )

    # Which split the reported predictions come from. Falling back to val, or
    # worse to train, silently reports held-in performance as if it were test,
    # so resolve it once here and record it alongside the results.
    if test_loader is not None:
        inference_loader, eval_split = test_loader, "test"
    elif val_loader is not None:
        inference_loader, eval_split = val_loader, "val"
        warnings.warn(
            "No test split available; reporting predictions from the VALIDATION "
            "split. These are not test metrics.",
            RuntimeWarning, stacklevel=2,
        )
    else:
        inference_loader, eval_split = train_loader, "train"
        warnings.warn(
            "No test or validation split available; reporting predictions from "
            "the TRAINING split. These metrics are held-in and meaningless as a "
            "generalisation estimate.",
            RuntimeWarning, stacklevel=2,
        )

    # The task MUST be in the name. Without it, two arms of the same comparison
    # at the same seed, for example --task labs_only and --task notes_labs,
    # resolve to one directory and the second run overwrites the first run's
    # metrics_history.json, run_config.json and predictions CSV. The loss is
    # silent: the surviving directory looks like a complete run.
    exp_name = f"{args.task}_{args.model}_seed{args.seed}"
    output_dir = Path(args.output_dir)

    trainer = Trainer(
        model=model,
        # f1_opt is a threshold-optimised F1 that this checkout's
        # binary_metrics_fn does not implement, and requesting it aborts
        # validation at the end of epoch 1. Model selection uses pr_auc, which
        # is a rank metric and needs no threshold.
        metrics=["pr_auc", "roc_auc", "f1", "accuracy"],
        device=args.device,
        enable_logging=True,
        output_path=str(output_dir),
        exp_name=exp_name,
    )

    # BottleneckTransformer is more fragile on full MIMIC-IV with no warmup.
    # Use safer defaults unless explicitly overridden from CLI.
    effective_lr = args.lr
    effective_max_grad_norm = args.max_grad_norm
    optimizer_params = {}

    if args.model == "bottleneck_transformer":
        if effective_lr is None:
            effective_lr = 1e-4
        if effective_max_grad_norm is None:
            effective_max_grad_norm = 0.5
        optimizer_params["eps"] = args.adam_eps if args.adam_eps is not None else 1e-6
    else:
        # All non-BT models: 1e-4 (was 1e-3). With pos_weight correction,
        # effective gradient magnitude for positives is ~10x higher, so a
        # smaller LR is needed to avoid training oscillation.
        if effective_lr is None:
            effective_lr = 1e-4
        # Universal grad clipping: prevents runaway updates from the weighted
        # positive-class loss (pos_weight ≈ 10 scales positive gradients 10x).
        if effective_max_grad_norm is None:
            effective_max_grad_norm = 1.0
        if args.adam_eps is not None:
            optimizer_params["eps"] = args.adam_eps

    optimizer_params["lr"] = effective_lr

    # Record the resolved conditions, not the raw flags: text_finetune_mode and
    # the learning rate are both derived, so the CLI alone does not identify the
    # run. Without this the artifacts cannot say whether the encoder was frozen.
    write_run_config(
        str(output_dir / exp_name),
        {
            **vars(args),
            "resolved_text_finetune_mode": _resolve_finetune_mode(args),
            "resolved_lr": effective_lr,
            "resolved_max_grad_norm": effective_max_grad_norm,
            "split_mode": split_mode,
            "eval_split": eval_split,
            "split_seed_pinned": split_seed_pinned,
            "note_availability": note_availability,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
        },
    )

    if args.epochs > 0 and len(train_ds) > 0:
        # PR-AUC/ROC-AUC are undefined for a single-class validation fold.
        # Full MIMIC patient splits contain both labels; tiny real-data demo
        # runs do not, so use finite validation loss for checkpoint selection.
        val_labels = {
            int(float(val_ds[i][label_key])) for i in range(len(val_ds))
        }
        monitor = "pr_auc" if len(val_labels) == 2 else "loss"
        monitor_criterion = "max" if monitor == "pr_auc" else "min"
        if monitor != "pr_auc":
            print("[monitor] Validation fold has one class; selecting checkpoints by loss.")
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs,
            optimizer_params=optimizer_params,
            weight_decay=args.weight_decay,
            max_grad_norm=effective_max_grad_norm,
            monitor=monitor,
            monitor_criterion=monitor_criterion,
            load_best_model_at_last=True,
            patience=args.patience,
            encoder_lr=args.encoder_lr,
            # Mixed precision is a property of the training loop, not of the
            # Trainer object.
            use_amp=args.use_amp,
            amp_dtype=args.amp_dtype,
        )

    y_true, y_prob, _, patient_ids = trainer.inference(
        inference_loader, return_patient_ids=True
    )

    output_csv = output_dir / exp_name / f"predictions_{args.model}.csv"
    _write_predictions(output_csv, patient_ids, y_true, y_prob)
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E2E unified embedding on MIMIC-IV with any of six sequence heads."
    )
    parser.add_argument("--ehr-root", type=str, required=True)
    parser.add_argument("--note-root", type=str, default=None)
    parser.add_argument("--cxr-root", type=str, default=None)
    parser.add_argument(
        "--cxr-variant",
        choices=["default", "sunlab"],
        default="default",
        help=(
            "Layout of the CXR root. 'default' reads "
            "mimic-cxr-2.0.0-metadata-pyhealth.csv and needs a "
            "studytime_normalized column. 'sunlab' reads the resized set, "
            "normalises StudyTime itself and derives image paths from "
            "dicom_id. Choosing the wrong one fails at dataset build with "
            "KeyError: 'studytime_normalized'."
        ),
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./output/unified_e2e")
    parser.add_argument(
        "--numeric-input-stats-path", type=str, default=None,
        help="Optional JSON artifact: first lab tensor entering the numeric encoder.",
    )
    parser.add_argument(
        "--pretrained-ckpt",
        type=str,
        default=None,
        help=(
            "Path to a SSL pretraining checkpoint (e.g., from "
            "scripts/pretrain_ssl.py).  Loads embedding_model and backbone "
            "weights into the downstream model."
        ),
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["stagenet", "icd_labs", "clinical_notes_icd_labs", "notes_labs", "notes_only", "labs_only", "cxr_only", "cxr_labs", "cxr_notes_labs"],
        default="stagenet",
        help=(
            "notes_labs: admission-context text (CC/HPI/PMH/MedsOnAdm) + labs. "
            "No ICD codes (discharge-coded = leakage). Recommended for multimodal."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "rnn", "transformer", "bottleneck_transformer",
                 "ehrmamba", "jambaehr"],
        default="rnn",
    )

    # Shared embedding / training
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "Learning rate. Default is 1e-4 for all models. "
            "(Previously 1e-3 for mlp/rnn/transformer/ehrmamba/jambaehr — "
            "reduced after pos_weight correction caused oscillation at 1e-3.)"
        ),
    )
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=None,
        help=(
            "Adam epsilon. Default is model-specific: 1e-8 for non-BT models, "
            "1e-6 for bottleneck_transformer."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--loader-num-workers",
        type=int,
        default=0,
        help="Worker processes for runtime batch loading/collation.",
    )
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--persistent-workers", action="store_true", default=False)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--use-amp", dest="use_amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=False)
    parser.add_argument(
        "--amp-dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="AMP compute dtype; bf16 is recommended on A100/H100.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Patient split seed. Defaults to --seed for backward compatibility.",
    )
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
        "--pos-weight",
        type=float,
        default=None,
        help=(
            "BCE pos_weight for the positive class (float). "
            "Default: auto-computed as n_neg/n_pos from training split. "
            "Set to 1.0 to disable class-imbalance correction."
        ),
    )

    # Task-specific
    parser.add_argument("--observation-window-hours", type=int, default=24)
    parser.add_argument(
        "--cxr-all-views", action="store_true", default=False,
        help="Keep non-frontal CXR views; default restricts to PA/AP frontal images.",
    )
    parser.add_argument("--cxr-image-size", type=int, default=224)
    parser.add_argument("--cxr-max-images", type=int, default=4)
    parser.add_argument(
        "--icd-codes",
        action="store_true",
        default=False,
        help=(
            "Include discharge-coded ICD codes in notes_labs task. "
            "Default: off (ICD codes are coded at discharge and constitute "
            "data leakage for in-hospital mortality prediction). "
            "Enable only for ablation / legacy comparison experiments."
        ),
    )
    parser.add_argument(
        "--discharge-note-policy",
        choices=["extraction", "charttime"],
        default="extraction",
        help="How the observation window applies to discharge summaries. "
             "extraction (default, Lee et al. 2023): retrieve across the "
             "admission and let admission-context section extraction be the "
             "temporal control. charttime: strictly non-anticipative, but "
             "drops the summary for ~90%% of admissions.",
    )
    parser.add_argument(
        "--no-text-cache",
        action="store_true",
        help="Disable the frozen-text [CLS] cache. Diagnostic: isolates the cache "
             "as a cause when a run fails to optimise.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        default=False,
        help=(
            "Freeze pretrained BERT text encoder weights and train only the "
            "downstream backbone (MLP/RNN/Transformer head + projection layer). "
            "Reduces VRAM by ~50%% for the text branch; useful when GPU memory "
            "is limited or for faster iteration on backbone architectures. "
            "Back-compat alias for --text-finetune-mode frozen."
        ),
    )
    parser.add_argument(
        "--text-finetune-mode",
        type=str,
        default="full",
        help=(
            "Text encoder fine-tuning regime: 'full' (train all encoder params), "
            "'frozen' (train only the head/projection), 'topk:N' (unfreeze the top "
            "N transformer layers, embeddings stay frozen), or 'lora:r' (rank-r "
            "LoRA adapters on attention, base frozen; needs peft). "
            "Overridden by --freeze-encoder when that flag is set."
        ),
    )
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=None,
        help=(
            "Discriminative learning rate for the pretrained text encoder. When "
            "set, the encoder trains at this LR while the projection + downstream "
            "head keep the base --lr. Recommended ~2e-5 for full/topk fine-tuning "
            "to avoid destabilizing the encoder. Default None = single global LR."
        ),
    )
    parser.add_argument(
        "--note-source",
        type=str,
        default="discharge",
        choices=["discharge", "radiology"],
        help=(
            "Which MIMIC note table to use for notes_labs. 'discharge' (default) "
            "uses admission-context discharge sections; 'radiology' uses radiology "
            "report Impression/Findings, concatenated per admission (mirrors the "
            "multimodal-EHR benchmark). Pair with a RadBERT --tokenizer-model."
        ),
    )
    parser.add_argument(
        "--include-vitals",
        action="store_true",
        default=False,
        help=(
            "Include ICU vital signs (HeartRate, SysBP, DiasBP, MeanBP, "
            "RespRate, SpO2, Temperature) from chartevents as an additional "
            "modality alongside labs and notes. Adds chartevents to EHR tables."
        ),
    )
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        default=False,
        help=(
            "Undersample the majority (negative) class in training to improve "
            "PR-AUC on imbalanced datasets. Uses sample_balanced() to create a "
            "1:--balanced-ratio pos:neg training set."
        ),
    )
    parser.add_argument(
        "--balanced-ratio",
        type=float,
        default=1.0,
        help=(
            "Negatives per positive in the balanced training set. "
            "Default: 1.0 (equal pos/neg). Used with undersample and oversample strategies."
        ),
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default=None,
        help=(
            "Override the tokenizer/encoder for notes. Must be a HuggingFace model ID. "
            "Default: None (uses task class default, emilyalsentzer/Bio_ClinicalBERT). "
            "Changes the task cache UUID so different tokenizers use isolated caches. "
            "Examples: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext, "
            "medicalai/ClinicalBERT, yikuan8/Clinical-Longformer."
        ),
    )
    parser.add_argument(
        "--no-normalize-content",
        action="store_true",
        default=False,
        help=(
            "Disable content normalization in the unified embedding, reproducing "
            "pre-repair behaviour where text events were ~94%% patient-independent "
            "constant while raw labs dominated. Use only to reproduce old runs."
        ),
    )
    parser.add_argument(
        "--no-lab-standardization",
        action="store_true",
        default=False,
        help=(
            "Disable train-split-only per-analyte lab z-scoring. Use only for "
            "the raw-lab ablation; default standardises immediately before the "
            "numeric projection and checkpoints the fitted statistics."
        ),
    )
    parser.add_argument(
        "--note-max-length",
        type=int,
        default=None,
        help=(
            "Note tokenization budget (TupleTimeTextProcessor.max_length; default "
            "128, which truncates ~96%% of extracted discharge notes). BERT-family "
            "encoders cap at 512 position embeddings; for longer budgets pass "
            "--tokenizer-model yikuan8/Clinical-Longformer (4096) and expect to "
            "need --batch-size 1."
        ),
    )
    parser.add_argument(
        "--text-normalize",
        type=str,
        default="none",
        choices=["none", "punct", "stopwords", "both"],
        help=(
            "Cut tokens per note before tokenization. 'punct': strip punctuation "
            "(decimal points and thousands separators inside numbers are kept, so "
            "lab values survive). 'stopwords': drop common English stopwords. "
            "'both': both. Changes task_name so each setting gets its own cache."
        ),
    )
    parser.add_argument(
        "--note-extraction",
        type=str,
        default="regex",
        choices=[
            "regex", "regex_priority", "compact", "tfidf",
            "section_hpi", "section_cc", "section_pmh", "section_meds",
            "section_social", "section_family", "section_allergies", "section_ros",
            "lab_retrieval",
        ],
        help=(
            "Note text extraction strategy. "
            "'regex' (default): section headers, document order. "
            "'regex_priority': HPI-first order (fixes 48%% truncation rate). "
            "'compact': HPI + CC only — always fits in 512 tokens. "
            "'tfidf': TF overlap paragraph retrieval, no headers required. "
            "'section_<name>': single-section ablation (hpi/cc/pmh/meds/"
            "social/family/allergies/ros). "
            "Non-regex values isolate the task cache UUID."
        ),
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "undersample", "oversample", "weighted"],
        help=(
            "Training-set class balance strategy. "
            "'none': no resampling (default). "
            "'undersample': drop majority-class (neg) samples via sample_balanced(). "
            "'oversample': duplicate minority-class (pos) samples via sample_oversample(). "
            "'weighted': WeightedRandomSampler for batch-level balance without dataset modification. "
            "--balanced-sampling is a legacy alias for 'undersample'."
        ),
    )

    # RNN-specific
    parser.add_argument("--rnn-type", type=str, default="GRU")
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")

    # Transformer / BottleneckTransformer shared.
    # Standardized compute: 64-dim, 1 layer, 2 heads (head_dim 32) across all archs.
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)

    # BottleneckTransformer-specific
    parser.add_argument("--bottlenecks-n", type=int, default=4)
    parser.add_argument("--fusion-startidx", type=int, default=1)

    # Training stability
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help=(
            "Gradient clipping max norm. Default is model-specific: None for "
            "non-BT models, 0.5 for bottleneck_transformer."
        ),
    )

    # Mamba / JambaEHR-specific
    parser.add_argument("--mamba-state-size", type=int, default=16,
                        help="SSM state size for EHRMamba and JambaEHR blocks.")
    parser.add_argument("--mamba-conv-kernel", type=int, default=4,
                        help="Causal conv kernel size for EHRMamba and JambaEHR blocks.")
    parser.add_argument("--jamba-transformer-layers", type=int, default=2,
                        help="Number of Transformer (attention) layers in JambaEHR. "
                             "Standard: 1 (a single Jamba block = 1 attn + 1 mamba).")
    parser.add_argument("--jamba-mamba-layers", type=int, default=6,
                        help="Number of Mamba (SSM) layers in JambaEHR. "
                             "Standard: 1 (a single Jamba block = 1 attn + 1 mamba).")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_csv_path = run(args)
    print(f"Saved predictions to: {output_csv_path}")
