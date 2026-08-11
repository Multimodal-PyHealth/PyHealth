"""Self-supervised pretraining script for PyHealth multimodal sequences.

Supports MAE, SimMIM, and I-JEPA over the unified embedding model.  Run after
this finishes, use ``scripts/train_unified.py --pretrained-ckpt ...`` to
fine-tune downstream.

Example:
    python scripts/pretrain_ssl.py \
        --ehr-root /data/mimic-iv/2.2 \
        --note-root /data/mimic-iv/note \
        --task notes_labs \
        --method mae \
        --epochs 50 --batch-size 32 \
        --output-dir ./output/pretrain_mae
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

# Make project root importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyhealth.datasets import MIMIC4Dataset, get_dataloader
from pyhealth.models import UnifiedMultimodalEmbeddingModel
from pyhealth.models.pretrain import (
    MultimodalIJEPA,
    MultimodalMaskedAutoencoder,
    MultimodalSimMIM,
    MultimodalVJEPA,
)
from pyhealth.models.pretrain.backbones import ARCH_CHOICES, build_backbone
from pyhealth.tasks.multimodal_mimic4 import (
    ClinicalNotesICDLabsMIMIC4,
    ICDLabsMIMIC4,
    LabsOnlyMIMIC4,
    NotesLabsMIMIC4,
)
from pyhealth.models.pretrain.trainer import PretrainTrainer
from pyhealth.utils import set_seed


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_config(config_path: Path) -> Dict[str, Any]:
    """Load config and recursively merge _inherit chain (child wins)."""
    cfg = _load_yaml(config_path)
    inherit = cfg.pop("_inherit", None)
    if inherit:
        parent = _resolve_config(config_path.parent / inherit)
        cfg = {**parent, **cfg}
    return cfg


def _dict_to_namespace(cfg: Dict[str, Any], defaults: argparse.Namespace) -> argparse.Namespace:
    """Convert a resolved config dict into an argparse Namespace.

    All CLI defaults are copied first so that optional config keys (e.g.
    ``cache_dir``) always have an attribute.  Config values then override
    defaults.
    """
    ns = argparse.Namespace(**vars(defaults))
    for k, v in cfg.items():
        setattr(ns, k, v)
    return ns


def _merge_cli_overrides(ns: argparse.Namespace, cli: argparse.Namespace) -> argparse.Namespace:
    """Explicit CLI-set values override config values."""
    for key in vars(cli):
        val = getattr(cli, key)
        if val is not None and key != "config":
            setattr(ns, key, val)
    return ns


# Fallback defaults applied after YAML config + CLI merging.  argparse defaults
# are intentionally None so that config values are never clobbered by CLI defaults.
_DEFAULTS: Dict[str, Any] = {
    "task": "labs_only",
    "method": "mae",
    # Encoder backbone architecture: transformer | jamba | mamba.
    "arch": "transformer",
    # Standardized compute: 128-dim, 2 layers, 4 heads (matches base.yaml and the
    # downstream e2e backbone). These are only fallbacks; base.yaml provides them.
    "embedding_dim": 128,
    "heads": 4,
    "num_layers": 2,
    "dropout": 0.1,
    # Mamba / Jamba backbone knobs.
    "state_size": 16,
    "conv_kernel": 4,
    "jamba_transformer_layers": 1,
    "jamba_mamba_layers": 1,
    "rope_max_seq_len": 8192,
    "rope_base": 10000.0,
    "rope_scaling": 1.0,
    "decoder_layers": 2,
    "decoder_heads": 2,
    "predictor_layers": 2,
    "predictor_heads": 2,
    "ema_decay": 0.996,
    "ema_end": 1.0,
    "num_target_blocks": 4,
    "target_block_len": 4,
    # V-JEPA: multi-scale spans + cross-modal windows.
    "target_block_scales": [2, 4, 8],
    "require_multimodal_blocks": False,
    "normalize_targets": True,
    # store_true flags: keep False fallbacks here and None argparse defaults so a
    # config value of true is not clobbered by the absent-flag default.
    "use_rope": False,
    "norm_pix_loss": False,
    "use_amp": False,
    "icd_codes": False,
    "include_vitals": False,
    "freeze_encoder": True,
    "mask_ratio": 0.5,
    "mask_strategy": "random",
    "epochs": 10,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 0.05,
    "max_grad_norm": 1.0,
    "scheduler": "cosine",
    "warmup_steps": 1000,
    "save_every_n_epochs": 5,
    "num_workers": 4,
    "seed": 42,
    "grad_accumulation_steps": 1,
    "local_rank": 0,
    "observation_window_hours": 24,
    "note_source": "discharge",
    "note_extraction": "regex",
    "dev": 0,
    "output_dir": "./output/pretrain_ssl",
}


def _apply_defaults(ns: argparse.Namespace) -> argparse.Namespace:
    for key, val in _DEFAULTS.items():
        if getattr(ns, key, None) is None:
            setattr(ns, key, val)
    return ns


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
        # notes_only reuses the same base dataset (tables) as notes_labs — only the
        # task differs (include_labs=False), so the base-dataset cache is shared.
        ehr_tables = (
            ["diagnoses_icd", "procedures_icd", "labevents"]
            if args.icd_codes
            else ["labevents"]
        )
        if args.include_vitals:
            if "chartevents" not in ehr_tables:
                ehr_tables.append("chartevents")

    if args.task == "labs_only":
        ehr_tables = ["labevents"]
        note_tables = None

    return MIMIC4Dataset(
        ehr_root=args.ehr_root,
        ehr_tables=ehr_tables,
        note_root=args.note_root if note_tables else None,
        note_tables=note_tables,
        cache_dir=args.cache_dir,
        dev=args.dev if args.dev else False,
        num_workers=args.num_workers,
    )


def _build_task(args: argparse.Namespace):
    if args.task == "stagenet":
        from pyhealth.tasks import MortalityPredictionStageNetMIMIC4
        return MortalityPredictionStageNetMIMIC4()
    if args.task == "icd_labs":
        return ICDLabsMIMIC4(window_hours=args.observation_window_hours)
    if args.task == "clinical_notes_icd_labs":
        return ClinicalNotesICDLabsMIMIC4(window_hours=args.observation_window_hours)
    if args.task in ("notes_labs", "notes_only"):
        task = NotesLabsMIMIC4(
            window_hours=args.observation_window_hours,
            include_icd=args.icd_codes,
            include_vitals=args.include_vitals,
            include_labs=(args.task != "notes_only"),
            note_extraction=getattr(args, "note_extraction", "regex"),
            note_source=getattr(args, "note_source", "discharge"),
        )
        if args.tokenizer_model:
            schema_key = "admission_note_times"
            _, opts = task.input_schema[schema_key]
            task.input_schema[schema_key] = (
                "tuple_time_text",
                {**opts, "tokenizer_model": args.tokenizer_model},
            )
            print(f"[tokenizer] Overriding tokenizer_model -> {args.tokenizer_model}")
        return task
    if args.task == "labs_only":
        return LabsOnlyMIMIC4(window_hours=args.observation_window_hours)
    raise ValueError(f"Unknown task: {args.task}")


def _build_model(args: argparse.Namespace, sample_dataset: Any):
    # Pretraining must match the downstream text path. Downstream runs freeze
    # the text encoder, so pretraining that updates all 110 million BERT
    # parameters trains a path that inference then discards. A frozen encoder
    # also lets the [CLS] cache operate.
    unified = UnifiedMultimodalEmbeddingModel(
        processors=sample_dataset.input_processors,
        embedding_dim=args.embedding_dim,
        freeze_text_encoder=bool(args.freeze_encoder),
    )

    backbone = build_backbone(
        arch=getattr(args, "arch", "transformer"),
        feature_size=args.embedding_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        use_rope=getattr(args, "use_rope", False),
        rope_max_seq_len=getattr(args, "rope_max_seq_len", 8192),
        rope_base=getattr(args, "rope_base", 10000.0),
        rope_scaling=getattr(args, "rope_scaling", 1.0),
        state_size=getattr(args, "state_size", 16),
        conv_kernel=getattr(args, "conv_kernel", 4),
        num_transformer_layers=getattr(args, "jamba_transformer_layers", 1),
        num_mamba_layers=getattr(args, "jamba_mamba_layers", 1),
    )

    per_modality_ratio = None
    if args.lab_mask_ratio is not None or args.text_mask_ratio is not None:
        per_modality_ratio = {}
        # Map modality strings to indices using the unified model's lookup.
        for field_name, modality in unified.modality_types.items():
            mod_idx = unified._modality_to_idx[modality]
            if modality.value == "numeric" and args.lab_mask_ratio is not None:
                per_modality_ratio[mod_idx] = args.lab_mask_ratio
            if modality.value == "text" and args.text_mask_ratio is not None:
                per_modality_ratio[mod_idx] = args.text_mask_ratio

    if args.method == "mae":
        model = MultimodalMaskedAutoencoder(
            embedding_model=unified,
            backbone=backbone,
            decoder_layers=args.decoder_layers,
            decoder_heads=args.decoder_heads,
            decoder_dim=args.decoder_dim,
            mask_ratio=args.mask_ratio,
            mask_strategy=args.mask_strategy,
            per_modality_ratio=per_modality_ratio,
            norm_pix_loss=args.norm_pix_loss,
        )
    elif args.method == "simmim":
        model = MultimodalSimMIM(
            embedding_model=unified,
            backbone=backbone,
            mask_ratio=args.mask_ratio,
            mask_strategy=args.mask_strategy,
            per_modality_ratio=per_modality_ratio,
            norm_targets=args.norm_pix_loss,
        )
    elif args.method == "ijepa":
        model = MultimodalIJEPA(
            embedding_model=unified,
            context_encoder=backbone,
            predictor_layers=args.predictor_layers,
            predictor_heads=args.predictor_heads,
            predictor_dim=args.predictor_dim,
            target_ema_decay=args.ema_decay,
            target_ema_end=args.ema_end,
            num_target_blocks=args.num_target_blocks,
            target_block_len=args.target_block_len,
        )
    elif args.method == "vjepa":
        scales = getattr(args, "target_block_scales", None) or [2, 4, 8]
        if isinstance(scales, str):
            scales = [int(s) for s in scales.split(",") if s.strip()]
        scales = tuple(int(s) for s in scales)
        normalize_targets = getattr(args, "normalize_targets", None)
        model = MultimodalVJEPA(
            embedding_model=unified,
            context_encoder=backbone,
            predictor_layers=args.predictor_layers,
            predictor_heads=args.predictor_heads,
            predictor_dim=args.predictor_dim,
            target_ema_decay=args.ema_decay,
            target_ema_end=args.ema_end,
            num_target_blocks=args.num_target_blocks,
            target_block_scales=scales,
            require_multimodal_blocks=getattr(args, "require_multimodal_blocks", False),
            normalize_targets=True if normalize_targets is None else normalize_targets,
        )
    else:
        raise ValueError(f"Unknown pretraining method: {args.method}")

    # Attach dataset metadata so the model can convert raw batches internally.
    model.feature_keys = list(sample_dataset.input_processors.keys())
    model.input_processors = sample_dataset.input_processors
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSL pretraining on unified multimodal sequences.")
    p.add_argument("--config", type=str, default=None, help="YAML config path (e.g. configs/pretrain/mae_labs_only.yaml).")
    p.add_argument("--ehr-root", type=str, default=None)
    p.add_argument("--note-root", type=str, default=None)
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./output/pretrain_ssl")
    p.add_argument(
        "--task",
        type=str,
        choices=["stagenet", "icd_labs", "clinical_notes_icd_labs", "notes_labs", "notes_only", "labs_only"],
        default=None,
    )
    p.add_argument(
        "--method",
        type=str,
        choices=["mae", "simmim", "ijepa", "vjepa"],
        default=None,
    )

    # Model
    p.add_argument("--arch", type=str, default=None, choices=list(ARCH_CHOICES),
                   help="Encoder backbone architecture (transformer | jamba | mamba).")
    p.add_argument("--embedding-dim", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)

    # Mamba / Jamba backbone knobs
    p.add_argument("--state-size", type=int, default=None, help="Mamba SSM state size (mamba/jamba).")
    p.add_argument("--conv-kernel", type=int, default=None, help="Mamba causal conv kernel (mamba/jamba).")
    p.add_argument("--jamba-transformer-layers", type=int, default=None, help="Attention layers in the Jamba stack.")
    p.add_argument("--jamba-mamba-layers", type=int, default=None, help="Mamba layers in the Jamba stack.")

    # RoPE options
    p.add_argument("--use-rope", action="store_true", default=None, help="Use RoPE in the Transformer backbone.")
    p.add_argument("--rope-max-seq-len", type=int, default=None)
    p.add_argument("--rope-base", type=float, default=None)
    p.add_argument("--rope-scaling", type=float, default=None, help="NTK-aware scaling factor.")

    # MAE decoder
    p.add_argument("--decoder-layers", type=int, default=None)
    p.add_argument("--decoder-heads", type=int, default=None)
    p.add_argument("--decoder-dim", type=int, default=None)
    p.add_argument("--norm-pix-loss", action="store_true", default=None)

    # I-JEPA predictor
    p.add_argument("--predictor-layers", type=int, default=None)
    p.add_argument("--predictor-heads", type=int, default=None)
    p.add_argument("--predictor-dim", type=int, default=None)
    p.add_argument("--ema-decay", type=float, default=None)
    p.add_argument("--ema-end", type=float, default=None)
    p.add_argument("--num-target-blocks", type=int, default=None)
    p.add_argument("--target-block-len", type=int, default=None)

    # V-JEPA specific
    p.add_argument(
        "--target-block-scales",
        type=str,
        default=None,
        help="Comma-separated multi-scale block lengths for V-JEPA, e.g. '2,4,8'.",
    )
    p.add_argument(
        "--require-multimodal-blocks",
        action="store_true",
        default=None,
        help="V-JEPA: prefer target windows that span more than one modality.",
    )
    p.add_argument(
        "--no-normalize-targets",
        dest="normalize_targets",
        action="store_false",
        default=None,
        help="V-JEPA: disable LayerNorm on EMA targets before the loss.",
    )

    # Masking
    p.add_argument("--mask-ratio", type=float, default=None)
    p.add_argument("--mask-strategy", type=str, default=None, choices=["random", "block"])
    p.add_argument("--lab-mask-ratio", type=float, default=None)
    p.add_argument("--text-mask-ratio", type=float, default=None)

    # Training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--max-grad-norm", type=float, default=None)
    p.add_argument("--scheduler", type=str, default=None, choices=["none", "cosine"])
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--save-every-n-epochs", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--grad-accumulation-steps", type=int, default=None)
    p.add_argument("--use-amp", action="store_true", default=None, help="Use automatic mixed precision (CUDA only).")

    # torchrun / distributed
    p.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank passed by torchrun (ignored, used for compatibility).",
    )

    # Task-specific
    p.add_argument("--observation-window-hours", type=int, default=None)
    p.add_argument("--icd-codes", action="store_true", default=None)
    p.add_argument("--include-vitals", action="store_true", default=None)
    p.add_argument("--note-source", type=str, default=None, choices=["discharge", "radiology"])
    p.add_argument(
        "--note-extraction",
        type=str,
        default=None,
        choices=[
            "regex", "regex_priority", "compact", "tfidf",
            "section_hpi", "section_cc", "section_pmh", "section_meds",
            "section_social", "section_family", "section_allergies", "section_ros",
            "lab_retrieval",
        ],
    )
    p.add_argument("--tokenizer-model", type=str, default=None)
    p.add_argument("--freeze-encoder", action="store_true", default=None)
    p.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false")
    p.add_argument(
        "--text-finetune-mode",
        type=str,
        default=None,
        help="frozen | full | topk:N | lora:r",
    )

    # Data
    p.add_argument(
        "--dev",
        nargs="?",
        type=int,
        const=1000,
        default=None,
        help="Dev mode: limit dataset to N patients.",
    )

    return p.parse_args()


def main() -> None:
    cli_args = parse_args()

    if cli_args.config:
        config_path = Path(cli_args.config)
        if not config_path.exists():
            raise SystemExit(f"Config not found: {config_path}")
        cfg = _resolve_config(config_path)
        args = _apply_defaults(_merge_cli_overrides(_dict_to_namespace(cfg, cli_args), cli_args))
    else:
        args = _apply_defaults(cli_args)

    set_seed(args.seed)

    if not getattr(args, "ehr_root", None):
        raise SystemExit("--ehr-root is required (either via CLI or config).")

    base_dataset = _build_base_dataset(args)
    task = _build_task(args)
    sample_dataset = base_dataset.set_task(task, num_workers=args.num_workers)

    if len(sample_dataset) == 0:
        raise RuntimeError("Task produced zero samples.")

    model = _build_model(args, sample_dataset)

    # EMA update hook for I-JEPA / V-JEPA (V-JEPA subclasses I-JEPA).
    ema_fn = None
    ema_every = 1
    if args.method in ("ijepa", "vjepa") and isinstance(model, MultimodalIJEPA):
        ema_fn = model.update_target_encoder

    train_loader = get_dataloader(sample_dataset, batch_size=args.batch_size, shuffle=True)

    exp_name = f"{args.arch}_{args.method}_{args.task}_seed{args.seed}"
    trainer = PretrainTrainer(
        model=model,
        device=args.device,
        enable_logging=True,
        output_path=args.output_dir,
        exp_name=exp_name,
        ema_update_fn=ema_fn,
        ema_update_every=ema_every,
    )

    trainer.train(
        train_dataloader=train_loader,
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        scheduler=args.scheduler if args.scheduler != "none" else None,
        warmup_steps=args.warmup_steps,
        save_every_n_epochs=args.save_every_n_epochs,
        grad_accumulation_steps=getattr(args, "grad_accumulation_steps", 1),
        use_amp=getattr(args, "use_amp", False),
    )

    print(f"Pretraining complete. Logs: {trainer.exp_path}")


if __name__ == "__main__":
    main()
