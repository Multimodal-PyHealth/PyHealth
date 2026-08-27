"""Seed content for the three wandb config artifacts.

Edit this file, then run `python ../scripts/push_experiments.py` to publish
new `:latest` versions of `defaults`, `tasks`, and `experiments` in the
`pyhealth-multimodal` wandb project. All future skill launches read those
artifacts; this file is only the edit surface.

Three sections mirror the retired `experiments.yaml`:
- DEFAULTS: shared params applied to every run unless overridden per-entry
- TASKS: modality combos and which roots they require
- EXPERIMENTS: named GPU-pinned combos. RAM-sensitive params (batch_size,
  embedding_dim, hidden_dim, lr, model-structural keys) are self-contained
  per entry - there is no per-model defaults layer.
"""

DEFAULTS = {
    "project_dir": "/home/joshua86/PyHealth",
    "conda_env": "pyhealth2",
    "ehr_root": "/shared/rsaas/physionet.org/files/mimiciv/2.2",
    "note_root": "/shared/rsaas/physionet.org/files/mimic-note",
    "cxr_root": "/shared/rsaas/physionet.org/files/MIMIC-CXR",
    "cxr_variant": "sunlab",
    "cache_dir": None,  # None -> ~/.cache/pyhealth (975 GB free). /shared/eng/pyhealth is full.
    "output_dir": "output/unified",
    "observation_window_hours": 24,
    "seed": 42,
    "dropout": 0.1,
    "weight_decay": 0.0,
    "num_workers": 1,
    "epochs": 10,
    "wandb_project": "pyhealth-multimodal",
}

TASKS = [
    {"name": "cxr",                     "class": "CXRMIMIC4",                 "roots": ["ehr", "cxr"]},
    {"name": "icd_cxr",                 "class": "ICDCXRMIMIC4",              "roots": ["ehr", "cxr"]},
    {"name": "labs_cxr",                "class": "LabsCXRMIMIC4",             "roots": ["ehr", "cxr"]},
    {"name": "icd_labs_cxr",            "class": "ICDLabsCXRMIMIC4",          "roots": ["ehr", "cxr"]},
    {"name": "icd_labs",                "class": "ICDLabsMIMIC4",             "roots": ["ehr"]},
    {"name": "clinical_notes_icd_labs", "class": "ClinicalNotesICDLabsMIMIC4","roots": ["ehr", "note"]},
]

# Column list is authoritative - push_experiments.py uses it to shape the
# wandb Table. Adding a new per-model knob? Add its column name here, then
# populate it on the relevant EXPERIMENTS entries. Missing values become None.
EXPERIMENT_COLUMNS = [
    "name", "task", "model", "gpu",
    "batch_size", "embedding_dim", "hidden_dim", "lr",
    "heads", "num_layers", "dropout", "weight_decay", "epochs", "seed",
    "rnn_type", "rnn_layers", "bidirectional",
    "bottlenecks_n", "fusion_startidx",
    "mamba_state_size", "mamba_conv_kernel",
    "jamba_transformer_layers", "jamba_mamba_layers",
    "vision_pool", "output_dir",
]

EXPERIMENTS = [
    {"name": "cxr_mlp",          "task": "cxr",          "model": "mlp", "gpu": 0,
     "batch_size": 16, "embedding_dim": 128, "hidden_dim": 128, "lr": 1.0e-3},

    {"name": "labs_cxr_mlp",     "task": "labs_cxr",     "model": "mlp", "gpu": 3,
     "batch_size": 16, "embedding_dim": 128, "hidden_dim": 128, "lr": 1.0e-3},

    {"name": "icd_cxr_mlp",      "task": "icd_cxr",      "model": "mlp", "gpu": 5,
     "batch_size": 16, "embedding_dim": 128, "hidden_dim": 128, "lr": 1.0e-3},

    {"name": "icd_labs_cxr_mlp", "task": "icd_labs_cxr", "model": "mlp", "gpu": 7,
     "batch_size": 8,  "embedding_dim": 128, "hidden_dim": 128, "lr": 1.0e-3,
     "epochs": 20},

    # ICD + labs sweep across four temporal models. GPU 5 / 7 rotation, 2 in parallel x 2 rounds.
    {"name": "icd_labs_mlp",     "task": "icd_labs",     "model": "mlp", "gpu": 5,
     "batch_size": 16, "embedding_dim": 64,  "hidden_dim": 64,  "lr": 1.0e-3},

    {"name": "icd_labs_ehrmamba","task": "icd_labs",     "model": "ehrmamba", "gpu": 7,
     "batch_size": 8,  "embedding_dim": 64,  "hidden_dim": 64,  "lr": 1.0e-3,
     "num_layers": 2, "mamba_state_size": 16, "mamba_conv_kernel": 4},

    {"name": "icd_labs_jambaehr","task": "icd_labs",     "model": "jambaehr", "gpu": 5,
     "batch_size": 8,  "embedding_dim": 64,  "hidden_dim": 64,  "lr": 1.0e-3,
     "heads": 4, "jamba_transformer_layers": 2, "jamba_mamba_layers": 6,
     "mamba_state_size": 16, "mamba_conv_kernel": 4},

    {"name": "icd_labs_bottleneck_transformer",
     "task": "icd_labs", "model": "bottleneck_transformer", "gpu": 7,
     "batch_size": 4,  "embedding_dim": 64,  "hidden_dim": 64,  "lr": 1.0e-4,
     "heads": 4, "num_layers": 2, "bottlenecks_n": 4, "fusion_startidx": 1},

    # CXR-only with JambaEHR. vision_pool=mean (1 vec/image).
    {"name": "cxr_jambaehr",     "task": "cxr",          "model": "jambaehr", "gpu": 5,
     "batch_size": 4,  "embedding_dim": 64,  "hidden_dim": 64,  "lr": 1.0e-3,
     "heads": 4, "jamba_transformer_layers": 2, "jamba_mamba_layers": 6,
     "mamba_state_size": 16, "mamba_conv_kernel": 4},

    # labs+CXR with Transformer, vision_pool=mean (1 vec/image).
    {"name": "labs_cxr_transformer_compressed",
     "task": "labs_cxr", "model": "transformer", "gpu": 5,
     "batch_size": 4,  "embedding_dim": 64,  "hidden_dim": 64,  "lr": 1.0e-3,
     "heads": 4, "num_layers": 2,
     "vision_pool": "mean",
     "output_dir": "output/unified_e2e/labs_cxr_transformer_compressed"},

    # labs+CXR with Transformer, vision_pool=none (all ~196 patch tokens/image).
    # Sequence length inflates ~196x; batch_size=1 is safe floor on 80GB.
    {"name": "labs_cxr_transformer_uncompressed",
     "task": "labs_cxr", "model": "transformer", "gpu": 5,
     "batch_size": 1,  "embedding_dim": 64,  "hidden_dim": 64,  "lr": 1.0e-3,
     "heads": 4, "num_layers": 2,
     "vision_pool": "none",
     "output_dir": "output/unified_e2e/labs_cxr_transformer_uncompressed"},
]
