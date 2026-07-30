"""Run MedFuse (EHR time-series + chest X-ray) in-hospital mortality on real MIMIC-IV.

Wires the merged MedFuse model to real MIMIC-IV + MIMIC-CXR via MedFuseMortalityMIMIC4
(no model edits). EHR selectable between ICU vitals (chartevents) and chemistry labs
(labevents); modality between EHR+CXR fusion and EHR-only (same paired cohort, CXR hidden).
Manual train loop; per-epoch curves to wandb project pyhealth-multimodal (team convention).

Self-contained: the two dataset configs (chartevents EHR + corrected CXR metadata) are
embedded as Python dicts and written to throwaway temp YAMLs at runtime, so this script +
the MedFuseMortalityMIMIC4 task are the only files needed.
"""

from __future__ import annotations

import argparse
import os
import platform
import random
import subprocess
import tempfile

import numpy as np
import pandas as pd
import torch
import yaml

from pyhealth.datasets import mimic4 as _mimic4_module

_mimic4_module.MIMIC4CXRDataset.prepare_metadata = lambda self, root: None  # no-op

import wandb

from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient
from pyhealth.metrics import binary_metrics_fn
from pyhealth.models import MedFuse
from pyhealth.tasks import MedFuseMortalityMIMIC4

HERE = os.path.dirname(os.path.abspath(__file__))
CORRECTED_METADATA_NAME = "mimic-cxr-2.0.0-metadata-medfuse.csv"
SOURCE_METADATA_NAME = "mimic-cxr-2.0.0-metadata-pyhealth.csv"
WANDB_PROJECT = "pyhealth-multimodal"
METRICS = ["pr_auc", "roc_auc", "f1", "accuracy"]

# --- Dataset configs as Python dicts (no committed YAML). Written to temp files at runtime
# --- because MIMIC4Dataset accepts a config *path*. The default mimic4_ehr.yaml lacks
# --- chartevents (needed for vitals); labs uses the default. The CXR config points at a
# --- corrected metadata CSV whose image_path resolves to the real on-disk images.
EHR_CHARTEVENTS_CFG = {
    "version": "2.2",
    "tables": {
        "patients": {"file_path": "hosp/patients.csv.gz", "patient_id": "subject_id",
                     "timestamp": None, "attributes": ["gender", "anchor_age", "dod"]},
        "admissions": {"file_path": "hosp/admissions.csv.gz", "patient_id": "subject_id",
                       "timestamp": "admittime",
                       "attributes": ["hadm_id", "dischtime", "hospital_expire_flag"]},
        "icustays": {"file_path": "icu/icustays.csv.gz", "patient_id": "subject_id",
                     "timestamp": "intime", "attributes": ["hadm_id", "stay_id", "outtime"]},
        "chartevents": {"file_path": "icu/chartevents.csv.gz", "patient_id": "subject_id",
                        "timestamp": "charttime",
                        "attributes": ["hadm_id", "stay_id", "itemid", "valuenum", "valueuom"]},
    },
}
CXR_METADATA_CFG = {
    "version": "2.1.0",
    "tables": {
        "metadata": {"file_path": CORRECTED_METADATA_NAME, "patient_id": "subject_id",
                     "timestamp": ["studydate", "studytime"], "timestamp_format": "%Y%m%d%H%M%S",
                     "attributes": ["image_path", "dicom_id", "study_id"]},
    },
}


def write_temp_yaml(config: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="medfuse_cfg_")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE).decode().strip()
    except Exception:
        return "unknown"


def ensure_corrected_cxr_metadata(cxr_root, image_subdir="resized_images"):
    dst = os.path.join(cxr_root, CORRECTED_METADATA_NAME)
    if os.path.exists(dst):
        return
    src = os.path.join(cxr_root, SOURCE_METADATA_NAME)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Expected {src} (pyhealth CXR metadata).")
    print(f"[cxr] building corrected metadata -> {dst}")
    df = pd.read_csv(src, dtype=str)
    img_dir = os.path.join(cxr_root, image_subdir)
    df["image_path"] = df["dicom_id"].map(lambda d: os.path.join(img_dir, f"{d}.jpg"))
    df.to_csv(dst, index=False)
    print(f"[cxr] wrote {len(df):,} rows")


def build_dataset(args):
    if args.ehr_source == "vitals":
        ehr_tables = ["chartevents"]
        ehr_config_path = write_temp_yaml(EHR_CHARTEVENTS_CFG)
    else:  # labs — default mimic4_ehr.yaml already defines labevents
        ehr_tables = ["labevents"]
        ehr_config_path = None
    return MIMIC4Dataset(
        ehr_root=args.ehr_root, cxr_root=args.cxr_root,
        ehr_tables=ehr_tables, cxr_tables=["metadata"],
        ehr_config_path=ehr_config_path, cxr_config_path=write_temp_yaml(CXR_METADATA_CFG),
        dev=args.dev, num_workers=args.num_workers,
    )


def make_inputs(batch, ehr_only, device):
    inputs = {"ehr": batch["ehr"].to(device), "mortality": batch["mortality"].to(device)}
    if not ehr_only:
        inputs["cxr"] = batch["cxr"].to(device)
        inputs["cxr_mask"] = batch["cxr_mask"].to(device)
    return inputs


@torch.no_grad()
def evaluate(model, loader, ehr_only, device):
    model.eval()
    pids, y_true, y_prob, losses = [], [], [], []
    for batch in loader:
        out = model(**make_inputs(batch, ehr_only, device))
        y_prob.append(out["y_prob"].detach().cpu().view(-1))
        y_true.append(out["y_true"].detach().cpu().view(-1))
        losses.append(float(out["loss"]))
        pids.extend(batch["patient_id"])
    return (pids, torch.cat(y_true).numpy(), torch.cat(y_prob).numpy(),
            float(np.mean(losses)) if losses else float("nan"))


def safe_metrics(y_true, y_prob):
    try:
        return binary_metrics_fn(y_true, y_prob, metrics=METRICS)
    except Exception as exc:
        print(f"[warn] metric computation failed: {exc}")
        return {m: float("nan") for m in METRICS}


def main():
    p = argparse.ArgumentParser(description="MedFuse on MIMIC-IV (EHR + CXR).")
    p.add_argument("--ehr-root", default="/srv/local/data/physionet.org/files/mimiciv/2.2")
    p.add_argument("--cxr-root", default="/srv/local/data/MIMIC-CXR")
    p.add_argument("--ehr-source", choices=["vitals", "labs"], default="vitals")
    p.add_argument("--modality", choices=["ehr_cxr", "ehr_only"], default="ehr_cxr")
    p.add_argument("--window-hours", type=float, default=48.0)
    p.add_argument("--fixed-window", action="store_true", default=True)
    p.add_argument("--variable-window", dest="fixed_window", action="store_false")
    p.add_argument("--bin-hours", type=float, default=2.0)
    p.add_argument("--min-los-hours", type=float, default=48.0)
    p.add_argument("--cxr-backbone", choices=["resnet18", "resnet34", "resnet50"], default="resnet34")
    p.add_argument("--no-cxr-pretrained", dest="cxr_pretrained", action="store_false")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=os.path.join(HERE, "output"))
    p.add_argument("--exp-name", default=None)
    p.add_argument("--dev", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    ehr_only = args.modality == "ehr_only"
    device = args.device if torch.cuda.is_available() else "cpu"
    exp_name = args.exp_name or f"medfuse_{args.ehr_source}_{args.modality}_seed{args.seed}"
    out_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    ensure_corrected_cxr_metadata(args.cxr_root)
    print(f"[data] building dataset (ehr_source={args.ehr_source}, dev={args.dev})")
    dataset = build_dataset(args)

    task = MedFuseMortalityMIMIC4(
        ehr_source=args.ehr_source, window_hours=args.window_hours,
        fixed_window=args.fixed_window, bin_hours=args.bin_hours,
        min_los_hours=args.min_los_hours, require_cxr=True,
    )
    print(f"[task] set_task {task.task_name} ({args.ehr_source})")
    samples = dataset.set_task(task, num_workers=args.num_workers)
    print(f"[task] {len(samples)} samples (F={task.n_features}, n_bins={task.n_bins})")

    train_ds, val_ds, test_ds = split_by_patient(samples, [0.7, 0.1, 0.2], seed=args.seed)
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = MedFuse(
        dataset=samples, ehr_feature_key="ehr",
        cxr_feature_key="__disabled__" if ehr_only else "cxr",
        cxr_mask_key=None if ehr_only else "cxr_mask",
        cxr_backbone=args.cxr_backbone,
        cxr_pretrained=False if ehr_only else args.cxr_pretrained,
        dropout=args.dropout,
    ).to(device)
    param_count = sum(x.numel() for x in model.parameters())

    wandb.init(
        project=WANDB_PROJECT, name=exp_name,
        tags=["medfuse", args.ehr_source, args.modality] + (["dev"] if args.dev else []),
        config={
            "task": task.task_name, "model": "medfuse", "exp_name": exp_name,
            "modalities": ["ehr"] if ehr_only else ["ehr", "cxr"],
            "seed": args.seed, "dev_mode": args.dev,
            "hp/batch_size": args.batch_size, "hp/lr": args.lr, "hp/epochs": args.epochs,
            "hp/dropout": args.dropout,
            "arch/cxr_backbone": args.cxr_backbone, "arch/ehr_hidden_dim": 256,
            "arch/ehr_num_layers": 2, "arch/fusion_hidden_dim": 512,
            "arch/projection_dim": 256, "arch/param_count": param_count,
            "fusion/ehr_source": args.ehr_source,
            "fusion/observation_window_hours": args.window_hours,
            "fusion/fixed_window": args.fixed_window, "fusion/bin_hours": args.bin_hours,
            "data/train_samples": len(train_ds), "data/val_samples": len(val_ds),
            "data/test_samples": len(test_ds), "data/cxr_variant": "resized",
            "paths/ehr_root": args.ehr_root, "paths/cxr_root": args.cxr_root,
            "paths/output_dir": out_dir,
            "env/gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "env/cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "env/torch_version": torch.__version__, "env/git_sha": git_sha(),
            "env/hostname": platform.node(),
        },
    )
    wandb.define_metric("epoch")
    for key in ["train_loss", "val_loss"] + [f"val_{m}" for m in METRICS]:
        wandb.define_metric(key, step_metric="epoch")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_pr, best_path = -1.0, os.path.join(out_dir, "best.ckpt")

    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(**make_inputs(batch, ehr_only, device))
            out["loss"].backward()
            optimizer.step()
            bs = len(batch["patient_id"])
            total += float(out["loss"]) * bs
            n += bs
        train_loss = total / max(n, 1)

        _, y_true, y_prob, val_loss = evaluate(model, val_loader, ehr_only, device)
        scores = safe_metrics(y_true, y_prob)
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   **{f"val_{m}": scores[m] for m in METRICS}})
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_pr_auc={scores['pr_auc']:.4f} val_roc_auc={scores['roc_auc']:.4f}")

        if scores["pr_auc"] > best_pr:
            best_pr = scores["pr_auc"]
            torch.save(model.state_dict(), best_path)
            print(f"[epoch {epoch}] new best val_pr_auc={best_pr:.4f}")

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    pids, y_true, y_prob, test_loss = evaluate(model, test_loader, ehr_only, device)
    test_scores = safe_metrics(y_true, y_prob)
    print(f"[test] loss={test_loss:.4f} " + " ".join(f"{m}={test_scores[m]:.4f}" for m in METRICS))

    for m in METRICS:
        wandb.summary[f"test_{m}"] = test_scores[m]
    wandb.summary["test_loss"] = test_loss
    wandb.summary["best_val_pr_auc"] = best_pr

    pred_csv = os.path.join(out_dir, "predictions_medfuse.csv")
    pd.DataFrame({"patient_id": pids, "y_true": y_true, "y_prob": y_prob,
                  "y_pred_threshold_0_5": (y_prob >= 0.5).astype(int)}).to_csv(pred_csv, index=False)
    print(f"[test] wrote predictions -> {pred_csv}")
    wandb.finish()


if __name__ == "__main__":
    main()
