"""Full-scale SSL pretraining at the tuned hyperparameters.

Reads a ``best_params_pt_<arch>_<method>_<task>.json`` produced by
``optuna_pretrain.py`` and launches ``pretrain_ssl.py`` on the FULL dataset for
the real run (50 epochs), passing the tuned HPs as CLI overrides on top of the
128/2 base config. The resulting encoder checkpoint is what initializes the
downstream Table-2 runs.

Example:
    python scripts/run_full_pretrain.py \
      --best-params output/optuna_pretrain/notes_only/best_params_pt_mamba_vjepa_notes_only.json \
      --ehr-root ... --note-root ... --cache-dir ... --output-dir output/pretrain_full \
      --epochs 50
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# best_params key -> pretrain_ssl.py flag. store_true flags handled separately.
_FLAG = {
    "lr": "--lr", "weight_decay": "--weight-decay", "batch_size": "--batch-size",
    "warmup_steps": "--warmup-steps", "max_grad_norm": "--max-grad-norm",
    "mask_ratio": "--mask-ratio", "mask_strategy": "--mask-strategy",
    "ema_decay": "--ema-decay", "num_target_blocks": "--num-target-blocks",
    "state_size": "--state-size", "conv_kernel": "--conv-kernel",
    "jamba_transformer_layers": "--jamba-transformer-layers",
    "jamba_mamba_layers": "--jamba-mamba-layers",
}
_STORE_TRUE = {"norm_pix_loss": "--norm-pix-loss", "use_rope": "--use-rope"}


def build_cmd(cli) -> list:
    meta = json.loads(Path(cli.best_params).read_text())
    arch, method, task = meta["arch"], meta["method"], meta["task"]
    hp = meta["best_params"]
    # task_only maps to the pretrain_ssl --task + flags (notes_only/labs_only/vitals)
    task_flags = []
    if task == "notes_labs" and cli.include_vitals:
        task_flags = ["--include-vitals"]

    args = ["--config", str(REPO_ROOT / "configs" / "pretrain" / "base.yaml"),
            "--arch", arch, "--method", method, "--task", task,
            "--ehr-root", cli.ehr_root, "--cache-dir", cli.cache_dir,
            "--output-dir", cli.output_dir, "--epochs", str(cli.epochs),
            "--num-workers", str(cli.num_workers), "--freeze-encoder",
            *task_flags]
    if cli.note_root:
        args += ["--note-root", cli.note_root]
    for k, flag in _FLAG.items():
        if k in hp and hp[k] is not None:
            args += [flag, str(hp[k])]
    for k, flag in _STORE_TRUE.items():
        if hp.get(k):
            args.append(flag)

    # Appended last so they override the tuned values above (argparse keeps the
    # final occurrence of a repeated flag).
    args += [a for a in getattr(cli, "extra", []) or [] if a != "--"]

    script = str(REPO_ROOT / "scripts" / "pretrain_ssl.py")
    if cli.nproc_per_node and cli.nproc_per_node > 1:
        # multi-GPU DDP via torchrun (pretrain_ssl/PretrainTrainer are DDP-aware).
        # Use the env's torchrun (next to sys.executable), not a bare PATH lookup
        # that can resolve to the read-only system miniconda.
        torchrun = str(Path(sys.executable).parent / "torchrun")
        return [torchrun, "--standalone", f"--nproc_per_node={cli.nproc_per_node}",
                script, *args]
    return [sys.executable, script, *args]


def main():
    p = argparse.ArgumentParser(description="Full-scale SSL pretraining at tuned HPs.")
    p.add_argument("--best-params", required=True)
    p.add_argument("--ehr-root", required=True)
    p.add_argument("--note-root", default=None)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--nproc-per-node", type=int, default=1, help="GPUs for DDP (torchrun).")
    p.add_argument("--include-vitals", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Flags appended verbatim to pretrain_ssl.py, overriding the "
                        "tuned values (e.g. --batch-size 32 --grad-accumulation-steps 2 "
                        "to fit a smaller GPU at the same effective batch size).")
    cli = p.parse_args()
    cmd = build_cmd(cli)
    print("[full-pretrain]", " ".join(cmd), flush=True)
    if cli.dry_run:
        return
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
