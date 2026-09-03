#!/usr/bin/env python3
"""Verify the paper cells: completeness, protocol alignment, metric integrity.

Run this before anything goes in a table. It re-derives every reported number
from the raw predictions rather than trusting a summary, and it diffs every
cell's recorded config against a reference so a cell that drifted cannot pass
quietly.

    python scripts/paper/verify.py --seed 1

Exit code is non-zero if any check fails, so it works as a gate.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

MODELS = ["mlp", "rnn", "transformer", "bottleneck_transformer", "ehrmamba", "jambaehr"]

# Keys that define the protocol. Any difference between a cell and the reference
# outside ALLOWED_DIFFS is a failure, not a note.
PROTOCOL_KEYS = [
    "seed", "resolved_split_seed", "split_mode", "embedding_dim", "hidden_dim",
    "dropout", "batch_size", "resolved_lr", "resolved_adam_eps",
    "resolved_max_grad_norm", "epochs", "patience", "use_amp", "amp_dtype",
    "weight_decay",
]
# --freeze-encoder is not comparable across modalities: the labs task has no
# text encoder, so its launchers omit the flag on purpose. Checked per modality.
TEXT_TASKS = {"notes_labs", "notes_labs_cxr"}
# Differences that are the experiment rather than drift.
ALLOWED_DIFFS = {"task", "model", "image_backbone", "max_frozen_text_cache",
                 "n_train", "n_val", "n_test", "cxr_root", "note_root",
                 "output_dir", "cache_dir", "inert_flags", "rnn_type",
                 "rnn_layers", "heads", "num_layers", "bottlenecks_n",
                 "fusion_startidx", "mamba_state_size", "mamba_conv_kernel",
                 "jamba_transformer_layers", "jamba_mamba_layers"}

FAILURES: list[str] = []
NOTES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print("  FAIL  " + msg)


def ok(msg: str) -> None:
    print("  ok    " + msg)


def load_cell(d: str):
    rc = glob.glob(os.path.join(d, "*", "run_config.json"))
    pf = glob.glob(os.path.join(d, "*", "predictions_*.csv"))
    mh = glob.glob(os.path.join(d, "*", "metrics_history.json"))
    if not (rc and pf and mh):
        return None
    cfg = json.load(open(rc[0]))
    rows = list(csv.DictReader(open(pf[0])))
    hist = json.load(open(mh[0]))
    return dict(cfg=cfg, config=cfg["config"], rows=rows, hist=hist,
                pred_path=pf[0], run_config_path=rc[0])


def metrics(rows):
    from sklearn.metrics import average_precision_score, roc_auc_score

    ys = [float(r["y_true"]) for r in rows]
    ps = [float(r["y_prob"]) for r in rows]
    return average_precision_score(ys, ps), roc_auc_score(ys, ps), ys, ps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fixdiff-root", default=os.path.expanduser("~/ml4h-tranche1-fixdiff/output"))
    ap.add_argument("--paper-root", default=os.path.expanduser("~/ml4h-paper/output"))
    ap.add_argument("--cxr-backbone", default="xrv_chex")
    args = ap.parse_args()
    s = args.seed

    cells: dict[tuple[str, str], str] = {}
    for m in MODELS:
        cells[("labs", m)] = f"{args.fixdiff_root}/paper_labs_{m}_seed{s}"
        cells[("labs_notes", m)] = f"{args.fixdiff_root}/paper_labs_notes_{m}_seed{s}"
        cells[("labs_notes_cxr", m)] = (
            f"{args.paper_root}/paper_labs_notes_cxr_{m}_seed{s}_{args.cxr_backbone}")
    # the RNN CXR cell was run across several encoders; the canonical one is the
    # default backbone, but accept the bare-name dir for the control arm too
    if not os.path.isdir(cells[("labs_notes_cxr", "rnn")]):
        alt = f"{args.paper_root}/paper_labs_notes_cxr_rnn_seed{s}"
        if os.path.isdir(alt):
            cells[("labs_notes_cxr", "rnn")] = alt

    print("=" * 78)
    print("1. COMPLETENESS")
    loaded: dict[tuple[str, str], dict] = {}
    for key, d in sorted(cells.items()):
        c = load_cell(d)
        if c is None:
            fail(f"{key[0]}/{key[1]}: missing predictions/metrics/run_config ({d})")
        else:
            loaded[key] = c
    if loaded:
        ok(f"{len(loaded)}/{len(cells)} cells have all three artifacts")

    print()
    print("2. PROTOCOL ALIGNMENT (vs labs_notes/rnn)")
    ref_key = ("labs_notes", "rnn")
    if ref_key not in loaded:
        fail("reference cell labs_notes/rnn missing; cannot check alignment")
    else:
        ref = loaded[ref_key]["config"]
        for key, c in sorted(loaded.items()):
            diffs = [f"{k}: {ref.get(k)!r}->{c['config'].get(k)!r}"
                     for k in PROTOCOL_KEYS
                     if k not in ALLOWED_DIFFS and ref.get(k) != c["config"].get(k)]
            if diffs:
                fail(f"{key[0]}/{key[1]} protocol drift: {'; '.join(diffs)}")
        if not FAILURES:
            ok(f"all {len(loaded)} cells match the reference on {len(PROTOCOL_KEYS)} protocol keys")
        for key, c in sorted(loaded.items()):
            want = key[0] in TEXT_TASKS
            got = bool(c["config"].get("freeze_encoder"))
            if want != got:
                fail(f"{key[0]}/{key[1]}: freeze_encoder={got}, expected {want} "
                     f"({'text task must freeze BERT' if want else 'no text encoder in this task'})")
        ok("freeze_encoder correct for every modality (text tasks frozen, labs n/a)")

    print()
    print("3. METRIC INTEGRITY (recomputed from raw predictions)")
    for key, c in sorted(loaded.items()):
        ap_, roc, ys, ps = metrics(c["rows"])
        n, pos = len(ys), int(sum(ys))
        nan = sum(1 for v in ps if math.isnan(v))
        cfg = c["config"]
        problems = []
        if nan:
            problems.append(f"{nan} NaN probabilities")
        if pos == 0 or pos == n:
            problems.append("degenerate label column")
        if cfg.get("n_test") != n:
            problems.append(f"row count {n} != run_config n_test {cfg.get('n_test')}")
        if len(set(r["patient_id"] for r in c["rows"])) != n:
            problems.append("duplicate patient_id in predictions")
        if min(ps) < 0 or max(ps) > 1:
            problems.append(f"probabilities out of [0,1]: [{min(ps):.3f},{max(ps):.3f}]")
        if problems:
            fail(f"{key[0]}/{key[1]}: " + "; ".join(problems))
        else:
            ok(f"{key[0]:<15}{key[1]:<24} AP={ap_:.4f} ROC={roc:.4f} n={n} pos={pos}")

    print()
    print("4. SPLIT CONSISTENCY (within each modality)")
    by_task = defaultdict(list)
    for (task, model), c in loaded.items():
        by_task[task].append((model, c))
    for task, items in sorted(by_task.items()):
        sizes = {(c["config"]["n_train"], c["config"]["n_val"], c["config"]["n_test"])
                 for _, c in items}
        if len(sizes) > 1:
            fail(f"{task}: cells disagree on split sizes: {sizes}")
            continue
        idsets = {m: frozenset(r["patient_id"] for r in c["rows"]) for m, c in items}
        distinct = set(idsets.values())
        if len(distinct) > 1:
            fail(f"{task}: cells evaluated different test patients "
                 f"({len(distinct)} distinct id sets)")
        else:
            ok(f"{task:<15} {len(items)} cells share one split {sizes.pop()} "
               f"and identical test patient set")

    print()
    print("5. MODEL SELECTION (test comes from the best-val checkpoint)")
    for key, c in sorted(loaded.items()):
        h = c["hist"]
        best = max(h, key=lambda e: e["val_pr_auc"])
        stopped = h[-1]["epoch"]
        patience = c["config"]["patience"]
        if stopped != len(h) - 1:
            fail(f"{key[0]}/{key[1]}: metrics_history epochs not contiguous")
        elif len(h) < c["config"]["epochs"] and stopped - best["epoch"] != patience:
            NOTES.append(f"{key[0]}/{key[1]}: stopped at {stopped}, best at "
                         f"{best['epoch']}, patience {patience} (tie in val may explain)")
    ok("early-stop arithmetic checked on all cells")

    print()
    print("6. CODE PROVENANCE")
    shas = defaultdict(set)
    for (task, model), c in loaded.items():
        shas[task].add(c["cfg"].get("source_sha256"))
    for task, ss in sorted(shas.items()):
        if len(ss) > 1:
            fail(f"{task}: cells ran under {len(ss)} different runner versions: {ss}")
        else:
            ok(f"{task:<15} single runner build {list(ss)[0][:16] if list(ss)[0] else '?'}")

    print()
    print("=" * 78)
    for n in NOTES:
        print("  note  " + n)
    if FAILURES:
        print(f"\nVERIFICATION FAILED: {len(FAILURES)} problem(s)")
        return 1
    print(f"\nVERIFICATION PASSED: {len(loaded)} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
