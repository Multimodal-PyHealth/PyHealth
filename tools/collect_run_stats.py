"""Collect one standard-schema stats row per run directory.

Reads run directories (metrics.json, run_config.json, test_metrics.json,
metrics_history.json — whichever are present) and writes one row per run
in the standard run-stats schema, as CSV and Markdown. A field that is
absent stays missing and is listed; nothing is defaulted or inferred.
Code identity comes from ``run_config.json``'s ``source_sha256``,
written on git and archive trees alike; the digest covers the
``pyhealth`` package only, not ``examples/``. ``tree_sha`` is still
collected when git metadata exists, but it is not compared.

Comparability follows the run-identity rule: every pair of runs is
checked on the protocol and hardware fields both runs record, and pairs
that fail (or cannot be checked) are flagged with the reason.

Examples:
    >>> import collect_run_stats as c
    >>> row = c.collect_run("output/arm_on/transformer_seed1")
    >>> isinstance(row["missing_fields"], str)
    True

Usage:
    python tools/collect_run_stats.py RUN_DIR [RUN_DIR ...] --out PREFIX
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Run-identity and protocol fields two runs must share to be comparable.
COMPARE_FIELDS = [
    "source_sha256",
    "task",
    "backbone",
    "window_hours",
    "embedding_dim",
    "hidden_dim",
    "n_layers",
    "heads",
    "batch_size",
    "use_amp",
    "amp_dtype",
    "freeze_encoder",
    "seed",
    "gpu_model",
    "thread_pinning",
    "drop_modality",
    "lr",
    "adam_eps",
    "max_grad_norm",
    "dev",
    "split_mode",
    "eval_split",
    "cache_frozen_text",
]


def _read_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _get(mapping, *keys, default=None):
    for key in keys:
        if isinstance(mapping, dict) and key in mapping:
            return mapping[key]
    return default


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_run(run_dir: str | Path) -> dict:
    """Collect one schema row from a run directory.

    Args:
        run_dir: path to a run directory holding metrics.json and,
            when the runner records them, run_config.json,
            test_metrics.json, and metrics_history.json.

    Returns:
        A dict with one entry per schema field; absent fields hold
        None and are named in ``missing_fields`` (semicolon-joined).
        No field is defaulted or inferred.

    Examples:
        >>> row = collect_run("output/arm_on/transformer_seed1")
        >>> isinstance(row["missing_fields"], str)
        True
        >>> row["missing_fields"].split(";")[0] in row
        True
    """
    root = Path(run_dir)
    metrics = _read_json(root / "metrics.json") or {}
    run_config = _read_json(root / "run_config.json") or {}
    test_metrics = _read_json(root / "test_metrics.json") or {}
    history = _read_json(root / "metrics_history.json")
    if isinstance(history, dict):
        history = history.get("epochs", history.get("history", []))
    if not isinstance(history, list):
        history = []

    config = run_config.get("config", {}) if isinstance(run_config, dict) else {}
    git = run_config.get("git", {}) if isinstance(run_config, dict) else {}
    blob = metrics if isinstance(metrics, dict) else {}
    # Older runs nest the protocol under their own "config" key.
    blob_config = blob.get("config") if isinstance(blob.get("config"), dict) else {}
    # The runner records GPU identity, thread pinning, and VRAM peaks in a
    # nested hardware object; older layouts only have flat keys, which the
    # same picks below still find.
    hw = blob.get("hardware") if isinstance(blob.get("hardware"), dict) else {}

    def pick(*names):
        for source in (hw, blob, blob_config, config):
            value = _get(source, *names)
            if value is not None:
                return value
        return None

    split = pick("split") or {}
    cache = pick("frozen_text_cache", "text_cache") or {}
    scores = pick("test_scores", "test") or {}
    if not isinstance(scores, dict) and isinstance(test_metrics, dict):
        scores = test_metrics
    elif isinstance(test_metrics, dict):
        scores = {**scores, **{k: v for k, v in test_metrics.items() if k not in scores}}

    epochs = pick("epoch_train_seconds", "train_epoch_s") or []
    if not isinstance(epochs, list):
        epochs = []
    warm = [_num(e) for e in epochs[1:]]
    warm = [e for e in warm if e is not None]
    eval_s = _num(pick("eval_seconds", "eval_s"))
    infer_s = _num(pick("inference_seconds", "inference_s"))
    n_train = _num(split.get("train")) if isinstance(split, dict) else None
    n_test = _num(split.get("test")) if isinstance(split, dict) else None

    train_cpu, cpu_utils, vram_alloc, vram_res = [], [], [], []
    for e in history:
        if not isinstance(e, dict):
            continue
        for target, keys in (
            (train_cpu, ("train_cpu_seconds",)),
            (cpu_utils, ("train_cpu_util_pct",)),
            (vram_alloc, ("train_vram_allocated_mb", "vram_allocated_mb")),
            (vram_res, ("train_vram_peak_mb", "vram_peak_mb")),
        ):
            for k in keys:
                v = _num(e.get(k))
                if v is not None:
                    target.append(v)
                    break

    pin = pick("thread_pinning", "omp_num_threads", "OMP_NUM_THREADS")

    # The runner records CPU totals at the metrics top level (plus per-epoch
    # lists and a frozen-cache cap in the recorded config); per-epoch
    # history files are an older layout and only fill in when present.
    cache_cap = cache.get("cap", cache.get("max_entries")) \
        if isinstance(cache, dict) else None

    row = {
        "run_dir": str(root),
        "tree_sha": _get(git, "commit", "sha") or pick("tree_sha", "git_sha"),
        "source_sha256": run_config.get("source_sha256"),
        "dirty": _get(git, "dirty"),
        "seed": pick("seed"),
        "task": pick("task"),
        "backbone": pick("backbone", "model"),
        "window_hours": pick("window_hours", "observation_window_hours"),
        "embedding_dim": pick("embedding_dim"),
        "hidden_dim": pick("hidden_dim"),
        "n_layers": pick("n_layers", "num_layers"),
        "heads": pick("heads"),
        "batch_size": pick("batch_size"),
        "use_amp": pick("use_amp"),
        "amp_dtype": pick("amp_dtype"),
        "freeze_encoder": pick("freeze_encoder", "freeze_text_encoder"),
        "drop_modality": pick("drop_modality"),
        "lr": pick("lr"),
        "adam_eps": pick("adam_eps"),
        "max_grad_norm": pick("max_grad_norm"),
        "dev": pick("dev"),
        "split_mode": pick("split_mode"),
        "eval_split": pick("eval_split"),
        "cache_frozen_text": pick("cache_frozen_text"),
        "note_length": pick("note_length", "max_length"),
        "text_encoder": pick("text_encoder", "tokenizer_model"),
        "split_train": split.get("train") if isinstance(split, dict) else None,
        "split_val": split.get("val", split.get("valid")) if isinstance(split, dict) else None,
        "split_test": split.get("test") if isinstance(split, dict) else None,
        "positive_rate": pick("positive_rate", "prevalence"),
        "train_epoch_s": json.dumps(epochs) if epochs else None,
        "train_epoch_mean_warm_s": (sum(warm) / len(warm)) if warm else None,
        "eval_s": eval_s,
        "inference_s": infer_s,
        "throughput_train_pts_per_s": (n_train / sum(_f for _f in
            [_num(e) for e in epochs] if _f is not None)) if (n_train and epochs
            and all(_num(e) is not None for e in epochs)) else None,
        "throughput_infer_pts_per_s": (n_test / infer_s) if (n_test and infer_s) else None,
        "gpu_model": pick("gpu_model"),
        "gpu_count": pick("gpu_count", "n_gpus"),
        "thread_pinning": pin,
        "vram_allocated_peak_mb": max(vram_alloc) if vram_alloc else pick(
            "vram_allocated_peak_mb", "peak_vram_allocated_mb"),
        "vram_reserved_peak_mb": max(vram_res) if vram_res else pick(
            "vram_reserved_peak_mb", "peak_vram_reserved_mb"),
        "peak_rss_mb": pick("peak_rss_mb", "peak_rss"),
        "peak_pss_mb": pick("peak_pss_mb", "peak_pss_bytes"),
        "train_cpu_seconds_total": sum(train_cpu) if train_cpu
        else _num(pick("train_cpu_seconds")),
        "train_cpu_util_pct_mean": sum(cpu_utils) / len(cpu_utils)
        if cpu_utils else _num(pick("train_cpu_util_pct")),
        "cache_hits": cache.get("hits") if isinstance(cache, dict) else None,
        "cache_misses": cache.get("misses") if isinstance(cache, dict) else None,
        "cache_entries": cache.get("size", cache.get("entries")) if isinstance(cache, dict) else None,
        "cache_bytes": cache.get("bytes") if isinstance(cache, dict) else None,
        "cache_cap": cache_cap if cache_cap is not None
        else _num(pick("max_frozen_text_cache")),
        "val_pr_auc_best": pick("val_pr_auc_best", "best_val_pr_auc"),
        "test_pr_auc": _get(scores, "pr_auc") if isinstance(scores, dict) else None,
        "test_roc_auc": _get(scores, "roc_auc") if isinstance(scores, dict) else None,
    }
    missing = sorted(k for k, v in row.items() if v is None and k != "run_dir")
    row["missing_fields"] = ";".join(missing)
    return row



def check_pair(a: dict, b: dict) -> str:
    """Judge whether two collected rows are comparable.

    Args:
        a: first row from :func:`collect_run`.
        b: second row from :func:`collect_run`.

    Returns:
        ``"comparable"``, ``"NON-COMPARABLE: <fields>"`` listing
        shared fields that differ, or ``"UNVERIFIABLE: <fields>"``
        listing compare fields missing on either side.

    Examples:
        >>> a = {f: 1 for f in COMPARE_FIELDS}
        >>> check_pair(a, dict(a))
        'comparable'
    """
    unknown = [f for f in COMPARE_FIELDS if a.get(f) is None or b.get(f) is None]
    if unknown:
        return "UNVERIFIABLE: missing " + ",".join(unknown)
    differed = [f for f in COMPARE_FIELDS if a.get(f) != b.get(f)]
    if differed:
        return "NON-COMPARABLE: differ " + ",".join(differed)
    return "comparable"


def write_outputs(rows: list[dict], prefix: str) -> tuple[str, str]:
    """Write collected rows as CSV and Markdown.

    Args:
        rows: rows from :func:`collect_run`.
        prefix: output path prefix; writes ``prefix.csv`` and ``prefix.md``.

    Returns:
        The (csv_path, md_path) pair that was written.

    Examples:
        >>> write_outputs([], "/tmp/empty_stats")
        ('/tmp/empty_stats.csv', '/tmp/empty_stats.md')
    """
    csv_path, md_path = prefix + ".csv", prefix + ".md"
    fields = [k for k in rows[0] if k != "missing_fields"] + ["missing_fields"] \
        if rows else ["run_dir", "missing_fields"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Collected run stats", ""]
    if rows:
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(
                "" if row.get(k) is None else str(row.get(k)) for k in fields) + " |")
    lines += ["", "## Pairwise comparability", ""]
    if len(rows) < 2:
        lines.append("Fewer than two runs: no pairs to check.")
    else:
        lines.append("| run_a | run_b | verdict |")
        lines.append("| --- | --- | --- |")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                lines.append(
                    f"| {rows[i]['run_dir']} | {rows[j]['run_dir']} "
                    f"| {check_pair(rows[i], rows[j])} |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return csv_path, md_path


def main(argv=None) -> int:
    """CLI entry point: collect run dirs and write CSV + Markdown.

    Examples:
        >>> main(["only_arg_check"])  # doctest: +SKIP
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="run directories to collect")
    parser.add_argument("--out", required=True, help="output path prefix")
    args = parser.parse_args(argv)
    rows = [collect_run(d) for d in args.run_dirs]
    csv_path, md_path = write_outputs(rows, args.out)
    print(f"wrote {csv_path} and {md_path} ({len(rows)} runs)")
    for row in rows:
        print(f"- {row['run_dir']} missing: {row['missing_fields'] or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
