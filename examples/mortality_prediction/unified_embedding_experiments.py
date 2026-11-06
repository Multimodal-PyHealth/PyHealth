"""One-knob experiment driver over the unified-embedding e2e runner.

Three modes build arms that differ in exactly one declared knob, run
each through the existing runner's ``run()``, and write ``summary.csv``
and ``summary.md`` (arm, the one differing knob, seconds per phase,
cache counters, scores)::

    experiments.py ab --flag cache-frozen-text -- <runner args...>
    experiments.py sweep --flag window-hours --values 0 24 48 -- <runner args...>
    experiments.py ablate --modalities notes -- <runner args...>

``ablate`` drops each modality at the model stage via the runner's
``--drop-modality``: same task, same samples, same split, same seed.
Extra arguments pass through to every arm unchanged. Each arm gets its
own output directory under ``--output-root``. The driver refuses to run
when the flag does not exist, is inert for the chosen backbone or task,
or the arms would differ in more than the declared knob, or a
passed-through ``--drop-modality`` would be silently replaced by the
ablation arms. The summary checks (not asserts) that seed, split sizes,
and sample identity match across arms and says which differ.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_RUNNER_PATH = Path(__file__).resolve().parent / "unified_embedding_e2e_mimic4.py"

ALIASES = {"window-hours": "observation-window-hours"}


def _bool(value: str) -> bool:
    text = value.strip().lower()
    if text in ("1", "true", "on", "yes"):
        return True
    if text in ("0", "false", "off", "no"):
        return False
    raise ValueError(f"Not a boolean value: {value!r} (use on/off).")


# dest -> converter for --values parsing. Reachability (which backbone or
# task a flag is live for) comes from the runner, not from here: task
# scope below, backbone scope from the runner's own wiring map, so the
# two can never disagree about what the runner passes to the model.
_CONVERTERS: Dict[str, Callable[[str], Any]] = {
    "ehr_root": str, "note_root": str,
    "cache_dir": str, "task": str,
    "model": str, "embedding_dim": int,
    "hidden_dim": int,
    "dropout": float, "epochs": int, "batch_size": int,
    "lr": float, "adam_eps": float,
    "weight_decay": float, "device": str,
    "use_amp": _bool, "amp_dtype": str,
    "accumulation_steps": int, "num_workers": int,
    "pin_memory": _bool, "persistent_workers": _bool,
    "prefetch_factor": int, "seed": int,
    "patience": int, "dev": int,
    "observation_window_hours": float,
    "freeze_encoder": _bool,
    "cache_frozen_text": _bool,
    "max_frozen_text_cache": int,
    "rnn_type": str, "rnn_layers": int,
    "bidirectional": _bool,
    "heads": int,
    "num_layers": int,
    "bottlenecks_n": int,
    "fusion_startidx": int,
    "max_grad_norm": float,
    "wandb": _bool, "wandb_project": str,
    "wandb_entity": str, "wandb_run_name": str,
    "wandb_tags": str,
    "mamba_state_size": int,
    "mamba_conv_kernel": int,
    "jamba_transformer_layers": int,
    "jamba_mamba_layers": int,
}

# dest -> tasks it is live for. Everything else applies to every task.
_TASK_ONLY: Dict[str, set] = {
    "note_root": {"notes_labs"},
    "freeze_encoder": {"notes_labs"},
    "cache_frozen_text": {"notes_labs"},
    "max_frozen_text_cache": {"notes_labs"},
}


@lru_cache(maxsize=1)
def _load_runner() -> Any:
    """Load the runner module once; arms of one driver run share it."""
    spec = importlib.util.spec_from_file_location("unified_e2e_runner", _RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_flag(name: str) -> str:
    """Resolve a ``--flag`` name to the runner argparse dest.

    Accepts leading dashes and either spelling, plus the
    ``window-hours`` alias for ``observation-window-hours``.

    Example:
        >>> resolve_flag("window-hours")
        'observation_window_hours'
    """
    key = ALIASES.get(name.lstrip("-"), name.lstrip("-")).replace("-", "_")
    if key in _CONVERTERS:
        return key
    raise ValueError(
        f"Unknown --flag {name!r}. See "
        "unified_embedding_e2e_mimic4.py --help for the flag list."
    )


def _check_live(dest: str, task: str, model: str) -> None:
    tasks = _TASK_ONLY.get(dest)
    if tasks is not None and task not in tasks:
        raise ValueError(
            f"--flag {dest} is inert for --task {task} "
            f"(reaches only {sorted(tasks)}). Refusing."
        )
    arch_used = _load_runner()._ARCH_FLAGS_USED
    if dest in _load_runner()._ARCH_FLAGS_ALL:
        if model not in arch_used or dest not in arch_used[model]:
            raise ValueError(
                f"--flag {dest} is inert for --model {model} "
                "(the runner never passes it to that backbone). Refusing."
            )


def build_arms(
    mode: str,
    base: argparse.Namespace,
    flag: Optional[str] = None,
    values: Optional[List[str]] = None,
    modalities: Optional[List[str]] = None,
    offs: Optional[List[str]] = None,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Build ``(arm_name, differs, overrides)`` triples for one mode.

    Pure function (no runs): ``ab`` gives two arms, ``sweep`` gives one
    per value, ``ablate`` gives full plus one arm per dropped modality
    or switched-off boolean component. ``ablate`` drops modalities at the
    model stage (same task, samples, split, seed); dropping everything is
    refused. Raises ``ValueError`` on unknown flags, inert flags, or arms
    that would not differ in exactly one knob.

    Example:
        >>> import argparse
        >>> base = argparse.Namespace(task="labs", model="rnn")
        >>> [a[0] for a in build_arms("ab", base, flag="bidirectional")]
        ['ab_bidirectional_off', 'ab_bidirectional_on']
    """
    if mode == "ab":
        dest = resolve_flag(flag or "")
        _check_live(dest, base.task, base.model)
        converter = _CONVERTERS[dest]
        if values:
            if len(values) != 2:
                raise ValueError("ab takes exactly 2 --values (or none for booleans).")
            pair = [converter(v) for v in values]
        elif converter is _bool:
            pair = [False, True]
        else:
            raise ValueError(f"ab --flag {dest} needs --values (not a boolean).")
        if pair[0] == pair[1]:
            raise ValueError("ab arms would be identical. Refusing.")
        return [
            (f"ab_{dest}_{'on' if v is True else 'off' if v is False else v}",
             f"{dest}={v}", {dest: v})
            for v in pair
        ]
    if mode == "sweep":
        dest = resolve_flag(flag or "")
        _check_live(dest, base.task, base.model)
        converter = _CONVERTERS[dest]
        if not values or len(values) < 2:
            raise ValueError("sweep needs --values with at least 2 entries.")
        seen = [converter(v) for v in values]
        if len(set(map(repr, seen))) != len(seen):
            raise ValueError("sweep --values contains duplicates. Refusing.")
        return [(f"sweep_{dest}_{v}", f"{dest}={v}", {dest: v}) for v in seen]
    if mode == "ablate":
        runner = _load_runner()
        modalities = modalities or []
        if len(set(modalities)) != len(modalities):
            raise ValueError("ablate --modalities contains duplicates. Refusing.")
        for modality in modalities:
            if modality not in runner._DROP_MODALITY_FIELDS:
                raise ValueError(
                    f"Cannot drop modality {modality!r}: "
                    f"droppable modalities are "
                    f"{sorted(runner._DROP_MODALITY_FIELDS)}."
                )
        if len(set(offs or [])) != len(offs or []):
            raise ValueError("ablate --off contains duplicates. Refusing.")
        arms = [("ablate_full", "none (reference)", {"drop_modality": []})]
        for modality in modalities:
            arms.append(
                (f"ablate_no_{modality}", f"drop {modality}",
                 {"drop_modality": [modality]})
            )
        for name in offs or []:
            dest = resolve_flag(name)
            _check_live(dest, base.task, base.model)
            if _CONVERTERS[dest] is not _bool:
                raise ValueError(f"--off {dest} is not a boolean component. Refusing.")
            if getattr(base, dest, False) is not True:
                raise ValueError(
                    f"--off {dest}: base run already has it off, "
                    "the arm would be identical. Refusing."
                )
            arms.append((f"ablate_no_{dest}", f"{dest}=False", {dest: False}))
        if len(arms) < 2:
            raise ValueError("ablate needs --modalities and/or --off. Refusing.")
        for _, _, overrides in arms[1:]:
            if "drop_modality" in overrides:
                probe = argparse.Namespace(
                    task=base.task, drop_modality=overrides["drop_modality"])
                try:
                    runner._check_drop_modality(probe)
                except ValueError as exc:
                    raise ValueError(str(exc))
        return arms
    raise ValueError(f"Unknown mode {mode!r} (ab, sweep, ablate).")


def assert_single_knob(
    base: Dict[str, Any],
    arm: Dict[str, Any],
    declared: str,
) -> None:
    """Refuse when an arm differs from base in more than the declared knob.

    ``output_dir`` is always per-arm and excluded. An arm may match base
    on the knob (it still differs from its siblings); distinctness of the
    declared values is enforced in :func:`build_arms`.

    Example:
        >>> assert_single_knob({"a": 1, "seed": 2}, {"a": 9, "seed": 2}, "a")
    """
    extra = {
        key for key in arm
        if key != "output_dir" and arm[key] != base.get(key)
    }
    allowed = {declared} if declared is not None else set()
    if not extra <= allowed:
        raise ValueError(
            f"Arm differs in {sorted(extra)}, not only {declared!r}. Refusing."
        )


def _summarize_arm(output_csv: Path) -> Dict[str, Any]:
    metrics = json.loads((output_csv.parent / "metrics.json").read_text())
    inference_patient_ids = None
    try:
        with output_csv.open(newline="") as handle:
            ids = sorted(
                {row["patient_id"] for row in csv.DictReader(handle)
                 if row.get("patient_id")}
            )
            inference_patient_ids = ids or None
    except OSError:
        inference_patient_ids = None
    return {
        "train_s": round(sum(metrics["epoch_train_seconds"]), 3),
        "eval_s": round(metrics["eval_seconds"], 3),
        "infer_s": round(metrics["inference_seconds"], 3),
        "cache": metrics["frozen_text_cache"],
        "split": metrics["split"],
        "inference_patient_ids": inference_patient_ids,
        "scores": metrics["test_scores"],
        "seed": metrics["config"]["seed"],
    }


def _checked_lines(rows: List[Dict[str, Any]]) -> List[str]:
    """Seed, split-size, and sample-identity comparison lines.

    The identity check compares unique inference-split patient ids only:
    train/val splits and labels are not compared. Each check states what
    is true; when arms differ it names which arms differ and how, instead
    of asserting identity unconditionally.
    """
    by_seed: Dict[Any, List[str]] = {}
    by_split: Dict[str, List[str]] = {}
    by_ids: Dict[Any, List[str]] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], []).append(row["arm"])
        by_split.setdefault(
            json.dumps(row["split"], sort_keys=True), []).append(row["arm"])
        ids = (None if row["inference_patient_ids"] is None
               else tuple(row["inference_patient_ids"]))
        by_ids.setdefault(ids, []).append(row["arm"])
    if len(by_seed) == 1:
        lines = [f"Seeds identical across arms: all {len(rows)} use "
                 f"seed {rows[0]['seed']}."]
    else:
        detail = "; ".join(
            f"{', '.join(arms)} use seed {seed}"
            for seed, arms in by_seed.items())
        lines = [f"Seeds DIFFER across arms: {detail}."]
    if len(by_split) == 1:
        split = rows[0]["split"]
        lines.append(
            f"Split sizes identical across arms: train {split['train']}, "
            f"val {split['val']}, test {split['test']}.")
    else:
        detail = "; ".join(
            f"{', '.join(arms)}: {split}"
            for split, arms in by_split.items())
        lines.append(f"Split sizes DIFFER across arms: {detail}.")
    if len(by_ids) == 1 and rows[0]["inference_patient_ids"] is not None:
        lines.append(
            f"Sample identity identical across arms: "
            f"{len(rows[0]['inference_patient_ids'])} unique "
            f"inference-split patient ids match (train/val splits and "
            f"labels not compared).")
    elif len(by_ids) == 1:
        lines.append("Sample identity not recorded: no inference-split "
                     "patient ids recorded for the arms.")
    else:
        detail = "; ".join(
            f"{', '.join(arms)}: "
            f"{'not recorded' if ids is None else f'{len(ids)} ids'}"
            for ids, arms in by_ids.items())
        lines.append(f"Sample identity DIFFERS across arms "
                     f"(unique inference-split ids; train/val splits and "
                     f"labels not compared): {detail}.")
    return lines


def write_summary(root: Path, rows: List[Dict[str, Any]]) -> None:
    """Write ``summary.csv`` and ``summary.md`` for the finished arms.

    Example:
        >>> import tempfile
        >>> d = tempfile.mkdtemp()
        >>> write_summary(Path(d), [{"arm": "a", "differs": "x=1",
        ...   "train_s": 1.0, "eval_s": 0.5, "infer_s": 0.5,
        ...   "cache": {"hits": 0, "misses": 0, "size": 0},
        ...   "split": {"train": 1, "val": 0, "test": 0},
        ...   "inference_patient_ids": ["p1"],
        ...   "scores": {"pr_auc": 0.5}, "seed": 2}])
        >>> (Path(d) / "summary.csv").is_file()
        True
    """
    score_keys = sorted({k for r in rows for k in r["scores"]})
    fieldnames = (["arm", "differs", "seed", "train_s", "eval_s", "infer_s",
                   "cache_hits", "cache_misses", "cache_size"]
                  + [f"score_{k}" for k in score_keys])
    with (root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "arm": r["arm"], "differs": r["differs"], "seed": r["seed"],
                "train_s": r["train_s"], "eval_s": r["eval_s"],
                "infer_s": r["infer_s"],
                "cache_hits": r["cache"]["hits"],
                "cache_misses": r["cache"]["misses"],
                "cache_size": r["cache"]["size"],
                **{f"score_{k}": r["scores"].get(k) for k in score_keys},
            })
    lines = ["# Experiment summary", "",
             "| " + " | ".join(fieldnames) + " |",
             "|" + "|".join(["---"] * len(fieldnames)) + "|"]
    with (root / "summary.csv").open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            lines.append("| " + " | ".join(row[c] for c in fieldnames) + " |")
    lines += [""] + _checked_lines(rows)
    (root / "summary.md").write_text("\n".join(lines) + "\n")


def _driver_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="./output/unified_experiments")
    sub = parser.add_subparsers(dest="mode", required=True)
    ab = sub.add_parser("ab")
    ab.add_argument("--flag", required=True)
    ab.add_argument("--values", nargs=2, default=None)
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--flag", required=True)
    sweep.add_argument("--values", nargs="+", required=True)
    ablate = sub.add_parser("ablate")
    ablate.add_argument("--modalities", nargs="*", default=[])
    ablate.add_argument("--off", nargs="*", default=[], dest="offs")
    return parser


def main(argv: Optional[List[str]] = None, run_fn: Optional[Callable] = None) -> Path:
    """Entry point (``argv`` injectable for tests).

    Driver options come first, then ``--``, then the runner arguments::

        experiments.py --output-root ./out ab --flag bidirectional -- <runner args...>

    Example:
        >>> main(["--help"])  # doctest: +SKIP
    """
    parser = _driver_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        cut = argv.index("--")
        driver_argv, passthrough = argv[:cut], argv[cut + 1:]
    else:
        driver_argv, passthrough = argv, []
        if any(a.startswith("--ehr-root") for a in driver_argv):
            parser.error("put runner arguments after '--'.")
    args = parser.parse_args(driver_argv)
    if any(a == "--output-dir" or a.startswith("--output-dir=") for a in passthrough):
        parser.error("pass --output-root instead; each arm gets its own --output-dir.")
    runner = _load_runner()
    base = runner.parse_args(passthrough)
    if args.mode == "ablate" and (getattr(base, "drop_modality", None) or []):
        parser.error(
            "ablate sets --drop-modality per arm; a passed-through value "
            "would be silently replaced by the arm overrides. Drop it from "
            "the runner arguments."
        )
    try:
        arms = build_arms(args.mode, base, getattr(args, "flag", None),
                          getattr(args, "values", None),
                          getattr(args, "modalities", None),
                          getattr(args, "offs", None))
    except ValueError as exc:
        parser.error(str(exc))
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    run = run_fn or runner.run
    rows = []
    for name, differs, overrides in arms:
        arm_args = argparse.Namespace(**{**vars(base), **overrides})
        declared = next(iter(overrides)) if overrides else None
        arm_args.output_dir = str(root / name)
        try:
            assert_single_knob(vars(base), vars(arm_args), declared)
        except ValueError as exc:
            parser.error(str(exc))
        output_csv = Path(run(arm_args))
        rows.append({"arm": name, "differs": differs,
                     **_summarize_arm(output_csv)})
    write_summary(root, rows)
    print(f"Wrote {root / 'summary.csv'} and {root / 'summary.md'}")
    return root


if __name__ == "__main__":
    main()
