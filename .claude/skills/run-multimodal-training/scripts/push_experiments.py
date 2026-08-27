"""Publish the three wandb config artifacts from experiments_bootstrap.py.

Reads DEFAULTS, TASKS, EXPERIMENTS from ../config/experiments_bootstrap.py
and logs each as a wandb.Table wrapped in an Artifact. A new artifact version
is created every push; the skill always fetches `:latest`, so this run is
also how you deploy config changes.

Prereqs: `wandb login` on this machine. Run from anywhere; paths are resolved
relative to this file.

    python .claude/skills/run-multimodal-training/scripts/push_experiments.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "config"))

from experiments_bootstrap import (  # noqa: E402
    DEFAULTS,
    EXPERIMENT_COLUMNS,
    EXPERIMENTS,
    TASKS,
)

WANDB_PROJECT = "pyhealth-multimodal"


def _defaults_table() -> wandb.Table:
    columns = list(DEFAULTS.keys())
    row = [DEFAULTS[c] for c in columns]
    return wandb.Table(columns=columns, data=[row])


def _tasks_table() -> wandb.Table:
    # roots joined with commas because wandb.Table cells can't hold Python lists reliably
    rows = [[t["name"], t["class"], ",".join(t["roots"])] for t in TASKS]
    return wandb.Table(columns=["name", "class", "roots"], data=rows)


def _experiments_table() -> wandb.Table:
    rows = [[e.get(c) for c in EXPERIMENT_COLUMNS] for e in EXPERIMENTS]
    return wandb.Table(columns=EXPERIMENT_COLUMNS, data=rows)


def main() -> None:
    run = wandb.init(
        project=WANDB_PROJECT,
        job_type="config-publish",
        name="publish-experiments-config",
    )
    try:
        for name, table in [
            ("defaults", _defaults_table()),
            ("tasks", _tasks_table()),
            ("experiments", _experiments_table()),
        ]:
            artifact = wandb.Artifact(name=name, type="config-table")
            artifact.add(table, name)
            run.log_artifact(artifact)
            print(f"logged artifact: {name} ({len(table.data)} rows)")
    finally:
        run.finish()

    print(f"\ndone. view: {run.url}")


if __name__ == "__main__":
    main()
