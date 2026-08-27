"""Fetch the three wandb config artifacts and reassemble the launch config.

The skill's workflow step 1 calls `load_config()` to get the same dict shape
the retired experiments.yaml used to provide. The runner uses
`use_artifacts(run)` right after `wandb.init` so every training run records
which config versions produced its launch flags.

Standalone use, mostly for verification:

    python .claude/skills/run-multimodal-training/scripts/fetch_experiments.py
    python .claude/skills/run-multimodal-training/scripts/fetch_experiments.py v3
"""
from __future__ import annotations

import sys
from typing import Any, Dict, Tuple

import wandb

WANDB_PROJECT = "pyhealth-multimodal"
ARTIFACT_NAMES = ("defaults", "tasks", "experiments")


def _table_rows(artifact: "wandb.Artifact", key: str) -> list[dict]:
    table = artifact.get(key)
    return [dict(zip(table.columns, row)) for row in table.data]


def load_config(version: str = "latest") -> Dict[str, Any]:
    """Fetch defaults / tasks / experiments artifacts and rebuild the launch config.

    Returns a dict with the same shape the old YAML produced, plus
    `artifact_versions` for provenance:

        {
          "defaults":    {...},        # flat dict
          "tasks":       {name: {"class": str, "roots": [str]}, ...},
          "experiments": {name: {...}, ...},   # None values dropped
          "artifact_versions": {"defaults": "v3", "tasks": "v2", "experiments": "v7"},
        }
    """
    api = wandb.Api()
    artifacts = {
        name: api.artifact(f"{WANDB_PROJECT}/{name}:{version}")
        for name in ARTIFACT_NAMES
    }

    defaults_row = _table_rows(artifacts["defaults"], "defaults")[0]
    tasks_rows = _table_rows(artifacts["tasks"], "tasks")
    exp_rows = _table_rows(artifacts["experiments"], "experiments")

    tasks = {
        r["name"]: {"class": r["class"], "roots": r["roots"].split(",")}
        for r in tasks_rows
    }
    experiments = {
        r["name"]: {k: v for k, v in r.items() if v is not None}
        for r in exp_rows
    }

    return {
        "defaults": defaults_row,
        "tasks": tasks,
        "experiments": experiments,
        "artifact_versions": {name: art.version for name, art in artifacts.items()},
    }


def use_artifacts(run: "wandb.sdk.wandb_run.Run", version: str = "latest") -> None:
    """Link the three config artifacts to a training run for provenance."""
    for name in ARTIFACT_NAMES:
        run.use_artifact(f"{name}:{version}")


def _summarize(cfg: Dict[str, Any]) -> None:
    print(f"artifact versions: {cfg['artifact_versions']}")
    print(f"defaults: {len(cfg['defaults'])} keys")
    print(f"tasks: {len(cfg['tasks'])} entries -> {sorted(cfg['tasks'].keys())}")
    print(f"experiments: {len(cfg['experiments'])} entries")
    for name, entry in cfg["experiments"].items():
        print(f"  {name}: task={entry['task']} model={entry['model']} "
              f"gpu={entry['gpu']} batch={entry['batch_size']} lr={entry['lr']}")


if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv) > 1 else "latest"
    _summarize(load_config(v))