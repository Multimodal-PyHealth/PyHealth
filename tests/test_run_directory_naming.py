"""Two arms of one comparison must not write to the same directory.

The run directory was named from the model and the seed only. A paired
comparison holds both of those fixed and varies the task, so the two arms
resolved to one path and the second run overwrote the first run's
``metrics_history.json``, ``run_config.json`` and predictions CSV. The loss is
silent: the surviving directory looks like a complete run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "mortality_prediction"
    / "unified_embedding_e2e_mimic4.py"
)


def _exp_name_template() -> str:
    """The literal assignment, read from source.

    The name is computed deep inside ``main()`` after a dataset build, so the
    assignment itself is what this test pins.
    """
    for line in RUNNER.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("exp_name = "):
            return stripped
    raise AssertionError("exp_name is no longer assigned in the runner")


def _render(task: str, model: str, seed: int) -> str:
    template = _exp_name_template().split("=", 1)[1].strip()
    args = SimpleNamespace(task=task, model=model, seed=seed)
    return eval(template, {"args": args})  # noqa: S307 - a literal f-string


def test_the_run_directory_name_includes_the_task():
    assert "args.task" in _exp_name_template(), (
        "two arms of the same comparison would share one output directory"
    )


@pytest.mark.parametrize(
    "left,right",
    [
        (("labs_only", "transformer", 42), ("notes_labs", "transformer", 42)),
        (("labs_only", "rnn", 1), ("cxr_notes_labs", "rnn", 1)),
    ],
)
def test_two_arms_at_the_same_seed_get_different_directories(left, right):
    assert _render(*left) != _render(*right)


def test_the_seed_still_separates_repeats_of_one_arm():
    assert _render("notes_labs", "transformer", 42) != _render(
        "notes_labs", "transformer", 43
    )


def test_the_model_still_separates_backbones():
    assert _render("notes_labs", "transformer", 42) != _render(
        "notes_labs", "ehrmamba", 42
    )
