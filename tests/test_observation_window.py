"""Regression tests for observation-window correctness.

Each targets a defect that reached real results, and each is written to fail
against the pre-fix code rather than to restate the implementation.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest


LAB_TASKS = [
    "LabsOnlyMIMIC4",
    "ICDLabsMIMIC4",
    "ClinicalNotesICDLabsMIMIC4",
    "ClinicalNotesICDLabsCXRMIMIC4",
    "NotesLabsMIMIC4",
]


@pytest.mark.parametrize("task_name", LAB_TASKS)
def test_every_lab_task_honours_its_observation_window(task_name):
    """Collecting labs through DISCHARGE under a declared 24h window leaks.

    Labs drawn hours before death are near-deterministic for a mortality label.
    All four task bodies computed an observation window and then passed
    ``admission_dischtime`` anyway, so ``window_hours`` was inert. Parameterised
    over every lab-emitting task rather than pinned to the one found first.
    """
    from pyhealth.tasks import multimodal_mimic4 as m

    task = getattr(m, task_name)(window_hours=24)
    admit = datetime(2180, 5, 6, 8, 0, 0)
    discharge = admit + timedelta(days=9)

    end = task._admission_window_end(admit, discharge)
    horizon = (end - admit).total_seconds() / 3600.0

    assert horizon == pytest.approx(24.0, abs=0.01), (
        f"{task_name} collects labs {horizon:.0f}h past admission against a "
        f"declared 24h window; anything beyond it leaks the outcome"
    )
    assert end < discharge


@pytest.mark.parametrize("task_name", LAB_TASKS)
def test_observation_window_is_anchored_per_admission(task_name):
    """The window anchored on the FIRST admission globally.

    Later admissions then received a span ending before it began, collected
    nothing, and the task injected a placeholder row for each, so sequence
    length encoded the patient's future admission count.
    """
    from pyhealth.tasks import multimodal_mimic4 as m

    task = getattr(m, task_name)(window_hours=24)
    first = datetime(2180, 5, 6, 8, 0, 0)
    later = first + timedelta(days=400)

    # A long stay is bounded by the window.
    assert task._admission_window_end(first, first + timedelta(days=9)) == \
        first + timedelta(hours=24)
    # A stay shorter than the window is bounded by discharge, not the window.
    assert task._admission_window_end(first, first + timedelta(hours=6)) == \
        first + timedelta(hours=6)
    # A later admission gets its OWN window, not one anchored 400 days earlier.
    end = task._admission_window_end(later, later + timedelta(days=5))
    assert end == later + timedelta(hours=24)
    assert end > later, "later admission received an already-expired window"


@pytest.mark.parametrize("task_name", LAB_TASKS)
def test_window_change_invalidates_the_cache(task_name):
    """A code fix alone leaves existing caches serving superseded samples.

    The task cache key is uuid5 over ``{**vars(task), schemas}``, so without a
    version marker every previously built cache is silently reused.
    """
    from pyhealth.tasks import multimodal_mimic4 as m

    task = getattr(m, task_name)(window_hours=24)
    assert vars(task).get("emitted_data_version") is not None, (
        f"{task_name} emits different data after the window fix but carries no "
        f"version marker, so stale leaky samples would be reused silently"
    )

    def cache_key(t, drop_version=False):
        v = dict(vars(t))
        if drop_version:
            v.pop("emitted_data_version", None)
        params = json.dumps(
            {**v, "input_schema": t.input_schema, "output_schema": t.output_schema},
            sort_keys=True, default=str,
        )
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, params))

    assert cache_key(task) != cache_key(task, drop_version=True)


def test_window_none_still_collects_through_discharge():
    """window_hours=None is the explicit whole-stay mode and must be preserved."""
    from pyhealth.tasks.multimodal_mimic4 import LabsOnlyMIMIC4

    task = LabsOnlyMIMIC4(window_hours=None)
    admit = datetime(2180, 5, 6, 8, 0, 0)
    discharge = admit + timedelta(days=9)
    assert task._admission_window_end(admit, discharge) == discharge


def test_icd_labs_task_is_defined_once():
    """A shadowed duplicate silently swaps behaviour on any future edit."""
    import inspect
    from pyhealth.tasks import multimodal_mimic4 as m

    source = inspect.getsource(m)
    assert source.count("\nclass ICDLabsMIMIC4(") == 1
