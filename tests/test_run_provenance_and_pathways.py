"""Regression tests for run provenance and the two silent fallbacks.

Each defect below made a measurement mean something other than what it said. A
run reported validation performance as test performance. A patient split became
a sample split, which leaks. A frozen-encoder run and a fine-tuned run left
identical artefacts on disk, so the condition of an earlier result could not be
recovered.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Run provenance
# ─────────────────────────────────────────────────────────────────────────────


def test_the_run_configuration_is_written_beside_the_metrics(tmp_path):
    from pyhealth.utils import write_run_config

    path = write_run_config(str(tmp_path), {"task": "notes_labs", "seed": 42})
    record = json.loads(Path(path).read_text())

    assert Path(path).name == "run_config.json"
    assert record["config"]["task"] == "notes_labs"
    assert record["config"]["seed"] == 42


def test_code_identity_survives_a_run_from_an_unpacked_archive(tmp_path):
    """A cluster run starts from a tarball, where `git rev-parse` gives nothing.
    A digest of the package source identifies the code in that case.
    """
    from pyhealth.utils import write_run_config

    record = json.loads(Path(write_run_config(str(tmp_path), {})).read_text())

    assert record["source_sha256"], "no code identity was recorded"
    assert len(record["source_sha256"]) >= 16
    assert "torch" in record


def test_the_digest_is_stable_for_unchanged_source():
    """Two runs of the same code must record the same digest, otherwise the
    digest cannot show that a set of arms shared one version of the code.
    """
    from pyhealth import utils

    assert utils._source_digest() == utils._source_digest()


def test_a_value_that_json_cannot_hold_does_not_lose_the_whole_record(tmp_path):
    """A configuration holds Paths, enums and devices. If one of them raises,
    the run finishes with no provenance at all, which is the case this file
    exists to prevent.
    """
    from pyhealth.utils import write_run_config

    config = {
        "output_dir": Path("/scratch/run"),
        "device": torch.device("cpu"),
        "window_hours": 24,
        "model": nn.Linear(2, 2),
    }
    record = json.loads(Path(write_run_config(str(tmp_path), config)).read_text())

    assert set(record["config"]) == set(config)
    assert record["config"]["window_hours"] == 24


def test_the_temporary_file_does_not_remain_after_a_write(tmp_path):
    """A partially written run_config.json is worse than none, so the write is
    atomic. No temporary file may survive it.
    """
    from pyhealth.utils import write_run_config

    write_run_config(str(tmp_path), {"seed": 1})

    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert leftovers == [], f"temporary files remained: {leftovers}"


# ─────────────────────────────────────────────────────────────────────────────
# Discriminative learning rate for the text pathway
# ─────────────────────────────────────────────────────────────────────────────


TEXT_FIELDS = {"notes"}


def test_a_trainable_encoder_keeps_the_projection_at_the_base_rate():
    """``encoder_lr`` exists to give a PRETRAINED encoder a gentler rate. A
    projection with random values must not receive that rate.
    """
    from pyhealth.trainer import is_text_pathway_param

    assert is_text_pathway_param(
        "embedding_model.encoders.notes.layer.0.weight", TEXT_FIELDS
    )
    assert not is_text_pathway_param(
        "embedding_model.projections.notes.weight", TEXT_FIELDS
    )


def test_a_frozen_encoder_puts_the_projection_in_the_group():
    """With the encoder frozen, every ``encoders.*`` parameter has
    ``requires_grad=False``, so the group is empty and ``encoder_lr`` controls
    nothing. The projection is then the only trainable text parameter, so it IS
    the text pathway.
    """
    from pyhealth.trainer import is_text_pathway_param

    assert is_text_pathway_param(
        "embedding_model.projections.notes.weight", TEXT_FIELDS, frozen_text_fields={"notes"}
    )


def test_a_non_text_parameter_never_joins_the_group():
    from pyhealth.trainer import is_text_pathway_param

    for name in (
        "_unified_backbone.layers.0.weight",
        "fc.weight",
        "embedding_model.encoders.labs.weight",
        "embedding_model.projections.labs.weight",
    ):
        assert not is_text_pathway_param(name, TEXT_FIELDS, frozen_text_fields={"notes"})


def test_a_field_name_that_is_a_prefix_of_another_does_not_match():
    """``notes`` must not capture ``notes_extra``."""
    from pyhealth.trainer import is_text_pathway_param

    assert not is_text_pathway_param(
        "embedding_model.encoders.notes_extra.weight", TEXT_FIELDS
    )


# ─────────────────────────────────────────────────────────────────────────────
# Within-epoch loss trajectory
# ─────────────────────────────────────────────────────────────────────────────


def test_the_epoch_record_holds_the_within_epoch_trajectory():
    """An epoch mean cannot show the difference between a run that starts badly
    and a run that becomes worse inside the epoch.
    """
    import inspect

    from pyhealth.trainer import Trainer

    source = inspect.getsource(Trainer.train)
    for field in (
        "train_loss_first_step",
        "train_loss_first100",
        "train_loss_last100",
    ):
        assert f'"{field}"' in source, f"{field} is not recorded"


# ─────────────────────────────────────────────────────────────────────────────
# Chest X-ray as a third modality
# ─────────────────────────────────────────────────────────────────────────────


CXR_ARMS = [
    ("cxr_only", {}, {"cxr"}),
    ("cxr_labs", {"include_labs": True}, {"cxr", "labs", "labs_mask"}),
    (
        "cxr_notes_labs",
        {"include_labs": True, "include_notes": True},
        {"cxr", "labs", "labs_mask", "admission_note_times"},
    ),
]


@pytest.mark.parametrize("name,kwargs,expected", CXR_ARMS)
def test_each_cxr_arm_declares_the_fields_it_uses(name, kwargs, expected):
    from pyhealth.tasks import CXRMultimodalMIMIC4

    task = CXRMultimodalMIMIC4(window_hours=24, **kwargs)

    assert set(task.input_schema) == expected


def test_the_lab_mask_matches_the_other_tasks():
    """``StageNetTensorProcessor`` takes no arguments. A processor option here
    that the other tasks do not use would raise at dataset build time.
    """
    from pyhealth.processors import StageNetTensorProcessor
    from pyhealth.tasks import CXRMultimodalMIMIC4

    task = CXRMultimodalMIMIC4(window_hours=24, include_labs=True)
    name, options = task.input_schema["labs_mask"]

    assert name == "stagenet_tensor"
    assert options == {}
    StageNetTensorProcessor(**options)  # must not raise


def test_an_image_uses_the_same_time_convention_as_a_laboratory_value():
    """``StudyDate`` and ``StudyTime`` give hours from admission, which is what
    the unified embedding reads for every modality.
    """
    from pyhealth.tasks import CXRMultimodalMIMIC4

    task = CXRMultimodalMIMIC4(window_hours=24)
    assert task.input_schema["cxr"][0] == "time_image"
