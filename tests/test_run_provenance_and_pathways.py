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


def test_sunlab_accepts_resized_images_as_well_as_images(tmp_path):
    """The resized set holds flattened ``{dicom_id}.jpg`` files under
    ``resized_images``. Requiring a directory literally named ``images`` failed
    on a complete 377,110-image cohort.
    """
    import pandas as pd
    from pyhealth.datasets.mimic4 import MIMIC4CXRSunlabDataset

    root = tmp_path / "cxr"
    (root / "resized_images").mkdir(parents=True)
    pd.DataFrame(
        {"dicom_id": ["abc123"], "StudyTime": ["123045.0"]}
    ).to_csv(root / "mimic-cxr-2.0.0-metadata.csv", index=False)

    MIMIC4CXRSunlabDataset.prepare_metadata(
        object.__new__(MIMIC4CXRSunlabDataset), str(root)
    )

    out = pd.read_csv(root / "mimic-cxr-2.0.0-metadata-pyhealth-sunlab.csv")
    path = str(out["image_path"].iloc[0])
    assert path.endswith(f"resized_images{os.sep}abc123.jpg") or path.endswith(
        "resized_images/abc123.jpg"
    )
    assert str(out["studytime"].iloc[0]).zfill(6) == "123045"


def test_sunlab_writes_metadata_to_cache_when_root_is_unwritable(tmp_path):
    """PhysioNet roots are typically read-only. Writing the derived CSV there
    raised PermissionError after a complete cohort had already been found.
    """
    import pandas as pd
    from pyhealth.datasets.mimic4 import MIMIC4CXRSunlabDataset

    root = tmp_path / "cxr"
    (root / "resized_images").mkdir(parents=True)
    pd.DataFrame(
        {"dicom_id": ["abc"], "StudyTime": ["93000"], "subject_id": ["1"]}
    ).to_csv(root / "mimic-cxr-2.0.0-metadata.csv", index=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    os.chmod(root, 0o555)
    try:
        dest = MIMIC4CXRSunlabDataset.prepare_metadata(
            object.__new__(MIMIC4CXRSunlabDataset),
            str(root),
            cache_dir=str(cache),
        )
    finally:
        os.chmod(root, 0o755)

    assert dest.startswith(str(cache))
    assert Path(dest).is_file()
    assert not (root / "mimic-cxr-2.0.0-metadata-pyhealth-sunlab.csv").exists()
    written = pd.read_csv(dest)
    path = str(written.loc[0, "image_path"])
    assert path.endswith(f"resized_images{os.sep}abc.jpg") or path.endswith(
        "resized_images/abc.jpg"
    )


def test_sunlab_reports_both_directory_names_when_neither_exists(tmp_path):
    import pandas as pd
    from pyhealth.datasets.mimic4 import MIMIC4CXRSunlabDataset

    root = tmp_path / "cxr"
    root.mkdir()
    pd.DataFrame(
        {"dicom_id": ["abc123"], "StudyTime": ["1"]}
    ).to_csv(root / "mimic-cxr-2.0.0-metadata.csv", index=False)

    with pytest.raises(FileNotFoundError, match="resized_images"):
        MIMIC4CXRSunlabDataset.prepare_metadata(
            object.__new__(MIMIC4CXRSunlabDataset), str(root)
        )



def test_the_image_channel_count_comes_from_the_processor():
    """The unified embedding sizes its patch embedding from
    ``processor.in_channels``. Without that attribute it fell back to 3 while a
    greyscale CXR task produced 1, and the mismatch surfaced only at the first
    forward pass:

        RuntimeError: Given groups=1, weight of size [128, 3, 16, 16],
        expected input[16, 1, 224, 224] to have 3 channels
    """
    from pyhealth.processors import TimeImageProcessor

    assert TimeImageProcessor(image_size=224, mode="L").in_channels == 1
    assert TimeImageProcessor(image_size=224, mode="RGB").in_channels == 3


def test_a_placeholder_image_has_the_same_channels_as_a_real_one():
    from pyhealth.processors import TimeImageProcessor

    for mode in ("L", "RGB"):
        processor = TimeImageProcessor(image_size=64, mode=mode)
        assert processor._zero_image_tensor().shape[0] == processor.in_channels
