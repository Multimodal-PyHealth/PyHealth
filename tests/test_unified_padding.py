"""Batch padding must never reach the model as data.

The collator pads short samples to the longest in the batch, filling both value
and time with 0.0. Nothing recorded that padding, and no processor emits an
event-level mask, so the unified embedding marked every slot valid. A padded
slot then looked exactly like a real measurement taken at admission time.

Ordering made it worse. Padding carries time 0.0, so an ascending sort placed it
BEFORE every real event, while all three backbone families assume the opposite:
``RNNLayer`` packs the first ``mask.sum()`` steps, ``get_last_visit`` indexes
``mask.sum() - 1``, and ``TransformerLayer`` reads position 0 as its CLS vector.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from pyhealth.datasets.collate import collate_temporal
from pyhealth.models.embedding.unified import UnifiedMultimodalEmbeddingModel
from pyhealth.models.rnn import RNNLayer
from pyhealth.models.utils import get_last_visit
from pyhealth.processors import StageNetTensorProcessor


def _batch():
    """Sample A has three lab events; sample B has one, so B gets padded."""
    return [
        {
            "labs": {"value": torch.ones(3, 2), "time": torch.tensor([6.0, 12.0, 24.0])},
            "mortality": 0,
        },
        {
            "labs": {"value": torch.ones(1, 2), "time": torch.tensor([6.0])},
            "mortality": 1,
        },
    ]


def _model(**kwargs):
    model = UnifiedMultimodalEmbeddingModel(
        processors={"labs": StageNetTensorProcessor()}, embedding_dim=16, **kwargs
    )
    model.encoders["labs"] = nn.Linear(2, 16)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# The collator records the padding it creates
# ─────────────────────────────────────────────────────────────────────────────


def test_the_collator_reports_which_slots_are_padding():
    collated = collate_temporal(_batch())["labs"]

    assert "pad_mask" in collated, "padding was created but never recorded"
    assert collated["pad_mask"].tolist() == [[True, True, True], [True, False, False]]


def test_a_batch_that_needs_no_padding_is_still_reported_as_valid():
    batch = [
        {"labs": {"value": torch.ones(2, 2), "time": torch.tensor([1.0, 2.0])}},
        {"labs": {"value": torch.ones(2, 2), "time": torch.tensor([3.0, 4.0])}},
    ]
    assert collate_temporal(batch)["labs"]["pad_mask"].all()


def test_the_padding_mask_is_not_the_observation_mask():
    """``labs_mask`` answers "was this value measured"; ``pad_mask`` answers "is
    this slot real". Conflating them tells the standardiser that every real
    event was observed.
    """
    collated = collate_temporal(_batch())["labs"]
    assert "pad_mask" in collated and "mask" not in collated


# ─────────────────────────────────────────────────────────────────────────────
# The unified sequence is left-aligned and padding contributes nothing
# ─────────────────────────────────────────────────────────────────────────────


def test_padding_is_marked_invalid_in_the_unified_sequence():
    out = _model()({"labs": collate_temporal(_batch())["labs"]})

    assert out["mask"][1].tolist() == [1.0, 0.0, 0.0], (
        "padding is reported valid, so the model reads it as a measurement"
    )


def test_padding_sorts_after_every_real_event():
    out = _model()({"labs": collate_temporal(_batch())["labs"]})
    mask = out["mask"][1].bool()

    assert bool(mask[0]), "position 0 is padding; TransformerLayer reads it as CLS"
    valid = mask.tolist()
    assert valid == sorted(valid, reverse=True), "the sequence is not left-aligned"


def test_get_last_visit_lands_on_a_real_event():
    out = _model()({"labs": collate_temporal(_batch())["labs"]})
    mask = out["mask"][1]

    positions = torch.arange(mask.numel()).float().view(1, -1, 1).repeat(1, 1, 2)
    picked = int(get_last_visit(positions, mask.view(1, -1))[0, 0])

    assert bool(mask[picked]), f"pooled from padded position {picked}"


def test_padded_slots_contribute_nothing_to_the_sequence():
    out = _model()({"labs": collate_temporal(_batch())["labs"]})
    padded = out["sequence"][1][~out["mask"][1].bool()]

    assert padded.abs().sum().item() == 0.0


def test_the_event_order_is_stable_under_tied_times():
    """The sort key is heavily tied: all padding shares 0.0, and events from one
    admission share offsets. An unstable sort changes RNN and Mamba outputs
    between torch builds and between CPU and CUDA.
    """
    tied = [
        {"labs": {"value": torch.arange(6.0).view(3, 2), "time": torch.zeros(3)}},
        {"labs": {"value": torch.ones(1, 2), "time": torch.tensor([0.0])}},
    ]
    model = _model()
    first = model({"labs": collate_temporal(tied)["labs"]})["sequence"]
    for _ in range(4):
        again = model({"labs": collate_temporal(tied)["labs"]})["sequence"]
        assert torch.equal(first, again)


# ─────────────────────────────────────────────────────────────────────────────
# The consequence the backbones actually see
# ─────────────────────────────────────────────────────────────────────────────


def test_the_rnn_output_does_not_depend_on_padded_content():
    """With every slot marked valid, ``lengths`` covered the whole padded
    sequence and the RNN consumed the fabricated t=0 slots as real events.
    """
    layer = RNNLayer(input_size=1, hidden_size=4, num_layers=1).eval()
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
    x = torch.zeros(1, 6, 1)
    x[0, :3, 0] = 1.0

    with torch.no_grad():
        _, base = layer(x, mask)
        perturbed = x.clone()
        perturbed[0, 3:, 0] = 99.0
        _, after = layer(perturbed, mask)

    assert torch.allclose(base, after), "the RNN is reading the padded region"


def test_a_sample_with_no_valid_event_does_not_crash_the_rnn():
    """Correct masks make a zero length reachable for the first time.
    ``pack_padded_sequence`` rejects it.
    """
    layer = RNNLayer(input_size=2, hidden_size=4).eval()

    with torch.no_grad():
        _, last = layer(torch.zeros(1, 3, 2), torch.tensor([[0, 0, 0]]))

    assert last.shape == (1, 4)
    assert torch.isfinite(last).all()


# ─────────────────────────────────────────────────────────────────────────────
# The standardiser reads observation flags, not padding flags
# ─────────────────────────────────────────────────────────────────────────────


def test_the_standardiser_uses_the_sibling_observation_field():
    """``feat_dict["mask"]`` is never populated by any processor, so the
    standardiser path raised on every run that enabled it.
    """
    from pyhealth.processors import fit_lab_standardizer

    standardizer = fit_lab_standardizer(
        [
            {"labs": torch.tensor([[140.0, 1.0]]), "labs_mask": torch.tensor([[True, True]])},
            {"labs": torch.tensor([[142.0, 1.2]]), "labs_mask": torch.tensor([[True, True]])},
        ]
    )
    # ``labs_mask`` is a declared field with its own processor, exactly as the
    # task schema builds it.
    model = UnifiedMultimodalEmbeddingModel(
        processors={
            "labs": StageNetTensorProcessor(),
            "labs_mask": StageNetTensorProcessor(),
        },
        embedding_dim=16,
        numeric_standardizers={"labs": standardizer},
    )
    model.encoders["labs"] = nn.Linear(2, 16)
    model.encoders["labs_mask"] = nn.Linear(2, 16)

    out = model(
        {
            "labs": {"value": torch.tensor([[[140.0, 1.0]]]), "time": torch.tensor([[6.0]])},
            "labs_mask": {
                "value": torch.tensor([[[1.0, 1.0]]]),
                "time": torch.tensor([[6.0]]),
            },
        }
    )

    assert torch.isfinite(out["sequence"]).all()


def test_a_missing_observation_field_is_reported_clearly():
    from pyhealth.processors import fit_lab_standardizer

    standardizer = fit_lab_standardizer(
        [{"labs": torch.tensor([[1.0, 2.0]]), "labs_mask": torch.tensor([[True, True]])}]
    )
    model = _model(numeric_standardizers={"labs": standardizer})

    with pytest.raises(ValueError, match="labs_mask"):
        model({"labs": {"value": torch.ones(1, 1, 2), "time": torch.tensor([[6.0]])}})


# ─────────────────────────────────────────────────────────────────────────────
# The path production actually uses
# ─────────────────────────────────────────────────────────────────────────────


def test_the_dataloader_collate_records_padding():
    """``get_dataloader`` uses ``collate_fn_dict_with_padding``, not
    ``collate_temporal``. A padding fix applied only to the latter is inert in
    production, which is exactly what a real run revealed.
    """
    from pyhealth.datasets.utils import PAD_MASK_SUFFIX, collate_fn_dict_with_padding

    # StageNet processors emit (time, value) tuples.
    batch = [
        {"labs": (torch.tensor([6.0, 12.0, 24.0]), torch.ones(3, 2)), "mortality": 0},
        {"labs": (torch.tensor([6.0]), torch.ones(1, 2)), "mortality": 1},
    ]
    collated = collate_fn_dict_with_padding(batch)

    assert f"labs{PAD_MASK_SUFFIX}" in collated
    assert collated[f"labs{PAD_MASK_SUFFIX}"].tolist() == [
        [True, True, True],
        [True, False, False],
    ]


def test_a_batch_of_equal_lengths_reports_no_padding():
    from pyhealth.datasets.utils import PAD_MASK_SUFFIX, collate_fn_dict_with_padding

    batch = [
        {"labs": (torch.tensor([1.0, 2.0]), torch.ones(2, 2))},
        {"labs": (torch.tensor([3.0, 4.0]), torch.ones(2, 2))},
    ]
    collated = collate_fn_dict_with_padding(batch)

    # Nothing was padded, so no mask is needed and none is invented.
    assert f"labs{PAD_MASK_SUFFIX}" not in collated


def test_the_backbone_threads_the_pad_mask_into_the_unified_inputs():
    """``_build_unified_inputs`` keys the field dict off ``schema()``, and no
    processor's schema contains ``mask``. The padding validity therefore has to
    arrive on a parallel batch key or it never reaches the model.
    """
    from types import SimpleNamespace

    from pyhealth.datasets.utils import PAD_MASK_SUFFIX, collate_fn_dict_with_padding
    from pyhealth.models.transformer import Transformer

    batch = [
        {"labs": (torch.tensor([6.0, 12.0, 24.0]), torch.ones(3, 2))},
        {"labs": (torch.tensor([6.0]), torch.ones(1, 2))},
    ]
    collated = collate_fn_dict_with_padding(batch)

    host = SimpleNamespace(
        feature_keys=["labs"],
        device="cpu",
        dataset=SimpleNamespace(
            input_processors={"labs": StageNetTensorProcessor()}
        ),
    )
    inputs = Transformer._build_unified_inputs(host, collated)

    assert "pad_mask" in inputs["labs"], "padding validity never reaches the model"
    assert inputs["labs"]["pad_mask"].tolist() == [
        [True, True, True],
        [True, False, False],
    ]
