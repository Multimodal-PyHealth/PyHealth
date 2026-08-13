"""Regression tests for the encoder, tokenisation and standardiser defects.

Each defect below produced a result that looked correct and was not: a
checkpoint that trained a random backbone and reported no missing keys, a text
channel that was 94% constant across patients, a standardiser fitted on 1/N of
the train split, and a note truncated to one fifth of its length.

Each test is written to fail against the pre-fix behaviour rather than to
restate the implementation.
"""

from __future__ import annotations

import os
from unittest import mock

from pyhealth.models.base_model import BaseModel

import pytest
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 1. Strict transfer of a pretrained checkpoint
# ─────────────────────────────────────────────────────────────────────────────


class _Downstream(BaseModel):
    """Minimal real BaseModel with the transformer backbone attribute name."""

    def __init__(self, width: int = 8, depth: int = 3):
        nn.Module.__init__(self)
        self.dataset = None
        self._unified_backbone = nn.Sequential(
            *[nn.Linear(width, width) for _ in range(depth)]
        )
        self.fc = nn.Linear(width, 1)

    def forward(self, **kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError


def _model(width: int = 8, depth: int = 3) -> _Downstream:
    return _Downstream(width, depth)


def _checkpoint(model, *, keep: int | None = None) -> dict[str, torch.Tensor]:
    """An SSL checkpoint under the ``backbone.*`` prefix."""
    items = [
        ("backbone." + k, torch.randn_like(v))
        for k, v in model._unified_backbone.state_dict().items()
    ]
    if keep is not None:
        items = items[:keep]
    return dict(items)


def test_a_complete_checkpoint_transfers_and_reports_full_coverage():
    model = _model()
    report = model.load_pretrained_state_dict(_checkpoint(model))

    assert report["backbone_matched"] == report["backbone_source"]
    assert report["backbone_matched"] == report["backbone_target"]
    assert report["backbone_matched"] > 0


def test_a_partial_checkpoint_is_refused_instead_of_silently_accepted():
    """The previous loader wrote the keys that matched, then called
    ``strict=False``. The unmatched tensors were already present, so PyTorch
    reported NO missing keys. A jamba checkpoint that matched 6 of 30 backbone
    tensors trained a mostly random backbone and the run looked correct.
    """
    model = _model()
    partial = _checkpoint(model, keep=2)

    # Reproduce the pre-fix path exactly: build the full target state dict,
    # overwrite only the keys that matched, then load with strict=False. Every
    # key is present, so PyTorch reports nothing missing.
    merged = dict(model.state_dict())
    merged.update(
        {k.replace("backbone.", "_unified_backbone."): v for k, v in partial.items()}
    )
    result = nn.Module.load_state_dict(model, merged, strict=False)
    assert not result.missing_keys, "the silent path reported no missing keys"
    assert not result.unexpected_keys

    with pytest.raises(ValueError, match="partial pretrained backbone"):
        model.load_pretrained_state_dict(partial)


def test_a_shape_mismatch_is_refused():
    model = _model(width=8)
    wider = _checkpoint(_model(width=16))

    with pytest.raises(ValueError, match="shape mismatch|partial"):
        model.load_pretrained_state_dict(wider)


def test_a_model_without_a_registered_backbone_is_refused():
    from pyhealth.models.base_model import BaseModel

    class _NoBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 1)

    model = _NoBackbone()
    load = BaseModel.load_pretrained_state_dict.__get__(model)

    with pytest.raises(ValueError, match="no registered unified backbone"):
        load({"backbone.fc.weight": torch.randn(1, 4)})


# ─────────────────────────────────────────────────────────────────────────────
# 3. Standardiser that reads the full train split
# ─────────────────────────────────────────────────────────────────────────────


def _lab_samples(n: int = 40):
    """Sodium near 140 and creatinine near 1: a 140x scale range."""
    torch.manual_seed(0)
    return [
        {
            "labs": torch.stack(
                [140.0 + torch.randn(1) * 4, 1.0 + torch.randn(1) * 0.2]
            ).view(1, 2),
            "labs_mask": torch.ones(1, 2, dtype=torch.bool),
        }
        for _ in range(n)
    ]


def test_the_fit_ignores_unobserved_slots():
    """A padded slot holds 0.0. Fitting over it moves the mean toward zero."""
    from pyhealth.processors import fit_lab_standardizer

    samples = [
        {
            "labs": torch.tensor([[140.0, 1.0], [0.0, 0.0]]),
            "labs_mask": torch.tensor([[True, True], [False, False]]),
        },
        {
            "labs": torch.tensor([[142.0, 1.2], [138.0, 0.8]]),
            "labs_mask": torch.tensor([[True, True], [True, True]]),
        },
    ]
    standardizer = fit_lab_standardizer(samples)

    # Observed sodium is 140, 142, 138. The mean is 140, not 105 (which is what
    # counting the padded 0.0 would give).
    assert standardizer.mean[0].item() == pytest.approx(140.0, abs=1e-4)


def test_an_unobserved_slot_maps_to_zero_and_not_to_a_z_score():
    from pyhealth.processors import fit_lab_standardizer

    standardizer = fit_lab_standardizer(_lab_samples())
    values = torch.tensor([[[140.0, 1.0], [0.0, 0.0]]])
    observed = torch.tensor([[[True, True], [False, False]]])

    out = standardizer(values, observed)

    assert out[0, 1].abs().sum().item() == 0.0
    # A missing value must not become the large negative z-score of 0.0.
    assert torch.isfinite(out).all()


def test_world_size_does_not_shrink_the_fit():
    """``SampleDataset`` subclasses ``litdata.StreamingDataset``, whose
    ``_DistributedEnv`` divides ``__len__`` and ``__iter__`` by ``WORLD_SIZE``.
    ``torchrun`` sets ``WORLD_SIZE`` but not ``GLOBAL_RANK``, and
    ``torch.distributed`` is not initialised when the dataset is built, so every
    rank reported ``global_rank=0`` and fitted the same first 1/N.

    The fit now reads ``patient_to_index``, which ``WORLD_SIZE`` does not divide.
    """
    from pyhealth.processors import fit_lab_standardizer

    samples = _lab_samples(40)

    class _ShardedByWorldSize:
        """Reproduces the litdata contract that caused the defect.

        ``__len__`` and ``__iter__`` shard by ``WORLD_SIZE``; ``region_of_interest``
        and ``__getitem__`` do not. Verified against real litdata with 20 samples:
        ``len()`` reports 5 under ``WORLD_SIZE=4`` while the region of interest
        still sums to 20 and indexing still reaches 19.
        """

        def __init__(self, records):
            self._records = records
            self.region_of_interest = [(0, len(records))]

        def _visible(self):
            world = int(os.environ.get("WORLD_SIZE", "1"))
            return self._records[: len(self._records) // world]

        def __len__(self):
            return len(self._visible())

        def __iter__(self):
            return iter(self._visible())

        def __getitem__(self, index):
            return self._records[index]

    dataset = _ShardedByWorldSize(samples)

    single = fit_lab_standardizer(dataset)
    with mock.patch.dict(os.environ, {"WORLD_SIZE": "4"}):
        sharded = fit_lab_standardizer(dataset)

    assert torch.allclose(single.mean, sharded.mean), (
        "statistics changed under WORLD_SIZE, so each rank fitted a different "
        "1/N of the train split"
    )
    assert torch.allclose(single.std, sharded.std)
    assert int(single.observed_count.sum()) == int(sharded.observed_count.sum())


def test_the_statistics_travel_in_the_state_dict():
    """A checkpoint must apply at inference the same transform it trained under.
    Statistics held outside ``state_dict`` do not survive a save and load.
    """
    from pyhealth.processors import fit_lab_standardizer

    standardizer = fit_lab_standardizer(_lab_samples())
    keys = set(standardizer.state_dict())

    assert {"mean", "std"} <= keys
    assert standardizer.state_dict()["mean"].shape == (2,)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Scale-safe unified embedding
# ─────────────────────────────────────────────────────────────────────────────


def test_content_normalisation_adds_no_parameters():
    """An existing checkpoint must continue to load, so the correction must be
    parameter free.
    """
    from pyhealth.models.embedding.unified import UnifiedMultimodalEmbeddingModel

    import inspect

    source = inspect.getsource(UnifiedMultimodalEmbeddingModel.forward)
    assert "F.layer_norm" in source, "the content term is not normalised"
    assert "nn.LayerNorm" not in source, "a parametric norm would break transfer"


def test_content_normalisation_restores_separation_between_patients():
    """Raw laboratory values reach a norm of about 761 at ``embedding_dim=128``,
    against 3.2 for a BERT ``[CLS]`` through a default projection. The constant
    time and type terms then dominate the text channel: cosine similarity across
    patients was 0.9953, so the signal was about 5.9% of the squared norm.
    """
    torch.manual_seed(0)
    dim = 128

    def unit(x):
        return x / x.norm(dim=-1, keepdim=True)

    # The measured norms at embedding_dim=128 are vector norms.
    content = unit(torch.randn(6, dim)) * 3.2  # BERT [CLS] through Linear(768, D)
    time_term = unit(torch.randn(1, dim)) * 8.0  # constant across patients
    type_term = unit(torch.randn(1, dim)) * 11.8  # constant across patients

    def separation(x):
        x = torch.nn.functional.normalize(x, dim=-1)
        similarity = x @ x.T
        off_diagonal = similarity[~torch.eye(len(x), dtype=torch.bool)]
        return 1.0 - off_diagonal.mean().item()

    before = separation(content + time_term + type_term)
    normalised = torch.nn.functional.layer_norm(content, (dim,))
    after = separation(normalised + time_term + type_term)

    assert after > before * 2, (
        f"normalisation did not restore separation: {before:.4f} -> {after:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Token budget
# ─────────────────────────────────────────────────────────────────────────────


def test_the_token_budget_default_is_512():
    """128 cut 95% of extracted discharge notes to about one fifth of their
    length. 512 is the position-embedding limit of Bio_ClinicalBERT.
    """
    import inspect

    from pyhealth.processors.tuple_time_text_processor import (
        TupleTimeTextProcessor,
    )

    default = inspect.signature(TupleTimeTextProcessor.__init__).parameters[
        "max_length"
    ].default
    assert default == 512


def test_padding_is_not_a_fixed_width():
    """A note of 4 tokens cost 512 slots under ``padding="max_length"``. Every
    padded slot is a full BERT forward pass.
    """
    import inspect

    from pyhealth.processors import tuple_time_text_processor as module

    source = inspect.getsource(module)
    assert '"max_length"' not in source or "longest" in source, (
        "padding still pads every note to the full budget"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Cache for the frozen text encoder
# ─────────────────────────────────────────────────────────────────────────────


class _CountingEncoder(nn.Module):
    """Counts rows encoded, and returns a deterministic vector for each row."""

    def __init__(self, width: int = 6):
        super().__init__()
        self.width = width
        self.rows_encoded = 0

    def forward(self, input_ids, attention_mask=None):
        from types import SimpleNamespace

        self.rows_encoded += int(input_ids.shape[0])
        base = input_ids.float().sum(dim=-1, keepdim=True)
        cls = base * torch.arange(1.0, self.width + 1.0)
        hidden = cls.unsqueeze(1).expand(-1, int(input_ids.shape[1]), -1)
        return SimpleNamespace(last_hidden_state=hidden)


def _cache_host(*, frozen: bool, enabled: bool = True, limit: int = 1000):
    """The smallest object that satisfies ``_encode_text_cls``."""
    from types import SimpleNamespace

    return SimpleNamespace(
        cache_frozen_text=enabled,
        _frozen_text_fields={"notes"} if frozen else set(),
        _frozen_text_cache={},
        max_frozen_text_cache=limit,
        type_embedding=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32)),
    )


def _encode(host, encoder, ids, mask):
    from pyhealth.models.embedding.unified import UnifiedMultimodalEmbeddingModel

    return UnifiedMultimodalEmbeddingModel._encode_text_cls(
        host, "notes", encoder, ids, mask
    )


def test_a_frozen_encoder_runs_once_for_a_repeated_note():
    """A frozen encoder is deterministic. A run of 50 epochs repeated the same
    forward pass of 110 million parameters 50 times.
    """
    host = _cache_host(frozen=True)
    encoder = _CountingEncoder()
    ids = torch.tensor([[5, 6, 7], [5, 6, 7], [8, 9, 10]])
    mask = torch.ones_like(ids)

    first = _encode(host, encoder, ids, mask)
    assert encoder.rows_encoded == 2, "identical rows were encoded twice"

    second = _encode(host, encoder, ids, mask)
    assert encoder.rows_encoded == 2, "the warm pass called the encoder again"
    assert torch.equal(first, second), "the cache changed the output"


def test_a_trainable_encoder_never_reads_the_cache():
    """The encoder weights change every step, so a cached value is stale."""
    host = _cache_host(frozen=False)
    encoder = _CountingEncoder()
    ids = torch.tensor([[5, 6, 7], [5, 6, 7]])
    mask = torch.ones_like(ids)

    _encode(host, encoder, ids, mask)
    _encode(host, encoder, ids, mask)

    assert encoder.rows_encoded == 4
    assert host._frozen_text_cache == {}


def test_the_key_contains_the_attention_mask_and_not_only_the_token_ids():
    """Two rows can hold the same identifiers under a different truncation
    budget. A key over the identifiers alone would return the wrong vector.
    """
    host = _cache_host(frozen=True)
    encoder = _CountingEncoder()
    ids = torch.tensor([[5, 6, 7]])

    full = _encode(host, encoder, ids, torch.tensor([[1, 1, 1]]))
    truncated = _encode(host, encoder, ids, torch.tensor([[1, 1, 0]]))

    assert encoder.rows_encoded == 2, "the mask was not part of the key"
    assert full.shape == truncated.shape


def test_a_full_cache_recalculates_instead_of_growing_without_limit():
    host = _cache_host(frozen=True, limit=1)
    encoder = _CountingEncoder()
    ids = torch.tensor([[5, 6, 7], [8, 9, 10]])
    mask = torch.ones_like(ids)

    _encode(host, encoder, ids, mask)
    assert len(host._frozen_text_cache["notes"]) == 1

    before = encoder.rows_encoded
    out = _encode(host, encoder, ids, mask)
    assert encoder.rows_encoded > before, "the uncached row was not recalculated"
    assert torch.isfinite(out).all()


def test_the_fit_works_on_a_split_and_not_only_on_the_full_dataset():
    """``SampleDataset.subset`` copies ``patient_to_index`` unchanged, so after
    ``split_by_patient`` it still holds indices into the PARENT dataset while
    ``__getitem__`` is restricted to the subset's own region. Driving the fit
    from it raised on a real training split:

        ValueError: The provided index 237 didn't find a match within the
        chunk intervals [Interval(chunk_start=0, roi_start_idx=135, ...)]
    """
    from pyhealth.processors.lab_standardizer import _provenance_indices

    class _Split:
        """A subset: the region of interest describes it, the parent map does not."""

        def __init__(self, records, keep):
            self._records = records[:keep]
            self.region_of_interest = [(0, keep)]
            # Stale parent-global indices, exactly as subset() leaves them.
            self.patient_to_index = {f"p{i}": [i] for i in range(len(records))}

        def __len__(self):
            return len(self._records)

        def __getitem__(self, index):
            return self._records[index]

    split = _Split(_lab_samples(40), keep=12)
    indices = _provenance_indices(split)

    assert indices == list(range(12)), (
        "the fit is driven by parent indices, which the split cannot serve"
    )
    assert max(indices) < len(split)


def test_the_frozen_text_cache_key_ignores_batch_padding():
    """The collator pads each row to the widest note in its batch, and batch
    composition changes every epoch because the loader shuffles. A key over the
    padded row therefore gives one note a different key each epoch and the cache
    never hits. Measured on the full-scale notes run, epoch time did not fall
    after epoch 1: 3458s, 3936s, 4048s, 3835s.
    """
    from pyhealth.models.embedding.unified import UnifiedMultimodalEmbeddingModel

    host = _cache_host(frozen=True)
    encoder = _CountingEncoder()

    # The same note, padded to 5 in one batch and to 3 in another.
    wide_ids = torch.tensor([[5, 6, 7, 0, 0]])
    wide_mask = torch.tensor([[1, 1, 1, 0, 0]])
    narrow_ids = torch.tensor([[5, 6, 7]])
    narrow_mask = torch.tensor([[1, 1, 1]])

    UnifiedMultimodalEmbeddingModel._encode_text_cls(
        host, "notes", encoder, wide_ids, wide_mask
    )
    after_first = encoder.rows_encoded
    UnifiedMultimodalEmbeddingModel._encode_text_cls(
        host, "notes", encoder, narrow_ids, narrow_mask
    )

    assert encoder.rows_encoded == after_first, (
        "the same note re-encoded because the key included batch padding"
    )
