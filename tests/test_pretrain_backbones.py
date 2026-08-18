"""Tests for the SSL backbone factory (pyhealth/models/pretrain/backbones.py).

Every backbone must satisfy the contract ``emb, cls = backbone(x, mask)`` with
``emb:(B,S,E)``, ``cls:(B,E)``, and padded positions must not contaminate the
outputs of valid positions.
"""
import pytest
import torch

from pyhealth.models.pretrain.backbones import ARCH_CHOICES, build_backbone

B, S, E = 3, 10, 16


def _make(arch):
    kw = dict(feature_size=E, num_layers=2, heads=4, dropout=0.1)
    if arch == "jamba":
        kw.update(num_transformer_layers=1, num_mamba_layers=1)
    return build_backbone(arch, **kw)


@pytest.mark.parametrize("arch", ARCH_CHOICES)
def test_backbone_contract_shapes(arch):
    m = _make(arch)
    x = torch.randn(B, S, E)
    mask = torch.ones(B, S)
    mask[0, 6:] = 0
    emb, cls = m(x, mask)
    assert emb.shape == (B, S, E)
    assert cls.shape == (B, E)


@pytest.mark.parametrize("arch", ARCH_CHOICES)
def test_backbone_backward(arch):
    m = _make(arch)
    x = torch.randn(B, S, E, requires_grad=True)
    emb, _ = m(x, torch.ones(B, S))
    emb.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("arch", ARCH_CHOICES)
def test_padding_does_not_leak_into_valid(arch):
    """Perturbing padded positions must leave valid-position outputs unchanged."""
    m = _make(arch).eval()
    x = torch.randn(B, S, E)
    mask = torch.ones(B, S)
    mask[0, 6:] = 0
    with torch.no_grad():
        emb, _ = m(x, mask)
        x2 = x.clone()
        x2[0, 6:] = 999.0
        emb2, _ = m(x2, mask)
    assert torch.allclose(emb[0, :6], emb2[0, :6], atol=1e-5)


def test_unknown_arch_raises():
    with pytest.raises(ValueError):
        build_backbone("gru", feature_size=E)


def test_transformer_rope_variant():
    m = build_backbone("transformer", feature_size=E, num_layers=1, heads=2, use_rope=True)
    emb, cls = m(torch.randn(B, S, E), torch.ones(B, S))
    assert emb.shape == (B, S, E) and cls.shape == (B, E)


def _make_stagenet_dataset(n_codes=6):
    from pyhealth.datasets import create_sample_dataset

    samples = [
        {"patient_id": "p0", "visit_id": "v0",
         "codes": ([float(i) for i in range(n_codes)], [f"c{i}" for i in range(n_codes)]), "label": 1},
        {"patient_id": "p1", "visit_id": "v1", "codes": ([0.0, 1.0], ["c0", "c1"]), "label": 0},
    ]
    return create_sample_dataset(
        samples, input_schema={"codes": "stagenet"}, output_schema={"label": "binary"},
        dataset_name="test_pretrain_backbones",
    )


@pytest.mark.parametrize("arch", ARCH_CHOICES)
def test_mae_accepts_every_backbone(arch):
    """Each backbone must plug into an SSL method and produce a finite,
    differentiable loss with a per-modality loss_dict."""
    from pyhealth.datasets import get_dataloader
    from pyhealth.models import UnifiedMultimodalEmbeddingModel
    from pyhealth.models.pretrain import MultimodalMaskedAutoencoder

    dim = 32
    ds = _make_stagenet_dataset(n_codes=6)
    batch = next(iter(get_dataloader(ds, batch_size=2, shuffle=False)))
    unified = UnifiedMultimodalEmbeddingModel(processors=ds.input_processors, embedding_dim=dim)
    kw = dict(feature_size=dim, num_layers=2, heads=4, dropout=0.0)
    if arch == "jamba":
        kw.update(num_transformer_layers=1, num_mamba_layers=1)
    model = MultimodalMaskedAutoencoder(
        embedding_model=unified, backbone=build_backbone(arch, **kw),
        decoder_layers=2, decoder_heads=4, decoder_dim=dim, mask_ratio=0.5,
    )
    model.feature_keys = list(ds.input_processors.keys())
    model.input_processors = ds.input_processors
    out = model(**batch)
    assert torch.isfinite(out["loss"]).item()
    assert "total" in out["loss_dict"]
    out["loss"].backward()
