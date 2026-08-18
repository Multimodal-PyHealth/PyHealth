"""Unit tests for self-supervised pretraining models.

Run with:
    TOKENIZERS_PARALLELISM=false pytest tests/test_pretrain.py -v
"""

import sys
import tempfile
from pathlib import Path

import torch


def _make_code_dataset_and_batch(batch_size=2, seq_len=5):
    """Build a minimal SampleDataset with one StageNetProcessor field."""
    from pyhealth.datasets import create_sample_dataset, get_dataloader

    codes_p0 = [f"c{i}" for i in range(seq_len)]
    times_p0 = [float(i) for i in range(seq_len)]
    codes_p1 = [f"c{i}" for i in range(2)]
    times_p1 = [0.0, 1.0]

    samples = [
        {
            "patient_id": "p0",
            "visit_id": "v0",
            "codes": (times_p0, codes_p0),
            "label": 1,
        },
        {
            "patient_id": "p1",
            "visit_id": "v1",
            "codes": (times_p1, codes_p1),
            "label": 0,
        },
    ]
    dataset = create_sample_dataset(
        samples,
        input_schema={"codes": "stagenet"},
        output_schema={"label": "binary"},
        dataset_name="test_pretrain",
    )
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    return dataset, batch


def test_unified_mask_generator_random():
    from pyhealth.models.pretrain import UnifiedMaskGenerator

    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
    gen = UnifiedMaskGenerator(mask_ratio=0.5, strategy="random")
    out = gen(mask)
    assert out.shape == mask.shape
    assert out.dtype == torch.bool
    # Padding should never be masked.
    assert not out[0, 4].item()
    assert not out[1, 2:].any().item()


def test_unified_mask_generator_block():
    from pyhealth.models.pretrain import UnifiedMaskGenerator

    mask = torch.ones(2, 40)
    gen = UnifiedMaskGenerator(mask_ratio=0.25, strategy="block", min_block_len=3, max_block_len=6)
    out = gen(mask)
    assert out.shape == mask.shape
    # Roughly the requested ratio should be masked; exact count depends on
    # block boundaries and trimming.
    for b in range(out.shape[0]):
        n_masked = out[b].sum().item()
        assert 0 < n_masked <= int(mask.shape[1] * 0.5)


def test_mae_forward_loss():
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.models.pretrain import MultimodalMaskedAutoencoder
    from pyhealth.models.transformer import TransformerLayer

    dataset, batch = _make_code_dataset_and_batch()
    unified = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors,
        embedding_dim=32,
    )
    backbone = TransformerLayer(feature_size=32, heads=2, dropout=0.0, num_layers=1)
    model = MultimodalMaskedAutoencoder(
        embedding_model=unified,
        backbone=backbone,
        decoder_layers=1,
        decoder_heads=2,
        mask_ratio=0.5,
    )
    model.feature_keys = list(dataset.input_processors.keys())
    model.input_processors = dataset.input_processors

    out = model(**batch)
    assert "loss" in out
    assert out["loss"].numel() == 1
    assert out["loss"].item() >= 0.0
    assert "pred" in out and "target" in out and "mask_token" in out
    assert out["pred"].shape == out["target"].shape
    out["loss"].backward()


def test_simmim_forward_loss():
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.models.pretrain import MultimodalSimMIM
    from pyhealth.models.transformer import TransformerLayer

    dataset, batch = _make_code_dataset_and_batch()
    unified = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors,
        embedding_dim=32,
    )
    backbone = TransformerLayer(feature_size=32, heads=2, dropout=0.0, num_layers=1)
    model = MultimodalSimMIM(
        embedding_model=unified,
        backbone=backbone,
        mask_ratio=0.5,
    )
    model.feature_keys = list(dataset.input_processors.keys())
    model.input_processors = dataset.input_processors

    out = model(**batch)
    assert "loss" in out
    assert out["loss"].item() >= 0.0
    out["loss"].backward()


def test_ijepa_forward_loss():
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.models.pretrain import MultimodalIJEPA
    from pyhealth.models.transformer import TransformerLayer

    dataset, batch = _make_code_dataset_and_batch(seq_len=12)
    unified = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors,
        embedding_dim=32,
    )
    backbone = TransformerLayer(feature_size=32, heads=2, dropout=0.0, num_layers=1)
    model = MultimodalIJEPA(
        embedding_model=unified,
        context_encoder=backbone,
        predictor_layers=1,
        predictor_heads=2,
        predictor_dim=32,
        num_target_blocks=2,
        target_block_len=2,
    )
    model.feature_keys = list(dataset.input_processors.keys())
    model.input_processors = dataset.input_processors

    out = model(**batch)
    assert "loss" in out
    assert out["loss"].item() >= 0.0
    # Target encoder should be an EMA copy and have no gradients.
    for p in model.target_encoder.parameters():
        assert p.requires_grad is False
    out["loss"].backward()


def test_rope_transformer_layer():
    from pyhealth.models.pretrain.rope import RoPETransformerLayer

    layer = RoPETransformerLayer(feature_size=32, heads=2, dropout=0.0, num_layers=1)
    x = torch.randn(2, 10, 32)
    mask = torch.ones(2, 10)
    out, cls = layer(x, mask)
    assert out.shape == (2, 10, 32)
    assert cls.shape == (2, 32)
    out.mean().backward()


def test_rope_extrapolation():
    from pyhealth.models.pretrain.rope import RotaryPositionEmbedding

    rope = RotaryPositionEmbedding(dim=16, max_seq_len=128, scaling_factor=2.0)
    x = torch.randn(1, 200, 16)
    out = rope(x, seq_len=200)
    assert out.shape == x.shape


def test_load_pretrained_into_downstream_transformer():
    """Save a pretraining checkpoint and load it into a supervised Transformer."""
    import tempfile
    from pyhealth.models import Transformer
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.models.pretrain import MultimodalSimMIM
    from pyhealth.models.transformer import TransformerLayer

    dataset, batch = _make_code_dataset_and_batch()
    unified = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors,
        embedding_dim=32,
    )
    backbone = TransformerLayer(feature_size=32, heads=2, dropout=0.0, num_layers=1)
    pretrain_model = MultimodalSimMIM(
        embedding_model=unified,
        backbone=backbone,
        mask_ratio=0.5,
    )
    pretrain_model.feature_keys = list(dataset.input_processors.keys())
    pretrain_model.input_processors = dataset.input_processors

    out = pretrain_model(**batch)
    out["loss"].backward()

    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        ckpt_path = f.name
    torch.save(pretrain_model.state_dict(), ckpt_path)

    # Build a supervised Transformer with the same unified embedding.
    downstream = Transformer(
        dataset=dataset,
        embedding_dim=32,
        heads=2,
        num_layers=1,
        unified_embedding=unified,
    )

    # The transfer goes through the guarded loader on BaseModel, which maps the
    # architecture-specific backbone name and requires full coverage before it
    # calls strict=False for the absent classification head.
    report = downstream.load_pretrained_state_dict(torch.load(ckpt_path))
    assert report["backbone_matched"] == report["backbone_target"], report
    assert report["backbone_matched"] > 0, report

    # Forward should still work.
    out2 = downstream(**batch)
    assert "loss" in out2
    out2["loss"].backward()

    import os
    os.remove(ckpt_path)


def test_ijepa_blocks_not_dropped_and_distinct():
    """Regression: base I-JEPA must predict every target position (no dropped
    blocks, no N x N broadcast) with distinct per-position predictions."""
    import torch
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.models.pretrain import MultimodalIJEPA
    from pyhealth.models.transformer import TransformerLayer

    torch.manual_seed(0)
    dataset, batch = _make_code_dataset_and_batch(seq_len=24)
    unified = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors, embedding_dim=32
    )
    backbone = TransformerLayer(feature_size=32, heads=2, dropout=0.0, num_layers=1)
    model = MultimodalIJEPA(
        embedding_model=unified,
        context_encoder=backbone,
        predictor_layers=1,
        predictor_heads=2,
        predictor_dim=32,
        num_target_blocks=3,
        target_block_len=2,
    )
    model.feature_keys = list(dataset.input_processors.keys())
    model.input_processors = dataset.input_processors
    model.eval()

    out = model(**batch)
    pred = out["context_pred"]
    tgt = out["target_embs"]
    # Per-position (N, E), NOT an (N, N, E) broadcast.
    assert pred.ndim == 2 and tgt.ndim == 2
    assert pred.shape == tgt.shape
    # Number of predicted positions equals number of target positions (nothing
    # silently dropped).
    assert pred.shape[0] == int(out["target_mask"].sum().item())
    assert out["loss"].item() > 0.0
    import itertools

    dup = sum(
        1
        for i, j in itertools.combinations(range(pred.shape[0]), 2)
        if torch.allclose(pred[i], pred[j], atol=1e-6)
    )
    assert dup == 0


def test_ijepa_single_block_nonzero_loss():
    """Regression: a single contiguous target block must still produce a real
    (nonzero, gradient-bearing) loss, not the empty-fallback no-op."""
    import torch
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.models.pretrain import MultimodalIJEPA
    from pyhealth.models.transformer import TransformerLayer

    torch.manual_seed(0)
    dataset, batch = _make_code_dataset_and_batch(seq_len=16)
    unified = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors, embedding_dim=32
    )
    backbone = TransformerLayer(feature_size=32, heads=2, dropout=0.0, num_layers=1)
    model = MultimodalIJEPA(
        embedding_model=unified,
        context_encoder=backbone,
        predictor_layers=1,
        predictor_heads=2,
        predictor_dim=32,
        num_target_blocks=1,
        target_block_len=3,
    )
    model.feature_keys = list(dataset.input_processors.keys())
    model.input_processors = dataset.input_processors

    out = model(**batch)
    assert out["loss"].item() > 0.0
    out["loss"].backward()
    # The predictor must have received gradient (loss is graph-connected).
    assert any(p.grad is not None for p in model.predictor.parameters())


def test_block_mask_preserves_contiguity():
    """Regression: block masking must yield a small number of contiguous runs,
    not shattered single positions (the random-trim bug)."""
    import torch
    from pyhealth.models.pretrain import UnifiedMaskGenerator

    gen = UnifiedMaskGenerator(mask_ratio=0.5, strategy="block", min_block_len=3, max_block_len=12)
    total_runs = 0
    trials = 30
    torch.manual_seed(0)
    for _ in range(trials):
        out = gen(torch.ones(1, 50))[0]
        # Count contiguous runs of True.
        runs = 0
        prev = False
        for v in out.tolist():
            if v and not prev:
                runs += 1
            prev = bool(v)
        total_runs += runs
    avg_runs = total_runs / trials
    # With proper contiguous spans this is ~2-3; the buggy random-trim gave ~4.4+.
    assert avg_runs < 4.0, f"avg contiguous runs {avg_runs} too high (spans shattered)"


def test_random_mask_floor():
    """Regression: every sample with valid positions masks at least one."""
    import torch
    from pyhealth.models.pretrain import UnifiedMaskGenerator

    gen = UnifiedMaskGenerator(mask_ratio=0.3, strategy="random")
    torch.manual_seed(0)
    valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0]]).float()
    for _ in range(200):
        out = gen(valid)
        # No padding masked.
        assert not (out & (valid == 0)).any()
        # Every row with valid positions has >= 1 masked.
        assert (out.any(dim=1) | (valid.sum(dim=1) == 0)).all()


def test_vjepa_forward_loss():
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.models.pretrain import MultimodalVJEPA
    from pyhealth.models.transformer import TransformerLayer

    dataset, batch = _make_code_dataset_and_batch(seq_len=20)
    unified = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors,
        embedding_dim=32,
    )
    backbone = TransformerLayer(feature_size=32, heads=2, dropout=0.0, num_layers=1)
    model = MultimodalVJEPA(
        embedding_model=unified,
        context_encoder=backbone,
        predictor_layers=1,
        predictor_heads=2,
        predictor_dim=32,
        num_target_blocks=3,
        target_block_scales=(2, 4),
    )
    model.feature_keys = list(dataset.input_processors.keys())
    model.input_processors = dataset.input_processors

    out = model(**batch)
    assert "loss" in out and out["loss"].item() >= 0.0
    assert "scale_ids" in out
    # Target encoder must be a frozen EMA copy.
    for p in model.target_encoder.parameters():
        assert p.requires_grad is False
    out["loss"].backward()
    # Predictor + scale embedding + context encoder receive gradients.
    assert model.scale_embed.weight.grad is not None
    assert any(p.grad is not None for p in model.context_encoder.parameters())
    # Target encoder receives NO gradient.
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_vjepa_predictor_distinguishes_blocks():
    """The V-JEPA fix: distinct target positions get distinct predictions.

    The base I-JEPA predictor used a single shared query, so multiple target
    blocks in one sample produced identical predictions.  V-JEPA's location +
    scale aware queries must break that degeneracy.
    """
    import torch
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.models.pretrain import MultimodalVJEPA
    from pyhealth.models.transformer import TransformerLayer

    torch.manual_seed(0)
    dataset, batch = _make_code_dataset_and_batch(seq_len=24)
    unified = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors, embedding_dim=32
    )
    backbone = TransformerLayer(feature_size=32, heads=2, dropout=0.0, num_layers=1)
    model = MultimodalVJEPA(
        embedding_model=unified,
        context_encoder=backbone,
        predictor_layers=1,
        predictor_heads=2,
        predictor_dim=32,
        num_target_blocks=4,
        target_block_scales=(2, 3),
    )
    model.feature_keys = list(dataset.input_processors.keys())
    model.input_processors = dataset.input_processors
    model.eval()

    out = model(**batch)
    preds = out["context_pred"]  # (N_target_positions, E)
    assert preds.shape[0] >= 4
    # No two predicted positions should be exactly identical (degeneracy check).
    import itertools

    dup = sum(
        1
        for i, j in itertools.combinations(range(preds.shape[0]), 2)
        if torch.allclose(preds[i], preds[j], atol=1e-6)
    )
    assert dup == 0, f"found {dup} identical predictions (predictor degenerate)"


def test_vjepa_multiscale_sampling():
    import torch
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.models.pretrain import MultimodalVJEPA
    from pyhealth.models.transformer import TransformerLayer

    torch.manual_seed(1)
    dataset, batch = _make_code_dataset_and_batch(seq_len=40)
    unified = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors, embedding_dim=16
    )
    backbone = TransformerLayer(feature_size=16, heads=2, dropout=0.0, num_layers=1)
    model = MultimodalVJEPA(
        embedding_model=unified,
        context_encoder=backbone,
        predictor_dim=16,
        predictor_layers=1,
        predictor_heads=2,
        num_target_blocks=5,
        target_block_scales=(2, 4, 8),
        min_context_len=4,
    )
    # Sample directly on a long all-valid sequence to exercise multi-scale draws.
    event_mask = torch.ones(4, 40)
    target_mask, context_mask, scale_ids = model._sample_multiscale_blocks(event_mask)
    # Context always preserved.
    assert (context_mask.sum(1) >= model.min_context_len).all()
    # Targets and context are disjoint and cover only valid positions.
    assert not (target_mask & context_mask).any()
    # Over the batch, more than one scale index is used.
    used_scales = scale_ids[target_mask].unique().numel()
    assert used_scales >= 2


def test_rope_inv_freq_spectrum():
    """Regression for the RoPE frequency bug (all freqs had collapsed to 1/base)."""
    import torch
    from pyhealth.models.pretrain.rope import RotaryPositionEmbedding

    dim = 32
    rope = RotaryPositionEmbedding(dim=dim, max_seq_len=128)
    reference = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
    assert torch.allclose(rope.inv_freq, reference, atol=1e-6)
    # Frequencies must be distinct (not collapsed to a single value).
    assert rope.inv_freq.unique().numel() == dim // 2


def test_ema_schedule_advances():
    """Regression: set_ema_decay must move the momentum toward target_ema_end."""
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.models.pretrain import MultimodalIJEPA
    from pyhealth.models.transformer import TransformerLayer

    dataset, _ = _make_code_dataset_and_batch()
    unified = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors, embedding_dim=16
    )
    backbone = TransformerLayer(feature_size=16, heads=2, dropout=0.0, num_layers=1)
    model = MultimodalIJEPA(
        embedding_model=unified,
        context_encoder=backbone,
        predictor_dim=16,
        target_ema_decay=0.99,
        target_ema_end=1.0,
    )
    start = model.target_ema_decay
    model.set_ema_decay(0, 100)
    mid = model.target_ema_decay
    model.set_ema_decay(100, 100)
    end = model.target_ema_decay
    assert abs(start - 0.99) < 1e-6
    assert mid >= start
    assert abs(end - 1.0) < 1e-6


def test_mae_simmim_token_target():
    """MAE/SimMIM default to the content-only ('token') target, not the
    composed sequence (which leaks recoverable time/type into the loss)."""
    import torch
    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.models.pretrain import MultimodalMaskedAutoencoder, MultimodalSimMIM
    from pyhealth.models.transformer import TransformerLayer

    dataset, batch = _make_code_dataset_and_batch(seq_len=8)
    unified = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors, embedding_dim=32
    )
    backbone = TransformerLayer(feature_size=32, heads=2, dropout=0.0, num_layers=1)

    for Model, kwargs in [
        (MultimodalMaskedAutoencoder, dict(decoder_layers=1, decoder_heads=2)),
        (MultimodalSimMIM, dict()),
    ]:
        model = Model(embedding_model=unified, backbone=backbone, mask_ratio=0.5, **kwargs)
        assert model.target == "token"  # new default
        model.feature_keys = list(dataset.input_processors.keys())
        model.input_processors = dataset.input_processors

        from pyhealth.models.pretrain.utils import build_unified_inputs_from_batch
        inputs = build_unified_inputs_from_batch(
            dataset.input_processors, list(dataset.input_processors.keys()), batch
        )
        emb_out = unified(inputs)
        assert "token_emb" in emb_out
        # token_emb (content only) must differ from the composed sequence.
        assert not torch.allclose(emb_out["token_emb"], emb_out["sequence"])

        out = model(**batch)
        # The returned target is the content-only token_emb, not the sequence.
        assert torch.allclose(out["target"], emb_out["token_emb"], atol=1e-5)
        assert not torch.allclose(out["target"], emb_out["sequence"])
        assert out["loss"].item() >= 0.0
        # Target MUST be detached (otherwise the model shrinks it -> collapse).
        assert not out["target"].requires_grad
        out["loss"].backward()
        # But the embedding model must still train via the encoder INPUT path.
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.embedding_model.parameters()
        )


def test_per_modality_mae_decoder():
    from pyhealth.models.pretrain import PerModalityMAEDecoder

    decoder = PerModalityMAEDecoder(
        embedding_dim=32,
        output_specs={0: ("numeric", 10), 1: ("code", 50)},
    )
    emb = torch.randn(4, 8, 32)
    type_ids = torch.tensor([[0, 0, 1, 1, 0, 1, 0, 1]]).expand(4, -1)
    out = decoder(emb, type_ids)
    assert 0 in out and 1 in out
    assert out[0].shape == (out[0].shape[0], 10)
    assert out[1].shape == (out[1].shape[0], 50)
