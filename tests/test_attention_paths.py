"""The fused and explicit attention paths must agree.

The explicit path materialises several ``(B, H, S, S)`` tensors for each layer
and exists only for interpretability. Ordinary training uses fused SDPA. If the
two paths disagree, an interpretability pass describes a model that training
never produced.
"""

from __future__ import annotations

import pytest
import torch

from pyhealth.models.transformer import TransformerLayer


def _layer(seed: int = 0) -> TransformerLayer:
    torch.manual_seed(seed)
    return TransformerLayer(feature_size=128, heads=4, num_layers=2).eval()


def test_fused_and_explicit_paths_agree_without_padding():
    layer = _layer()
    x = torch.randn(4, 32, 128, requires_grad=True)
    mask = torch.ones(4, 32)

    fused, _ = layer(x, mask, register_hook=False)
    explicit, _ = layer(x, mask, register_hook=True)

    assert torch.allclose(fused, explicit, atol=1e-5)


def test_fused_and_explicit_paths_agree_with_padding():
    """Padding is where a mask convention error shows."""
    layer = _layer()
    x = torch.randn(4, 32, 128, requires_grad=True)
    mask = torch.cat([torch.ones(4, 20), torch.zeros(4, 12)], dim=1)

    fused, _ = layer(x, mask, register_hook=False)
    explicit, _ = layer(x, mask, register_hook=True)

    assert torch.allclose(fused, explicit, atol=1e-5)


def test_fused_path_does_not_retain_the_attention_map():
    """The map was kept in memory even when no caller read it."""
    layer = _layer()
    x = torch.randn(2, 16, 128)
    mask = torch.ones(2, 16)

    with torch.no_grad():
        layer(x, mask, register_hook=False)

    for module in layer.modules():
        if hasattr(module, "attn_map"):
            assert module.attn_map is None


def test_explicit_path_still_supplies_the_attention_map():
    layer = _layer()
    x = torch.randn(2, 16, 128, requires_grad=True)
    mask = torch.ones(2, 16)

    layer(x, mask, register_hook=True)

    maps = [m.attn_map for m in layer.modules() if hasattr(m, "attn_map")]
    assert any(m is not None for m in maps)


def test_mask_fill_value_is_representable_in_half_precision():
    """``-1e9`` is outside the fp16 range, so it cannot be the fill value.

    Under fp16 autocast the previous constant overflows. The fill value must come
    from the dtype.
    """
    from pyhealth.models.transformer import Attention

    attention = Attention()
    query = torch.randn(1, 1, 4, 8, dtype=torch.float16)
    key = torch.randn(1, 1, 4, 8, dtype=torch.float16)
    value = torch.randn(1, 1, 4, 8, dtype=torch.float16)
    mask = torch.tensor([[[[1, 1, 0, 0]]]])

    out, weights = attention(query, key, value, mask=mask)

    assert torch.isfinite(out).all()
    assert torch.isfinite(weights).all()
    # Masked positions must receive exactly zero weight.
    assert weights[..., 2:].abs().max().item() == 0.0


def test_gradients_flow_through_the_fused_path():
    layer = TransformerLayer(feature_size=64, heads=4, num_layers=2)
    x = torch.randn(2, 16, 64)
    out, _ = layer(x, torch.ones(2, 16))
    out.sum().backward()

    total = sum(
        p.grad.abs().sum().item() for p in layer.parameters() if p.grad is not None
    )
    assert total > 0


def test_amp_dtype_is_validated_not_silently_coerced():
    """Any value other than "bf16" previously selected fp16 with no message.

    fp16 also needs a GradScaler, so the silent path changed gradient behaviour
    as well as precision.
    """
    from pyhealth.trainer import resolve_amp_dtype

    assert resolve_amp_dtype("bf16") is torch.bfloat16
    assert resolve_amp_dtype("fp16") is torch.float16
    # The long spellings used to fall through to fp16.
    assert resolve_amp_dtype("bfloat16") is torch.bfloat16
    assert resolve_amp_dtype("float16") is torch.float16

    for bad in ("bfloat_16", "f16", "int8", ""):
        with pytest.raises(ValueError):
            resolve_amp_dtype(bad)
