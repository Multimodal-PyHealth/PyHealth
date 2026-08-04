"""Sequence-encoder backbones for SSL pretraining.

Every SSL method (MAE / SimMIM / I-JEPA / V-JEPA) injects a ``backbone``
(a.k.a. ``context_encoder``) that must satisfy the same contract as
:class:`pyhealth.models.transformer.TransformerLayer`::

    emb, cls = backbone(x, mask)
    # x:   (B, S, E)   input sequence
    # mask:(B, S)      1 = valid, 0 = pad   (may be float or bool)
    # emb: (B, S, E)   per-step encoded features
    # cls: (B, E)      pooled vector (unused by the SSL methods, kept for parity)

``build_backbone`` is the single place the training scripts and the Optuna
sweeps construct an encoder, so a new architecture only has to be added here.

Supported ``arch`` values:
    - ``"transformer"`` -> :class:`TransformerLayer` (or RoPE variant if ``use_rope``)
    - ``"jamba"``       -> :class:`JambaLayer` (interleaved attention + Mamba)
    - ``"mamba"``       -> :class:`MambaLayer` (this module; stacks ``MambaBlock``)

The Jamba layer already matches the contract; ``TransformerLayer`` does too.
``MambaBlock`` is a single ``forward(x) -> x`` residual block, so this module
wraps a stack of them into a mask-aware layer that returns ``(emb, cls)``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from pyhealth.models.ehrmamba import MambaBlock, RMSNorm
from pyhealth.models.jamba_ehr import JambaLayer
from pyhealth.models.transformer import TransformerLayer
from pyhealth.models.utils import get_last_visit

__all__ = ["MambaLayer", "build_backbone", "ARCH_CHOICES"]

ARCH_CHOICES = ("transformer", "jamba", "mamba")


class MambaLayer(nn.Module):
    """A stack of :class:`MambaBlock` layers exposing the standard backbone
    contract ``forward(x, mask) -> (emb, cls)``.

    Padded positions are zeroed on input.  Because ``MambaBlock`` is causal
    (left-padded conv + left-to-right SSM scan) and pad tokens sit at the end
    of each sequence, they cannot leak into earlier valid positions, so a
    single input masking is sufficient.

    Args:
        feature_size: hidden/embedding dimension ``E``.
        num_layers: number of stacked Mamba blocks. Default 2.
        dropout: dropout on the output features. Default 0.0.
        state_size: SSM state size per channel. Default 16.
        conv_kernel: causal conv kernel size inside each block. Default 4.
    """

    def __init__(
        self,
        feature_size: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        state_size: int = 16,
        conv_kernel: int = 4,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MambaBlock(d_model=feature_size, state_size=state_size, conv_kernel=conv_kernel)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(feature_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, register_hook: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        emb = self.dropout(x)
        cls = get_last_visit(emb, mask) if mask is not None else emb[:, -1, :]
        return emb, cls


def build_backbone(
    arch: str,
    feature_size: int,
    num_layers: int = 2,
    heads: int = 4,
    dropout: float = 0.1,
    *,
    # transformer
    use_rope: bool = False,
    rope_max_seq_len: int = 8192,
    rope_base: float = 10000.0,
    rope_scaling: float = 1.0,
    # mamba / jamba
    state_size: int = 16,
    conv_kernel: int = 4,
    # jamba layer mix (num_layers is ignored for jamba)
    num_transformer_layers: int = 1,
    num_mamba_layers: int = 1,
) -> nn.Module:
    """Construct an SSL encoder backbone satisfying the standard contract.

    ``arch`` is one of :data:`ARCH_CHOICES`.  ``feature_size`` (== embedding
    dim), ``num_layers`` and ``heads`` are the standardized-size knobs; the
    remaining kwargs are arch-specific and ignored where not applicable.
    """
    arch = arch.lower()
    if arch == "transformer":
        if use_rope:
            from pyhealth.models.pretrain.rope import RoPETransformerLayer

            return RoPETransformerLayer(
                feature_size=feature_size,
                heads=heads,
                dropout=dropout,
                num_layers=num_layers,
                rope_max_seq_len=rope_max_seq_len,
                rope_base=rope_base,
                rope_scaling=rope_scaling,
            )
        return TransformerLayer(
            feature_size=feature_size, heads=heads, dropout=dropout, num_layers=num_layers
        )
    if arch == "mamba":
        return MambaLayer(
            feature_size=feature_size,
            num_layers=num_layers,
            dropout=dropout,
            state_size=state_size,
            conv_kernel=conv_kernel,
        )
    if arch == "jamba":
        return JambaLayer(
            feature_size=feature_size,
            num_transformer_layers=num_transformer_layers,
            num_mamba_layers=num_mamba_layers,
            heads=heads,
            dropout=dropout,
            state_size=state_size,
            conv_kernel=conv_kernel,
        )
    raise ValueError(f"Unknown backbone arch '{arch}'. Choices: {ARCH_CHOICES}")
