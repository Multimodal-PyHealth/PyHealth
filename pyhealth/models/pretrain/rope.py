"""Rotary Position Embedding (RoPE) utilities.

RoPE encodes relative position by rotating query/key vectors in 2D subspaces.
It is especially useful for long clinical sequences because it generalizes to
lengths longer than those seen during training and supports extrapolation
techniques such as NTK-aware scaling or YaRN.

References:
    Jianlin Su et al., "RoFormer: Enhanced Transformer with Rotary Position
    Embedding", Neurocomputing 2024.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    """Rotary position embedding for sequences.

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Maximum sequence length for which to precompute angles.
        base: Base for the inverse frequency computation.  Default 10000.
        scaling_factor: Multiplicative scaling for sequence-length
            extrapolation (e.g., NTK-aware scaling).  Default 1.0.

    Shape:
        Input:  ``(..., seq_len, head_dim)``
        Output: ``(..., seq_len, head_dim)`` rotated by position.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor

        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin caches.
        self._update_cos_sin_cache(max_seq_len, device=inv_freq.device)

    def _compute_inv_freq(self) -> torch.Tensor:
        # Standard RoPE: inv_freq_i = 1 / base^(2i/dim).  NTK-aware extrapolation
        # rescales the base by scaling_factor^(dim/(dim-2)) so that longer
        # sequences interpolate smoothly (scaling_factor=1.0 -> vanilla RoPE).
        base = self.base
        if self.scaling_factor != 1.0:
            base = base * (self.scaling_factor ** (self.dim / (self.dim - 2)))
        exponent = torch.arange(0, self.dim, 2).float() / self.dim
        return 1.0 / (base ** exponent)

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device) -> None:
        if (
            hasattr(self, "cos_cached")
            and self.cos_cached.shape[1] >= seq_len  # axis 1 is the cached seq len
            and self.cos_cached.device == device
        ):
            return
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos()[None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, :], persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the last dimension by swapping pairs and negating."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """Apply rotary embedding to ``x`` of shape ``(..., seq_len, dim)``."""
        if seq_len is None:
            seq_len = x.shape[-2]
        self._update_cos_sin_cache(seq_len, device=x.device)
        cos = self.cos_cached[:, :seq_len, :]
        sin = self.sin_cached[:, :seq_len, :]
        return x * cos + self._rotate_half(x) * sin


class RoPEMultiHeadedAttention(nn.Module):
    """Multi-head attention with RoPE applied to Q and K.

    This is a drop-in replacement for
    :class:`pyhealth.models.transformer.MultiHeadedAttention`.

    Args:
        h: Number of attention heads.
        d_model: Model dimensionality (must be divisible by ``h``).
        dropout: Dropout probability on attention weights.
        rope_max_seq_len: Max sequence length for RoPE cache.
        rope_base: RoPE inverse-frequency base.
        rope_scaling: RoPE scaling factor for extrapolation.
    """

    def __init__(
        self,
        h: int,
        d_model: int,
        dropout: float = 0.1,
        rope_max_seq_len: int = 8192,
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
    ):
        super().__init__()
        if d_model % h != 0:
            raise ValueError("d_model must be divisible by h")
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(3)]
        )
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.rope = RotaryPositionEmbedding(
            dim=self.d_k,
            max_seq_len=rope_max_seq_len,
            base=rope_base,
            scaling_factor=rope_scaling,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        register_hook: bool = False,
    ) -> torch.Tensor:
        batch_size = query.size(0)
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        # Apply RoPE to Q and K; V is left unchanged.
        query = self.rope(query)
        key = self.rope(key)

        if mask is not None:
            # (B, S, S) -> (B, 1, S, S) so it broadcasts across heads (mirrors
            # MultiHeadedAttention).  Without this, multi-head crashes and the
            # B == h case silently mis-masks.
            mask = mask.unsqueeze(1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # dtype min keeps this safe under fp16 AMP (-1e9 overflows half).
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        p_attn = torch.softmax(scores, dim=-1)
        if mask is not None:
            p_attn = p_attn.masked_fill(mask == 0, 0)
        p_attn = self.dropout(p_attn)

        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


class RoPETransformerBlock(nn.Module):
    """Transformer block using :class:`RoPEMultiHeadedAttention`."""

    def __init__(
        self,
        hidden: int,
        attn_heads: int,
        dropout: float,
        rope_max_seq_len: int = 8192,
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
    ):
        super().__init__()
        from pyhealth.models.transformer import PositionwiseFeedForward, SublayerConnection

        self.attention = RoPEMultiHeadedAttention(
            h=attn_heads,
            d_model=hidden,
            dropout=dropout,
            rope_max_seq_len=rope_max_seq_len,
            rope_base=rope_base,
            rope_scaling=rope_scaling,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=4 * hidden, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        register_hook: bool = False,
    ) -> torch.Tensor:
        x = self.input_sublayer(
            x, lambda _x: self.attention(_x, _x, _x, mask=mask, register_hook=register_hook)
        )
        x = self.output_sublayer(x, lambda _x: self.feed_forward(_x, mask=mask))
        return self.dropout(x)


class RoPETransformerLayer(nn.Module):
    """RoPE-enabled Transformer layer matching the interface of
    :class:`pyhealth.models.transformer.TransformerLayer`.

    When used as the SSL backbone, the unified embedding model can optionally
    omit its sinusoidal time embedding (set ``time_embedding="none"`` once
    supported) because RoPE encodes relative position directly in attention.
    """

    def __init__(
        self,
        feature_size: int,
        heads: int = 1,
        dropout: float = 0.5,
        num_layers: int = 1,
        rope_max_seq_len: int = 8192,
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.heads = heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.transformer = nn.ModuleList(
            [
                RoPETransformerBlock(
                    hidden=feature_size,
                    attn_heads=heads,
                    dropout=dropout,
                    rope_max_seq_len=rope_max_seq_len,
                    rope_base=rope_base,
                    rope_scaling=rope_scaling,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        register_hook: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            mask = torch.einsum("ab,ac->abc", mask, mask)
        for transformer in self.transformer:
            x = transformer(x, mask, register_hook)
        return x, x[:, 0, :]
