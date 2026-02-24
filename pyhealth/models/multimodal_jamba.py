# =============================================================================
# Contributors: Joshua Steier
# Paper Title: Jamba: A Hybrid Transformer-Mamba Language Model (AI21 Labs)
# Paper Link: https://arxiv.org/abs/2403.19887
# Additional Reference: Multimodal Bottleneck Transformer (arXiv:2412.16178)
# Description: Multimodal Jamba-EHR model combining unified multimodal
#     embeddings with the JambaLayer hybrid Transformer-Mamba backbone
#     (from pyhealth.models.jamba_ehr) for patient-level prediction tasks.
#     Supports heterogeneous EHR modalities (images, text, timeseries,
#     codes) with temporal and modality-type embeddings, missing modality
#     handling, and configurable Transformer/Mamba layer interleaving.
#
#     Builds on JambaEHR (pyhealth.models.jamba_ehr) by adding:
#       - UnifiedMultimodalEmbedding for fusing heterogeneous modalities
#       - Sinusoidal temporal embeddings for observation timestamps
#       - Learnable modality-type embeddings
#       - Learnable missing-modality tokens
# =============================================================================

"""Multimodal Jamba-EHR model for heterogeneous patient data.

This module provides:
- ``ModalityType``: Enum for supported modality types.
- ``UnifiedMultimodalEmbedding``: Combines per-modality embeddings with
  temporal and modality-type encodings into a single token sequence.
- ``MultimodalJambaEHR``: End-to-end model that passes unified embeddings
  through a ``JambaLayer`` backbone for patient-level prediction.

Example:
    >>> from pyhealth.models.multimodal_jamba import (
    ...     MultimodalJambaEHR, ModalityType,
    ... )
    >>> model = MultimodalJambaEHR(
    ...     embedding_dim=128,
    ...     num_transformer_layers=2,
    ...     num_mamba_layers=6,
    ...     heads=4,
    ...     num_classes=2,
    ... )
    >>> inputs = {
    ...     ModalityType.IMAGE: (
    ...         torch.randn(2, 5, 128), torch.rand(2, 5),
    ...     ),
    ...     ModalityType.TEXT: (
    ...         torch.randn(2, 10, 128), torch.rand(2, 10),
    ...     ),
    ... }
    >>> out = model(inputs, labels=torch.tensor([0, 1]))
    >>> out["loss"].backward()
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.models.jamba_ehr import JambaLayer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class ModalityType(IntEnum):
    """Integer codes for each supported modality."""
    IMAGE = 0
    TEXT = 1
    TIMESERIES = 2
    SEQUENCE = 3


NUM_MODALITIES = len(ModalityType)


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Map scalar timestamps to dense vectors via sinusoidal encoding.

    Args:
        embed_dim: Dimensionality of the output embedding.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        half = (embed_dim + 1) // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode timestamps.

        Args:
            t: Tensor of shape ``(B, S)`` with scalar timestamps.

        Returns:
            Tensor of shape ``(B, S, embed_dim)``.
        """
        t = t.unsqueeze(-1).float()
        args = t * self.freqs
        out = torch.cat([args.sin(), args.cos()], dim=-1)
        return out[:, :, :self.embed_dim]


# ---------------------------------------------------------------------------
# Unified Multimodal Embedding
# ---------------------------------------------------------------------------

class UnifiedMultimodalEmbedding(nn.Module):
    """Combine per-modality embeddings into a single token sequence.

    For each modality present in a sample, this module:
    1. Assumes value embeddings are already projected to
       ``embed_dim`` (E').
    2. Adds sinusoidal **temporal embeddings** from observation
       timestamps.
    3. Adds learnable **modality-type embeddings** to distinguish
       modalities.
    4. Handles **missing modalities** with a learnable missing token.
    5. Concatenates all tokens across modalities into one sequence.

    Args:
        embed_dim: Shared embedding dimensionality E'.
        max_modalities: Number of distinct modality types.
        use_cls_token: Whether to prepend a learnable [CLS] token.
    """

    def __init__(
        self,
        embed_dim: int,
        max_modalities: int = NUM_MODALITIES,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        self.time_embed = SinusoidalTimeEmbedding(embed_dim)
        self.modality_embed = nn.Embedding(max_modalities, embed_dim)

        self.missing_tokens = nn.Parameter(
            torch.randn(max_modalities, 1, embed_dim) * 0.02
        )

        if use_cls_token:
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, embed_dim) * 0.02
            )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        modality_inputs: Dict[
            ModalityType,
            Tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse modality embeddings into a unified sequence.

        Args:
            modality_inputs: Dict mapping ``ModalityType`` to
                ``(values (B, S_i, E'), times (B, S_i))``.
                Omit a key to indicate missing modality.

        Returns:
            Tuple of:
                - ``embeddings``: ``(B, total_tokens, E')``
                - ``mask``: ``(B, total_tokens)`` float mask
                  (1.0 = valid, 0.0 = pad). Float to match
                  JambaLayer's mask convention.
        """
        B = None
        device = None
        all_embeds: List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []

        for mod_type in ModalityType:
            inp = modality_inputs.get(mod_type, None)
            if inp is not None:
                B = inp[0].shape[0]
                device = inp[0].device
                break

        if B is None:
            raise ValueError(
                "At least one modality must be present."
            )

        for mod_type in ModalityType:
            inp = modality_inputs.get(mod_type, None)

            if inp is None:
                missing = self.missing_tokens[
                    mod_type.value
                ].expand(B, -1, -1)
                mod_embed = self.modality_embed(
                    torch.tensor(
                        [mod_type.value], device=device
                    )
                ).unsqueeze(0).expand(B, 1, -1)
                all_embeds.append(missing + mod_embed)
                all_masks.append(
                    torch.ones(
                        B, 1, dtype=torch.float, device=device
                    )
                )
            else:
                values, times = inp
                t_emb = self.time_embed(times)
                mod_id = torch.full(
                    (B, values.shape[1]),
                    mod_type.value,
                    dtype=torch.long,
                    device=device,
                )
                m_emb = self.modality_embed(mod_id)
                combined = values + t_emb + m_emb
                mask = torch.ones(
                    B,
                    values.shape[1],
                    dtype=torch.float,
                    device=device,
                )
                all_embeds.append(combined)
                all_masks.append(mask)

        embeddings = torch.cat(all_embeds, dim=1)
        mask = torch.cat(all_masks, dim=1)

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            cls_mask = torch.ones(
                B, 1, dtype=torch.float, device=device
            )
            embeddings = torch.cat([cls, embeddings], dim=1)
            mask = torch.cat([cls_mask, mask], dim=1)

        return self.norm(embeddings), mask


# ---------------------------------------------------------------------------
# MultimodalJambaEHR
# ---------------------------------------------------------------------------

class MultimodalJambaEHR(nn.Module):
    """Multimodal Jamba-EHR model for patient-level prediction.

    Combines ``UnifiedMultimodalEmbedding`` with ``JambaLayer`` (from
    ``pyhealth.models.jamba_ehr``) and a linear classification head.

    Args:
        embedding_dim: Shared embedding dimensionality E'.
        num_transformer_layers: Transformer layers in JambaLayer.
        num_mamba_layers: Mamba layers in JambaLayer.
        heads: Number of attention heads.
        state_size: Mamba SSM state dimension.
        conv_kernel: Mamba causal conv kernel size.
        num_classes: Number of output classes.
        dropout: Dropout rate.
        use_cls_token: Whether to use a [CLS] token.
        pool: ``"last"`` (JambaLayer's get_last_visit),
            ``"cls"`` (CLS token), or ``"mean"``.

    Example:
        >>> model = MultimodalJambaEHR(
        ...     embedding_dim=128, num_classes=2,
        ... )
        >>> inputs = {
        ...     ModalityType.IMAGE: (
        ...         torch.randn(2, 5, 128), torch.rand(2, 5),
        ...     ),
        ... }
        >>> out = model(inputs, labels=torch.tensor([0, 1]))
        >>> out["loss"].backward()
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_transformer_layers: int = 2,
        num_mamba_layers: int = 6,
        heads: int = 4,
        state_size: int = 16,
        conv_kernel: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
        use_cls_token: bool = True,
        pool: str = "last",
    ) -> None:
        super().__init__()
        self.pool = pool
        self.use_cls_token = use_cls_token

        self.unified_embedding = UnifiedMultimodalEmbedding(
            embed_dim=embedding_dim,
            use_cls_token=use_cls_token,
        )

        self.backbone = JambaLayer(
            feature_size=embedding_dim,
            num_transformer_layers=num_transformer_layers,
            num_mamba_layers=num_mamba_layers,
            heads=heads,
            dropout=dropout,
            state_size=state_size,
            conv_kernel=conv_kernel,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(
        self,
        modality_inputs: Dict[
            ModalityType,
            Optional[Tuple[torch.Tensor, torch.Tensor]],
        ],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            modality_inputs: Dict mapping ``ModalityType`` to
                ``(values (B, S_i, E'), times (B, S_i))`` or
                ``None``.
            labels: Optional ``(B,)`` integer class labels.

        Returns:
            Dict with ``logit``, ``y_prob``, and optionally
            ``y_true`` and ``loss``.
        """
        embeddings, mask = self.unified_embedding(modality_inputs)

        # JambaLayer returns (emb, cls_emb) where cls_emb
        # is from get_last_visit
        emb, cls_emb = self.backbone(embeddings, mask=mask)

        if self.pool == "cls" and self.use_cls_token:
            pooled = emb[:, 0, :]
        elif self.pool == "mean":
            mask_f = mask.unsqueeze(-1)
            pooled = (
                (emb * mask_f).sum(1)
                / mask_f.sum(1).clamp(min=1)
            )
        else:
            # "last" — JambaLayer's built-in get_last_visit
            pooled = cls_emb

        logits = self.head(pooled)
        y_prob = F.softmax(logits, dim=-1)

        result: Dict[str, torch.Tensor] = {
            "logit": logits,
            "y_prob": y_prob,
        }
        if labels is not None:
            result["y_true"] = labels
            result["loss"] = F.cross_entropy(logits, labels)

        return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    B, E = 4, 128

    def _test(name: str, inputs: dict, **kwargs):
        print(f"\n=== {name} ===")
        model = MultimodalJambaEHR(embedding_dim=E, **kwargs)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params:,}")
        labels = torch.randint(0, 2, (B,))
        out = model(inputs, labels=labels)
        print(
            f"  logit: {out['logit'].shape}, "
            f"loss: {out['loss'].item():.4f}"
        )
        out["loss"].backward()
        print("  backward OK")

    _test("All modalities (2T+6M)", {
        ModalityType.IMAGE: (
            torch.randn(B, 5, E), torch.rand(B, 5),
        ),
        ModalityType.TEXT: (
            torch.randn(B, 10, E), torch.rand(B, 10),
        ),
        ModalityType.TIMESERIES: (
            torch.randn(B, 20, E), torch.rand(B, 20),
        ),
        ModalityType.SEQUENCE: (
            torch.randn(B, 8, E), torch.rand(B, 8),
        ),
    })

    _test("EHR only", {
        ModalityType.TIMESERIES: (
            torch.randn(B, 20, E), torch.rand(B, 20),
        ),
        ModalityType.SEQUENCE: (
            torch.randn(B, 8, E), torch.rand(B, 8),
        ),
    })

    _test("Text only", {
        ModalityType.TEXT: (
            torch.randn(B, 15, E), torch.rand(B, 15),
        ),
    })

    _test("Pure Transformer (4T+0M)", {
        ModalityType.IMAGE: (
            torch.randn(B, 5, E), torch.rand(B, 5),
        ),
        ModalityType.TEXT: (
            torch.randn(B, 10, E), torch.rand(B, 10),
        ),
    }, num_transformer_layers=4, num_mamba_layers=0)

    _test("Pure Mamba (0T+4M)", {
        ModalityType.IMAGE: (
            torch.randn(B, 5, E), torch.rand(B, 5),
        ),
        ModalityType.TIMESERIES: (
            torch.randn(B, 20, E), torch.rand(B, 20),
        ),
    }, num_transformer_layers=0, num_mamba_layers=4)

    _test("CLS pooling", {
        ModalityType.TEXT: (
            torch.randn(B, 10, E), torch.rand(B, 10),
        ),
        ModalityType.SEQUENCE: (
            torch.randn(B, 8, E), torch.rand(B, 8),
        ),
    }, pool="cls")

    _test("Mean pooling", {
        ModalityType.TEXT: (
            torch.randn(B, 10, E), torch.rand(B, 10),
        ),
        ModalityType.SEQUENCE: (
            torch.randn(B, 8, E), torch.rand(B, 8),
        ),
    }, pool="mean")

    print("\n All smoke tests passed!")