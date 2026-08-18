"""Masked Autoencoder (MAE) for unified multimodal event sequences.

The model embeds heterogeneous temporal features with
:class:`UnifiedMultimodalEmbeddingModel`, masks a fraction of the unified event
sequence, encodes the visible subset with a Transformer backbone, then decodes
the masked positions with a lightweight transformer decoder.  The reconstruction
target is the original unified event embedding (time + type + modality token)
before masking.

References:
    Kaiming He et al., "Masked Autoencoders Are Scalable Vision Learners",
    CVPR 2022.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import torch
import torch.nn as nn

from ...processors.base_processor import ModalityType
from ..embedding.unified import UnifiedMultimodalEmbeddingModel
from ..transformer import TransformerLayer
from .masking import UnifiedMaskGenerator, apply_mask_token


class MultimodalMaskedAutoencoder(nn.Module):
    """True MAE pretrainer over a unified temporal event sequence.

    Args:
        embedding_model: The unified embedding model that produces a single
            temporally-sorted event sequence.
        backbone: Transformer (or other sequence encoder) backbone.
        decoder_layers: Number of layers in the lightweight reconstruction
            decoder.  Default 4.
        decoder_heads: Attention heads in the decoder.  Default 8.
        decoder_dim: Hidden dim of the decoder.  Defaults to ``embedding_dim``.
        mask_ratio: Fraction of valid positions to mask.  Default 0.5.
        mask_strategy: ``"random"`` or ``"block"``.
        per_modality_ratio: Optional override of ``mask_ratio`` per modality
            index.
        norm_pix_loss: If True, normalize targets by their mean/std before
            computing MSE (MAE default for images).
        target: Reconstruction target.  ``"token"`` (default) reconstructs the
            content-only per-event embedding (``token_emb``) before time/type
            are added — this is the recommended objective because the time/type
            components are largely recoverable from event position and otherwise
            dilute the content signal.  ``"unified"`` reconstructs the full
            composed ``sequence`` (legacy behaviour).

    Inputs:
        Same dict expected by ``embedding_model.forward``:
        ``{field: {"value": ..., "time": ..., "mask": ...}}``.

    Outputs:
        Dict with keys:
            - ``loss``: scalar MSE reconstruction loss.
            - ``loss_dict``: per-modality MSE breakdown.
            - ``pred``: ``(B, S, E)`` decoder predictions for masked positions.
            - ``target``: ``(B, S, E)`` reconstruction target.
            - ``mask_token``: ``(B, S)`` bool, True = masked.
    """

    def __init__(
        self,
        embedding_model: UnifiedMultimodalEmbeddingModel,
        backbone: nn.Module,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        decoder_dim: Optional[int] = None,
        mask_ratio: float = 0.5,
        mask_strategy: str = "random",
        per_modality_ratio: Optional[dict[int, float]] = None,
        norm_pix_loss: bool = False,
        target: str = "token",
    ):
        super().__init__()
        if target not in ("token", "unified"):
            raise ValueError(f"target must be 'token' or 'unified', got {target}")
        self.target = target
        self.embedding_model = embedding_model
        self.backbone = backbone
        self.embedding_dim = embedding_model.embedding_dim
        self.decoder_dim = decoder_dim or self.embedding_dim

        self.mask_generator = UnifiedMaskGenerator(
            mask_ratio=mask_ratio,
            strategy=mask_strategy,
            per_modality_ratio=per_modality_ratio,
        )
        self.mask_token = nn.Parameter(torch.zeros(self.embedding_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Transformer decoder: project encoder output to decoder dim, add mask
        # tokens, then run a shallow Transformer.
        self.encoder_to_decoder = (
            nn.Linear(self.embedding_dim, self.decoder_dim)
            if self.decoder_dim != self.embedding_dim
            else nn.Identity()
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, 1, self.decoder_dim)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.decoder = TransformerLayer(
            feature_size=self.decoder_dim,
            heads=decoder_heads,
            dropout=0.0,
            num_layers=decoder_layers,
        )
        self.decoder_norm = nn.LayerNorm(self.decoder_dim)
        self.prediction_head = nn.Linear(self.decoder_dim, self.embedding_dim)

        self.norm_pix_loss = norm_pix_loss

    def forward(
        self,
        inputs: Optional[dict[str, dict[str, torch.Tensor]]] = None,
        mask_token: Optional[torch.Tensor] = None,
        feature_keys: Optional[list[str]] = None,
        input_processors: Optional[dict[str, Any]] = None,
        **raw_kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        if inputs is None:
            from .utils import build_unified_inputs_from_batch
            fk = feature_keys or getattr(self, "feature_keys", None)
            ip = input_processors or getattr(self, "input_processors", None)
            if fk is None or ip is None:
                raise ValueError(
                    "When 'inputs' is not provided, both 'feature_keys' and "
                    "'input_processors' are required (either as arguments or "
                    "as attributes on the model)."
                )
            inputs = build_unified_inputs_from_batch(ip, fk, raw_kwargs)

        # 1. Embed all events.  The encoder input is the composed sequence
        # (content + time + type); the reconstruction target is content-only by
        # default (see ``target``).
        emb_out = self.embedding_model(inputs)
        sequence = emb_out["sequence"]            # (B, S, E) encoder input
        target = self._resolve_target(emb_out)    # (B, S, E) recon target
        event_mask = emb_out["mask"]              # (B, S)
        type_ids = emb_out.get("type_ids")        # (B, S)
        B, S, E = sequence.shape

        # 2. Generate masking pattern.
        if mask_token is None:
            mask_token = self.mask_generator(event_mask, type_ids)

        # 3. Replace masked positions with the learnable mask token and encode.
        masked_input = apply_mask_token(sequence, mask_token, self.mask_token)
        encoded, _ = self.backbone(masked_input, event_mask)  # (B, S, E)
        encoded = self.encoder_to_decoder(encoded)             # (B, S, D)

        # 4. Decoder: process full sequence.  Per-event localization already
        # comes from the unified embedding's sinusoidal time embedding; this is
        # just a small global learnable decoder bias.
        decoder_input = encoded + self.decoder_pos_embed
        decoded, _ = self.decoder(decoder_input, event_mask)  # (B, S, D)
        decoded = self.decoder_norm(decoded)
        pred = self.prediction_head(decoded)                   # (B, S, E)

        # 5. Reconstruction loss on masked positions only.
        loss, loss_dict = self._reconstruction_loss(
            pred, target, mask_token, event_mask, type_ids
        )

        return {
            "loss": loss,
            "loss_dict": loss_dict,
            "pred": pred,
            "target": target,
            "mask_token": mask_token,
            "event_mask": event_mask,
            "type_ids": type_ids,
        }

    def _resolve_target(self, emb_out: dict[str, torch.Tensor]) -> torch.Tensor:
        """Pick the reconstruction target (content-only by default).

        The target is always **detached**: it is a function of the (trainable)
        embedding model, so back-propagating through it lets the model trivially
        shrink the target to zero (representation collapse).  ``token`` targets
        are additionally normalized in the loss to be scale-invariant.
        """
        if self.target == "token":
            if "token_emb" in emb_out:
                return emb_out["token_emb"].detach()
            warnings.warn(
                "target='token' requested but the embedding model did not return "
                "'token_emb'; falling back to the composed 'sequence' target.",
                stacklevel=2,
            )
        return emb_out["sequence"].detach()

    def _reconstruction_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask_token: torch.Tensor,
        event_mask: torch.Tensor,
        type_ids: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute per-modality normalized MSE on masked valid positions."""
        valid_mask = event_mask.bool() & mask_token
        if not valid_mask.any():
            # Graph-connected zero so every parameter still receives a (zero)
            # gradient; a disconnected leaf tensor breaks DDP / AMP GradScaler.
            return pred.sum() * 0.0, {"total": 0.0}

        pred_masked = pred[valid_mask]      # (N, E)
        target_masked = target[valid_mask]  # (N, E)

        # Normalize when requested, and always for the learnable ``token``
        # target — a scale-invariant target removes the remaining slow,
        # weight-decay-driven shrink path toward collapse.
        if self.norm_pix_loss or self.target == "token":
            mean = target_masked.mean(dim=-1, keepdim=True)
            var = target_masked.var(dim=-1, keepdim=True, unbiased=False)
            target_masked = (target_masked - mean) / (var + 1e-6).sqrt()

        per_pos_loss = ((pred_masked - target_masked) ** 2).mean(dim=-1)  # (N,)

        loss_dict: dict[str, float] = {}
        if type_ids is not None:
            type_ids_masked = type_ids[valid_mask]
            unique_types = type_ids_masked.unique()
            for t in unique_types:
                name = f"modality_{int(t.item())}"
                loss_dict[name] = per_pos_loss[type_ids_masked == t].mean().item()

        loss = per_pos_loss.mean()
        loss_dict["total"] = loss.item()
        return loss, loss_dict


class PerModalityMAEDecoder(nn.Module):
    """Optional add-on that decodes unified embeddings back to raw modality features.

    This is kept separate from :class:`MultimodalMaskedAutoencoder` so that the
    default MAE can predict in the unified embedding space (stable, fast),
    while experiments that want true raw-value reconstruction can attach
    per-modality heads without touching the core MAE code.

    Args:
        embedding_dim: Dimension of unified event embeddings.
        output_specs: Dict mapping modality index to output dimension and
            prediction type.  Example::

                {0: ("numeric", 10), 1: ("code", vocab_size)}

        hidden_dim: Hidden size of the small MLP decoder per modality.
    """

    def __init__(
        self,
        embedding_dim: int,
        output_specs: dict[int, tuple[str, int]],
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.heads = nn.ModuleDict()
        for mod_idx, (task, out_dim) in output_specs.items():
            key = str(mod_idx)
            if task == "numeric":
                self.heads[key] = nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, out_dim),
                )
            elif task == "code":
                self.heads[key] = nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, out_dim),
                )
            else:
                raise ValueError(f"Unknown per-modality task: {task}")

    def forward(
        self,
        unified_embedding: torch.Tensor,
        type_ids: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Return per-modality predictions keyed by modality index."""
        outputs: dict[int, torch.Tensor] = {}
        for key, head in self.heads.items():
            mod_idx = int(key)
            mask = type_ids == mod_idx
            if not mask.any():
                continue
            outputs[mod_idx] = head(unified_embedding[mask])
        return outputs
