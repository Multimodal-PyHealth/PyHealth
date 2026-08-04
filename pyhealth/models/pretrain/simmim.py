"""SimMIM-style pretraining for unified multimodal event sequences.

Unlike MAE, SimMIM feeds the *full* sequence (with a learnable mask token at
masked positions) through the encoder and applies a simple linear head on the
final feature map to reconstruct the original per-modality token embedding.
This avoids a separate decoder transformer and is therefore cheaper per step.

References:
    Zhenda Xie et al., "SimMIM: A Simple Framework for Masked Image Modeling",
    CVPR 2022.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import torch
import torch.nn as nn

from ..embedding.unified import UnifiedMultimodalEmbeddingModel
from .masking import UnifiedMaskGenerator, apply_mask_token


class MultimodalSimMIM(nn.Module):
    """SimMIM pretrainer over a unified temporal event sequence.

    Args:
        embedding_model: Unified embedding model.
        backbone: Sequence encoder (e.g., TransformerLayer, Mamba stack).
        mask_ratio: Fraction of valid positions to mask.
        mask_strategy: ``"random"`` or ``"block"``.
        per_modality_ratio: Optional per-modality mask ratio overrides.
        target: What to reconstruct.  ``"token"`` (default) predicts the
            content-only per-event embedding (``token_emb``) *before* time/type
            are added — the recommended objective, since time/type are largely
            recoverable from position and otherwise dilute the content signal.
            ``"unified"`` predicts the full composed embedding (legacy).
        norm_targets: Normalize targets by mean/std before MSE.

    Inputs:
        Same dict as ``UnifiedMultimodalEmbeddingModel.forward``.

    Outputs:
        Dict with keys ``loss``, ``loss_dict``, ``pred``, ``target``,
        ``mask_token``, ``event_mask``, ``type_ids``.
    """

    def __init__(
        self,
        embedding_model: UnifiedMultimodalEmbeddingModel,
        backbone: nn.Module,
        mask_ratio: float = 0.5,
        mask_strategy: str = "random",
        per_modality_ratio: Optional[dict[int, float]] = None,
        target: str = "token",
        norm_targets: bool = False,
    ):
        super().__init__()
        if target not in ("token", "unified"):
            raise ValueError(f"target must be 'token' or 'unified', got {target}")
        self.embedding_model = embedding_model
        self.backbone = backbone
        self.embedding_dim = embedding_model.embedding_dim
        self.target = target
        self.norm_targets = norm_targets

        self.mask_generator = UnifiedMaskGenerator(
            mask_ratio=mask_ratio,
            strategy=mask_strategy,
            per_modality_ratio=per_modality_ratio,
        )
        self.mask_token = nn.Parameter(torch.zeros(self.embedding_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # SimMIM uses a single linear prediction head.
        self.head = nn.Linear(self.embedding_dim, self.embedding_dim)

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

        emb_out = self.embedding_model(inputs)
        sequence = emb_out["sequence"]            # (B, S, E) encoder input
        target = self._resolve_target(emb_out)    # (B, S, E) recon target
        event_mask = emb_out["mask"]              # (B, S)
        type_ids = emb_out.get("type_ids")        # (B, S)

        if mask_token is None:
            mask_token = self.mask_generator(event_mask, type_ids)

        # SimMIM: feed full sequence with mask tokens.
        masked_input = apply_mask_token(sequence, mask_token, self.mask_token)
        encoded, _ = self.backbone(masked_input, event_mask)  # (B, S, E)
        pred = self.head(encoded)                              # (B, S, E)

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

        Always **detached**: the target is a function of the trainable embedding
        model, so back-propagating through it lets the model shrink the target to
        zero (representation collapse).  ``token`` targets are also normalized in
        the loss for scale-invariance.
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
        valid_mask = event_mask.bool() & mask_token
        if not valid_mask.any():
            # Graph-connected zero so every parameter still receives a (zero)
            # gradient; a disconnected leaf tensor breaks DDP / AMP GradScaler.
            return pred.sum() * 0.0, {"total": 0.0}

        pred_masked = pred[valid_mask]
        target_masked = target[valid_mask]

        # Normalize when requested, and always for the learnable ``token``
        # target (scale-invariance removes the slow shrink-to-collapse path).
        if self.norm_targets or self.target == "token":
            mean = target_masked.mean(dim=-1, keepdim=True)
            var = target_masked.var(dim=-1, keepdim=True, unbiased=False)
            target_masked = (target_masked - mean) / (var + 1e-6).sqrt()

        per_pos_loss = ((pred_masked - target_masked) ** 2).mean(dim=-1)

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
