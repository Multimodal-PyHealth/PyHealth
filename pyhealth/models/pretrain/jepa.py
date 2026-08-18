"""I-JEPA / V-JEPA pretraining for unified multimodal event sequences.

A context encoder processes unmasked context positions, a target encoder
(processing the full input) provides regression targets via an EMA update, and
a small predictor network maps context representations to the EMA-target latents
of masked "target blocks".

Because the objective operates purely in representation space, the model does
not reconstruct noisy raw values.  This makes it attractive for multimodal EHR
data where labs, codes, and text have very different noise characteristics.

The predictor is **location aware**: every target position receives its own
query (``base_query + sinusoidal_pos_emb(position)`` and, for V-JEPA, a
``scale_embed``), so the predictor can produce a distinct prediction per target
position.  Prediction and loss are computed in a fully batched, per-position way
(``pred[target] vs target_latent[target]``), so there is no block-splitting and
no all-pairs broadcasting.

References:
    Mahmoud Assran et al., "Self-Supervised Learning from Images with a Joint
    Embedding Predictive Architecture", CVPR 2023.
    Adrien Bardes et al., "Revisiting Feature Prediction for Learning Visual
    Representations from Video" (V-JEPA), 2024.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Optional

import torch
import torch.nn as nn

from ..embedding.unified import UnifiedMultimodalEmbeddingModel
from ..transformer import TransformerLayer


class MultimodalIJEPA(nn.Module):
    """I-JEPA pretrainer for unified multimodal event sequences.

    Args:
        embedding_model: Unified embedding model.
        context_encoder: Sequence encoder backbone.
        predictor_layers: Number of layers in the predictor network.
        predictor_heads: Attention heads in the predictor.
        predictor_dim: Hidden dim of the predictor.
        target_ema_decay: EMA momentum for the target encoder.
            Default 0.996, increased to ``target_ema_end`` over training via
            ``set_ema_decay`` (the trainer wires this automatically).
        target_ema_end: Final EMA momentum (cosine schedule).  Default 1.0.
        num_target_blocks: Number of contiguous target blocks to predict per
            sample.  Default 4.
        target_block_len: Length of each target block.  Default 4.
        min_context_len: Minimum number of visible context positions required.
            Default 4.
        normalize_targets: LayerNorm (no affine) the EMA targets before the
            loss (standard JEPA practice; reduces representation collapse).

    Inputs:
        Same dict as ``UnifiedMultimodalEmbeddingModel.forward``.

    Outputs:
        Dict with ``loss``, ``loss_dict``, ``context_pred`` (per-position
        predictions for target positions), ``target_embs`` (detached EMA
        targets), ``target_mask``, ``context_mask``, ``event_mask``,
        ``type_ids``.
    """

    def __init__(
        self,
        embedding_model: UnifiedMultimodalEmbeddingModel,
        context_encoder: nn.Module,
        predictor_layers: int = 6,
        predictor_heads: int = 8,
        predictor_dim: Optional[int] = None,
        target_ema_decay: float = 0.996,
        target_ema_end: float = 1.0,
        num_target_blocks: int = 4,
        target_block_len: int = 4,
        min_context_len: int = 4,
        normalize_targets: bool = True,
    ):
        super().__init__()
        self.embedding_model = embedding_model
        self.context_encoder = context_encoder
        self.embedding_dim = embedding_model.embedding_dim
        self.predictor_dim = predictor_dim or self.embedding_dim

        # Target encoder is an EMA copy of the context encoder.
        self.target_encoder = copy.deepcopy(context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor: small Transformer that consumes context latents at context
        # positions and location-aware queries at target positions.
        self.predictor_context_proj = (
            nn.Linear(self.embedding_dim, self.predictor_dim)
            if self.predictor_dim != self.embedding_dim
            else nn.Identity()
        )
        self.predictor_query = nn.Parameter(torch.randn(1, 1, self.predictor_dim))
        nn.init.trunc_normal_(self.predictor_query, std=0.02)
        self.predictor = TransformerLayer(
            feature_size=self.predictor_dim,
            heads=predictor_heads,
            dropout=0.0,
            num_layers=predictor_layers,
        )
        self.predictor_norm = nn.LayerNorm(self.predictor_dim)
        self.predictor_out_proj = nn.Linear(self.predictor_dim, self.embedding_dim)

        self.normalize_targets = normalize_targets
        self.target_norm = (
            nn.LayerNorm(self.embedding_dim, elementwise_affine=False)
            if normalize_targets
            else nn.Identity()
        )

        self.target_ema_decay = target_ema_decay
        self.target_ema_end = target_ema_end
        self.num_target_blocks = num_target_blocks
        self.target_block_len = target_block_len
        self.min_context_len = min_context_len
        self._ema_start = target_ema_decay
        self._global_step = 0

    def set_ema_decay(self, step: int, total_steps: int) -> None:
        """Cosine schedule from ``target_ema_decay`` to ``target_ema_end``."""
        progress = min(1.0, step / max(1, total_steps))
        self._global_step = step
        self.target_ema_decay = (
            self._ema_start
            + (self.target_ema_end - self._ema_start)
            * (1 - math.cos(progress * math.pi))
            / 2
        )

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """EMA update of target encoder from context encoder."""
        m = self.target_ema_decay
        for param_q, param_k in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

    @staticmethod
    def _sinusoidal_pos_emb(
        positions: torch.Tensor, dim: int, device: torch.device
    ) -> torch.Tensor:
        """Parameter-free sinusoidal embedding of integer positions.

        Args:
            positions: ``(S,)`` long tensor of position indices.
            dim: Output dimension.

        Returns:
            ``(S, dim)`` float tensor.
        """
        half = dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(1, half - 1))
        )
        ang = positions.float().unsqueeze(-1) * freqs  # (S, half)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < dim:  # pad odd dims
            emb = torch.cat(
                [emb, torch.zeros(emb.shape[0], dim - emb.shape[-1], device=device)],
                dim=-1,
            )
        return emb

    def _sample_target_blocks(
        self,
        event_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample non-overlapping single-scale target blocks + context.

        Returns:
            target_mask: ``(B, S)`` bool, True = target positions.
            context_mask: ``(B, S)`` bool, True = context positions.
        """
        B, S = event_mask.shape
        device = event_mask.device
        valid = event_mask.bool()
        target_mask = torch.zeros_like(event_mask, dtype=torch.bool)

        for b in range(B):
            valid_positions = valid[b].nonzero(as_tuple=False).flatten()
            n_valid = int(valid_positions.numel())
            if n_valid < self.min_context_len + self.target_block_len:
                # Too short: mask a small random subset, leaving context.
                n_target = min(
                    max(0, n_valid - self.min_context_len),
                    self.num_target_blocks * self.target_block_len,
                )
                if n_target > 0:
                    perm = torch.randperm(n_valid, device=device)
                    target_mask[b, valid_positions[perm[:n_target]]] = True
                continue

            max_targets = n_valid - self.min_context_len
            used = torch.zeros(n_valid, dtype=torch.bool, device=device)
            n_blocks = 0
            attempts = 0
            max_attempts = self.num_target_blocks * 20
            while n_blocks < self.num_target_blocks and attempts < max_attempts:
                attempts += 1
                block_len = min(self.target_block_len, max_targets)
                if block_len <= 0:
                    break
                start = int(
                    torch.randint(0, n_valid - block_len + 1, (1,), device=device).item()
                )
                span = torch.arange(start, start + block_len, device=device)
                if used[span].any():
                    continue
                if int(used.sum().item()) + block_len > max_targets:
                    continue
                used[span] = True
                target_mask[b, valid_positions[span]] = True
                n_blocks += 1

        context_mask = valid & ~target_mask
        return target_mask, context_mask

    def _encode_and_predict(
        self,
        sequence: torch.Tensor,
        event_mask: torch.Tensor,
        target_mask: torch.Tensor,
        context_mask: torch.Tensor,
        scale_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared context/target encode + location-aware batched prediction.

        Returns ``(pred, target_latent)`` both of shape ``(B, S, E)``.
        """
        B, S, E = sequence.shape
        device = sequence.device

        # Context encoder: attend ONLY over context positions (no target leak).
        context_input = sequence.clone()
        context_input[~context_mask] = 0.0
        context_latent, _ = self.context_encoder(context_input, context_mask.float())

        # Target encoder: full sequence, no gradients; normalized targets.
        with torch.no_grad():
            target_latent, _ = self.target_encoder(sequence, event_mask)
            target_latent = self.target_norm(target_latent)

        # Predictor input: context positions carry projected context latents;
        # target positions carry a location (+scale) aware query.
        P = self.predictor_dim
        ctx_proj = self.predictor_context_proj(context_latent)            # (B, S, P)
        positions = torch.arange(S, device=device)
        pos_emb = self._sinusoidal_pos_emb(positions, P, device)          # (S, P)
        pos_emb = pos_emb.unsqueeze(0).to(ctx_proj.dtype)                 # (1, S, P)
        query = self.predictor_query.to(ctx_proj.dtype) + pos_emb        # (1, S, P)
        if scale_ids is not None and hasattr(self, "scale_embed"):
            query = query + self.scale_embed(scale_ids)                  # (B, S, P)
        # Keep the predictor input in the context dtype (under AMP nn.Embedding
        # stays fp32, which would otherwise upcast the whole predictor input).
        query = query.to(ctx_proj.dtype)
        pred_in = torch.where(target_mask.unsqueeze(-1), query, ctx_proj)

        pred_out, _ = self.predictor(pred_in, event_mask)                # (B, S, P)
        pred_out = self.predictor_norm(pred_out)
        pred = self.predictor_out_proj(pred_out)                         # (B, S, E)
        return pred, target_latent

    def forward(
        self,
        inputs: Optional[dict[str, dict[str, torch.Tensor]]] = None,
        target_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
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
        sequence = emb_out["sequence"]      # (B, S, E)
        event_mask = emb_out["mask"]        # (B, S)
        type_ids = emb_out.get("type_ids")

        if target_mask is None or context_mask is None:
            target_mask, context_mask = self._sample_target_blocks(event_mask)

        pred, target_latent = self._encode_and_predict(
            sequence, event_mask, target_mask, context_mask
        )

        valid_target = target_mask & event_mask.bool()
        if not valid_target.any():
            # Graph-connected zero (keeps every parameter in the autograd graph
            # so backward/DDP do not break); contributes no signal this step.
            loss = pred.sum() * 0.0
            return {
                "loss": loss,
                "loss_dict": {"total": 0.0},
                "context_pred": pred[valid_target],
                "target_embs": target_latent[valid_target].detach(),
                "target_mask": target_mask,
                "context_mask": context_mask,
                "event_mask": event_mask,
                "type_ids": type_ids,
            }

        pred_t = pred[valid_target]                       # (N, E)
        tgt_t = target_latent[valid_target].detach()      # (N, E)
        per_pos_loss = ((pred_t - tgt_t) ** 2).mean(dim=-1)  # (N,)
        loss = per_pos_loss.mean()
        loss_dict = {"total": loss.item()}
        if type_ids is not None:
            tt = type_ids[valid_target]
            for t in tt.unique():
                loss_dict[f"modality_{int(t.item())}"] = per_pos_loss[tt == t].mean().item()
        return {
            "loss": loss,
            "loss_dict": loss_dict,
            "context_pred": pred_t,
            "target_embs": tgt_t,
            "target_mask": target_mask,
            "context_mask": context_mask,
            "event_mask": event_mask,
            "type_ids": type_ids,
        }


class MultimodalVJEPA(MultimodalIJEPA):
    """V-JEPA-style pretrainer for unified multimodal event sequences.

    This extends :class:`MultimodalIJEPA` along the axes that distinguish
    V-JEPA (video JEPA) from I-JEPA, adapted to temporal EHR sequences:

    1. **Multi-scale span (tube) masking.**  Target blocks are sampled with
       lengths drawn from ``target_block_scales`` (short blocks capture
       fine-grained dynamics, long blocks force trend prediction).
    2. **Scale aware predictor.**  In addition to the location (positional)
       query shared with I-JEPA, each target position adds a learned
       ``scale_embed`` for the scale of the block it belongs to.
    3. **Per-position latent prediction.**  Inherited from the base class: the
       predictor forecasts the EMA-target latent at every masked position.
    4. **Cross-modal target windows (optional).**  With
       ``require_multimodal_blocks=True`` the sampler prefers contiguous spans
       that contain more than one modality, forcing cross-modal temporal
       reasoning rather than within-modality interpolation.

    Context and target encoders share architecture (the target encoder is an EMA
    copy of the context encoder); an asymmetric design is incompatible with the
    EMA-copy update and is therefore intentionally not used.

    Args:
        embedding_model: Unified embedding model.
        context_encoder: Sequence encoder backbone.
        predictor_layers / predictor_heads / predictor_dim: Predictor config.
        target_ema_decay / target_ema_end: EMA momentum schedule endpoints.
        num_target_blocks: Number of target blocks to sample per sample.
        target_block_scales: Candidate block lengths (the multi-scale set).
        min_context_len: Minimum visible context positions to keep.
        require_multimodal_blocks: Prefer spans spanning >1 modality.
        normalize_targets: LayerNorm (no affine) targets before the loss.
    """

    def __init__(
        self,
        embedding_model: UnifiedMultimodalEmbeddingModel,
        context_encoder: nn.Module,
        predictor_layers: int = 6,
        predictor_heads: int = 8,
        predictor_dim: Optional[int] = None,
        target_ema_decay: float = 0.996,
        target_ema_end: float = 1.0,
        num_target_blocks: int = 4,
        target_block_scales: tuple[int, ...] = (2, 4, 8),
        min_context_len: int = 4,
        require_multimodal_blocks: bool = False,
        normalize_targets: bool = True,
    ):
        if not target_block_scales:
            raise ValueError("target_block_scales must be non-empty")
        mean_len = max(1, int(round(sum(target_block_scales) / len(target_block_scales))))
        super().__init__(
            embedding_model=embedding_model,
            context_encoder=context_encoder,
            predictor_layers=predictor_layers,
            predictor_heads=predictor_heads,
            predictor_dim=predictor_dim,
            target_ema_decay=target_ema_decay,
            target_ema_end=target_ema_end,
            num_target_blocks=num_target_blocks,
            target_block_len=mean_len,
            min_context_len=min_context_len,
            normalize_targets=normalize_targets,
        )
        self.target_block_scales = list(target_block_scales)
        self.require_multimodal_blocks = require_multimodal_blocks

        # Scale-aware component of the predictor query (one vector per scale).
        self.scale_embed = nn.Embedding(len(self.target_block_scales), self.predictor_dim)
        nn.init.trunc_normal_(self.scale_embed.weight, std=0.02)

    def _sample_multiscale_blocks(
        self,
        event_mask: torch.Tensor,
        type_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample multi-scale, non-overlapping target spans.

        Returns:
            target_mask: ``(B, S)`` bool, True = target position.
            context_mask: ``(B, S)`` bool, True = context position.
            scale_ids: ``(B, S)`` long, index into ``target_block_scales`` for
                each target position (0 elsewhere).
        """
        B, S = event_mask.shape
        device = event_mask.device
        valid = event_mask.bool()
        target_mask = torch.zeros_like(event_mask, dtype=torch.bool)
        scale_ids = torch.zeros_like(event_mask, dtype=torch.long)
        scales = self.target_block_scales

        for b in range(B):
            valid_positions = valid[b].nonzero(as_tuple=False).flatten()
            n_valid = int(valid_positions.numel())
            if n_valid <= self.min_context_len + 1:
                # Too short for span masking: optionally hide one position.
                if n_valid > self.min_context_len:
                    j = int(torch.randint(0, n_valid, (1,), device=device).item())
                    target_mask[b, valid_positions[j]] = True
                continue

            max_targets = n_valid - self.min_context_len
            used = torch.zeros(n_valid, dtype=torch.bool, device=device)
            n_blocks = 0
            attempts = 0
            max_attempts = self.num_target_blocks * 20
            while n_blocks < self.num_target_blocks and attempts < max_attempts:
                attempts += 1
                s_idx = int(torch.randint(0, len(scales), (1,), device=device).item())
                block_len = min(scales[s_idx], max_targets)
                if block_len <= 0:
                    break
                max_start = n_valid - block_len
                if max_start < 0:
                    continue
                start = int(torch.randint(0, max_start + 1, (1,), device=device).item())
                span = torch.arange(start, start + block_len, device=device)
                if used[span].any():
                    continue
                if int(used.sum().item()) + block_len > max_targets:
                    continue
                seg_positions = valid_positions[span]
                # Prefer cross-modal windows when requested (soft constraint).
                if (
                    self.require_multimodal_blocks
                    and type_ids is not None
                    and attempts < max_attempts // 2
                    and int(type_ids[b, seg_positions].unique().numel()) < 2
                ):
                    continue
                used[span] = True
                target_mask[b, seg_positions] = True
                scale_ids[b, seg_positions] = s_idx
                n_blocks += 1

        context_mask = valid & ~target_mask
        return target_mask, context_mask, scale_ids

    def forward(
        self,
        inputs: Optional[dict[str, dict[str, torch.Tensor]]] = None,
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
        sequence = emb_out["sequence"]      # (B, S, E)
        event_mask = emb_out["mask"]        # (B, S)
        type_ids = emb_out.get("type_ids")

        target_mask, context_mask, scale_ids = self._sample_multiscale_blocks(
            event_mask, type_ids
        )

        pred, target_latent = self._encode_and_predict(
            sequence, event_mask, target_mask, context_mask, scale_ids
        )

        valid_target = target_mask & event_mask.bool()
        if not valid_target.any():
            loss = pred.sum() * 0.0  # graph-connected zero (no signal this step)
            return {
                "loss": loss,
                "loss_dict": {"total": 0.0},
                "context_pred": pred[valid_target],
                "target_embs": target_latent[valid_target].detach(),
                "target_mask": target_mask,
                "context_mask": context_mask,
                "scale_ids": scale_ids,
                "event_mask": event_mask,
                "type_ids": type_ids,
            }

        pred_t = pred[valid_target]                       # (N, E)
        tgt_t = target_latent[valid_target].detach()      # (N, E)
        per_pos_loss = ((pred_t - tgt_t) ** 2).mean(dim=-1)
        loss = per_pos_loss.mean()

        loss_dict: dict[str, float] = {"total": loss.item()}
        scale_t = scale_ids[valid_target]
        for s in scale_t.unique():
            block_len = self.target_block_scales[int(s.item())]
            loss_dict[f"scale_{block_len}"] = per_pos_loss[scale_t == s].mean().item()
        if type_ids is not None:
            tt = type_ids[valid_target]
            for t in tt.unique():
                loss_dict[f"modality_{int(t.item())}"] = per_pos_loss[tt == t].mean().item()

        return {
            "loss": loss,
            "loss_dict": loss_dict,
            "context_pred": pred_t,
            "target_embs": tgt_t,
            "target_mask": target_mask,
            "context_mask": context_mask,
            "scale_ids": scale_ids,
            "event_mask": event_mask,
            "type_ids": type_ids,
        }
