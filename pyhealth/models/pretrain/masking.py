"""Masking utilities for self-supervised pretraining on unified event sequences.

A unified event sequence has shape ``(B, S_total, E)`` with an accompanying
validity mask ``(B, S_total)`` and modality type ids ``(B, S_total)``.  The
collators below generate boolean ``mask_token`` tensors that select which
positions are hidden during pretraining.
"""

from __future__ import annotations

from typing import Optional

import torch


class UnifiedMaskGenerator:
    """Generate masking patterns for a unified temporal event sequence.

    Args:
        mask_ratio: Fraction of valid (non-padding) positions to mask.
        strategy: ``"random"`` or ``"block"``.  Block masking hides contiguous
            spans, which is closer to MAE/SimMIM and more realistic for EHR
            (a missing lab window rather than random single events).
        min_block_len: Minimum span length for block masking.
        max_block_len: Maximum span length for block masking.
        per_modality_ratio: Optional dict ``{modality_index: ratio}`` that
            overrides ``mask_ratio`` for specific modality types.  Useful when
            text should be masked less aggressively than labs.
        seed: Not used; callers should set the global/random generator for
            reproducibility.

    Shape:
        Input ``mask`` is ``(B, S)`` with 1 = valid, 0 = padding.
        Output ``mask_token`` is ``(B, S)`` bool, True = hide this position.
    """

    def __init__(
        self,
        mask_ratio: float = 0.5,
        strategy: str = "random",
        min_block_len: int = 3,
        max_block_len: int = 12,
        per_modality_ratio: Optional[dict[int, float]] = None,
    ):
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")
        self.mask_ratio = mask_ratio
        self.strategy = strategy
        self.min_block_len = min_block_len
        self.max_block_len = max_block_len
        self.per_modality_ratio = per_modality_ratio or {}

    def __call__(
        self,
        mask: torch.Tensor,
        type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a boolean mask_token tensor of the same shape as ``mask``."""
        if self.strategy == "random":
            return self._random_mask(mask, type_ids)
        if self.strategy == "block":
            return self._block_mask(mask, type_ids)
        raise ValueError(f"Unknown masking strategy: {self.strategy}")

    def _effective_ratio(
        self,
        ref: torch.Tensor,
        type_ids: Optional[torch.Tensor],
        default_ratio: float,
    ) -> torch.Tensor:
        """Build a per-position target mask ratio with shape ``ref.shape``."""
        if type_ids is None or not self.per_modality_ratio:
            return torch.full_like(ref, default_ratio, dtype=torch.float32)
        ratio = torch.full_like(type_ids, default_ratio, dtype=torch.float32)
        for mod_idx, mod_ratio in self.per_modality_ratio.items():
            ratio = torch.where(type_ids == mod_idx, mod_ratio, ratio)
        return ratio

    def _random_mask(
        self,
        mask: torch.Tensor,
        type_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Independent Bernoulli masking over valid positions."""
        valid = mask.bool()
        ratio = self._effective_ratio(mask, type_ids, self.mask_ratio)
        probs = torch.zeros_like(mask, dtype=torch.float32)
        probs[valid] = ratio[valid]
        mask_token = torch.bernoulli(probs).bool() & valid
        # Floor: every sample that has valid positions must mask at least one,
        # otherwise that sample contributes no reconstruction/prediction signal.
        empty_rows = valid.any(dim=1) & ~mask_token.any(dim=1)
        for b in empty_rows.nonzero(as_tuple=False).flatten():
            vp = valid[b].nonzero(as_tuple=False).flatten()
            j = vp[torch.randint(0, vp.numel(), (1,), device=mask.device)]
            mask_token[b, j] = True
        return mask_token

    def _block_mask(
        self,
        mask: torch.Tensor,
        type_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Sample contiguous spans until the target ratio is reached per sample."""
        B, S = mask.shape
        device = mask.device
        mask_token = torch.zeros_like(mask, dtype=torch.bool)

        for b in range(B):
            valid_positions = mask[b].nonzero(as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue

            # Determine effective ratio per position.
            if type_ids is not None and self.per_modality_ratio:
                ratios = self._effective_ratio(
                    mask[b : b + 1], type_ids[b : b + 1], self.mask_ratio
                ).squeeze(0)
                target_count = int(
                    round(ratios[valid_positions].float().mean().item() * valid_positions.numel())
                )
            else:
                target_count = int(round(self.mask_ratio * valid_positions.numel()))

            chosen = torch.zeros_like(valid_positions, dtype=torch.bool)
            attempts = 0
            max_attempts = target_count * 4 + 10
            while chosen.sum().item() < target_count and attempts < max_attempts:
                attempts += 1
                remaining = target_count - int(chosen.sum().item())
                block_len = int(
                    torch.randint(
                        self.min_block_len,
                        self.max_block_len + 1,
                        (1,),
                        device=device,
                    ).item()
                )
                # Cap the block to the remaining budget so we never overshoot and
                # never need a random trim (random trimming shatters the spans,
                # defeating the purpose of block masking).
                block_len = min(block_len, remaining)
                start_idx = int(
                    torch.randint(
                        0, max(1, valid_positions.numel() - block_len + 1), (1,), device=device
                    ).item()
                )
                chosen[start_idx : start_idx + block_len] = True

            # `chosen` marks contiguous runs in valid-event order; map back to the
            # actual sequence positions (kept contiguous over valid events).
            mask_token[b, valid_positions[chosen]] = True

        return mask_token


def apply_mask_token(
    sequence: torch.Tensor,
    mask_token: torch.Tensor,
    learnable_mask_token: torch.Tensor,
) -> torch.Tensor:
    """Replace masked positions in ``sequence`` with a learnable mask token.

    Args:
        sequence: ``(B, S, E)`` unified event embeddings.
        mask_token: ``(B, S)`` bool, True = hide.
        learnable_mask_token: ``(E,)`` shared mask embedding.

    Returns:
        ``(B, S, E)`` sequence with masked positions replaced.
    """
    masked = sequence.clone()
    masked[mask_token] = learnable_mask_token
    return masked


def random_mask_like(
    mask: torch.Tensor,
    mask_ratio: float = 0.5,
) -> torch.Tensor:
    """Convenience one-liner for independent random masking."""
    gen = UnifiedMaskGenerator(mask_ratio=mask_ratio, strategy="random")
    return gen(mask)
