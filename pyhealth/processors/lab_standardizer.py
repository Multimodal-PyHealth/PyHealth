"""Train-split-only z-scoring for masked temporal laboratory values.

Labs and their observation mask are separate temporal fields. Only rows whose
mask is true are fitted: zero-filled / forward-filled missing values must never
affect a lab's mean or variance.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

import torch
from torch import nn


def _indices(dataset: Any) -> Optional[list[int]]:
    """Explicit sample indices for this split, or None for a plain iterable.

    ``SampleDataset`` subclasses ``litdata.StreamingDataset``, whose ``__iter__``
    and ``__len__`` are sharded by ``WORLD_SIZE``: under torchrun, iterating
    would silently fit on 1/WORLD_SIZE of the train split. Indexing is not
    sharded, and ``region_of_interest`` is the only unsharded description of
    what the split holds. (``patient_to_index`` is not usable — ``subset()``
    copies it unchanged, so after ``split_by_patient`` it still indexes the
    parent and raises "index ... didn't find a match within the chunk
    intervals".)
    """
    roi = getattr(dataset, "region_of_interest", None)
    return list(range(sum(end - start for start, end in roi))) if roi else None


class LabStandardizer(nn.Module):
    """Per-feature z-score with persistent train-only statistics.

    ``mean``/``std``/``observed_count`` are buffers, so they travel in the
    model ``state_dict`` and a checkpoint transforms serving inputs exactly as
    it did at training time.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        observed_count: torch.Tensor,
    ) -> None:
        super().__init__()
        if (std <= 0).any() or not torch.isfinite(mean).all():
            raise ValueError("Lab statistics must be finite with positive std.")
        self.register_buffer("mean", mean.detach().to(torch.float32).clone())
        self.register_buffer("std", std.detach().to(torch.float32).clone())
        self.register_buffer("observed_count", observed_count.detach().clone())

    @property
    def feature_dim(self) -> int:
        return int(self.mean.numel())

    @classmethod
    def fit(
        cls,
        samples: Iterable[dict[str, Any]],
        *,
        value_field: str = "labs",
        observation_mask_field: Optional[str] = None,
    ) -> "LabStandardizer":
        """Fit on observed, finite values of the supplied (already split) data."""
        mask_field = observation_mask_field or f"{value_field}_mask"
        idx = _indices(samples)
        stream = samples if idx is None else (samples[i] for i in idx)

        count = total = total_sq = None
        for sample in stream:
            if value_field not in sample or mask_field not in sample:
                continue
            v = sample[value_field]
            m = sample[mask_field]
            v = v[1] if isinstance(v, (tuple, list)) else v
            m = m[1] if isinstance(m, (tuple, list)) else m
            v = torch.as_tensor(v, dtype=torch.float64)
            m = torch.as_tensor(m).bool() & torch.isfinite(v)
            if v.ndim == 1:
                v, m = v.unsqueeze(0), m.unsqueeze(0)
            if count is None:
                z = torch.zeros(v.shape[-1], dtype=torch.float64)
                count, total, total_sq = z.clone(), z.clone(), z.clone()
            obs = torch.where(m, v, torch.zeros_like(v))
            count += m.sum(0).to(torch.float64)
            total += obs.sum(0)
            total_sq += (obs * obs).sum(0)

        if count is None:
            raise ValueError(
                f"No samples carried both {value_field!r} and {mask_field!r}."
            )

        seen = count > 0
        mean = torch.where(seen, total / count.clamp(min=1), torch.zeros_like(total))
        var = torch.where(
            seen,
            (total_sq / count.clamp(min=1) - mean.square()).clamp_min(0),
            torch.ones_like(total),
        )
        # A constant train feature maps to zero; unit std keeps that finite.
        std = torch.where(var > 0, var.sqrt(), torch.ones_like(var))
        return cls(mean.to(torch.float32), std.to(torch.float32), count)

    def forward(
        self, values: torch.Tensor, observed_mask: torch.Tensor
    ) -> torch.Tensor:
        """Z-score observed values; missing or unfittable features map to zero.

        Deliberately not clipped: there is no universally valid physiological
        range for these MIMIC category aggregates, so values outside train
        support stay as large finite z-scores and remain auditable.
        """
        if values.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim} lab features, got {values.shape[-1]}."
            )
        values = values.to(dtype=self.mean.dtype)
        observed = observed_mask.bool() & torch.isfinite(values)
        z = (values - self.mean) / self.std
        return torch.where(observed & (self.observed_count > 0), z, torch.zeros_like(z))


def fit_lab_standardizer(
    train_dataset: Iterable[dict[str, Any]], **kwargs: Any
) -> LabStandardizer:
    """Fit on the training split only. Train-only is enforced by the caller."""
    return LabStandardizer.fit(train_dataset, **kwargs)
