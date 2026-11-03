"""Universal collation utilities for TemporalFeatureProcessor outputs.

Because every ``TemporalFeatureProcessor.process_temporal()`` returns a
``dict[str, Tensor]``, batching is trivial: stack/pad each key independently.
No per-processor custom collation logic is needed.

Usage::

    from pyhealth.datasets.collate import collate_temporal
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_temporal)

"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _pad_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Right-pad same-rank tensors to the per-dimension max, then stack.

    ``pad_sequence`` only pads dimension 0 and requires every trailing dimension
    to already match. Tokenized notes are ``(n_notes, seq_len)`` and, once the
    text processor pads to the longest note in a sample rather than to a fixed
    ``max_length``, BOTH dimensions vary across samples.
    """
    if len({t.dim() for t in tensors}) != 1:
        raise ValueError("cannot pad tensors of differing rank")
    target = [max(t.shape[d] for t in tensors) for d in range(tensors[0].dim())]
    padded = []
    for t in tensors:
        spec: list[int] = []
        for d in range(t.dim() - 1, -1, -1):
            spec.extend([0, target[d] - t.shape[d]])
        padded.append(F.pad(t, spec) if any(spec) else t)
    return torch.stack(padded)


def _stack_positions(values: list[tuple]) -> tuple:
    """Collates same-length tuples one position at a time.

    Tensor positions stack when their shapes match and pad every ragged
    dimension otherwise. A position that is None in every sample stays None,
    and one that is None in only some samples is dropped to None. Other
    positions, such as a text tag, stay lists.

    Example:
        >>> import torch
        >>> from pyhealth.datasets.collate import _stack_positions
        >>> _stack_positions([(torch.ones(2), "a"), (torch.ones(1), "b")])[1]
        ['a', 'b']
    """
    stacked = []
    for pos in range(len(values[0])):
        elems = [v[pos] for v in values]
        if all(e is None for e in elems):
            stacked.append(None)
        elif all(e is None or isinstance(e, torch.Tensor) for e in elems):
            if any(e is None for e in elems):
                stacked.append(None)
            elif elems[0].dim() == 0:
                stacked.append(torch.stack(elems))
            else:
                stacked.append(_pad_stack(elems))
        else:
            stacked.append(elems)
    return tuple(stacked)


def _event_pad_mask(values: list[tuple]) -> torch.Tensor | None:
    """Event-level validity of a batch of tuples, or None if nothing is ragged.

    Read from the first position holding non-scalar tensors. Those tensors
    share dimension 0 (the event count), so their lengths give the mask.
    """
    for pos in range(len(values[0])):
        elems = [v[pos] for v in values]
        if all(isinstance(e, torch.Tensor) for e in elems) and elems[0].dim() > 0:
            if all(e.shape == elems[0].shape for e in elems):
                return None
            lengths = torch.tensor([e.shape[0] for e in elems])
            return torch.arange(int(lengths.max()))[None, :] < lengths[:, None]
    return None


def _stack_or_pad(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Stack if all shapes match; pad every ragged dimension otherwise."""
    if all(t.shape == tensors[0].shape for t in tensors):
        return torch.stack(tensors)
    return _pad_stack(tensors)


def collate_temporal(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Universal collator for datasets that contain ``TemporalFeatureProcessor``
    dict outputs alongside ordinary tensors and labels.

    Handling rules (applied recursively for nested dicts):

    * ``Tensor``         — stack if same shape, pad to longest otherwise
    * ``dict[str, ...]`` — recursively collate each sub-key  *(temporal feature)*
    * ``int / float``    — ``torch.tensor(...)``
    * anything else      — kept as a plain Python list

    Args:
        batch: List of sample dicts as returned by the DataLoader's dataset.

    Returns:
        A single collated dict ready for model ``forward(**batch)``.
    """
    if not batch:
        return {}

    result: dict[str, Any] = {}

    for key in batch[0]:
        vals = [s[key] for s in batch]
        first = vals[0]

        if isinstance(first, dict):
            # ── Temporal feature dict — collate each sub-key ─────────────
            sub_result: dict[str, Any] = {}
            for sub_key in first:
                sub_vals = [v[sub_key] for v in vals]
                if sub_vals[0] is None:
                    sub_result[sub_key] = [None] * len(sub_vals)
                elif isinstance(sub_vals[0], torch.Tensor):
                    sub_result[sub_key] = _stack_or_pad(sub_vals)
                else:
                    sub_result[sub_key] = sub_vals
            result[key] = sub_result

        elif isinstance(first, torch.Tensor):
            result[key] = _stack_or_pad(vals)

        elif isinstance(first, (int, float)):
            result[key] = torch.tensor(vals)

        else:
            result[key] = vals

    return result
