"""Utilities for wiring SSL pretraining models to PyHealth batches."""

from __future__ import annotations

import torch

from ...processors.base_processor import TemporalFeatureProcessor


def build_unified_inputs_from_batch(
    processors: dict[str, TemporalFeatureProcessor],
    feature_keys: list[str],
    batch: dict[str, torch.Tensor | tuple[torch.Tensor, ...]],
    device: torch.device | str | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Convert a PyHealth collated batch into unified-embedding inputs.

    Each feature in ``batch`` is either a tensor or a tuple of tensors ordered
    according to the processor's ``schema()``.  This mirrors
    :meth:`pyhealth.models.transformer.Transformer._build_unified_inputs`.

    Args:
        processors: ``dataset.input_processors``.
        feature_keys: Ordered list of input feature names.
        batch: Collated batch dict from a PyHealth DataLoader.
        device: If provided, move tensors to this device.

    Returns:
        ``{field_name: {"value": Tensor, "time": Tensor, "mask": Tensor}}``.
    """
    inputs: dict[str, dict[str, torch.Tensor]] = {}
    for field_name in feature_keys:
        feature = batch[field_name]
        if isinstance(feature, torch.Tensor):
            feature = (feature,)
        schema = processors[field_name].schema()
        field_dict: dict[str, torch.Tensor] = {}
        if "value" in schema:
            field_dict["value"] = feature[schema.index("value")]
        if "time" in schema:
            field_dict["time"] = feature[schema.index("time")]
        if "mask" in schema:
            field_dict["mask"] = feature[schema.index("mask")]
        if device is not None:
            field_dict = {k: v.to(device) for k, v in field_dict.items()}
        inputs[field_name] = field_dict
    return inputs
