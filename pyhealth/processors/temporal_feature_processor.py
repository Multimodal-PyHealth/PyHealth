# =============================================================================
# Contributors: Joshua Steier
# Description: Abstract base class for temporal feature processors in PyHealth's
#     multimodal pipeline. All time-aware processors (images, text, timeseries,
#     codes) should inherit from this class to ensure consistent
#     (value, time, modality_type) tuple output format for the
#     UnifiedMultimodalEmbeddingModel.
# =============================================================================

"""Temporal feature processor abstract base class.

This module defines ``TemporalFeatureProcessor``, which extends PyHealth's
``FeatureProcessor`` with temporal metadata requirements. Any processor that
emits time-aligned data for the multimodal embedding pipeline must inherit
from this class.

Expected output format from ``process()``:
    ``(value_tensor, time_tensor, modality_type_str)``

Where:
    - ``value_tensor``: The processed feature data (images, token IDs, floats, etc.)
    - ``time_tensor``: 1D tensor of timestamps (e.g., days since first admission)
    - ``modality_type_str``: One of ``"image"``, ``"text"``,
      ``"timeseries"``, ``"sequence"``

Example:
    >>> class MyTemporalProcessor(TemporalFeatureProcessor):
    ...     def __init__(self):
    ...         super().__init__(modality_type="timeseries")
    ...
    ...     def process(self, value):
    ...         values, times = value
    ...         val_tensor = torch.tensor(values, dtype=torch.float32)
    ...         time_tensor = torch.tensor(times, dtype=torch.float32)
    ...         return (val_tensor, time_tensor, self.modality_type)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Tuple

import torch

from pyhealth.processors.base_processor import FeatureProcessor


VALID_MODALITY_TYPES = {"image", "text", "timeseries", "sequence"}


class TemporalFeatureProcessor(FeatureProcessor):
    """Abstract base class for time-aware feature processors.

    Extends ``FeatureProcessor`` to enforce that the output of ``process()``
    is a tuple of ``(value_tensor, time_tensor, modality_type_str)``. This
    standardized format is required by ``UnifiedMultimodalEmbeddingModel``,
    which uses the modality type string to route data to the correct encoder
    and applies temporal embeddings from the time tensor.

    Subclasses must:
        1. Call ``super().__init__(modality_type=...)`` with a valid modality type.
        2. Implement ``process()`` returning ``(value, time, modality_type)``.

    Args:
        modality_type: One of ``"image"``, ``"text"``, ``"timeseries"``,
            ``"sequence"``. Used for modality routing in the unified
            embedding model.

    Raises:
        ValueError: If ``modality_type`` is not one of the valid types.
    """

    def __init__(self, modality_type: str, **kwargs) -> None:
        super().__init__(**kwargs)
        if modality_type not in VALID_MODALITY_TYPES:
            raise ValueError(
                f"Invalid modality_type '{modality_type}'. "
                f"Must be one of {VALID_MODALITY_TYPES}."
            )
        self._modality_type = modality_type

    @property
    def modality_type(self) -> str:
        """The modality type string for this processor.

        Returns:
            One of ``"image"``, ``"text"``, ``"timeseries"``, ``"sequence"``.
        """
        return self._modality_type

    @abstractmethod
    def process(self, value: Any) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Process a raw feature value into the temporal tuple format.

        Subclasses must implement this method. The return format must be:
            ``(value_tensor, time_tensor, modality_type_str)``

        Args:
            value: Raw input value. Typically a tuple of
                ``(List[raw_data], List[time_diffs])`` from the task.

        Returns:
            Tuple of:
                - ``value_tensor``: Processed feature tensor. Shape depends
                  on modality (e.g., ``(N, C, H, W)`` for images,
                  ``(S, D)`` for timeseries).
                - ``time_tensor``: 1D float tensor of shape ``(N,)`` with
                  timestamps (e.g., days since first admission).
                - ``modality_type``: String literal identifying the modality.
        """
        ...

    def is_token(self) -> bool:
        """Whether the value output represents discrete token indices.

        Returns:
            ``True`` if the output values are discrete indices (e.g., ICD codes),
            ``False`` if continuous (e.g., lab values, pixel intensities).
        """
        raise NotImplementedError(
            "Subclasses must implement is_token() to indicate whether "
            "output values are discrete tokens or continuous values."
        )

    def schema(self) -> tuple:
        """Returns the output schema for this processor.

        All temporal processors emit a 3-tuple: ``("value", "time", "modality")``.

        Returns:
            ``("value", "time", "modality")``
        """
        return ("value", "time", "modality")

    def dim(self) -> tuple:
        """Number of dimensions for each output tensor.

        Returns:
            Tuple of ints. Must be overridden by subclasses since value tensor
            dimensionality varies by modality (e.g., 4D for images, 2D for
            timeseries).
        """
        raise NotImplementedError(
            "Subclasses must implement dim() to specify the number of "
            "dimensions for each output tensor."
        )

    def spatial(self) -> tuple:
        """Whether each axis of the value tensor is spatial.

        Returns:
            Tuple of bools. Must be overridden by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement spatial() to specify which "
            "dimensions are spatial."
        )


def validate_temporal_processor(processor: FeatureProcessor) -> None:
    """Validate that a processor is a TemporalFeatureProcessor.

    Intended for use in ``UnifiedMultimodalEmbeddingModel`` to enforce that
    all input processors conform to the temporal interface.

    Args:
        processor: A feature processor instance to validate.

    Raises:
        TypeError: If the processor does not inherit from
            ``TemporalFeatureProcessor``.
    """
    if not isinstance(processor, TemporalFeatureProcessor):
        raise TypeError(
            f"Processor {type(processor).__name__} must inherit from "
            f"TemporalFeatureProcessor to be used in the multimodal "
            f"embedding pipeline. Got {type(processor).__mro__}."
        )