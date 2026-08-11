from abc import ABC
from typing import Callable, Any, Optional, Mapping
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..datasets import SampleDataset
from ..processors import PROCESSOR_REGISTRY


class BaseModel(ABC, nn.Module):
    """Abstract class for PyTorch models.

    Args:
        dataset (SampleDataset): The dataset to train the model. It is used to query certain
            information such as the set of all tokens.
            
    Interpretability
    --------
        To use a model with interpretability methods, the model must implement a method
        `forward_from_embedding` that takes in embeddings as input instead of raw features;
        for the models that already take in dense features as input, this method can simply
        call the existing `forward` method. 
        
        For certain gradient-based interpretability methods (e.g., DeepLIFT), the model must also
        ensure all non-linearity (e.g. ReLU, Sigmoid, Softmax) are using nn.Module versions instead of
        functional versions (e.g., F.relu, F.sigmoid, F.softmax) so that hooks can be registered properly.
    """

    def __init__(self, dataset: SampleDataset):
        """
        Initializes the BaseModel.

        Args:
            dataset (SampleDataset): The dataset to train the model.
        """
        super(BaseModel, self).__init__()
        self.dataset = dataset
        self.feature_keys = []
        self.label_keys = []
        if dataset:
            self.feature_keys = list(dataset.input_schema.keys())
            self.label_keys = list(dataset.output_schema.keys())
            # if single label, try to resolve mode for legacy trainer usage
            if len(self.label_keys) == 1:
                try:
                    m = self._resolve_mode(dataset.output_schema[self.label_keys[0]])
                    if m in {"binary", "multiclass", "multilabel", "regression"}:
                        self.mode = m
                except Exception:
                    pass
        # used to query the device of the model
        self._dummy_param = nn.Parameter(torch.empty(0))

        self.mode = getattr(self, "mode", None)  # legacy API
        
    def forward(self, 
            **kwargs: torch.Tensor | tuple[torch.Tensor, ...]
        ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            **kwargs: A variable number of keyword arguments representing input features.
                Each keyword argument is a tensor or a tuple of tensors of shape (batch_size, ...).
        
        Returns:
            A dictionary with the following keys:
                logit: a tensor of predicted logits.
                y_prob: a tensor of predicted probabilities.
                loss [optional]: a scalar tensor representing the final loss, if self.label_keys in kwargs.
                y_true [optional]: a tensor representing the true labels, if self.label_keys in kwargs.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_mode(self, schema_entry: Any) -> str:
        """Resolve a mode string from an output_schema entry.

        Supports:
          - direct string ("binary", ...)
          - processor class
          - processor instance
        Returns the registered processor name if found.
        """
        if isinstance(schema_entry, str):
            return schema_entry.lower()

        # Get class reference
        cls = schema_entry if inspect.isclass(schema_entry) else schema_entry.__class__
        for name, registered_cls in PROCESSOR_REGISTRY.items():
            if cls is registered_cls or issubclass(
                cls, registered_cls
            ):  # allow subclassing
                return name.lower()
        raise ValueError(
            f"Cannot resolve mode from output_schema entry {schema_entry}. Use a supported string"
        )

    @property
    def device(self) -> torch.device:
        """
        Gets the device of the model.

        Returns:
            torch.device: The device on which the model is located.
        """
        return self._dummy_param.device

    def get_output_size(self) -> int:
        """
        Gets the default output size using the label tokenizer and `self.mode`.

        If the mode is "binary", the output size is 1. If the mode is "multiclass"
        or "multilabel", the output size is the number of classes or labels.

        Returns:
            int: The output size of the model.
        """
        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported if get_output_size is called"
        output_size = self.dataset.output_processors[self.label_keys[0]].size()
        return output_size

    def get_loss_function(self) -> Callable:
        """
        Gets the default loss function using `self.mode`.

        The default loss functions are:
            - binary: `F.binary_cross_entropy_with_logits`
            - multiclass: `F.cross_entropy`
            - multilabel: `F.binary_cross_entropy_with_logits`
            - regression: `F.mse_loss`

        Returns:
            Callable: The default loss function.
        """
        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported if get_loss_function is called"
        label_key = self.label_keys[0]
        mode = self._resolve_mode(self.dataset.output_schema[label_key])
        if mode == "binary":
            return F.binary_cross_entropy_with_logits
        elif mode == "multiclass":
            return F.cross_entropy
        elif mode == "multilabel":
            return F.binary_cross_entropy_with_logits
        elif mode == "regression":
            return F.mse_loss
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def prepare_y_prob(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Prepares the predicted probabilities for model evaluation.

        This function converts the predicted logits to predicted probabilities
        depending on the mode. The default formats are:
            - binary: a tensor of shape (batch_size, 1) with values in [0, 1],
                which is obtained with `torch.sigmoid()`
            - multiclass: a tensor of shape (batch_size, num_classes) with
                values in [0, 1] and sum to 1, which is obtained with
                `torch.softmax()`
            - multilabel: a tensor of shape (batch_size, num_labels) with values
                in [0, 1], which is obtained with `torch.sigmoid()`
            - regression: a tensor of shape (batch_size, 1) with raw logits

        Args:
            logits (torch.Tensor): The predicted logit tensor.

        Returns:
            torch.Tensor: The predicted probability tensor.
        """
        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported if get_loss_function is called"
        label_key = self.label_keys[0]
        mode = self._resolve_mode(self.dataset.output_schema[label_key])
        if mode in ["binary"]:
            y_prob = torch.sigmoid(logits)
        elif mode in ["multiclass"]:
            y_prob = F.softmax(logits, dim=-1)
        elif mode in ["multilabel"]:
            y_prob = torch.sigmoid(logits)
        elif mode in ["regression"]:
            y_prob = logits
        else:
            raise NotImplementedError
        return y_prob


    def load_pretrained_state_dict(
        self,
        checkpoint: Mapping[str, Any],
        *,
        min_backbone_match: float = 1.0,
        min_embedding_match: float = 1.0,
    ) -> dict[str, Any]:
        """Safely transfer a unified SSL encoder into a downstream model.

        SSL checkpoints use ``backbone.*`` (or ``context_encoder.*``), while
        downstream architectures register their unified encoders under three
        different names. This method performs the architecture-specific mapping,
        rejects shape/configuration mismatches, and requires explicit source and
        target coverage before using ``strict=False`` for the intentionally absent
        classification head.
        """

        if not 0 < min_backbone_match <= 1:
            raise ValueError("min_backbone_match must be in (0, 1].")
        if not 0 <= min_embedding_match <= 1:
            raise ValueError("min_embedding_match must be in [0, 1].")

        raw = checkpoint.get("state_dict", checkpoint)
        if not isinstance(raw, Mapping):
            raise TypeError("checkpoint must be a state-dict mapping.")
        source = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in raw.items()
            if isinstance(key, str) and isinstance(value, torch.Tensor)
        }

        backbone_prefixes = [
            prefix
            for prefix in ("backbone.", "context_encoder.")
            if any(key.startswith(prefix) for key in source)
        ]
        if len(backbone_prefixes) != 1:
            raise ValueError(
                "Checkpoint must contain exactly one of 'backbone.*' or "
                "'context_encoder.*'."
            )
        source_prefix = backbone_prefixes[0]

        modules = self._modules
        if "_unified_jamba" in modules:
            relative_maps = (("", "_unified_jamba."),)
        elif "_unified_blocks" in modules:
            relative_maps = (
                ("blocks.", "_unified_blocks."),
                ("norm.", "_unified_norm."),
            )
        elif "_unified_backbone" in modules:
            relative_maps = (("", "_unified_backbone."),)
        else:
            raise ValueError(
                f"{type(self).__name__} has no registered unified backbone; "
                "refusing a potentially partial transfer."
            )

        target = self.state_dict()
        loadable: dict[str, torch.Tensor] = {}
        shape_errors: list[str] = []
        source_backbone = {
            key[len(source_prefix) :]: value
            for key, value in source.items()
            if key.startswith(source_prefix)
        }
        recognized_source: set[str] = set()
        # Bookkeeping buffers are not learned encoder weights, so a pretraining
        # checkpoint legitimately does not contain them. Counting them toward
        # target coverage makes a COMPLETE load look partial and rejects it
        # (e.g. TransformerLayer carries `_checkpoint_config`, which alone drops
        # target_ratio to 24/25 = 0.96 and trips min_backbone_match=1.0).
        _NON_WEIGHT_SUFFIXES = ("_checkpoint_config", "_dummy_param")

        def _non_transferable(key: str) -> bool:
            return key.endswith(_NON_WEIGHT_SUFFIXES)

        def _standardizer_state(keys: set[str]) -> set[str]:
            return {key for key in keys if ".numeric_standardizers." in key}

        target_backbone_keys = {
            key
            for key in target
            if any(key.startswith(target_prefix) for _, target_prefix in relative_maps)
            and not _non_transferable(key)
        }
        source_backbone = {
            key: value for key, value in source_backbone.items() if not _non_transferable(key)
        }
        for relative_key, value in source_backbone.items():
            for source_relative_prefix, target_prefix in relative_maps:
                if relative_key.startswith(source_relative_prefix):
                    target_key = target_prefix + relative_key[len(source_relative_prefix) :]
                    recognized_source.add(relative_key)
                    if target_key in target:
                        if target[target_key].shape != value.shape:
                            shape_errors.append(
                                f"{relative_key}: checkpoint {tuple(value.shape)} != "
                                f"model {tuple(target[target_key].shape)}"
                            )
                        else:
                            loadable[target_key] = value
                    break

        matched_backbone = target_backbone_keys.intersection(loadable)
        source_ratio = len(matched_backbone) / max(len(source_backbone), 1)
        target_ratio = len(matched_backbone) / max(len(target_backbone_keys), 1)
        if shape_errors:
            raise ValueError(
                "Pretrained backbone tensor shape mismatch: " + "; ".join(shape_errors)
            )
        if (
            not source_backbone
            or source_ratio < min_backbone_match
            or target_ratio < min_backbone_match
            or len(recognized_source) != len(source_backbone)
        ):
            raise ValueError(
                "Refusing partial pretrained backbone load: "
                f"matched={len(matched_backbone)}, source={len(source_backbone)}, "
                f"target={len(target_backbone_keys)}, "
                f"source_ratio={source_ratio:.3f}, target_ratio={target_ratio:.3f}."
            )

        source_embedding = {
            key: value
            for key, value in source.items()
            if key.startswith("embedding_model.") and not _non_transferable(key)
        }
        target_embedding_keys = {
            key
            for key in target
            if key.startswith("embedding_model.") and not _non_transferable(key)
        }
        for key, value in source_embedding.items():
            if key in target:
                if target[key].shape != value.shape:
                    shape_errors.append(
                        f"{key}: checkpoint {tuple(value.shape)} != "
                        f"model {tuple(target[key].shape)}"
                    )
                else:
                    loadable[key] = value
        if shape_errors:
            raise ValueError(
                "Pretrained embedding tensor shape mismatch: " + "; ".join(shape_errors)
            )
        matched_embedding = target_embedding_keys.intersection(loadable)
        if source_embedding or target_embedding_keys:
            source_embedding_ratio = len(matched_embedding) / max(len(source_embedding), 1)
            target_embedding_ratio = len(matched_embedding) / max(
                len(target_embedding_keys), 1
            )
            if (
                source_embedding_ratio < min_embedding_match
                or target_embedding_ratio < min_embedding_match
            ):
                raise ValueError(
                    "Refusing partial pretrained embedding load: "
                    f"matched={len(matched_embedding)}, source={len(source_embedding)}, "
                    f"target={len(target_embedding_keys)}."
                )

        # This method has already validated source and target coverage, so call
        # nn.Module directly rather than the defensive public strict=False guard.
        incompatible = super().load_state_dict(loadable, strict=False)
        return {
            "backbone_matched": len(matched_backbone),
            "backbone_source": len(source_backbone),
            "backbone_target": len(target_backbone_keys),
            "embedding_matched": len(matched_embedding),
            "missing_keys": incompatible.missing_keys,
            "unexpected_keys": incompatible.unexpected_keys,
        }
