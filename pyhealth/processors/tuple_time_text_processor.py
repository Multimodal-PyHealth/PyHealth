from typing import Any, List, Tuple, Optional, Union
import torch
import logging
from .base_processor import FeatureProcessor, ModalityType, TemporalFeatureProcessor
from . import register_processor

logger = logging.getLogger(__name__)

@register_processor("tuple_time_text")
class TupleTimeTextProcessor(TemporalFeatureProcessor):
    """Processes (text, time_diff) tuples for multimodal temporal fusion.
    
    Converts paired text and temporal data into a format suitable for models
    that need to distinguish between different modality types automatically.
    
    If `tokenizer_model` is provided, the text will be tokenized using a HuggingFace
    AutoTokenizer, and the output will differ from the raw text version.
    """
    
    def __init__(
        self, 
        type_tag: str = "note",
        tokenizer_model: Optional[str] = None,
        max_length: int = 128,
        padding: bool = True,
        truncation: bool = True,
    ):
        """Initialize the processor.
        
        Args:
            type_tag: Modality identifier for automatic routing. Default: "note"
            tokenizer_model: Name or path of the HuggingFace tokenizer to use.
                If None, texts are returned as raw strings. Default: None
            max_length: Maximum sequence length for tokenization. Default: 128
            padding: Whether to pad sequences to max_length. Default: True
            truncation: Whether to truncate sequences to max_length. Default: True
        """
        super().__init__()
        self.type_tag = type_tag
        self.tokenizer_model = tokenizer_model
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
        self.tokenizer = None
        if self.tokenizer_model is not None:
            try:
                from transformers import AutoTokenizer
                # Suppress tokenizer warnings
                logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model)
            except ImportError:
                raise ImportError(
                    "The 'transformers' library is required when 'tokenizer_model' is provided. "
                    "Please install it via `pip install transformers`."
                )
            except Exception as e:
                raise ValueError(f"Failed to load tokenizer '{self.tokenizer_model}': {e}")

    def process(self, value: Tuple[List[str], List[float]]) -> Union[Tuple[List[str], torch.Tensor, torch.Tensor, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]]:
        """Process a tuple of texts and time differences.
        
        Args:
            value: Tuple containing:
                - List[str]: Text entries (clinical notes, observations, etc.)
                - List[float]: Time differences corresponding to each text entry.
                    NaN values are treated as missing timesteps: they are replaced
                    with 0.0 in the time tensor and marked False in ``time_mask``.
        
        Returns:
            If tokenizer_model is None:
                Tuple containing:
                    - List[str]: Original text entries (unmodified)
                    - torch.Tensor: 1D float tensor of time differences, NaN→0.0 [shape: (N,)]
                    - torch.Tensor: Boolean mask, True=real timestep [shape: (N,)]
                    - str: Type tag for modality routing
            
            If tokenizer_model is provided:
                Tuple containing:
                    - torch.Tensor: input_ids [shape: (N, max_length)]
                    - torch.Tensor: attention_mask [shape: (N, max_length)]
                    - torch.Tensor: token_type_ids [shape: (N, max_length)]
                    - torch.Tensor: time differences, NaN→0.0 [shape: (N,)]
                    - torch.Tensor: Boolean mask, True=real timestep [shape: (N,)]
                    - str: Type tag
        """
        texts, time_diffs = value
        time_tensor = torch.tensor(time_diffs, dtype=torch.float32)

        # Boolean mask: True where time is a real observation, False where NaN (missing).
        # NaN slots are zeroed out so downstream ops never receive NaN.
        time_mask = ~torch.isnan(time_tensor)          # (N,)  True = real
        time_tensor = time_tensor.nan_to_num(nan=0.0)  # (N,)  safe for all ops

        if self.tokenizer is not None:
            # Tokenize the list of texts
            encoded = self.tokenizer(
                texts,
                padding="max_length" if self.padding else False,
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            
            # Not all tokenizers return token_type_ids (e.g. RoBERTa might not, BERT does)
            if "token_type_ids" in encoded:
                token_type_ids = encoded["token_type_ids"]
            else:
                token_type_ids = torch.zeros_like(input_ids)

            return input_ids, attention_mask, token_type_ids, time_tensor, time_mask, self.type_tag

        return texts, time_tensor, time_mask, self.type_tag
    
    def size(self):
        """Return the size of the processor vocabulary (not applicable for this processor)."""
        if self.tokenizer is not None:
            return self.tokenizer.vocab_size
        return None
    
    def is_token(self) -> bool:
        """Returns True if the processor outputs discrete tokens (when tokenizer is used)."""
        return self.tokenizer is not None

    def schema(self) -> tuple[str, ...]:
        """Returns the schema of the processed feature."""
        if self.tokenizer is not None:
            # "value"=input_ids, "mask"=attention_mask, "time_mask"=valid timestep mask
            return ("value", "mask", "token_type_ids", "time", "time_mask", "type_tag")
        return ("text", "time", "time_mask", "type_tag")

    def dim(self) -> tuple[int, ...]:
        """Number of dimensions for each output tensor."""
        if self.tokenizer is not None:
            # input_ids (N,L), attention_mask (N,L), token_type_ids (N,L), time (N,), time_mask (N,)
            return (2, 2, 2, 1, 1)
        return (0, 1, 1, 0)  # text list, time (N,), time_mask (N,), type_tag str

    def modality(self) -> ModalityType:
        """Clinical text → TEXT modality."""
        return ModalityType.TEXT

    def value_dim(self) -> int:
        """Tokenizer vocabulary size (used with transformer encoder).
        Returns 0 if no tokenizer is loaded."""
        return self.tokenizer.vocab_size if self.tokenizer is not None else 0

    def process_temporal(self, value) -> dict:
        """Return dict output for UnifiedMultimodalEmbeddingModel.

        Requires ``tokenizer_model`` to be set (raw strings are not
        litdata-serialisable and cannot be embedded without tokenisation).

        Returns:
            {
                "value":     LongTensor  (N, L)  – input_ids
                "mask":      LongTensor  (N, L)  – attention_mask
                "time":      FloatTensor (N,)    – time offsets, NaN replaced with 0.0
                "time_mask": BoolTensor  (N,)    – True = real timestep, False = missing
            }

        Raises:
            ValueError: If processor was created without a tokenizer.
        """
        if self.tokenizer is None:
            raise ValueError(
                "TupleTimeTextProcessor.process_temporal() requires a tokenizer. "
                "Pass tokenizer_model='...' when creating the processor."
            )
        result = self.process(value)  # (input_ids, attn_mask, type_ids, time, time_mask, tag)
        return {
            "value":     result[0],  # input_ids       (N, L)
            "mask":      result[1],  # attention_mask  (N, L)
            "time":      result[3],  # time offsets    (N,)
            "time_mask": result[4],  # valid mask      (N,)  True=real
        }

    def __repr__(self):
        if self.tokenizer_model:
            return f"TupleTimeTextProcessor(type_tag='{self.type_tag}', tokenizer='{self.tokenizer_model}')"
        return f"TupleTimeTextProcessor(type_tag='{self.type_tag}')"
