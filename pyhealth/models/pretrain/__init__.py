"""Self-supervised pretraining models for PyHealth multimodal sequences.

Models:
- :class:`MultimodalMaskedAutoencoder` — true MAE with a transformer decoder.
- :class:`MultimodalSimMIM` — SimMIM-style linear-head reconstruction.
- :class:`MultimodalIJEPA` — I-JEPA latent predictive architecture.
- :class:`MultimodalVJEPA` — V-JEPA: multi-scale spans + location/scale-aware
  per-position latent prediction (extends I-JEPA).

Utilities:
- :class:`UnifiedMaskGenerator` — masking strategies for unified event seqs.
- :func:`apply_mask_token` — replace masked positions with a learnable token.
"""

from .jepa import MultimodalIJEPA, MultimodalVJEPA
from .mae import MultimodalMaskedAutoencoder, PerModalityMAEDecoder
from .masking import UnifiedMaskGenerator, apply_mask_token, random_mask_like
from .rope import RoPEMultiHeadedAttention, RoPETransformerLayer, RotaryPositionEmbedding
from .simmim import MultimodalSimMIM
from .trainer import PretrainTrainer

__all__ = [
    "MultimodalMaskedAutoencoder",
    "MultimodalSimMIM",
    "MultimodalIJEPA",
    "MultimodalVJEPA",
    "PerModalityMAEDecoder",
    "UnifiedMaskGenerator",
    "apply_mask_token",
    "random_mask_like",
    "RotaryPositionEmbedding",
    "RoPEMultiHeadedAttention",
    "RoPETransformerLayer",
    "PretrainTrainer",
]
