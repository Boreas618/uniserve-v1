"""Qwen3.5 numerical models and checkpoint definitions.

``uniserve.models.registry`` selects this package for
``Qwen3_5ForConditionalGeneration`` checkpoints and reads the names exported
here as its model package contract: configuration normalization, the model
class, its entry points, checkpoint sources and mappings, and precision
presets. The package serves the checkpoint's language model: gated full
attention and gated delta attention layers, whose recurrent state each
request keeps per request slot, with the multi-token prediction layers
loaded beside them. The vision tower is not loaded.
"""

from .config import Config, config_sources, read_config
from .model import Model, MultiTokenPredictor, entry_points
from .transformer import (
    Attention,
    LinearAttention,
    Transformer,
    TransformerLayer,
)
from .weights import (
    checkpoint_mappings,
    checkpoint_precision,
    checkpoint_sources,
    precisions,
)

# The revision of how this package derives its parameters from its
# checkpoint: its tensor mappings, fusions and conversions
# (`uniserve.numerics.loading`).
LOADING = "models.qwen3_5/1"

# The package serves text alone: the loading contract reads ``None`` as no
# image preprocessing and no flow prompt.
image_processor = None
flow_prompt = None

__all__ = [
    "config_sources",
    "image_processor",
    "flow_prompt",
    "Config",
    "read_config",
    "Model",
    "MultiTokenPredictor",
    "entry_points",
    "Attention",
    "LinearAttention",
    "Transformer",
    "TransformerLayer",
    "checkpoint_sources",
    "checkpoint_mappings",
    "precisions",
    "checkpoint_precision",
]
