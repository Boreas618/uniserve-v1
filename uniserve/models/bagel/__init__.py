"""BAGEL numerical models and checkpoint definitions.

BAGEL is a unified understanding and generation model. One mixture of
transformers (MoT) backbone carries a text expert for language tokens and a
flow expert for image latents; a SigLIP tower encodes input images; and a
FLUX autoencoder maps between pixels and the latents the flow expert
denoises. ``uniserve.models.registry`` selects this package for checkpoints
declaring ``BagelForConditionalGeneration``.
"""

from .config import Config, TransformerConfig, config_sources, read_config
from .denoiser import Denoiser
from .model import Model, entry_points
from .processing import flow_prompt, image_processor
from .transformer import Transformer, TransformerLayer
from .weights import (
    checkpoint_mappings,
    checkpoint_precision,
    checkpoint_sources,
    precisions,
)

# The revision of how this package derives its parameters from its
# checkpoint: its tensor mappings, fusions and conversions
# (`uniserve.numerics.loading`).
LOADING = "models.bagel/1"

__all__ = [
    "config_sources",
    "image_processor",
    "flow_prompt",
    "Config",
    "TransformerConfig",
    "read_config",
    "Denoiser",
    "Model",
    "entry_points",
    "Transformer",
    "TransformerLayer",
    "checkpoint_sources",
    "checkpoint_mappings",
    "precisions",
    "checkpoint_precision",
]
