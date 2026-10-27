"""Deterministic weightless model that stands in for a checkpoint.

A worker launched with ``no_model`` builds ``Model`` and ``image_processor``
instead of loading a checkpoint, so the serving path runs end to end without
weights and yields predictable tokens, cache writes and images. No checkpoint
catalog registers ``Model``.
"""

from .config import Config
from .denoiser import Denoiser
from .model import (
    STUB_EOS_TOKEN_ID,
    STUB_IMG_START_TOKEN_ID,
    STUB_STALL_TOKEN_ID,
    Model,
    TokenCount,
    entry_points,
)
from .processing import image_processor

# The revision of how this package derives its parameters from its
# checkpoint: its tensor mappings, fusions and conversions
# (`uniserve.numerics.loading`).
LOADING = "models.stub/1"

__all__ = [
    "image_processor",
    "Config",
    "Denoiser",
    "Model",
    "STUB_EOS_TOKEN_ID",
    "STUB_IMG_START_TOKEN_ID",
    "STUB_STALL_TOKEN_ID",
    "TokenCount",
    "entry_points",
]
