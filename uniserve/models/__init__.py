"""Capability classes, model inputs and outputs, and concrete model packages.

The capability classes (``CausalLM``, ``DiffusionLM``, the encoders,
denoisers and decoders) and their inputs and outputs are exported here. The
concrete model packages are the subpackages beside them.
``uniserve.models.registry`` maps the one architecture a checkpoint declares to
a package here and reads that package's module-level interface:
``read_config`` normalizes checkpoint metadata into a typed config and may
read tensor headers through ``sources``, the loader's resolution of the
sources ``config_sources`` names (downloaded for a Hub checkpoint, or
header-only for a dummy Hub load); ``Model`` builds the module tree
from that config; ``checkpoint_sources`` and ``checkpoint_mappings`` place
checkpoint tensors into the tree; ``precisions(config)`` names the weight
precision presets of a configuration and ``checkpoint_precision(config)`` is
its base configuration for a calibrated ModelOpt checkpoint; a package may
also offer a ``weight_config(config, *, preset, **components)`` factory for
per-component selections; ``entry_points`` declares the methods
serving ranks may call; and ``image_processor`` and ``flow_prompt`` describe
caller-side input preparation. Packages absent from that catalog serve other
roles: ``siglip`` is a vision tower that other models compose, ``qwen3_vl``
is the Qwen3-VL vision tower and multimodal text encoding that MiniMax H3
composes, and ``stub`` holds the deterministic weightless model a worker
builds when launched without a checkpoint.
"""

from .causal_lm import CausalLM
from .decoder import (
    AudioDecoder,
    ImageDecoder,
    VideoDecoder,
    VideoPostprocessor,
)
from .denoiser import (
    Condition,
    ConditionRole,
    ConditionTiles,
    Denoiser,
    ImageDenoiser,
    VideoDenoiser,
    VideoSize,
)
from .diffusion_lm import (
    CanvasTokens,
    DiffusionLM,
    DiffusionLMInput,
    SelfConditioning,
)
from .encoder import (
    AudioEncoder,
    Encoder,
    PatchEncoder,
    TextConditioner,
    TextEncoder,
    VideoEncoder,
)
from .inputs import (
    DEFAULT_COMPONENT,
    ComponentEntry,
    DenoiserInput,
    EmbeddingReplacement,
    EntryPoint,
    LatentInput,
    TextInput,
    TextSize,
    VisionInput,
)
from .logits import Logits, VocabShard
from .transformer import PhasedLayer, TransformerDecoder, TransformerEncoder

__all__ = [
    "DenoiserInput",
    "EmbeddingReplacement",
    "ComponentEntry",
    "DEFAULT_COMPONENT",
    "EntryPoint",
    "LatentInput",
    "TextInput",
    "TextSize",
    "VisionInput",
    "Logits",
    "VocabShard",
    "CausalLM",
    "DiffusionLMInput",
    "CanvasTokens",
    "SelfConditioning",
    "DiffusionLM",
    "PhasedLayer",
    "TransformerDecoder",
    "TransformerEncoder",
    "Denoiser",
    "ImageDenoiser",
    "VideoDenoiser",
    "VideoSize",
    "Condition",
    "ConditionRole",
    "ConditionTiles",
    "ImageDecoder",
    "AudioDecoder",
    "AudioEncoder",
    "VideoDecoder",
    "VideoEncoder",
    "VideoPostprocessor",
    "Encoder",
    "PatchEncoder",
    "TextConditioner",
    "TextEncoder",
]
