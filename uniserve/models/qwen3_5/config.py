"""Immutable Qwen3.5 text architecture and checkpoint metadata normalization.

``read_config`` turns a ``Qwen3_5ForConditionalGeneration`` checkpoint's
``config.json`` into the frozen ``Config`` that ``Model`` consumes: the
``text_config`` of its language model. The vision tower the checkpoint also
carries is not part of the model, so the model serves text alone.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from uniserve import loading
from uniserve.loading import checkpoint

from ...rotary import read_rotary

# The token mixer of each decoder layer, as ``layer_types`` names it.
FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"

# The recurrent state element types ``mamba_ssm_dtype`` may name.
_STATE_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable numerical parameters of the Qwen3.5 language model.

    Field names follow the keys of the checkpoint's ``text_config``.
    ``layer_types`` names each decoder layer's token mixer: gated full
    attention over ``num_attention_heads`` query heads of ``head_dim``, or
    gated delta attention over ``linear_num_key_heads`` key heads of
    ``linear_key_head_dim`` and ``linear_num_value_heads`` value heads of
    ``linear_value_head_dim`` behind a causal convolution of
    ``linear_conv_kernel_dim`` taps. ``partial_rotary_factor`` of each
    full-attention head rotates, by the interleaved multimodal sections
    ``mrope_sections``. ``state_dtype`` stores the recurrent state.
    ``mtp_num_hidden_layers`` multi-token prediction layers follow the
    model, sharing its embedding and head.

    Raises:
        ValueError: From ``__post_init__`` when a field has the wrong type or
            an invalid value.
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    mrope_sections: tuple[int, ...] | None
    max_position_embeddings: int
    attention_bias: bool
    tie_word_embeddings: bool
    layer_types: tuple[str, ...]
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    state_dtype: torch.dtype
    mtp_num_hidden_layers: int

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"Qwen3.5 {name} must be a positive integer")
        if (
            not isinstance(self.mtp_num_hidden_layers, int)
            or isinstance(self.mtp_num_hidden_layers, bool)
            or self.mtp_num_hidden_layers < 0
        ):
            raise ValueError(
                "Qwen3.5 mtp_num_hidden_layers must be a non-negative integer"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "Qwen3.5 attention heads must be divisible by KV heads"
            )
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError(
                "Qwen3.5 linear value heads must be divisible by its key heads"
            )
        if self.linear_conv_kernel_dim < 2:
            raise ValueError(
                "Qwen3.5 linear_conv_kernel_dim must be at least 2"
            )
        if (
            not isinstance(self.layer_types, tuple)
            or len(self.layer_types) != self.num_hidden_layers
            or any(
                kind not in (FULL_ATTENTION, LINEAR_ATTENTION)
                for kind in self.layer_types
            )
            or FULL_ATTENTION not in self.layer_types
        ):
            raise ValueError(
                "Qwen3.5 layer_types must name full_attention or "
                "linear_attention for every layer, with at least one "
                "full_attention layer"
            )
        for name in ("rms_norm_eps", "rope_theta"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Qwen3.5 {name} must be finite and positive")
        rotated = int(self.head_dim * self.partial_rotary_factor)
        if (
            not 0 < self.partial_rotary_factor <= 1
            or rotated < 2
            or rotated % 2
        ):
            raise ValueError(
                "Qwen3.5 partial_rotary_factor must rotate a positive even "
                "width of each head"
            )
        sections = self.mrope_sections
        if sections is not None and (
            not isinstance(sections, tuple)
            or any(type(width) is not int or width < 1 for width in sections)
            or sum(sections) != rotated // 2
        ):
            raise ValueError(
                "Qwen3.5 mrope_sections must be positive widths summing to "
                "half the rotated head width"
            )
        for name in ("attention_bias", "tie_word_embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"Qwen3.5 {name} must be boolean")
        if self.state_dtype not in _STATE_DTYPES.values():
            raise ValueError("Qwen3.5 state_dtype must be a floating type")

    def linear(self, layer: int) -> bool:
        """Whether decoder layer ``layer`` mixes tokens by gated delta
        attention rather than full attention.
        """  # noqa: D205
        return self.layer_types[layer] == LINEAR_ATTENTION


def _integer(config: Mapping[str, object], name: str, default=None) -> int:
    """Read an integer setting, rejecting booleans; required without a
    default.
    """  # noqa: D205
    raw = config.get(name, default)
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"Qwen3.5 config requires integer field {name!r}")
    return raw


def _number(config: Mapping[str, object], name: str, default: float) -> float:
    raw = config.get(name, default)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ValueError(f"Qwen3.5 config field {name!r} must be numeric")
    return float(raw)


def _boolean(config: Mapping[str, object], name: str, default: bool) -> bool:
    raw = config.get(name, default)
    if not isinstance(raw, bool):
        raise ValueError(f"Qwen3.5 config field {name!r} must be boolean")
    return raw


def read_config(
    root: Path,
    io: loading.Config,
    *,
    sources: Mapping[str, checkpoint.Source],
) -> Config:
    """Normalize checkpoint metadata into the immutable language model config.

    The checkpoint declares ``Qwen3_5ForConditionalGeneration``, the
    architecture the model catalog selected this package by, and nests its
    language model under ``text_config``. Absent fields take Transformers'
    ``Qwen3_5TextConfig`` defaults. The layers' token mixers come from
    ``layer_types``, or from ``full_attention_interval`` (every fourth
    layer full attention unless it says otherwise) as Transformers derives
    them.

    Args:
        root: Local checkpoint directory containing ``config.json``.
        io: Loading options of the package ``read_config`` contract; unused.
        sources: The resolved ``config_sources``; Qwen3.5 declares none.

    Returns:
        The validated ``Config``.

    Raises:
        ValueError: For invalid metadata, including another architecture, a
            missing ``text_config``, a field of the wrong type or value, an
            ungated attention output, a rotary recipe other than the default
            one, or multi-token prediction layers with embeddings of their
            own.
    """
    config = json.loads((root / "config.json").read_text())
    architectures = config.get("architectures")
    if architectures != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError(
            "a Qwen3.5 checkpoint declares the architecture "
            "Qwen3_5ForConditionalGeneration"
        )
    text = config.get("text_config")
    if not isinstance(text, Mapping):
        raise ValueError("a Qwen3.5 checkpoint requires a text_config object")

    # The attention output is gated by a sigmoid of the second half of each
    # query head's projection, and the linear attention output by a SiLU:
    # the only gating Transformers implements for the family.
    if not _boolean(text, "attn_output_gate", True):
        raise ValueError("Qwen3.5 attention outputs are gated")
    if text.get("output_gate_type", "swish") not in ("swish", "silu"):
        raise ValueError("Qwen3.5 linear attention outputs gate by SiLU")
    if text.get("hidden_act", "silu") != "silu":
        raise ValueError("Qwen3.5 MLPs gate by SiLU")
    if _boolean(text, "mtp_use_dedicated_embeddings", False):
        raise ValueError(
            "Qwen3.5 multi-token prediction shares the model's embedding"
        )

    layers = _integer(text, "num_hidden_layers", 32)
    layer_types = text.get("layer_types")
    if layer_types is None:
        interval = _integer(text, "full_attention_interval", 4)
        layer_types = [
            LINEAR_ATTENTION if (index + 1) % interval else FULL_ATTENTION
            for index in range(layers)
        ]
    if not isinstance(layer_types, list):
        raise ValueError("Qwen3.5 layer_types must be a list")

    max_position_embeddings = _integer(text, "max_position_embeddings", 32768)
    rotary = read_rotary(
        text,
        owner="Qwen3.5",
        default_theta=10000.0,
        default_original=max_position_embeddings,
    )
    if rotary.kind != "default":
        raise ValueError(
            f"Qwen3.5 serves the default rotary recipe, not {rotary.kind!r}"
        )
    parameters = text.get("rope_parameters") or {}
    # Transformers' Qwen3.5 rotates a quarter of each head unless the
    # checkpoint states another factor.
    partial = (
        rotary.partial_rotary_factor
        if "partial_rotary_factor" in text
        or "partial_rotary_factor" in parameters
        else 0.25
    )
    if not isinstance(partial, (int, float)) or isinstance(partial, bool):
        raise ValueError("Qwen3.5 partial_rotary_factor must be numeric")
    sections = parameters.get("mrope_section")
    if sections is not None and (
        not isinstance(sections, list)
        or not parameters.get("mrope_interleaved", True)
    ):
        raise ValueError(
            "Qwen3.5 mrope_section must be a list of interleaved widths"
        )

    state_dtype = text.get("mamba_ssm_dtype", "float32")
    if state_dtype not in _STATE_DTYPES:
        raise ValueError(
            f"Qwen3.5 mamba_ssm_dtype {state_dtype!r} is not one of "
            f"{sorted(_STATE_DTYPES)}"
        )

    return Config(
        vocab_size=_integer(text, "vocab_size", 248320),
        hidden_size=_integer(text, "hidden_size", 4096),
        intermediate_size=_integer(text, "intermediate_size", 12288),
        num_hidden_layers=layers,
        num_attention_heads=_integer(text, "num_attention_heads", 16),
        num_key_value_heads=_integer(text, "num_key_value_heads", 4),
        head_dim=_integer(text, "head_dim", 256),
        rms_norm_eps=_number(text, "rms_norm_eps", 1e-6),
        rope_theta=float(rotary.theta),
        partial_rotary_factor=float(partial),
        mrope_sections=None if sections is None else tuple(sections),
        max_position_embeddings=max_position_embeddings,
        attention_bias=_boolean(text, "attention_bias", False),
        tie_word_embeddings=_boolean(
            config,
            "tie_word_embeddings",
            _boolean(text, "tie_word_embeddings", False),
        ),
        layer_types=tuple(layer_types),
        linear_conv_kernel_dim=_integer(text, "linear_conv_kernel_dim", 4),
        linear_key_head_dim=_integer(text, "linear_key_head_dim", 128),
        linear_value_head_dim=_integer(text, "linear_value_head_dim", 128),
        linear_num_key_heads=_integer(text, "linear_num_key_heads", 16),
        linear_num_value_heads=_integer(text, "linear_num_value_heads", 32),
        state_dtype=_STATE_DTYPES[state_dtype],
        mtp_num_hidden_layers=_integer(text, "mtp_num_hidden_layers", 0),
    )


# Checkpoint sources whose tensor headers read_config needs before module
# selection. Qwen3.5 derives every dimension from config.json, so it has none.
config_sources = ()
