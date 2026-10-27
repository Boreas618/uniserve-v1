"""Qwen3.5 decoder layers: gated full attention and gated delta attention.

``Transformer`` is the Qwen3.5 language model backbone. Each decoder layer
mixes tokens either by full attention, whose output a sigmoid gate scales
per channel, or by gated delta attention, a linear attention with a
recurrent state per request (``uniserve.nn.gated_delta``); every layer then
applies a SiLU-gated MLP. Normalizations store zero-centered weights
(``GemmaRMSNorm``). The arithmetic follows the Transformers reference
(``transformers.models.qwen3_5``): the residual stream is rounded to the
activation dtype after each sublayer, and query and key normalization,
rotation and MLP gating round after each operation.
"""

from __future__ import annotations

import torch
from torch import nn

from uniserve.models import TransformerDecoder
from uniserve.nn import functional
from uniserve.nn.attention import Attention as ScaledAttention
from uniserve.nn.attention import AttentionBatch
from uniserve.nn.functional import Rounding
from uniserve.nn.gated_delta import GatedDeltaNet
from uniserve.nn.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from uniserve.nn.mlp import GatedMLP
from uniserve.nn.norm import GatedRMSNorm, GemmaRMSNorm
from uniserve.nn.rope import RotaryEmbedding

from .config import Config


class Attention(nn.Module):
    """Gated full attention with partially rotated, normalized heads.

    The query projection holds, per head, the query and then the gate of
    that head (``2 * num_attention_heads`` heads of ``head_dim``). Queries
    and keys are RMS-normalized over the head, their leading
    ``partial_rotary_factor`` share is rotated split-half, and the attended
    values are multiplied by ``sigmoid(gate)`` before the output projection.
    """

    function = "models.qwen3_5.transformer.Attention/1"

    def __init__(self, config: Config, cache_name: str | None):
        super().__init__()
        self.head_dim = config.head_dim
        self.qkv = QKVParallelLinear(
            config.hidden_size,
            2 * config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            bias=config.attention_bias,
        )
        self.query_norm = GemmaRMSNorm(config.head_dim, config.rms_norm_eps)
        self.key_norm = GemmaRMSNorm(config.head_dim, config.rms_norm_eps)
        self.attention = ScaledAttention(
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            cache_name=cache_name,
        )
        self.output = RowParallelLinear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.rotary = RotaryEmbedding(
            config.head_dim,
            theta=config.rope_theta,
            partial_rotary_factor=config.partial_rotary_factor,
            sections=config.mrope_sections,
            max_position_embeddings=config.max_position_embeddings,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        attention: AttentionBatch,
    ) -> torch.Tensor:
        """Attend over packed tokens at their rotary ``positions``.

        Positions are ``[tokens]``, or ``[axes, tokens]`` multimodal
        coordinates with ``mrope_sections``; one-dimensional positions place
        every axis at the same coordinate.
        """
        if self.rotary.sections is not None and positions.ndim == 1:
            positions = positions.expand(len(self.rotary.sections), -1)
        cos, sin = self.rotary(
            positions,
            dtype=torch.float32,
            sequence_length=positions.shape[-1],
        )

        # hidden: [tokens, hidden_size]. The query branch holds each local
        # head's query and gate side by side: [tokens, heads, 2, head_dim].
        values = self.qkv(hidden)
        tokens = hidden.shape[0]
        paired = values["q"].reshape(tokens, -1, 2, self.head_dim)
        query = paired[:, :, 0].contiguous()
        gate = paired[:, :, 1].reshape(tokens, -1)
        key = values["k"].reshape(tokens, -1, self.head_dim)
        value = values["v"].reshape(tokens, -1, self.head_dim)
        query, key = functional.qk_norm_rope(
            query,
            key,
            (self.query_norm.scale,),
            (self.key_norm.scale,),
            (cos,),
            (sin,),
            eps=self.query_norm.eps,
            axis_dims=(self.head_dim,),
            rounding=Rounding.STEPWISE,
        )
        attended = self.attention(query, key, value.to(query.dtype), attention)
        # The gate scales each attended channel in the activation dtype; the
        # row-parallel output projection sums the tensor-parallel shards.
        return self.output(attended.flatten(1) * torch.sigmoid(gate))


class LinearAttention(nn.Module):
    """Gated delta attention between its projections.

    One merged projection yields the convolution's query, key and value
    channels, the output gate ``z`` and the per-head inputs ``b`` and ``a``
    of the write strength and the decay; ``GatedDeltaNet`` mixes the tokens,
    a gated RMS norm normalizes each value head and gates it by
    ``silu(z)``, and a row-parallel projection returns to the hidden width.
    """

    function = "models.qwen3_5.transformer.LinearAttention/1"

    def __init__(self, config: Config, cache_name: str):
        super().__init__()
        keys = config.linear_num_key_heads * config.linear_key_head_dim
        values = config.linear_num_value_heads * config.linear_value_head_dim
        heads = config.linear_num_value_heads
        self.value_dim = config.linear_value_head_dim
        self.projection = MergedColumnParallelLinear(
            config.hidden_size,
            {
                "q": keys,
                "k": keys,
                "v": values,
                "z": values,
                "b": heads,
                "a": heads,
            },
            bias=False,
        )
        self.mixer = GatedDeltaNet(
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_conv_kernel_dim,
            cache_name=cache_name,
            state_dtype=config.state_dtype,
        )
        self.norm = GatedRMSNorm(
            config.linear_value_head_dim, config.rms_norm_eps
        )
        self.output = RowParallelLinear(values, config.hidden_size, bias=False)

    def forward(
        self, hidden: torch.Tensor, attention: AttentionBatch
    ) -> torch.Tensor:
        values = self.projection(hidden)
        tokens = hidden.shape[0]
        mixed = torch.cat((values["q"], values["k"], values["v"]), dim=-1)
        mixed = self.mixer(mixed, values["a"], values["b"], attention)
        gate = values["z"].reshape(tokens, -1, self.value_dim)
        return self.output(self.norm(mixed, gate).flatten(1))


class TransformerLayer(nn.Module):
    """Pre-normalized token mixer and MLP, each added to the residual stream.

    The layer receives and returns the complete stream: each sublayer's
    output is added to it and the sum rounded to the activation dtype, as
    the Transformers reference adds them.
    """

    function = "models.qwen3_5.transformer.TransformerLayer/1"

    def __init__(self, config: Config, linear: bool, cache_name: str | None):
        super().__init__()
        self.input_norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention: Attention | None = None
        self.linear_attention: LinearAttention | None = None
        if linear:
            assert cache_name is not None
            self.linear_attention = LinearAttention(config, cache_name)
        else:
            self.attention = Attention(config, cache_name)
        self.post_attention_norm = GemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = GatedMLP(
            config.hidden_size,
            config.intermediate_size,
            activation="silu",
            rounding=Rounding.STEPWISE,
        )

    def forward(self, hidden, residual, positions, attention):
        # Without a separate residual stream ``residual`` is always None and
        # ``hidden`` the complete stream.
        normalized = self.input_norm(hidden)
        if self.linear_attention is not None:
            mixed = self.linear_attention(normalized, attention)
        else:
            assert self.attention is not None
            mixed = self.attention(normalized, positions, attention)
        hidden = hidden + mixed
        hidden = hidden + self.mlp(self.post_attention_norm(hidden))
        return hidden, None


class Transformer(TransformerDecoder):
    """Qwen3.5 decoder stack: embedding, mixed decoder layers and final norm."""

    function = "models.qwen3_5.transformer.Transformer/1"

    def __init__(self, config: Config):
        # Layer keys are global layer indices, kept by pipeline binding, so
        # parameter paths and cache names match across stages.
        super().__init__(
            VocabParallelEmbedding(config.vocab_size, config.hidden_size),
            nn.ModuleDict(
                {
                    str(index): TransformerLayer(
                        config,
                        config.linear(index),
                        _cache_name(config, index),
                    )
                    for index in range(config.num_hidden_layers)
                }
            ),
            GemmaRMSNorm(config.hidden_size, config.rms_norm_eps),
            separate_residual=False,
        )
        self.config = config


def _cache_name(config: Config, layer: int) -> str:
    """The cache layer of decoder layer ``layer``: its attention's K/V
    pages, or its gated delta attention's state per request.
    """  # noqa: D205
    if config.linear(layer):
        return f"backbone.layers.{layer}.linear_attention.mixer"
    return f"backbone.layers.{layer}.attention.attention"
