"""Qwen3.5 language model composition and numerical entry points."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch
from torch import nn

from uniserve.models import (
    DEFAULT_COMPONENT,
    CausalLM,
    ComponentEntry,
    Drafter,
    EntryPoint,
)
from uniserve.nn.attention import AttentionBatch
from uniserve.nn.linear import Linear, VocabParallelHead
from uniserve.nn.norm import GemmaRMSNorm

from .config import Config
from .transformer import Transformer, TransformerLayer


class MultiTokenPredictor(Drafter):
    """The checkpoint's multi-token prediction layers, the model's drafter.

    Predicts the token after next from a position's final hidden state and
    the embedding of the token that follows it, as vLLM's
    ``Qwen3_5MultiTokenPredictor`` composes it: both are RMS-normalized,
    concatenated (embedding first) and projected to the hidden width, pass a
    full-attention decoder layer, and the final norm's output shares the
    model's vocabulary head and feeds the next draft step. Each layer's
    attention keeps its keys and values in the draft cache
    (``mtp.layers.<index>.attention.attention``).
    """

    function = "models.qwen3_5.model.MultiTokenPredictor/1"

    def __init__(self, config: Config):
        super().__init__()
        self.embedding_norm = GemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.hidden_norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.projection = Linear(
            2 * config.hidden_size, config.hidden_size, bias=False
        )
        self.layers = nn.ModuleDict(
            {
                str(index): TransformerLayer(
                    config, False, f"mtp.layers.{index}.attention.attention"
                )
                for index in range(config.mtp_num_hidden_layers)
            }
        )
        self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        embeddings: torch.Tensor,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        attention: AttentionBatch,
        *,
        step: int = 0,
    ) -> torch.Tensor:
        """Normalized hidden states of the prediction ``step`` layer.

        ``embeddings`` are the next tokens' embeddings and ``hidden`` the
        model's final hidden states at the same packed positions, both
        ``[tokens, hidden_size]``.
        """
        joined = torch.cat(
            (self.embedding_norm(embeddings), self.hidden_norm(hidden)), dim=-1
        )
        stream = self.projection(joined)
        layer = self.layers[str(step % len(self.layers))]
        stream, _ = layer(stream, None, positions, attention)
        return self.norm(stream)


class Model(CausalLM):
    """The same loaded Qwen3.5 language model for Python and serving callers.

    ``CausalLM`` supplies ``forward``, ``embed_input_ids`` and
    ``compute_logits`` over the ``Transformer`` backbone and a
    vocabulary-parallel head. ``mtp`` holds the checkpoint's multi-token
    prediction layers when it has any: the model's ``Drafter``, which a
    deployment that speculates runs after each forward.
    """

    function = "models.qwen3_5.model.Model/1"

    def __init__(self, config: Config):
        super().__init__(
            Transformer(config),
            VocabParallelHead(config.hidden_size, config.vocab_size),
        )
        self.config = config
        self.mtp = (
            MultiTokenPredictor(config)
            if config.mtp_num_hidden_layers
            else None
        )

        # Tied checkpoints store one embedding matrix; the head shares that
        # same Parameter. Construction precedes pipeline binding, so both
        # modules are still present.
        if config.tie_word_embeddings:
            embedding, head = self.backbone.embedding, self.lm_head
            if embedding is None or head is None:
                raise ValueError(
                    "tied embeddings require the embedding and head"
                )
            head.weight = embedding.weight


def entry_points(config: Config) -> Mapping[str, ComponentEntry]:
    """Declare the numerical methods serving ranks may invoke on this model.

    Qwen3.5 exports the model root as its single component. ``forward``
    runs on every pipeline stage; token embedding runs only on the first
    stage and logits only on the last.
    """
    return MappingProxyType(
        {
            DEFAULT_COMPONENT: ComponentEntry(
                "",
                (
                    EntryPoint("forward", groups=("tp", "sp", "pp")),
                    EntryPoint(
                        "embed_input_ids", stage="first", groups=("tp",)
                    ),
                    EntryPoint("compute_logits", stage="last", groups=("tp",)),
                ),
            ),
        }
    )
