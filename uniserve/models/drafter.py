"""The draft model a causal language model proposes speculative tokens with.

A :class:`Drafter` is the capability a causal LM supplies for speculative
decoding (Leviathan, Kalman and Matias, "Fast Inference from Transformers
via Speculative Decoding", 2023): from the target model's final hidden
state at a position and the embedding of the token that follows it, it
computes a hidden state whose logits, through the target's vocabulary head,
predict the token after that one. Its output feeds its next step as the
hidden state, so a chain of steps proposes several tokens; the target then
verifies them in one forward.

The drafter's layers attend to a cache of their own, its *draft cache*:
entry ``i`` holds what the drafter computed from the target's hidden state
at position ``i`` and the token at ``i + 1``, so it depends on the tokens
before ``i + 2``. The cache is optional derived state: a request without it
decodes correctly, verifying no drafts. ``cache_config`` declares its layers;
the prefix cache keeps them in a group of their own.
"""

from __future__ import annotations

import torch
from torch import nn

from uniserve import prefix_cache
from uniserve.nn.attention import AttentionBatch

from .transformer import _cache_layers


class Drafter(nn.Module):
    """Draft layers that read the target's hidden states.

    Subclasses hold their layers in ``layers`` (a ``ModuleDict`` of decoder
    layers whose attention names a ``cache_name``) and implement
    :meth:`forward`. The drafter shares the target's token embedding and
    vocabulary head, which the caller applies.
    """

    function = "models.drafter.Drafter/1"

    layers: nn.ModuleDict

    @property
    def cache_config(self) -> prefix_cache.Config:
        """The draft cache's layers, by cache name."""
        return prefix_cache.Config(_cache_layers(self.layers))

    def forward(
        self,
        embeddings: torch.Tensor,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        attention: AttentionBatch,
        *,
        step: int = 0,
    ) -> torch.Tensor:
        """The drafter's hidden states of draft step ``step``.

        ``embeddings`` are ``[tokens, hidden]`` embeddings of the tokens
        that follow the positions of ``hidden``, the target's final hidden
        states (or the drafter's own output of the step before), at rotary
        ``positions``; ``attention`` addresses the draft cache. Returns
        ``[tokens, hidden]`` states the vocabulary head projects and the
        next step reads.
        """
        raise NotImplementedError


__all__ = ["Drafter"]
