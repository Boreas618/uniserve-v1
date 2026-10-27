"""Shared transformer traversal and mathematical activation communication."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from uniserve import prefix_cache
from uniserve.distributed import Communicator, DeviceMesh
from uniserve.distributed.tokens import TokenShard
from uniserve.nn.attention import AttentionBatch, AttentionParallelConfig
from uniserve.nn.functional import add_rms_norm
from uniserve.nn.norm import RMSNorm
from uniserve.nn.routing import IndexedRoutes
from uniserve.prefix_cache import mha


@dataclass(frozen=True, slots=True)
class CacheLayer:
    """Logical K/V geometry of one cache layer, before any partitioning.

    ``window`` is the layer's attention history bound (``None`` reads the
    whole history), and ``num_kv_heads`` counts every KV head of the
    architecture. Cache layers that agree on all three share one cache
    group of the unit pool.
    """

    window: int | None
    num_kv_heads: int
    head_dim: int


def _cache_layers(layers) -> dict[str, prefix_cache.StateConfig]:
    """The cache layers the decoder ``layers`` own, by cache name: every
    module with a ``cache_name`` and a ``cache_state`` method
    (``prefix_cache.StateConfig``), in the layers' dtype.
    """  # noqa: D205
    result = {}
    for layer in layers.values():
        # Parameters expose their logical dtype even for encoded weights.
        dtype = next(layer.parameters()).dtype
        for child in layer.modules():
            name = getattr(child, "cache_name", None)
            if name is not None and hasattr(child, "cache_state"):
                result[name] = child.cache_state(dtype)
    return result


class PhasedLayer(nn.Module):
    """A decoder layer whose evaluation splits after its attention sublayer.

    ``forward(hidden, residual, positions, attention)`` equals
    ``feed_forward(*attend(hidden, residual, positions, attention))``.
    ``attend`` evaluates the attention sublayer, which mixes tokens, and
    returns the layer's state after it as a tuple of ``[tokens, ...]``
    tensors; ``feed_forward`` evaluates the rest of the layer from that
    state and returns its ``(hidden, residual)`` streams. The rest acts on
    each token alone, so ``feed_forward`` may evaluate any subset of the
    state's rows. ``write_cache`` evaluates the layer only up to its prefix
    cache write and returns nothing, for a pass whose only product is the
    cache. ``TransformerDecoder`` uses these phases for its final layer; a
    final layer without them runs whole.
    """

    function = "models.transformer.PhasedLayer/1"

    def forward(self, hidden, residual, positions, attention):
        """Compose token-mixing attention and the per-token remainder."""
        return self.feed_forward(
            *self.attend(hidden, residual, positions, attention)
        )

    def attend(
        self, hidden, residual, positions, attention
    ) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def feed_forward(self, *state: torch.Tensor):
        raise NotImplementedError

    def write_cache(self, hidden, residual, positions, attention) -> None:
        raise NotImplementedError

    @property
    def state_widths(self) -> tuple[int, ...]:
        """Per-token widths of the tensors ``attend`` returns, in order.

        Every state tensor is ``[tokens, width]`` in the layer's stream
        dtype, so a caller can hold the state of a bounded token count in
        storage it allocates before any evaluation.
        """
        raise NotImplementedError


class TransformerDecoder(nn.Module):
    """Traverse resident layers, carrying a separate residual stream.

    Layers are called as ``layer(hidden, residual, positions, attention)`` and
    return ``(hidden, residual)``. With ``separate_residual`` (the default) a
    layer may defer its residual addition: ``hidden`` is its unsummed update
    and ``residual`` the stream it carries forward, and the first layer of
    the first stage receives ``residual=None``. Without it every layer
    receives and returns ``residual=None``, and ``hidden`` is the complete
    stream; architectures whose layer output rescales the whole stream
    cannot defer the addition. Pipeline peers exchange the carried values
    with their original dtypes; only the final stage normalizes and gathers
    the token shards. Placement binding selects resident modules before
    loading.

    Besides the complete pass (``forward``), two partial passes evaluate the
    model's final layer in its ``PhasedLayer`` phases: ``fill_cache`` stops
    at the final layer's prefix cache write, for prompts whose only product is
    the cache, and ``attend`` stops after the final layer's attention, so
    ``finish`` completes the layer and the output norm for selected rows
    alone. Every other layer, and every layer of an earlier pipeline stage,
    runs whole.
    """

    function = "models.transformer.TransformerDecoder/1"

    # Pipeline binding drops the embedding outside the first stage and the
    # output norm outside the last one.
    embedding: nn.Module | None
    norm: nn.Module | None
    # Factor ``embed_input_ids`` multiplies embedding-table rows by; an
    # architecture that scales its token embeddings overrides it.
    embedding_scale: float = 1.0

    # Recorded by parallelize_ once the decoder is bound to its partition.
    _parallel_mesh: DeviceMesh
    _attention_parallel: AttentionParallelConfig

    def __init__(
        self,
        embedding,
        layers: nn.ModuleDict,
        norm: nn.Module,
        *,
        default_route: str | None = None,
        separate_residual: bool = True,
    ):
        super().__init__()
        if not layers:
            raise ValueError(
                "a transformer decoder requires at least one layer"
            )
        self.embedding, self.layers, self.norm = embedding, layers, norm
        self.separate_residual = separate_residual
        self.mesh = DeviceMesh(ranks=(0,), shape=(1,), axes=("tp",), rank=0)
        self._pipeline = Communicator()
        self._tokens = Communicator()
        self.hidden_size = embedding.embedding_dim
        self.vocab_size = embedding.num_embeddings
        # Logical cache identities and geometry survive pipeline pruning.
        # Resource owners use this order to describe global K/V transfers,
        # including each cache group's complete layer axis, without retaining
        # parameters or modules belonging to another pipeline stage. Only
        # paged layers transfer; a request's per-slot states do not.
        paged: list[tuple[str, mha.Config]] = []
        for name, layout in _cache_layers(layers).items():
            if layout.indexing != "tokens":
                continue
            if not isinstance(layout, mha.Config):
                raise TypeError(
                    "a paged cache layer type subclasses mha.Config"
                )
            paged.append((name, layout))
        self.cache_names = tuple(name for name, _ in paged)
        self.cache_layers = tuple(
            CacheLayer(layout.window, layout.num_kv_heads, layout.head_dim)
            for _, layout in paged
        )
        if default_route is not None and (
            not isinstance(norm, nn.ModuleDict) or default_route not in norm
        ):
            raise ValueError(
                "a default expert route must name one of the decoder's "
                "output norms"
            )
        self._default_route = default_route

    @property
    def cache_config(self) -> prefix_cache.Config:
        return prefix_cache.Config(_cache_layers(self.layers))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._pipeline.rank != 0 or self.embedding is None:
            raise ValueError(
                "token embeddings belong to the first pipeline stage"
            )
        return self.embedding(input_ids)

    def forward(
        self,
        embeddings: torch.Tensor | None,
        positions: torch.Tensor,
        attention: AttentionBatch,
        *,
        routes: IndexedRoutes | None = None,
        deepstack: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Evaluate every layer and, on the final stage, the output norm.

        ``routes`` assign the packed tokens of a routed decoder to its
        experts; without them every token takes the default route. Returns
        the final stage's normalized ``[tokens, hidden]`` rows, or the
        activations an earlier stage forwards.
        """
        traversal = self._traverse(
            embeddings,
            positions,
            attention,
            routes,
            split=False,
            deepstack=deepstack,
        )
        hidden, residual, local_routes = (
            traversal.hidden,
            traversal.residual,
            traversal.routes,
        )
        if self._pipeline.rank != self._pipeline.size - 1:
            return self._forward_stage(hidden, residual)
        return traversal.partition.gather(
            self._normalize(hidden, residual, local_routes)
        )

    def fill_cache(
        self,
        embeddings: torch.Tensor | None,
        positions: torch.Tensor,
        attention: AttentionBatch,
        *,
        routes: IndexedRoutes | None = None,
    ) -> None:
        """Write every layer's prefix cache without evaluating any output.

        The final layer stops at its K/V write (``PhasedLayer.write_cache``)
        and the output norm is skipped; earlier stages forward their
        activations as ``forward`` does. The cache holds exactly what
        ``forward`` writes.
        """
        traversal = self._traverse(
            embeddings, positions, attention, routes, split=True
        )
        if self._pipeline.rank != self._pipeline.size - 1:
            self._forward_stage(traversal.hidden, traversal.residual)
        elif traversal.final is not None:
            traversal.final.write_cache(
                traversal.hidden,
                traversal.residual,
                traversal.positions,
                attention,
            )

    def attend(
        self,
        embeddings: torch.Tensor | None,
        positions: torch.Tensor,
        attention: AttentionBatch,
    ) -> tuple[torch.Tensor, ...]:
        """Evaluate every layer up to the final layer's attention output.

        Returns the final stage's per-token state of every token for
        ``finish``: the final layer's ``PhasedLayer.attend`` state, or, when
        that layer does not split, its ``(hidden, residual)`` streams
        (``residual`` omitted for complete-stream layers). An earlier stage
        returns the activations it forwards. Routed passes are not split.
        """
        if self._default_route is not None:
            raise ValueError("routed decoders evaluate their layers whole")
        traversal = self._traverse(
            embeddings, positions, attention, None, split=True
        )
        hidden, residual = traversal.hidden, traversal.residual
        if self._pipeline.rank != self._pipeline.size - 1:
            return (self._forward_stage(hidden, residual),)
        if traversal.final is not None:
            state = traversal.final.attend(
                hidden, residual, traversal.positions, attention
            )
        else:
            state = (hidden,) if residual is None else (hidden, residual)
        return tuple(traversal.partition.gather(value) for value in state)

    @property
    def attend_widths(self) -> tuple[int, ...]:
        """Per-token widths of the final stage's ``attend`` state, in order.

        The final layer's ``PhasedLayer.state_widths`` when it splits,
        otherwise its ``(hidden, residual)`` streams (one stream for
        complete-stream layers). Each state tensor is ``[tokens, width]``
        in the stream dtype.
        """
        layer = tuple(self.layers.values())[-1]
        if isinstance(layer, PhasedLayer):
            return layer.state_widths
        if self.separate_residual:
            return (self.hidden_size, self.hidden_size)
        return (self.hidden_size,)

    def finish(
        self, state: tuple[torch.Tensor, ...], rows: torch.Tensor
    ) -> torch.Tensor:
        """Complete the final layer and the output norm for ``rows`` alone.

        ``state`` is ``attend``'s per-token state on the final stage and
        ``rows`` the int64 token rows to evaluate. Returns their normalized
        ``[rows, hidden]`` outputs, which equal the same rows of ``forward``
        up to the rounding that fewer rows allow.
        """
        if self._pipeline.rank != self._pipeline.size - 1:
            raise ValueError("the output norm belongs to the final stage")
        selected = tuple(value.index_select(0, rows) for value in state)
        layer = tuple(self.layers.values())[-1]
        if isinstance(layer, PhasedLayer):
            hidden, residual = layer.feed_forward(*selected)
        else:
            hidden, residual = selected[0], (selected[1:] or (None,))[0]
        return self._normalize(hidden, residual, None)

    def _traverse(
        self, embeddings, positions, attention, routes, *, split, deepstack=None
    ):
        """Evaluate the resident layers, all but a split final layer.

        With ``split``, an unrouted pass leaves the model's final layer on
        the final pipeline stage unevaluated when it is a ``PhasedLayer``,
        returning it as ``final`` with the streams it would receive.
        """
        count = positions.shape[-1]
        if routes is None and self._default_route is not None:
            routes = IndexedRoutes(self._default_route)
        if routes is not None and self._default_route is None:
            raise ValueError("only a routed decoder takes token routes")
        if deepstack and routes is not None:
            raise ValueError("DeepStack features require one packed stream")
        partition = TokenShard(count, self._tokens)
        positions = partition.local(positions, dim=positions.ndim - 1)

        # The first stage starts without a residual stream.
        hidden: torch.Tensor
        residual: torch.Tensor | None
        if self._pipeline.rank == 0:
            if embeddings is None or embeddings.shape != (
                count,
                self.hidden_size,
            ):
                raise ValueError(
                    "the first stage requires complete packed token embeddings"
                )
            hidden = partition.local(embeddings)
            residual = None
        else:
            reference = next(self.layers.parameters())
            hidden = reference.new_empty((partition.count, self.hidden_size))
            residual = (
                torch.empty_like(hidden) if self.separate_residual else None
            )
            for value in (hidden, residual):
                if value is not None:
                    self._pipeline.recv(src=self._pipeline.rank - 1, out=value)

        # Sparse route rows become local to this rank's token shard; rows
        # outside it route nothing here (``IndexedRoutes.shard``).
        local_routes = (
            None
            if routes is None
            else routes.shard(partition.token_slice.start, partition.count)
        )

        layers = tuple(self.layers.items())
        final = None
        if (
            split
            and routes is None
            and self._pipeline.rank == self._pipeline.size - 1
            and isinstance(layers[-1][1], PhasedLayer)
        ):
            final, layers = layers[-1][1], layers[:-1]
        for key, layer in layers:
            if local_routes is not None:
                hidden, residual = layer(
                    hidden, residual, positions, attention, routes=local_routes
                )
            else:
                hidden, residual = layer(hidden, residual, positions, attention)

            features = None if deepstack is None else deepstack.get(key)
            if features is not None:
                # DeepStack adds to the complete residual stream, after the
                # decoder layer's own residual addition has rounded.
                assert residual is not None
                residual = residual + hidden
                residual = residual + partition.local(features).to(
                    residual.dtype
                )
                hidden = torch.zeros_like(residual)
        # Layers that defer their residual addition return the stream they
        # carry forward; complete-stream layers return none.
        if final is None and (residual is None) == self.separate_residual:
            raise ValueError(
                "decoder layers must return a residual stream exactly when "
                "the decoder carries one separately"
            )
        return _Traversal(
            hidden, residual, positions, partition, local_routes, final
        )

    def _forward_stage(self, hidden, residual):
        """Send an earlier stage's streams on and return its hidden stream."""
        for stream in (hidden, residual):
            if stream is not None:
                self._pipeline.send(stream, dst=self._pipeline.rank + 1)
        return hidden

    def _normalize(self, hidden, residual, local_routes):
        """Fold the residual into the final stage's output norm."""
        # Only the final stage retains the norm. A routed decoder applies
        # each route's norm to its own tokens.
        norm = self.norm
        assert norm is not None
        if local_routes is not None:
            assert isinstance(norm, nn.ModuleDict)
            stream = hidden if residual is None else hidden + residual
            return local_routes.apply(norm, stream)
        if residual is None:
            return norm(hidden)
        if isinstance(norm, RMSNorm):
            result, _ = add_rms_norm(hidden, residual, norm.weight, norm.eps)
            return result
        return norm(hidden + residual)


@dataclass(frozen=True, slots=True)
class _Traversal:
    """The streams one ``TransformerDecoder._traverse`` leaves, and how.

    ``positions`` and ``routes`` are local to this rank's token shard of
    ``partition``; ``final`` is the unevaluated final layer of a split pass.
    """

    hidden: torch.Tensor
    residual: torch.Tensor | None
    positions: torch.Tensor
    partition: TokenShard
    routes: IndexedRoutes | None
    final: PhasedLayer | None


class TransformerEncoder(nn.Module):
    """Evaluate ordinary encoder layers over the same packed sequence domain."""

    function = "models.transformer.TransformerEncoder/1"

    def __init__(self, layers: nn.ModuleList, norm: nn.Module):
        super().__init__()
        self.layers, self.norm = layers, norm

    def forward(
        self, features: torch.Tensor, attention: AttentionBatch
    ) -> torch.Tensor:
        """Encode packed features over one uncached variable-length batch."""
        if (
            attention.queries is not None
            and attention.queries.num_tokens is not None
            and features.shape[0] != attention.queries.num_tokens
        ):
            raise ValueError(
                "encoder features must cover the declared query sequences"
            )
        for layer in self.layers:
            features = layer(features, attention)
        return self.norm(features)
