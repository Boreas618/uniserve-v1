"""Indexed routes give every token its route's module at static shapes.

A routed layer's result on a token must be its own route's module applied to
that token, whichever packing, shard or padding entries surround it: the
contract contiguous route spans stated, which the reference below evaluates
route by route over each route's own tokens.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.python.fixtures.checkpoints import bagel_checkpoint, load_bagel
from uniserve.nn.attention import AttentionBatch, SequenceLengths, VarlenInput
from uniserve.nn.routing import IndexedRoutes, SparseRoute

pytestmark = pytest.mark.unit


class _Scale(nn.Module):
    """A per-token module that also reads a per-token argument."""

    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def forward(self, value, offsets):
        return value * self.factor + offsets[:, None], value - self.factor


MODULES = {"flow": _Scale(2.0), "text": _Scale(-3.0)}


def _by_route(value, offsets, route_of):
    """Each token through its own route's module, as spans assign them."""
    first = torch.empty_like(value)
    second = torch.empty_like(value)
    for name, module in MODULES.items():
        tokens = torch.tensor(
            [index for index, route in enumerate(route_of) if route == name],
            dtype=torch.int64,
        )
        if len(tokens):
            a, b = module(value[tokens], offsets[tokens])
            first[tokens], second[tokens] = a, b
    return first, second


def _routes(text_rows):
    return IndexedRoutes(
        "flow",
        (SparseRoute.of("text", torch.tensor(text_rows, dtype=torch.int64)),),
    )


def test_sparse_rows_take_their_route_and_absent_entries_route_nothing():
    value = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    offsets = torch.arange(8, dtype=torch.float32) * 10
    # Two rows' markers, then two absent entries of padding rows.
    routes = _routes([0, 4, 3, 7, -1, -1]).shard(0, 8)
    route_of = ["text", "flow", "flow", "text", "text", "flow", "flow", "text"]

    actual = routes.apply(MODULES, value, offsets)

    for got, expected in zip(
        actual, _by_route(value, offsets, route_of), strict=True
    ):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("start,count", [(0, 5), (5, 5), (3, 4), (10, 0)])
def test_a_shard_routes_its_own_tokens_as_the_whole_domain_does(start, count):
    value = torch.randn(10, 4, generator=torch.Generator().manual_seed(1))
    offsets = torch.linspace(-1, 1, 10)
    text = [0, 2, 9, 4, -1, -1]
    route_of = ["text" if index in text else "flow" for index in range(10)]
    expected = _by_route(value, offsets, route_of)

    shard = slice(start, start + count)
    actual = (
        _routes(text)
        .shard(start, count)
        .apply(MODULES, value[shard], offsets[shard])
    )

    for got, whole in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, whole[shard], rtol=0, atol=0)


def test_a_shard_keeps_its_static_entry_count():
    routes = _routes([1, 6, 2, 8]).shard(5, 5)
    (text,) = routes.sparse
    # Every entry stays, so every shard evaluates the sparse route on the
    # same row count; entries outside the shard are not kept.
    assert text.rows.shape == (4,)
    assert text.keep.tolist() == [False, True, False, True]
    assert text.rows[text.keep].tolist() == [1, 3]
    # Entries not kept name a shard row that no kept entry names.
    free = set(text.rows[~text.keep].tolist())
    assert free.isdisjoint(text.rows[text.keep].tolist())
    assert free <= set(range(5))


def test_routes_name_each_sparse_route_once_apart_from_the_dense_one():
    rows = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="distinct"):
        IndexedRoutes("flow", (SparseRoute.of("flow", rows),))
    with pytest.raises(ValueError, match="distinct"):
        IndexedRoutes(
            "flow", (SparseRoute.of("text", rows), SparseRoute.of("text", rows))
        )
    with pytest.raises(ValueError, match="module"):
        IndexedRoutes("flow", (SparseRoute.of("audio", rows),)).shard(
            0, 2
        ).apply(MODULES, torch.zeros(2, 1), torch.zeros(2))


def _text_expert(backbone, embedded, positions, attention):
    """The text expert's equations over every token of a text prefill.

    The composition a contiguous text span gave every token: per layer the
    text input norm, rotary projection, shared attention, output projection
    and residual, then the text post-attention norm and MLP in the experts'
    BF16, and the text output norm.
    """
    hidden, residual = embedded, None
    for layer in backbone.layers.values():
        stream = hidden if residual is None else hidden + residual
        normalized = layer.input_norms["text"](stream)
        cos, sin = layer.position_embeddings(positions)
        query, key, value = layer.projections["text"](normalized, cos, sin)
        attended = layer.attention(query, key, value, attention).flatten(1)
        residual = stream + layer.outputs["text"](attended)
        normalized = layer.post_attention_norms["text"](residual)
        hidden = layer.mlps["text"](normalized.to(layer.mlp_dtype))
    return backbone.norm["text"](hidden + residual)


def test_a_text_prefill_evaluates_exactly_the_text_expert(tmp_path):
    _, _, config = bagel_checkpoint(tmp_path)
    backbone = load_bagel(tmp_path, config).text.backbone
    tokens = torch.tensor([1, 4, 8, 3, 9, 2, 30])
    lengths = SequenceLengths.from_lengths((4, 3), device="cpu")
    attention = AttentionBatch.single(
        VarlenInput(lengths, lengths, (True, True))
    )
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2])

    with torch.no_grad():
        embedded = backbone.embed_input_ids(tokens)
        actual = backbone(embedded, positions, attention)
        expected = _text_expert(backbone, embedded, positions, attention)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
