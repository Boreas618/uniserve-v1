"""The order and contents of a text computation's warmup phases.

A computation's phases depend only on its graph catalogs, its pool and
model facts, so every rank of a group runs the same forwards in the same
order. The catalogs here are those of a computation with decode sizes 1, 2
and 4, prefill token sizes 16 and 32 within 64 batch tokens, and pages of
four tokens over a pool of 96 units.
"""

from __future__ import annotations

import numpy as np
import pytest

from uniserve.executor.text.buffers import PageGroup, TextBounds
from uniserve.executor.text.context import VisionBlock
from uniserve.executor.text.protocol import CallKind
from uniserve.executor.text.rows import RowSource, TokenSelection
from uniserve.executor.text.shapes import PrefillVariant
from uniserve.executor.text.warmup import (
    context_layout,
    decode_layouts,
    least,
    prefill_layouts,
    top_buckets,
)
from uniserve.processing import PositionLayout

UNITS = 96


def _bounds(media_axes=None) -> TextBounds:
    return TextBounds.derive(
        max_batch_calls=4,
        max_batch_tokens=64,
        max_model_len=48,
        slots=8,
        groups=[PageGroup(block_size=4, units_per_page=1, window=None)],
        units=UNITS + 1,
        graphs=True,
        decode_sizes=[1, 2, 4],
        prefill_token_sizes=[16, 32],
        prefill_outputs=True,
        media_axes=media_axes,
    )


def _one(tokens):
    return [None]


@pytest.mark.parametrize("units", [None, UNITS])
def test_prefill_phases_capture_the_largest_footprint_first(units):
    bounds = _bounds()
    layouts = prefill_layouts(
        bounds.shapes(),
        pages=bounds.pages,
        row_tokens=48,
        units=units,
        capacities=lambda tokens: [2 * tokens, tokens],
    )
    # One row bucket of five rows (four live and the padding sequence) at
    # 64, 32 and 16 tokens.
    shapes = sorted(
        bounds.prefill, key=lambda shape: shape.token_bucket, reverse=True
    )
    assert [layout.lengths for layout in layouts] == [(32, 32), (32,), (16,)]
    for layout, shape in zip(layouts, shapes):
        # One step per capacity, largest first, over the scratch rows.
        assert [step.capacity for step in layout.steps] == [
            2 * shape.token_bucket,
            shape.token_bucket,
        ]
        for step in layout.steps:
            assert step.kind == CallKind.PREFILL
            assert step.attention() == [
                [slot, 0, length]
                for slot, length in enumerate(layout.lengths, start=1)
            ]
            assert list(step.rows.source) == [RowSource.HOST] * len(
                layout.lengths
            )
            assert set(step.rows.selection) == {TokenSelection.LAST_LOGITS}


def test_media_prefill_phases_capture_rows_at_explicit_positions():
    bounds = _bounds(media_axes=3)
    layouts = prefill_layouts(
        bounds.shapes(),
        pages=bounds.pages,
        row_tokens=48,
        units=UNITS,
        capacities=_one,
    )
    media = [
        shape
        for shape in sorted(
            bounds.prefill,
            key=lambda shape: shape.token_bucket,
            reverse=True,
        )
        if shape.variant == PrefillVariant.media(3)
    ]
    assert len(layouts) == 2 * len(media)
    placed = [
        layout
        for layout in layouts
        if layout.steps[0].rows.source[0] == RowSource.PLACED
    ]
    assert len(placed) == len(media)
    for layout in placed:
        assert all(layout.steps[0].rows.spatial)


def test_decode_phases_prime_commit_and_decode_largest_first():
    bounds = _bounds()
    layouts = decode_layouts(
        bounds.shapes(),
        prefill_outputs=True,
        capacities=lambda tokens: [8, 4],
    )
    assert [layout.lengths for layout in layouts] == [(2,) * 4, (2,) * 2, (2,)]
    assert [layout.primed for layout in layouts] == [4, 2, 1]
    largest = layouts[0]
    # Priming prefills of one token per slot, at most the largest prefill
    # bucket's live rows (four) at a time, then greedy and sampled decodes
    # at every capacity.
    kinds = [step.kind for step in largest.steps]
    assert kinds == [CallKind.PREFILL] + [CallKind.DECODE] * 4
    assert largest.steps[0].attention() == [
        [slot, 0, 1] for slot in (1, 2, 3, 4)
    ]
    decodes = largest.steps[1:]
    assert [(step.capacity, step.sampled) for step in decodes] == [
        (8, False),
        (8, True),
        (4, False),
        (4, True),
    ]
    for step in decodes:
        assert step.attention() == [[slot, 1, 1] for slot in (1, 2, 3, 4)]
        assert list(step.rows.selection) == [TokenSelection.LAST_LOGITS] * 4


def test_without_prefill_buckets_a_decode_bucket_primes_at_once():
    bounds = TextBounds.derive(
        max_batch_calls=4,
        max_batch_tokens=64,
        max_model_len=48,
        slots=8,
        groups=[PageGroup(block_size=4, units_per_page=1, window=None)],
        units=UNITS + 1,
        graphs=True,
        decode_sizes=[4],
        prefill_token_sizes=[],
        prefill_outputs=False,
        media_axes=None,
    )
    shapes = bounds.shapes()
    shapes = type(shapes)(shapes.decode, ())
    (layout,) = decode_layouts(shapes, prefill_outputs=False, capacities=_one)
    prime = layout.steps[0]
    assert prime.attention() == [[slot, 0, 1] for slot in (1, 2, 3, 4)]
    assert set(prime.rows.selection) == {TokenSelection.CACHE}


class FramedTemporal:
    positions = PositionLayout.TEMPORAL

    def block(self, feature_tokens, position, close_image, grid):
        tokens = np.array([7] + [1] * feature_tokens + [8], dtype=np.int64)
        return VisionBlock(
            token_ids=tokens,
            features=range(1, 1 + feature_tokens),
            positions=np.full(len(tokens), position, dtype=np.int64),
        )


def test_context_warmup_writes_a_block_behind_text_and_alone():
    layout = context_layout(
        FramedTemporal(), spatial=True, prefill_outputs=True
    )
    # A prompt token, a framed block of four features (six slots) and a
    # second block, all on slot 1.
    assert layout.lengths == (1 + 6 + 6,)
    mixed, alone = layout.steps
    assert mixed.attention() == [[1, 0, 1], [1, 1, 6]]
    assert alone.attention() == [[1, 7, 6]]
    assert list(mixed.rows.causal) == [True, False]
    # The features replace each block's slots: behind the prompt token
    # from the forward's third token, alone from its second.
    assert (mixed.features, alone.features) == (2, 1)
    # The second block starts where the first one's positions end.
    assert alone.rows.placed[0][0, 0] == 2


def test_top_buckets_hold_every_reported_width_through_their_least():
    assert top_buckets(37) == [0, 1, 2, 4, 8, 16, 32, 37]
    buckets = top_buckets(37)
    assert [least(buckets, position) for position in range(len(buckets))] == [
        0,
        1,
        2,
        3,
        5,
        9,
        17,
        33,
    ]
