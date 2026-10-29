"""Graph buckets of a text computation (``uniserve.executor.text.shapes``)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from uniserve.executor.text.shapes import (
    PrefillShape,
    PrefillVariant,
    Selected,
    TextForward,
    TextShapes,
    bucket,
    capture_order,
    prefill_row_buckets,
    prefill_shapes,
)

pytestmark = pytest.mark.unit

TEXT = PrefillVariant.TEXT

# A causal prefill forward without embeddings, outputs at every position.
PREFILL = TextForward(
    queries=(),
    tokens=0,
    decode=False,
    variant=TEXT,
    last_logits=False,
    cache_only=False,
)

# A causal decode step computing the last logits.
DECODE = replace(PREFILL, decode=True, last_logits=True)


def _prefill(
    rows: int, tokens: int, variant: PrefillVariant, outputs: bool
) -> PrefillShape:
    return PrefillShape(
        token_bucket=tokens,
        row_bucket=rows,
        live_rows=1,
        variant=variant,
        outputs=outputs,
    )


def _ceil(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def test_scratch_capture_fits_hybrid_pages_and_per_request_limits():
    shape = PrefillShape(
        token_bucket=257,
        row_bucket=9,
        live_rows=4,
        variant=TEXT,
        outputs=True,
    )
    lengths = shape.scratch_lengths(128, [(16, 2), (64, 1)], 24)
    assert len(lengths) == 4
    # Twenty-four units hold ten small pages and four large pages. The
    # small pages therefore bound the useful capture to 160 tokens.
    assert sum(lengths) == 160
    assert all(length <= 128 for length in lengths)
    used = sum(2 * _ceil(length, 16) + _ceil(length, 64) for length in lengths)
    assert used == 24

    limited = replace(shape, token_bucket=64, row_bucket=5, live_rows=1)
    lengths = limited.scratch_lengths(8, [(16, 1)], 8)
    assert len(lengths) == 4
    assert sum(lengths) == 32
    assert all(length <= 8 for length in lengths)


def test_decode_keeps_configured_order_and_never_uses_prefill_capacity():
    # A decode takes the first configured size that holds it and never a
    # prefill bucket; a decode no size holds, or one no decode graph
    # computes (rows placing vision blocks, outputs other than the last
    # logits), selects none.
    shapes = TextShapes([8, 4, 16], [_prefill(32, 64, TEXT, True)])
    assert shapes.select(
        replace(DECODE, queries=[1] * 3, tokens=3)
    ) == Selected(
        8,
        8,
        True,
        True,
    )
    for forward in [
        replace(DECODE, queries=[1] * 17, tokens=17),
        replace(DECODE, queries=[1], tokens=1, variant=PrefillVariant.media(3)),
        replace(DECODE, queries=[1], tokens=1, last_logits=False),
    ]:
        assert shapes.select(forward) is None


def test_prefill_selects_padding_capacity_and_matching_numerical_semantics():
    shapes = TextShapes(
        [],
        [
            _prefill(8, 16, TEXT, True),
            _prefill(4, 64, TEXT, True),
            _prefill(4, 32, PrefillVariant.media(3), True),
            _prefill(4, 16, TEXT, False),
        ],
    )
    two_rows = replace(PREFILL, queries=[2, 3], tokens=5)
    assert shapes.select(two_rows) == Selected(4, 64, False, True)
    assert shapes.select(replace(two_rows, cache_only=True)) == Selected(
        4,
        16,
        False,
        False,
    )
    assert shapes.select(
        replace(two_rows, variant=PrefillVariant.media(3))
    ) == Selected(4, 32, False, True)
    assert shapes.select(
        replace(PREFILL, queries=[1] * 4, tokens=4, last_logits=True)
    ) == Selected(8, 16, False, True)

    # A variant no shape was captured for, rows beyond every bucket and an
    # empty row select none.
    for forward in [
        replace(
            PREFILL, queries=[2], tokens=2, variant=PrefillVariant.media(1)
        ),
        replace(PREFILL, queries=[1] * 8, tokens=8),
        replace(PREFILL, queries=[0], tokens=0),
    ]:
        assert shapes.select(forward) is None
    with pytest.raises(ValueError):
        shapes.select(PREFILL)

    output_only = TextShapes([], [_prefill(2, 16, TEXT, True)])
    assert output_only.select(
        replace(PREFILL, queries=[2], tokens=2, cache_only=True)
    ) == Selected(2, 16, False, True)
    cache_only = TextShapes([], [_prefill(2, 16, TEXT, False)])
    assert (
        cache_only.select(
            replace(PREFILL, queries=[2], tokens=2, last_logits=True)
        )
        is None
    )


def test_every_batch_within_the_bounds_selects_a_bucket():
    # Every prefill batch of at most `max_rows` rows and the token bound,
    # and every decode batch of at most `max_rows` rows, selects a captured
    # bucket that holds it.
    max_rows, max_tokens = 40, 96
    rows = prefill_row_buckets(max_rows)
    assert rows == [8, 16, 41]
    shapes = TextShapes(
        [1, 2, 4, 8, 16, 32, max_rows],
        prefill_shapes(
            [4, 16, 64], rows, max_rows, max_tokens, [TEXT], True, None
        ),
    )
    for count in range(1, max_rows + 1):
        selected = shapes.select(
            replace(DECODE, queries=[1] * count, tokens=count)
        )
        assert selected is not None and selected.rows >= count, count
        for tokens in range(count, max_tokens + 1):
            # The rows share the tokens; any split selects the same way.
            queries = [1] * count
            queries[0] += tokens - count
            selected = shapes.select(
                replace(PREFILL, queries=queries, tokens=tokens)
            )
            assert (
                selected is not None
                and selected.rows > count
                and selected.tokens >= tokens
            ), (count, tokens)


def test_verification_buckets_hold_every_admitted_draft_length():
    drafts = 3
    shapes = TextShapes(
        [1, 2, 4, 8, 16, 32, 64, 128],
        [],
        verify=[1, 2, 4, 8, 16, 32, 64, 128],
        drafts=drafts,
        prompts=False,
    )
    for count in range(1, 129):
        for longest in range(1, drafts + 2):
            # Requests lacking draft state participate with one token.
            lengths = [1 + index % longest for index in range(count)]
            selected = shapes.select(
                replace(
                    PREFILL,
                    queries=lengths,
                    tokens=sum(lengths),
                    verification=True,
                )
            )
            assert selected is not None
            assert selected.verification
            assert selected.rows > count
            assert selected.tokens > sum(lengths)
    assert (
        shapes.select(
            replace(
                PREFILL,
                queries=[drafts + 2],
                tokens=drafts + 2,
                verification=True,
            )
        )
        is None
    )


def test_prefill_row_buckets_reach_one_past_the_largest_batch():
    assert prefill_row_buckets(3) == [4]
    assert prefill_row_buckets(31) == [8, 16, 32]
    assert prefill_row_buckets(128) == [8, 16, 32, 64, 129]


def test_captures_exclude_buckets_no_admitted_batch_can_reach():
    shapes = prefill_shapes(
        [16, 32], [1, 2, 4, 8], 3, 64, [TEXT], True, ([(4, 1)], 4)
    )
    assert [
        (shape.row_bucket, shape.token_bucket, shape.live_rows)
        for shape in shapes
    ] == [(2, 16, 1), (4, 16, 2)]


def test_buckets_and_capture_order_follow_configured_sizes():
    assert bucket([8, 1, 32, 4], 3) == 4
    assert bucket([8, 1, 32, 4], 32) == 32
    assert bucket([8, 1, 32, 4], 33) is None

    # (rows, tokens): largest product first, then tokens; duplicates
    # removed; equal footprints keep their configured order.
    order = capture_order(
        [(2, 16), (4, 64), (8, 8), (2, 16), (4, 16)],
        lambda shape: (shape[0] * shape[1], shape[1]),
    )
    assert order == [(4, 64), (4, 16), (8, 8), (2, 16)]
