"""Graph buckets, padding and slot chunks of a diffusion LM computation
(``uniserve.executor.diffusion_lm.shapes``).
"""  # noqa: D205

from __future__ import annotations

import pytest

from uniserve.executor.diffusion_lm.shapes import (
    Shapes,
    attend_padding,
    chunk_rows,
    slot_chunks,
)

pytestmark = pytest.mark.unit


def test_buckets_follow_the_capacity_and_the_block_length():
    shapes = Shapes(256, 10, 10, 2560)
    assert shapes.rows() == (1, 2, 3, 4, 6, 8, 10)
    assert shapes.attend(5, 200, 40) == (6, 384)
    assert shapes.attend(10, 2560, 256) == (10, 2560)
    # A block longer than the model's is refused, and so is a pass beyond
    # the readout bounds.
    with pytest.raises(ValueError):
        shapes.attend(1, 257, 257)
    with pytest.raises(ValueError):
        shapes.attend(11, 16, 16)
    assert shapes.denoise(7) == 8
    with pytest.raises(ValueError):
        shapes.denoise(11)

    # Generating ranks never step more blocks than their slots.
    shapes = Shapes(48, 200, 200, 9600, slots=3)
    assert shapes.rows() == (1, 2, 3)
    assert shapes.capacity() == 3
    with pytest.raises(ValueError):
        Shapes(48, 0, 1, 48)


def test_every_readout_within_the_bounds_selects_a_bucket():
    # Every pass of at most the readout rows and the token bound, each block
    # at most the model's, selects an ``attend`` bucket that holds it and
    # the padding sequences the columns hold.
    length, readout_rows, tokens = 64, 20, 256
    shapes = Shapes(length, tokens // length, readout_rows, tokens)
    for rows in range(1, readout_rows + 1):
        for total in range(rows, tokens + 1):
            # One block as long as the model's when the tokens allow, the
            # others sharing the rest.
            longest = min(total - (rows - 1), length)
            if longest + (rows - 1) * length < total:
                continue
            bucket, size = shapes.attend(rows, total, longest)
            assert bucket >= rows and total <= size <= tokens
            padding = attend_padding(rows, total, bucket, size, length)
            assert rows + len(padding) == 2 * bucket
            assert sum(padding) + total == size
            assert all(count <= length for count in padding)


def test_short_blocks_launch_few_padding_tokens():
    # A pass of twelve 16-token blocks within a 4096-token bound.
    shapes = Shapes(256, 16, 128, 4096)
    assert shapes.attend(12, 192, 16) == (12, 192)
    # Beyond the full blocks the bound holds, the bound is the bucket.
    assert shapes.attend(100, 3000, 256) == (128, 4096)
    with pytest.raises(ValueError):
        shapes.attend(129, 129, 1)


def test_attend_padding_fills_the_bucket_with_bounded_sequences():
    # Two blocks of 16 and 9 tokens in a bucket of 3 sequences and 48
    # tokens: one more full sequence, the 7-token remainder, then empty
    # ones.
    assert attend_padding(2, 25, 3, 48, 16) == [16, 7, 0, 0]
    # A full bucket pads with empty sequences only.
    assert attend_padding(2, 32, 2, 32, 16) == [0, 0]
    with pytest.raises(AssertionError):
        attend_padding(3, 10, 2, 32, 16)
    with pytest.raises(AssertionError):
        attend_padding(1, 10, 2, 40, 16)


def test_finish_buckets_reach_the_token_bound():
    shapes = Shapes(256, 16, 128, 4096)
    assert shapes.finish_buckets() == (64, 128, 256, 512, 1024, 2048, 4096)
    assert shapes.finish(3) == 64
    assert shapes.finish(4096) == 4096
    with pytest.raises(ValueError):
        shapes.finish(4097)


def test_readouts_chunk_slots_at_the_largest_bucket():
    assert slot_chunks(600) == [(0, 256, 256), (256, 256, 256), (512, 88, 128)]
    assert slot_chunks(3) == [(0, 3, 64)]
    assert slot_chunks(0) == []


def test_step_chunks_bound_their_transient_logits():
    # DiffusionGemma: one block of 256 x 262144 entries per chunk.
    assert chunk_rows(256, 262_144, 64) == 1
    assert chunk_rows(16, 64, 4) == 4
    assert chunk_rows(16, 64, 0) == 1
