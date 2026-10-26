"""Rows of prefill calls placing image features in a request's sequence.

The image half answers a text lane's questions through the
:class:`~uniserve.executor.text.context.ImageFacts` and
:class:`~uniserve.executor.text.context.ImageDenoiser` protocols; these
tests answer them with framed temporal blocks (a start marker, the
features, an end marker, every slot at the conditioning position) and a
latent sequence of fixed length.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from uniserve.executor.text.context import (
    VisionBlock,
    VisionInput,
    block_row,
    concat_rows,
    context,
    host_rows,
    next_position,
    placed_rows,
    take_rows,
    text_row,
    visual,
)
from uniserve.executor.text.rows import RowSource, TokenSelection
from uniserve.processing import PositionLayout

START, END = 151_652, 151_653
HIDDEN = 8


class FramedTemporal:
    """A framed injection placing every slot at its conditioning position."""

    positions = PositionLayout.TEMPORAL

    def block(self, feature_tokens, position, close_image, grid):
        tokens = np.array(
            [START] + [1] * feature_tokens + [END], dtype=np.int64
        )
        return VisionBlock(
            token_ids=tokens,
            features=range(1, 1 + feature_tokens),
            positions=np.full(len(tokens), position, dtype=np.int64),
        )


class Facts:
    injection = FramedTemporal()
    rope_advance = 3

    def spatial_positions(self):
        return True

    def grid(self, height, width, close_image, input_images):
        raise AssertionError("temporal blocks need no patch grid")


class Denoiser:
    def sequence_length(self, height, width):
        return 6

    def conditioning_positions(self, height, width, start):
        return np.stack([np.full(6, start), np.arange(6), np.zeros(6)])

    def condition_latents(self, rows, pages, stream):
        raise AssertionError("rows only lay the conditioning out")


def _features(tokens: int) -> torch.Tensor:
    return torch.zeros((tokens, HIDDEN))


def test_text_rows_take_consecutive_temporal_positions():
    row = text_row(
        3, [7, 8, 9], prefix=5, position=11, selection=0, spatial=True
    )
    assert row.query == 3 and row.causal and row.prefix == 5
    np.testing.assert_array_equal(
        row.positions, [[11, 12, 13], [0, 0, 0], [0, 0, 0]]
    )
    assert next_position(row) == 14


def test_block_rows_attend_to_all_of_themselves_at_their_positions():
    block = FramedTemporal().block(2, 9, False, None)
    row = block_row(block, 4, 7, spatial=False, selection=2)
    assert not row.causal and row.query == 4 and row.prefix == 7
    np.testing.assert_array_equal(row.tokens, [START, 1, 1, END])
    np.testing.assert_array_equal(row.positions[0], [9, 9, 9, 9])
    np.testing.assert_array_equal(row.positions[1:], np.zeros((2, 4)))
    # A temporal block advances the sequence by one position.
    assert next_position(row) == 10

    spatial = VisionBlock(
        token_ids=np.ones(2, dtype=np.int64),
        features=range(2),
        positions=np.array([[5, 5], [0, 1], [1, 0]]),
    )
    np.testing.assert_array_equal(
        block_row(spatial, 1, 0, True, 0).positions, [[5, 5], [0, 1], [1, 0]]
    )


def test_a_context_prompt_alternates_text_runs_and_vision_blocks():
    tokens = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    features = _features(3)
    built = context(
        Facts(),
        slot=2,
        tokens=tokens,
        vision=[VisionInput(offset=2, features=features, height=4, width=4)],
        logical=20,
        visible=30,
        allocated=64,
        input_images=1,
        scores=False,
        token_output=True,
    )
    text, block, tail = built.rows
    # The prompt run before the block, the framed block at the next
    # position, then the rest of the prompt one position past the block.
    np.testing.assert_array_equal(text.tokens, [10, 11])
    assert (text.prefix, text.positions[0, 0]) == (30, 20)
    np.testing.assert_array_equal(block.tokens, [START, 1, 1, 1, END])
    assert (block.prefix, block.positions[0, 0], block.causal) == (
        32,
        22,
        False,
    )
    np.testing.assert_array_equal(tail.tokens, [12, 13, 14])
    assert (tail.prefix, tail.positions[0, 0]) == (37, 23)
    # Only the last row samples; the block's features replace its slots
    # from the call's third token on.
    assert [row.selection for row in built.rows] == [
        TokenSelection.CACHE,
        TokenSelection.CACHE,
        TokenSelection.LAST_LOGITS,
    ]
    assert built.embeddings[0][0] == 3 and built.embeddings[0][1] is features
    assert built.position == built.continued == built.sample == 26
    assert built.last is tail and built.latent is None


def test_a_scored_context_prompt_projects_text_and_block_ends():
    built = context(
        Facts(),
        slot=1,
        tokens=np.array([4, 5], dtype=np.int64),
        vision=[
            VisionInput(offset=2, features=_features(1), height=2, width=2)
        ],
        logical=0,
        visible=0,
        allocated=16,
        input_images=1,
        scores=True,
        token_output=True,
    )
    assert [row.selection for row in built.rows] == [
        TokenSelection.ALL_LOGITS,
        TokenSelection.LAST_LOGITS,
    ]


def test_context_prompts_refuse_misplaced_or_undeclared_blocks():
    arguments = {
        "slot": 1,
        "tokens": np.array([4, 5], dtype=np.int64),
        "logical": 0,
        "visible": 0,
        "allocated": 16,
        "scores": False,
        "token_output": False,
    }
    beyond = VisionInput(offset=3, features=_features(1), height=2, width=2)
    with pytest.raises(ValueError, match="in order"):
        context(Facts(), vision=[beyond], input_images=1, **arguments)
    inside = VisionInput(offset=1, features=_features(1), height=2, width=2)
    with pytest.raises(ValueError, match="no input images"):
        context(Facts(), vision=[inside], input_images=0, **arguments)
    with pytest.raises(NotImplementedError):
        context(None, vision=[inside], input_images=1, **arguments)


def test_a_feedback_closes_its_image_past_the_rope_advance():
    features = _features(2)
    common = {
        "slot": 5,
        "latent": None,
        "logical": 40,
        "visible": 50,
        "allocated": 64,
        "input_images": 0,
        "max_tokens": 8,
        "token_output": True,
    }
    vision = VisionInput(offset=0, features=features, height=2, width=2)
    completed = visual(
        Facts(), None, vision=vision, completion_output=True, **common
    )
    (row,) = completed.rows
    assert row.prefix == 50 and row.positions[0, 0] == 40
    assert completed.position == completed.continued == 43
    assert completed.sample == 43 and completed.embeddings == [(1, features)]
    # Without a completion the slot continues one position later.
    pending = visual(
        Facts(), None, vision=vision, completion_output=False, **common
    )
    assert (pending.position, pending.continued) == (40, 41)


def test_a_latent_conditioning_has_no_text_row():
    latent = VisionInput(offset=0, features=_features(6), height=8, width=8)
    common = {
        "slot": 2,
        "vision": None,
        "latent": latent,
        "logical": 9,
        "visible": 12,
        "allocated": 32,
        "input_images": 0,
        "max_tokens": 6,
        "completion_output": False,
    }
    built = visual(Facts(), Denoiser(), token_output=False, **common)
    assert built.rows == [] and built.embeddings == []
    condition = built.latent
    assert (condition.slot, condition.prefix, condition.position) == (2, 12, 9)
    assert (built.last.prefix, built.last.query) == (12, 6)
    with pytest.raises(ValueError, match="samples no token"):
        visual(Facts(), Denoiser(), token_output=True, **common)
    with pytest.raises(ValueError, match="exactly one feature"):
        visual(
            Facts(),
            Denoiser(),
            token_output=False,
            **(common | {"vision": latent}),
        )


def test_rows_of_calls_concatenate_over_their_tokens():
    placed = placed_rows(
        [
            text_row(1, [4, 5], 0, 0, TokenSelection.CACHE, False),
            text_row(1, [6], 2, 2, TokenSelection.LAST_LOGITS, False),
        ]
    )
    hosted = host_rows(
        [3, 4],
        [np.array([7, 8, 9]), np.array([10])],
        prefix=[5, 0],
        selection=TokenSelection.LAST_LOGITS,
    )
    assert list(hosted.source) == [RowSource.HOST] * 2
    assert list(hosted.position) == [5, 0]

    rows = concat_rows([placed, take_rows(hosted, [1, 0])])
    rows.validate()
    assert list(rows.slot) == [1, 1, 4, 3]
    assert [list(rows.host_tokens(row)) for row in range(4)] == [
        [4, 5],
        [6],
        [10],
        [7, 8, 9],
    ]
    assert sorted(rows.placed) == [0, 1]
    np.testing.assert_array_equal(rows.placed[1][0], [2])
