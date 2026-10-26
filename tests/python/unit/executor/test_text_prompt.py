"""Prompt logprob scoring of a prefill forward.

``uniserve.executor.text.prompt``.
"""

from __future__ import annotations

import numpy as np
import pytest

from uniserve.errors import OutOfMemory
from uniserve.executor.text.prompt import (
    PromptChunk,
    PromptChunks,
    media_chunks,
    prompt_chunk,
    prompt_columns,
    prompt_rows,
)

pytestmark = pytest.mark.unit


def _chunks(**columns: list[int]) -> PromptChunks:
    count = len(columns["rows"])
    arrays = {
        name: np.asarray(value, dtype=np.int64)
        for name, value in columns.items()
    }
    zeros = np.zeros(count, dtype=np.int64)
    return PromptChunks(
        calls=zeros,
        sizes=zeros,
        scored=zeros,
        top=4,
        end=0,
        **arrays,
    )


def test_prompt_rows_predict_each_chunks_tokens_in_order():
    # Candidate rows 0-2 project positions 10-12, a first chunk of three
    # tokens from token 0; rows 3-4 project 20-21, a chunk from token 3
    # continued from slot 5's retained logits; rows 5-6 project 30-31, a
    # chunk from token 5 continued from candidate row 4 (the last row of a
    # vision block); row 7 projects 40, a chunk that only retains. The
    # chunks report 4, 2, 1 and no top tokens.
    logit_rows = np.array([10, 11, 12, 20, 21, 30, 31, 40])
    chunks = _chunks(
        rows=[0, 3, 5, 7],
        row_counts=[3, 2, 2, 0],
        tokens=[0, 3, 5, 0],
        continued=[0, 1, 2 + 4, 0],
        slots=[1, 5, 1, 1],
        offsets=[100, 200, 300, 400],
        tails=[0, 1, 2, 3],
        tops=[4, 2, 1, 0],
    )
    spare = 64

    rows = prompt_rows(chunks, logit_rows, spare)

    # Per row: hidden position, retained slot, target token, chunk words,
    # index, scored rows, their top count and tail row.
    assert rows.tolist() == [
        # The first chunk's first two rows score its second and third
        # tokens; its last row retains its logits.
        [10, -1, 1, 100, 0, 2, 4, spare],
        [11, -1, 2, 100, 1, 2, 4, spare],
        [12, -1, 0, 0, 0, 0, -1, 0],
        # The slot's retained logits score the next chunk's first token.
        [0, 5, 3, 200, 0, 2, 2, spare],
        [20, -1, 4, 200, 1, 2, 2, spare],
        [21, -1, 0, 0, 0, 0, -1, 1],
        # The vision block's last row scores the first token after it.
        [21, -1, 5, 300, 0, 2, 1, spare],
        [30, -1, 6, 300, 1, 2, 1, spare],
        [31, -1, 0, 0, 0, 0, -1, 2],
        [40, -1, 0, 0, 0, 0, -1, 3],
    ]

    # A chunk naming a row its forward does not project is refused.
    chunks.rows[0] = 8
    with pytest.raises(AssertionError):
        prompt_rows(chunks, logit_rows, spare)


def test_chunks_report_their_own_tops_after_the_sampled_words():
    chunks = [
        # A first chunk of three rows scores two tokens.
        PromptChunk(call=0, first=0, count=3, token=0, continued=0, slot=1,
                    top=4),
        PromptChunk(call=1, first=3, count=2, token=3, continued=1, slot=5,
                    top=2),
        # A chunk that only retains scores nothing, whatever its top.
        PromptChunk(call=2, first=5, count=0, token=0, continued=0, slot=6,
                    top=9),
    ]  # fmt: skip
    planned = PromptChunks.plan(chunks, cursor=24, vocab=100, capacity=1000)
    assert planned.scored.tolist() == [2, 2, 0]
    assert planned.sizes.tolist() == [2 * (3 + 12), 2 * (3 + 6), 0]
    assert planned.offsets.tolist() == [24, 54, 72]
    assert planned.end == 72
    assert planned.tails.tolist() == [0, 1, 2]
    # Scored at the top bucket of the widest chunk scoring a token.
    assert planned.top == 4
    layout = planned.layout(0)
    assert layout is not None and layout.counts.tolist() == [4, 4]
    assert layout.words == planned.sizes[0]
    assert planned.layout(2) is None

    with pytest.raises(OutOfMemory):
        PromptChunks.plan(chunks, cursor=24, vocab=100, capacity=71)


def test_a_continued_prompt_scores_from_its_retained_logits():
    first = prompt_chunk(0, 1, 0, 3, 0, logical=0, prompt_ready=False, top=2)
    assert first.continued == 0 and first.scored == 2
    later = prompt_chunk(0, 1, 0, 3, 0, logical=12, prompt_ready=True, top=2)
    assert later.continued == 1 and later.scored == 3
    with pytest.raises(ValueError):
        prompt_chunk(0, 1, 0, 3, 0, logical=12, prompt_ready=False, top=2)


def test_text_after_a_vision_block_continues_from_its_last_row():
    # Text rows of candidate rows 0-1 and 5-6 around a vision block of
    # candidate rows 2-4, from the forward's token 10.
    chunks = media_chunks(
        call=3,
        slot=2,
        first=[0, 2, 5],
        count=[2, 3, 2],
        causal=[True, False, True],
        query=[2, 3, 2],
        token=10,
        logical=0,
        prompt_ready=False,
        top=5,
    )
    assert chunks == [
        PromptChunk(3, 0, 2, 10, 0, 2, 5),
        PromptChunk(3, 5, 2, 15, 2 + 4, 2, 5),
    ]

    # A prompt ending with a block retains the block's last logits.
    ending = media_chunks(
        3, 2, [0, 2], [2, 3], [True, False], [2, 3], 10, 7, True, 5
    )
    assert ending == [
        PromptChunk(3, 0, 2, 10, 1, 2, 5),
        PromptChunk(3, 4, 0, 0, 0, 2, 5),
    ]
    with pytest.raises(ValueError):
        media_chunks(3, 2, [0], [2], [True], [2], 10, 7, False, 5)


def test_prompt_rows_pad_to_whole_scoring_calls():
    rows = np.arange(24, dtype=np.int64).reshape(3, 8)
    columns = prompt_columns(rows, block=2, spare=9)
    assert columns.shape == (8, 4)
    assert columns[:, :3].T.tolist() == rows.tolist()
    assert columns[:, 3].tolist() == [0, -1, 0, 0, 0, 0, -1, 9]
