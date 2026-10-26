"""The token batch protocol of a text lane.

``uniserve.executor.text.protocol``.
"""

from __future__ import annotations

import numpy as np
import pytest

from uniserve.executor.text.protocol import (
    CALL_FIELD_NAMES,
    RECORD_MAGIC,
    GraphMode,
    RecordFlag,
    RecordWriter,
    combine_modes,
    forward_fields,
    page_count,
    page_start,
    record_capacity,
)

pytestmark = pytest.mark.unit


def test_call_and_forward_columns_follow_the_worker_layout():
    assert CALL_FIELD_NAMES == (
        "slot",
        "kind",
        "active",
        "logical",
        "visible",
        "computed",
        "rng",
        "prompt_ready",
        "allocated",
        "predicate",
        "row_start",
        "row_count",
        "token_start",
        "token_count",
        "max_tokens",
        "flags",
        "canvas_block",
        "canvas_step",
        "request_id",
        "epoch",
    )
    # Five row fields, then each cache group's first page and page count.
    assert (page_start(0), page_count(0), page_start(1)) == (5, 6, 7)
    assert forward_fields(2) == 9


def test_records_hold_a_header_call_records_segments_and_tails():
    calls, tokens = 3, 4
    records = np.full(record_capacity(calls, tokens), -7, dtype=np.int64)
    writer = RecordWriter(records, calls)
    # Calls 0 and 2 ran; call 1 was inactive.
    writer.ran([0, 2])
    writer.progress(
        [0, 2],
        logical=np.array([5, 9]),
        rng=np.array([1, 2]),
        visible=np.array([5, 9]),
        computed=np.array([5, 10]),
        prompt_ready=np.array([True, False]),
    )
    writer.sampling([0, 2], offset=0, extent=8, rows=np.array([0, 1]))
    sampled = writer.segment(8, 24, [0, 1], [3, 3])
    writer.logprobs([2], sampled, np.array([1]))
    chunk = writer.segment(32, 9, [0], [2])
    writer.tail(0, prompt=[(chunk, 0)])
    writer.tail(2, drafts=[7, 8], terminal=2)
    written = writer.finish(
        words_written=41,
        lent=50,
        mode=GraphMode.REPLAY,
        live_tokens=4,
        padded_tokens=4,
        rows=2,
        draft_tokens=2,
    )

    assert records[:12].tolist() == [
        RECORD_MAGIC, 3, 2, 41, 1, 4, 4, 2, 2, 0, 0, 0
    ]  # fmt: skip
    call = records[12:60].reshape(3, 16)
    assert call[0].tolist() == [
        int(
            RecordFlag.RAN
            | RecordFlag.PROGRESS
            | RecordFlag.SAMPLING
            | RecordFlag.PROMPT
            | RecordFlag.PROMPT_LOGITS_READY
        ),
        5, 1, 5, 5,  # logical, rng, visible, computed
        0, 8, 0,  # sampling offset, extent, row
        0, 0,  # no logprobs
        0, 0,  # no candidates
        72, 5,  # tail
        0, 0,
    ]  # fmt: skip
    # An inactive call's record is zero.
    assert not call[1].any()
    assert call[2].tolist() == [
        int(
            RecordFlag.RAN
            | RecordFlag.PROGRESS
            | RecordFlag.SAMPLING
            | RecordFlag.LOGPROB
            | RecordFlag.SPECULATION
        ),
        9, 2, 9, 10,
        0, 8, 1,
        0, 1,  # segment 0, packed row 1
        0, 0,
        77, 5,
        0, 0,
    ]  # fmt: skip
    # Segments `[offset, count, n, rows, counts]`, then tails `[n_prompt,
    # (segment, row)..., n_drafts, drafts..., terminal]`.
    assert records[60:67].tolist() == [8, 24, 2, 0, 1, 3, 3]
    assert records[67:72].tolist() == [32, 9, 1, 0, 2]
    assert records[72:77].tolist() == [1, 1, 0, 0, -1]
    assert records[77:82].tolist() == [0, 2, 7, 8, 2]
    assert written == 82


def test_records_refuse_what_the_worker_cannot_read():
    writer = RecordWriter(np.zeros(record_capacity(1, 1), np.int64), 1)
    writer.tail(0, drafts=[1])
    # Segments are read in order before every tail.
    with pytest.raises(AssertionError):
        writer.segment(0, 3, [0], [0])
    # A tail names only written segments.
    with pytest.raises(AssertionError):
        writer.tail(0, prompt=[(0, 0)])
    with pytest.raises(AssertionError):
        writer.finish(
            words_written=9,
            lent=8,
            mode=GraphMode.EAGER,
            live_tokens=1,
            padded_tokens=0,
            rows=1,
            draft_tokens=0,
        )

    # Records never exceed the worker's column.
    small = RecordWriter(np.zeros(12 + 16 + 4, np.int64), 1)
    with pytest.raises(AssertionError):
        small.segment(0, 9, [0], [2])
    with pytest.raises(AssertionError):
        RecordWriter(np.zeros(20, np.int64), 1)


def test_a_batch_reports_its_most_specific_graph_mode():
    assert combine_modes([]) is GraphMode.NONE
    assert combine_modes([GraphMode.EAGER, GraphMode.REPLAY]) is (
        GraphMode.REPLAY
    )
    assert combine_modes(
        [GraphMode.REPLAY, GraphMode.CAPTURE, GraphMode.EAGER]
    ) is (GraphMode.CAPTURE)


def test_a_prefill_scoring_its_whole_chunk_fits_the_worker_column():
    # One prefill call of 256 tokens samples its next token with logprobs
    # and scores every prompt token: a sampled-logprob segment of one task,
    # the chunk's segment of 256 rows and a tail naming each scored row.
    tokens = 256
    writer = RecordWriter(np.zeros(record_capacity(1, tokens), np.int64), 1)
    sampled = writer.segment(0, 4, [0], [5])
    chunk = writer.segment(4, 3 * tokens, np.arange(tokens), np.full(tokens, 3))
    writer.tail(0, prompt=[(chunk, row) for row in range(tokens)])

    assert sampled == 0
    assert chunk == 1
