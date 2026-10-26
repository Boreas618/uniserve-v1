"""Token rows of a text forward (``uniserve.executor.text.rows``)."""

from __future__ import annotations

import numpy as np
import pytest

from uniserve.executor.text.protocol import (
    CALL_FIELDS,
    CallField,
    CallFlag,
    CallKind,
    Predicate,
)
from uniserve.executor.text.rows import (
    Rows,
    RowSource,
    TokenSelection,
    graph_widths,
    logit_layout,
    logit_positions,
    pack_rows,
    pad_rows,
    prepare_calls,
    text_forward,
    written_length,
)
from uniserve.executor.text.shapes import PrefillVariant

pytestmark = pytest.mark.unit


def _rows(
    query: list[int],
    selection: list[int] | None = None,
    **columns: list,
) -> Rows:
    # Host rows of slot 1 at position 0, every token 5, unless changed.
    n = len(query)
    values = {
        "slot": [1] * n,
        "source": [RowSource.HOST] * n,
        "prefix": [0] * n,
        "query": query,
        "selection": selection or [TokenSelection.LAST_LOGITS] * n,
        "gated": [False] * n,
        "force": [False] * n,
        "causal": [True] * n,
        "position": [0] * n,
        "token_start": list(np.cumsum([0, *query[:-1]])),
        "token_count": query,
        "spatial": [False] * n,
    }
    values.update(columns)
    dtypes = {"gated": bool, "force": bool, "causal": bool, "spatial": bool}
    arrays = {
        name: np.asarray(value, dtype=dtypes.get(name, np.int64))
        for name, value in values.items()
    }
    return Rows(tokens=np.full(sum(query), 5, dtype=np.int64), **arrays)


def test_logit_layout_follows_each_row_selection():
    rows = _rows(
        [3, 2, 4, 1],
        [
            TokenSelection.LAST_LOGITS,
            TokenSelection.CACHE,
            TokenSelection.ALL_LOGITS,
            TokenSelection.LAST_LOGITS,
        ],
    )
    rows.validate()
    first, count = logit_layout(rows.selection, rows.query)
    assert list(zip(first.tolist(), count.tolist())) == [
        (0, 1),
        (1, 0),
        (1, 4),
        (5, 1),
    ]
    # The candidate rows project these token positions when all of them
    # are projected.
    assert logit_positions(rows.query, count).tolist() == [2, 5, 6, 7, 8, 9]

    # A relay row's query is its drafts and the relayed token.
    relay = _rows([3], [TokenSelection.ALL_LOGITS], source=[RowSource.RELAY])
    relay.token_count[0] = 2
    relay.validate()
    relay.query[0] = 2
    with pytest.raises(ValueError):
        relay.validate()


def test_rows_require_a_slot_and_placed_positions_per_token():
    with pytest.raises(ValueError):
        _rows([1], slot=[0]).validate()
    placed = _rows([2], source=[RowSource.PLACED])
    placed.placed[0] = np.zeros((3, 2), dtype=np.int64)
    placed.validate()
    placed.placed[0] = np.zeros((3, 1), dtype=np.int64)
    with pytest.raises(ValueError):
        placed.validate()


def test_packed_rows_hold_lengths_offsets_and_token_rows():
    # Three rows of capacity (stride 4), five tokens, no tables.
    words = pack_rows(
        slots=[3, 5],
        prefixes=[10, 4],
        queries=[2, 3],
        writes=[False, True],
        pages=[],
        stride=4,
        tokens=5,
    )
    assert words[0:4].tolist() == [3, 5, 0, 0]  # slots
    assert words[4:8].tolist() == [10, 4, 0, 0]  # prefixes
    assert words[8:12].tolist() == [2, 3, 0, 0]  # queries
    assert words[12:16].tolist() == [0, 1, 0, 0]  # writes
    assert words[16:20].tolist() == [0, 2, 5, 0]  # query offsets
    assert words[20:24].tolist() == [0, 10, 14, 0]  # prefix offsets
    assert words[24:29].tolist() == [0, 0, 1, 1, 1]  # token rows

    # Each table's page counts follow; a row past a table's counts selects
    # none.
    words = pack_rows([3, 5], [10, 4], [2, 3], [1, 1], [[4, 2], [7]], 4, 5)
    assert words[24:28].tolist() == [4, 2, 0, 0]
    assert words[28:32].tolist() == [7, 0, 0, 0]
    assert words[32:37].tolist() == [0, 0, 1, 1, 1]


def _calls(*calls: dict, tokens: list[int]) -> tuple[np.ndarray, np.ndarray]:
    # Call columns of active calls of slots 1.., at logical position and
    # visible length 10, whose input tokens follow each other in `tokens`.
    columns = np.zeros((CALL_FIELDS, len(calls)), dtype=np.int64)
    start = 0
    for index, call in enumerate(calls):
        count = call.pop("tokens", 0)
        columns[CallField.SLOT, index] = index + 1
        columns[CallField.ACTIVE, index] = 1
        columns[CallField.LOGICAL, index] = 10
        columns[CallField.VISIBLE, index] = 10
        columns[CallField.TOKEN_START, index] = start
        columns[CallField.TOKEN_COUNT, index] = count
        for field, value in call.items():
            columns[CallField[field.upper()], index] = value
        start += count
    return columns, np.asarray(tokens, dtype=np.int64)


def test_calls_read_their_tokens_from_the_host_or_their_slot():
    output = int(CallFlag.TOKEN_OUTPUT)
    calls, tokens = _calls(
        # A decode relayed by its slot's token reads everything there.
        {"kind": CallKind.DECODE, "predicate": Predicate.RELAY},
        # An ungated decode reads its first host token.
        {"kind": CallKind.DECODE, "tokens": 2},
        # A device-predicated decode relays its slot's committed token.
        {"kind": CallKind.DECODE, "predicate": Predicate.DEVICE},
        # Prefills sampling, only writing the cache, and scoring.
        {"kind": CallKind.PREFILL, "tokens": 3, "flags": output},
        {"kind": CallKind.PREFILL, "tokens": 2},
        {"kind": CallKind.PREFILL, "tokens": 2},
        # A verification of its current token and two drafts, and one
        # relayed by its slot's token.
        {"kind": CallKind.VERIFY, "tokens": 3},
        {"kind": CallKind.VERIFY, "tokens": 2, "predicate": Predicate.RELAY},
        # A gated prefill without input tokens relays its slot's token.
        {"kind": CallKind.PREFILL, "predicate": Predicate.RELAY},
        tokens=[40, 41, 1, 2, 3, 4, 5, 6, 7, 70, 80, 90, 81, 91],
    )
    scores = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=bool)
    prepared = prepare_calls(calls, tokens, np.arange(9), scores)
    rows = prepared.rows
    rows.validate()
    assert rows.source.tolist() == [
        RowSource.INDEXED,
        RowSource.HOST,
        RowSource.RELAY,
        RowSource.HOST,
        RowSource.HOST,
        RowSource.HOST,
        RowSource.HOST,
        RowSource.RELAY,
        RowSource.RELAY,
    ]
    assert rows.query.tolist() == [1, 1, 1, 3, 2, 2, 3, 3, 1]
    last, every, cache = (
        TokenSelection.LAST_LOGITS,
        TokenSelection.ALL_LOGITS,
        TokenSelection.CACHE,
    )
    assert rows.selection.tolist() == [
        last,
        last,
        last,
        last,
        cache,
        every,
        every,
        every,
        cache,
    ]
    assert rows.host_tokens(1).tolist() == [40]
    assert rows.host_tokens(3).tolist() == [1, 2, 3]
    assert rows.host_tokens(7).tolist() == [81, 91]
    assert rows.gated.tolist() == [1, 0, 1, 0, 0, 0, 0, 1, 1]
    assert rows.prefix.tolist() == [10] * 9
    assert rows.position.tolist() == [10] * 9

    # A verification's drafts follow its current token; a relayed one's
    # are all its input tokens.
    drafts = [
        tokens[start : start + count].tolist()
        for start, count in zip(prepared.draft_start, prepared.draft_count)
    ]
    assert drafts[6] == [80, 90]
    assert drafts[7] == [81, 91]
    assert all(not drafts[index] for index in range(6))


def test_a_device_predicated_prefill_reads_its_tokens_as_a_gated_row():
    output = int(CallFlag.TOKEN_OUTPUT)
    calls, tokens = _calls(
        {"kind": CallKind.PREFILL, "tokens": 2, "flags": output},
        {
            "kind": CallKind.PREFILL,
            "tokens": 2,
            "flags": output,
            "predicate": Predicate.DEVICE,
        },
        tokens=[1, 2, 3, 4],
    )
    rows = prepare_calls(
        calls, tokens, np.arange(2), np.zeros(2, dtype=bool)
    ).rows
    # Both read their host tokens; only the predicated one is gated by its
    # slot's continuation predicate.
    assert rows.source.tolist() == [RowSource.HOST, RowSource.HOST]
    assert rows.host_tokens(1).tolist() == [3, 4]
    assert rows.gated.tolist() == [0, 1]


@pytest.mark.parametrize(
    "call",
    [
        # An ungated decode without a committed token.
        {"kind": CallKind.DECODE},
        # A fixed-parent verification of its current token alone.
        {"kind": CallKind.VERIFY, "tokens": 1},
        # A prefill without tokens to extend the cache with.
        {"kind": CallKind.PREFILL},
        # A call no text forward computes.
        {"kind": CallKind.TOKEN_DENOISING, "tokens": 1},
    ],
)
def test_calls_without_their_tokens_are_refused(call):
    calls, tokens = _calls(call, tokens=[1])
    with pytest.raises(ValueError):
        prepare_calls(calls, tokens, [0], np.zeros(1, dtype=bool))


def test_a_decode_bucket_pads_with_inert_rows_of_slot_zero():
    rows = _rows(
        [1, 1, 1],
        slot=[1, 2, 3],
        prefix=[5, 6, 7],
        gated=[True, True, True],
        force=[True, False, True],
    )
    starts = np.array([[11, 12, 13]])
    padded = pad_rows(rows, starts, True, 8, 8)
    assert padded.queries.tolist() == [1] * 8
    assert padded.slots.tolist() == [1, 2, 3, 0, 0, 0, 0, 0]
    assert padded.prefixes.tolist() == [5, 6, 7, 0, 0, 0, 0, 0]
    assert padded.force.tolist() == [True, False, True] + [False] * 5
    assert padded.writes.tolist() == [True] * 3 + [False] * 5
    assert padded.starts.tolist() == [[11, 12, 13, 0, 0, 0, 0, 0]]

    # A forced finish applies only when every row is gated.
    rows.gated[1] = False
    assert not pad_rows(rows, starts, True, 8, 8).force.any()
    # A decode bucket holds one token per row.
    with pytest.raises(AssertionError):
        pad_rows(rows, starts, True, 8, 9)


def test_a_prefill_bucket_pads_with_one_sequence_then_empty_rows():
    rows = _rows([3, 2], slot=[4, 5], prefix=[8, 0], causal=[True, False])
    padded = pad_rows(rows, np.zeros((2, 2)), False, 4, 16)
    assert padded.queries.tolist() == [3, 2, 11, 0]
    assert padded.slots.tolist() == [4, 5, 0, 0]
    assert padded.prefixes.tolist() == [8, 0, 0, 0]
    assert padded.causal.tolist() == [True, False, True, True]
    assert padded.writes.tolist() == [True, True, False, False]
    assert padded.starts.shape == (2, 4)

    # Padding tokens need a padding sequence.
    with pytest.raises(AssertionError):
        pad_rows(rows, np.zeros((2, 2)), False, 2, 6)


def test_graphs_address_their_configured_widths_and_eager_forwards_their_own():
    assert graph_widths([3, 1], [4, 2], decode=False, capacity=8) == [4, 2]
    with pytest.raises(ValueError):
        graph_widths([5], [4], decode=True, capacity=8)
    # An eager decode rounds up to a power of two within the binding.
    assert graph_widths([3, 5], None, decode=True, capacity=6) == [4, 6]
    assert graph_widths([3], None, decode=False, capacity=8) == [3]
    with pytest.raises(ValueError):
        graph_widths([9], None, decode=False, capacity=8)


def test_prompts_placing_vision_blocks_take_the_media_variant():
    text = text_forward(_rows([3, 1]), decode=False, embeddings=False)
    assert text.variant == PrefillVariant.TEXT
    assert text.tokens == 4
    assert text.last_logits and not text.cache_only

    block = _rows([3, 1], causal=[True, False])
    assert text_forward(block, False, False).variant == PrefillVariant.media(1)
    assert text_forward(_rows([2]), False, True).variant == (
        PrefillVariant.media(1)
    )
    spatial = _rows([2], source=[RowSource.PLACED], spatial=[True])
    assert text_forward(spatial, False, False).variant == (
        PrefillVariant.media(3)
    )
    cache = _rows([2, 2], [TokenSelection.CACHE] * 2)
    assert text_forward(cache, False, False).cache_only


def test_a_call_writes_at_most_its_query_within_its_allocation():
    assert written_length([4, 0], [3, 2], [3, 1], [8, 8]).tolist() == [7, 1]
    with pytest.raises(AssertionError):
        written_length([4], [1], [2], [8])
    with pytest.raises(AssertionError):
        written_length([4], [3], [3], [6])
