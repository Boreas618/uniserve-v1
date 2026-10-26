"""Token rows of a text forward and the host columns that describe them.

A *row* is one call's query over its slot's prefix cache. :class:`Rows`
holds a forward's rows column-wise: per row its slot, token source
(:class:`RowSource`), visible prefix, query length, logits selection
(:class:`TokenSelection`), gate, forced finish, causality, first position
and host tokens. :func:`prepare_calls` derives the rows of a step's text
calls from their call columns alone, never from device values.

The forward's rows then become its graph inputs: the logits rows each row
projects (:func:`logit_layout`, :func:`logit_positions`), the numerical
variant and bucket request (:func:`text_forward`), the page widths
(:func:`graph_widths`), the padded lengths of a bucket (:func:`pad_rows`)
and the packed row-gather column (:func:`pack_rows`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .protocol import CallField, CallFlag, CallKind, Predicate
from .shapes import PrefillVariant, TextForward

__all__ = [
    "ROW_SECTIONS",
    "PaddedRows",
    "Prepared",
    "RowSource",
    "Rows",
    "TokenSelection",
    "graph_widths",
    "logit_layout",
    "logit_positions",
    "logit_rows",
    "pack_rows",
    "pad_rows",
    "page_width",
    "prepare_calls",
    "text_forward",
    "written_length",
]

# Per-row sections of the packed row column before the per-table page
# counts: slots, prefixes, queries, writes, query offsets, prefix offsets.
ROW_SECTIONS = 6


class RowSource(IntEnum):
    """Where a row's input tokens and positions come from."""

    # Host tokens at consecutive positions from the row's position.
    HOST = 0
    # The slot's committed next token (on the device) followed by host
    # draft tokens, at consecutive positions: a continuation whose
    # predecessor's token was never observed on the host.
    RELAY = 1
    # One token, its position and the slot's verified cache length read
    # from the slot state on the device: a decode following its request's
    # last committed selection.
    INDEXED = 2
    # Host tokens at explicit rotary positions (temporal, row, column): a
    # vision block, or text of a prompt holding one.
    PLACED = 3


class TokenSelection(IntEnum):
    """Which logits a row projects, independently of its cache writes."""

    LAST_LOGITS = 0
    ALL_LOGITS = 1
    CACHE = 2


def logit_rows(selection: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Return the candidate logits rows each row contributes.

    Its last position (none for an empty query), every position, or none.
    """
    selection = np.asarray(selection)
    query = np.asarray(query, dtype=np.int64)
    return np.select(
        [
            selection == TokenSelection.LAST_LOGITS,
            selection == TokenSelection.ALL_LOGITS,
        ],
        [np.minimum(query, 1), query],
        0,
    ).astype(np.int64)


def logit_layout(
    selection: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return each row's candidate logits rows ``(first, count)``.

    Rows contribute candidates in row order.
    """
    count = logit_rows(selection, query)
    first = np.cumsum(count) - count
    return first, count


def logit_positions(query: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Return the token positions projected to candidate logits rows.

    Each row projects its last ``count`` positions, rows in order: every
    position of a verification row or of a prefill row scoring its prompt.

    Args:
        query: ``[n]`` each row's query tokens.
        count: ``[n]`` each row's candidate rows (:func:`logit_rows`).

    Returns:
        ``[sum(count)]`` int64 positions in the forward's tokens.
    """
    query = np.asarray(query, dtype=np.int64)
    count = np.asarray(count, dtype=np.int64)
    stops = np.cumsum(query)
    within = np.arange(int(count.sum()), dtype=np.int64) - np.repeat(
        np.cumsum(count) - count, count
    )
    return np.repeat(stops - count, count) + within


@dataclass(slots=True)
class Rows:
    """A text forward's rows, column by column.

    Attributes:
        slot: ``int64 [n]`` request slots.
        source: ``int64 [n]`` :class:`RowSource`.
        prefix: ``int64 [n]`` visible cache tokens before the query. A
            relayed decode's value bounds the device's verified length.
        query: ``int64 [n]`` query tokens.
        selection: ``int64 [n]`` :class:`TokenSelection`.
        gated: ``bool [n]`` the call is gated by its continuation flag.
        force: ``bool [n]`` grammar state forces the call to finish.
        causal: ``bool [n]`` the query attends causally; a vision block
            attends to all of itself.
        position: ``int64 [n]`` the first position of a host or relay row.
        token_start: ``int64 [n]`` the row's first host token in
            ``tokens``.
        token_count: ``int64 [n]`` the row's host tokens: its query for a
            host or placed row, its drafts (one less) for a relay row, none
            for an indexed row.
        tokens: ``int64`` flat token ids the rows index.
        spatial: ``bool [n]`` a placed row's positions are spatial (three
            meaningful axes); only its temporal axis is otherwise.
        placed: Each placed row's ``int64 [3, token_count]`` positions.
    """

    slot: np.ndarray
    source: np.ndarray
    prefix: np.ndarray
    query: np.ndarray
    selection: np.ndarray
    gated: np.ndarray
    force: np.ndarray
    causal: np.ndarray
    position: np.ndarray
    token_start: np.ndarray
    token_count: np.ndarray
    tokens: np.ndarray
    spatial: np.ndarray
    placed: dict[int, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.slot)

    def logit_rows(self) -> np.ndarray:
        """Return the candidate logits rows each row contributes."""
        return logit_rows(self.selection, self.query)

    def host_tokens(self, row: int) -> np.ndarray:
        """Return row ``row``'s host tokens."""
        start = int(self.token_start[row])
        return self.tokens[start : start + int(self.token_count[row])]

    def validate(self) -> None:
        """Check every row's source against its query and slot.

        Raises:
            ValueError: A placed row's positions disagree with its tokens,
                a row's tokens disagree with its nonempty query, or a row
                has no request slot.
        """
        for row, positions in self.placed.items():
            if positions.shape != (3, int(self.token_count[row])):
                raise ValueError(
                    "placed row positions disagree with its tokens"
                )
        indexed = self.source == RowSource.INDEXED
        expected = np.where(
            self.source == RowSource.RELAY,
            self.token_count + 1,
            np.where(indexed, 1, self.token_count),
        )
        if np.any(
            (expected != self.query)
            | (self.query == 0)
            | (indexed & (self.token_count != 0))
        ):
            raise ValueError("text row tokens disagree with its query length")
        if np.any(self.slot == 0):
            raise ValueError("text rows require a request slot")


@dataclass(slots=True)
class Prepared:
    """The rows of a step's text calls, one row per call.

    Attributes:
        calls: ``int64 [n]`` each row's call index.
        rows: The calls' rows, in call order.
        tagged: ``bool [n]`` the call's predicate relays a token.
        scores: ``bool [n]`` the call's request scores its prompt.
        draft_start: ``int64 [n]`` a verification's first draft token in
            the step's tokens.
        draft_count: ``int64 [n]`` a verification's draft tokens; zero for
            other calls.
    """

    calls: np.ndarray
    rows: Rows
    tagged: np.ndarray
    scores: np.ndarray
    draft_start: np.ndarray
    draft_count: np.ndarray


def prepare_calls(
    calls: np.ndarray,
    tokens: np.ndarray,
    indices: Sequence[int] | np.ndarray,
    scores: np.ndarray,
    force: np.ndarray | None = None,
) -> Prepared:
    """Build the input rows of a step's text calls from their call columns.

    A decode gated by a relayed token is *indexed*: it reads its token,
    position and cache length from the slot state. A gated decode or
    verification, and a gated call without input tokens, is *relayed*: its
    first token is the slot's committed next token, followed by its input
    tokens as drafts (none for a decode). Any other call reads its host
    tokens: a decode its first, a prefill or verification all of them.

    A verification and a prefill scoring its prompt project every
    position; a decode and a call with a token output project their last;
    any other call only writes the cache. A verification's drafts are its
    input tokens after its current token, all of them when relayed.

    Args:
        calls: The step's ``int64 [CALL_FIELDS, C]`` call columns.
        tokens: The step's flat input tokens.
        indices: The calls to prepare: active text calls placing no media.
        scores: ``bool [n]`` whether each call's request scores its prompt
            (``RequestTable.scores_prompt`` of its slot).
        force: ``bool [n]`` whether each call's grammar state forces it to
            finish; none by default.

    Raises:
        ValueError: A call is no text call, a decode has no committed
            token, a fixed-parent verification lacks its current and draft
            tokens, or a call extending the cache has no input token.
    """
    index = np.asarray(indices, dtype=np.int64)
    columns = calls[:, index]
    kind = columns[CallField.KIND]
    if np.any((kind < CallKind.PREFILL) | (kind > CallKind.VERIFY)):
        raise ValueError("only prefill, decode and verify are text calls")
    decode = kind == CallKind.DECODE
    verify = kind == CallKind.VERIFY
    predicate = columns[CallField.PREDICATE]
    count = columns[CallField.TOKEN_COUNT]
    start = columns[CallField.TOKEN_START]
    token_output = (columns[CallField.FLAGS] & CallFlag.TOKEN_OUTPUT) != 0
    scores = np.asarray(scores, dtype=bool)

    tagged = predicate == Predicate.RELAY
    # A relayed token's call and a device-predicated call are gated by the
    # slot's continuation predicate.
    gated = tagged | (predicate == Predicate.DEVICE)
    indexed = decode & tagged
    relay = (
        (predicate != Predicate.NONE)
        & ~indexed
        & (decode | verify | (count == 0))
    )
    hosts = np.where(indexed | (decode & relay), 0, np.where(decode, 1, count))
    if np.any(decode & ~indexed & ~relay & (count < 1)):
        raise ValueError("last-sampled token source has no committed token")
    if np.any(verify & ~relay & (hosts < 2)):
        raise ValueError(
            "fixed-parent verification requires current and draft tokens"
        )
    if np.any((hosts == 0) & ~indexed & ~relay):
        raise ValueError("text extension requires input tokens")

    selection = np.where(
        verify | (~decode & scores),
        TokenSelection.ALL_LOGITS,
        np.where(
            decode | token_output,
            TokenSelection.LAST_LOGITS,
            TokenSelection.CACHE,
        ),
    ).astype(np.int64)
    source = np.where(
        indexed,
        RowSource.INDEXED,
        np.where(relay, RowSource.RELAY, RowSource.HOST),
    ).astype(np.int64)
    query = np.where(relay, hosts + 1, np.where(indexed, 1, hosts))
    # An ungated verification's first input token is its current token.
    first = (predicate == Predicate.NONE).astype(np.int64)
    draft_count = np.where(verify, np.maximum(count - first, 0), 0)

    n = len(index)
    rows = Rows(
        slot=columns[CallField.SLOT].copy(),
        source=source,
        prefix=columns[CallField.VISIBLE].copy(),
        query=query.astype(np.int64),
        selection=selection,
        gated=gated,
        force=(
            np.zeros(n, dtype=bool)
            if force is None
            else np.asarray(force, dtype=bool)
        ),
        causal=np.ones(n, dtype=bool),
        position=columns[CallField.LOGICAL].copy(),
        token_start=start.copy(),
        token_count=hosts.astype(np.int64),
        tokens=np.asarray(tokens, dtype=np.int64),
        spatial=np.zeros(n, dtype=bool),
    )
    return Prepared(
        calls=index,
        rows=rows,
        tagged=tagged,
        scores=scores,
        draft_start=start + first,
        draft_count=draft_count.astype(np.int64),
    )


def page_width(lengths: np.ndarray) -> int:
    """Return a table's page width: its rows' most pages, at least one."""
    return max(int(np.max(lengths, initial=0)), 1)


def graph_widths(
    live: Sequence[int],
    configured: Sequence[int] | None,
    decode: bool,
    capacity: int,
) -> list[int]:
    """Return the page width each block table's forward addresses.

    A graphed forward addresses the configured widths; an eager decode the
    next power of two of its rows' width, at most ``capacity``; an eager
    prefill its rows' own.

    Args:
        live: Each table's page width over the forward's rows
            (:func:`page_width`).
        configured: Each table's configured graph width; ``None`` for an
            eager forward.
        decode: The forward is a decode.
        capacity: The pages a row of the binding selects from one table.

    Raises:
        ValueError: A table selects more pages than its graph width or the
            binding holds.
    """
    widths = []
    for table, width in enumerate(live):
        if configured is not None:
            if width > configured[table]:
                raise ValueError(
                    f"table {table} selects {width} pages; its graph width "
                    f"is {configured[table]}"
                )
            width = configured[table]
        elif decode:
            width = min(1 << max(width - 1, 0).bit_length(), capacity)
        if width > capacity:
            raise ValueError(
                f"table {table} selects {width} pages; the binding holds "
                f"{capacity}"
            )
        widths.append(int(width))
    return widths


def text_forward(rows: Rows, decode: bool, embeddings: bool) -> TextForward:
    """Return the bucket request of a forward of ``rows``.

    A prompt placing vision blocks (placed rows, vision features or rows
    that do not attend causally) takes the media variant, on three axes
    when a placed row is spatial; causal text takes the text variant.

    Args:
        rows: The forward's rows.
        decode: The forward is a decode.
        embeddings: Vision features replace some rows' token embeddings.
    """
    placed = rows.source == RowSource.PLACED
    multimodal = bool(np.any(placed)) or embeddings or not np.all(rows.causal)
    axes = 3 if np.any(placed & rows.spatial) else 1
    variant = PrefillVariant.media(axes) if multimodal else PrefillVariant.TEXT
    return TextForward(
        queries=rows.query,
        tokens=int(np.sum(rows.query)),
        decode=decode,
        variant=variant,
        last_logits=bool(np.all(rows.selection == TokenSelection.LAST_LOGITS)),
        cache_only=bool(np.all(rows.selection == TokenSelection.CACHE)),
    )


@dataclass(slots=True)
class PaddedRows:
    """A forward's row lengths padded to its bucket.

    Attributes:
        queries: ``int64 [rows]`` query tokens.
        prefixes: ``int64 [rows]`` visible prefixes.
        slots: ``int64 [rows]`` slots, 0 for padding.
        causal: ``bool [rows]`` causality, causal for padding.
        force: ``bool [rows]`` forced finish of rows that are all gated.
        writes: ``bool [rows]`` rows writing the cache: the live rows.
        starts: ``int64 [tables, rows]`` first selected pages, 0 for
            padding.
    """

    queries: np.ndarray
    prefixes: np.ndarray
    slots: np.ndarray
    causal: np.ndarray
    force: np.ndarray
    writes: np.ndarray
    starts: np.ndarray


def pad_rows(
    rows: Rows,
    starts: np.ndarray,
    decode: bool,
    padded_rows: int,
    padded_tokens: int,
) -> PaddedRows:
    """Pad a forward's rows to a bucket of ``padded_rows`` and tokens.

    A decode bucket adds inert one-token rows; a prefill bucket adds one
    sequence holding every padding token, then empty sequences. Padding
    rows have prefix 0, slot 0, causal attention, no forced finish, and
    never write the cache. A row's forced finish applies only when every
    row of the forward is gated.

    Args:
        rows: The forward's live rows.
        starts: ``int64 [tables, live]`` each table's first selected page
            per live row.
        decode: The forward is a decode.
        padded_rows: The bucket's rows, at least the live rows.
        padded_tokens: The bucket's tokens, at least the live tokens.

    Raises:
        AssertionError: Token padding without a padding row, or padded
            lengths that disagree with the bucket.
    """
    live = len(rows)
    live_tokens = int(np.sum(rows.query))
    extra = padded_rows - live
    if extra < 0 or padded_tokens < live_tokens:
        raise AssertionError("a graph bucket is smaller than its forward")
    queries = rows.query.astype(np.int64)
    if extra > 0:
        if decode:
            padding = np.ones(extra, dtype=np.int64)
        else:
            padding = np.zeros(extra, dtype=np.int64)
            padding[0] = padded_tokens - live_tokens
        queries = np.concatenate([queries, padding])
    elif padded_tokens > live_tokens:
        raise AssertionError("token padding requires an additional sequence")
    if int(queries.sum()) != padded_tokens:
        raise AssertionError(
            "padded query lengths disagree with the graph bucket"
        )

    zeros = np.zeros(extra, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64).reshape(-1, live)
    gated_all = bool(np.all(rows.gated))
    return PaddedRows(
        queries=queries,
        prefixes=np.concatenate([rows.prefix, zeros]),
        slots=np.concatenate([rows.slot, zeros]),
        causal=np.concatenate([rows.causal, np.ones(extra, dtype=bool)]),
        force=np.concatenate(
            [rows.force & gated_all, np.zeros(extra, dtype=bool)]
        ),
        writes=np.arange(padded_rows) < live,
        starts=np.concatenate(
            [starts, np.zeros((len(starts), extra), dtype=np.int64)], axis=1
        ),
    )


def pack_rows(
    slots: np.ndarray,
    prefixes: np.ndarray,
    queries: np.ndarray,
    writes: np.ndarray,
    pages: Sequence[np.ndarray],
    stride: int,
    tokens: int,
) -> np.ndarray:
    """Pack rows into the row-gather column the device gathers from.

    Layout, ``stride`` words per section:

    | Section | Words |
    | --- | --- |
    | slots, prefixes, queries, writes | ``stride`` each |
    | query offsets, prefix offsets | ``stride`` each |
    | each table's selected page count | ``stride`` each |
    | each token's row | ``tokens`` |

    The offsets are exclusive prefix sums over the rows, then the total.

    Args:
        slots: ``[n]`` each row's slot.
        prefixes: ``[n]`` each row's prefix.
        queries: ``[n]`` each row's query tokens.
        writes: ``[n]`` whether each row writes the cache.
        pages: Each table's selected page counts per row; a row past a
            table's counts selects none.
        stride: Section words: the binding's rows plus one.
        tokens: Token-row words: the binding's tokens.

    Returns:
        ``int64 [(ROW_SECTIONS + tables) * stride + tokens]`` words.

    Raises:
        ValueError: The rows or their tokens exceed the sections.
    """
    queries = np.asarray(queries, dtype=np.int64)
    rows = len(queries)
    total = int(queries.sum())
    if rows + 1 > stride or total > tokens:
        raise ValueError("packed rows exceed the row column")
    words = np.zeros((ROW_SECTIONS + len(pages)) * stride + tokens, np.int64)
    sections = words[: (ROW_SECTIONS + len(pages)) * stride].reshape(-1, stride)
    sections[0, :rows] = slots
    sections[1, :rows] = prefixes
    sections[2, :rows] = queries
    sections[3, :rows] = np.asarray(writes, dtype=np.int64)
    sections[4, 1 : rows + 1] = np.cumsum(queries)
    sections[5, 1 : rows + 1] = np.cumsum(np.asarray(prefixes, np.int64))
    for table, lengths in enumerate(pages):
        counted = np.asarray(lengths, dtype=np.int64)[:rows]
        sections[ROW_SECTIONS + table, : len(counted)] = counted
    token_rows = words[(ROW_SECTIONS + len(pages)) * stride :]
    token_rows[:total] = np.repeat(np.arange(rows, dtype=np.int64), queries)
    return words


def written_length(
    prefix: np.ndarray,
    query: np.ndarray,
    count: np.ndarray,
    allocated: np.ndarray,
) -> np.ndarray:
    """Return the cache extents calls write: prefix plus written tokens.

    Args:
        prefix: ``[n]`` each call's row prefix.
        query: ``[n]`` each call's row query.
        count: ``[n]`` the tokens each call commits to the cache.
        allocated: ``[n]`` each call's allocated cache length.

    Raises:
        AssertionError: A count exceeds its query, or an extent exceeds the
            slot's allocated length.
    """
    count = np.asarray(count, dtype=np.int64)
    if np.any(count > query):
        raise AssertionError("a call commits more cache tokens than it queries")
    length = np.asarray(prefix, dtype=np.int64) + count
    if np.any(length > allocated):
        raise AssertionError(
            "a call's cache extent exceeds its scheduler block table"
        )
    return length
