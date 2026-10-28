"""The token batch protocol between the worker and a text computation.

Per batch the worker passes host int64 columns and receives completion
records (``Executor.execute``):

- ``calls`` ``int64 [CALL_FIELDS, C]``: one column per call of the plan,
  active or not, fields as :class:`CallField` numbers them;
- ``forward`` ``int64 [FORWARD_FIELDS + 2 * G, R]``: the scheduler's forward
  rows permuted into call order, fields as :class:`ForwardField` numbers
  them, then each cache group ``g``'s first selected page
  (:func:`page_start`) and selected page count (:func:`page_count`);
- ``tokens`` ``int64 [N]``: every call's input tokens, concatenated;
- ``records`` ``int64 [K]``: the completion records the computation
  writes (:class:`RecordWriter`), ``K = record_capacity(C, N)``
  (:func:`record_capacity`).

Records, in words:

| Words | Content |
| --- | --- |
| ``[0, 12)`` | header, :class:`HeaderField` |
| ``[12 + 16 i, 28 + 16 i)`` | call ``i``'s record, :class:`RecordField` |
| then | segments, each ``[offset, count, n, rows[n], counts[n]]`` |
| then | tails (below) |

A segment is one logprob layout over a range of the batch's output words: a
microbatch's compact sampled-logprob column, shared by its reporting calls,
or one prompt chunk's words (rows ``0..scored``, each ``top`` counts). A
call's record names segments by their index in segment order.

A tail is ``[n_prompt, (segment, row) * n_prompt, n_drafts, drafts,
terminal]``: the call's prompt logprob ranges, as ``(segment, row)`` pairs
in prompt order, and its speculation, the draft tokens and the one-based
terminal draft prefix, or -1. A call's record locates its tail by record
word offset.

A call that proposes drafts (:attr:`CallField.PROPOSALS`) names its
proposed draft tokens with its range words (:attr:`RecordFlag.PROPOSALS`):
one output word per draft, in order.

Every offset into the output words is relative to the start of the words
lent to the batch.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import IntEnum, IntFlag

import numpy as np

__all__ = [
    "CALL_FIELDS",
    "CALL_FIELD_NAMES",
    "CALL_WORDS",
    "COMPLETION_FIELDS",
    "FORWARD_FIELDS",
    "HEADER_WORDS",
    "RECORD_MAGIC",
    "CallField",
    "CallFlag",
    "CallKind",
    "ForwardField",
    "GraphMode",
    "HeaderField",
    "Predicate",
    "RecordField",
    "RecordFlag",
    "RecordWriter",
    "combine_modes",
    "forward_fields",
    "page_count",
    "page_start",
    "record_capacity",
]


class CallField(IntEnum):
    """Rows of the ``calls`` columns, one field per row."""

    # The call's request slot.
    SLOT = 0
    # The call's kind (:class:`CallKind`).
    KIND = 1
    # 1 when the call runs: its request was not refused. A false predicate
    # gates the call on the device instead (:class:`Predicate`).
    ACTIVE = 2
    # The request's accepted progress: logical position, visible and
    # computed cache lengths, sampling counter and prompt-logits readiness.
    LOGICAL = 3
    VISIBLE = 4
    COMPUTED = 5
    RNG = 6
    PROMPT_READY = 7
    # The slot's allocated cache length after the batch's table install.
    ALLOCATED = 8
    # The call's predicate (:class:`Predicate`).
    PREDICATE = 9
    # The call's forward rows: first row and count in the ``forward``
    # columns, which group rows by call.
    ROW_START = 10
    ROW_COUNT = 11
    # The call's input tokens: first and count in ``tokens``.
    TOKEN_START = 12
    TOKEN_COUNT = 13
    # The most tokens the call may emit.
    MAX_TOKENS = 14
    # The call's outputs and JSON entries (:class:`CallFlag`).
    FLAGS = 15
    # A token-denoising call's canvas block and step; -1 when absent.
    CANVAS_BLOCK = 16
    CANVAS_STEP = 17
    # The call's request key, checked against the slot's request.
    REQUEST_ID = 18
    EPOCH = 19
    # Draft tokens the call proposes after its last committed token for
    # its request's next verification; 0 for none.
    PROPOSALS = 20
    # The token after the call's last input token, which the draft cache
    # entry of its last position reads; -1 when the call samples it or
    # knows none.
    LOOKAHEAD = 21


CALL_FIELDS = len(CallField)

# The field names a computation's manifest echoes, in row order.
CALL_FIELD_NAMES = tuple(field.name.lower() for field in CallField)


class CallKind(IntEnum):
    """Values of :attr:`CallField.KIND`."""

    PREFILL = 0
    DECODE = 1
    VERIFY = 2
    TOKEN_DENOISING = 3


class Predicate(IntEnum):
    """Values of :attr:`CallField.PREDICATE`."""

    # The call is ungated.
    NONE = 0
    # A uint8 predicate the computation gates the call on, on the device:
    # the call's JSON names its leased tensor (``predicate.tensor``), which
    # is copied into the slot's continuation predicate before the forward;
    # the call then runs as a gated row, as a relayed decode does.
    DEVICE = 1
    # A relayed int64 token: the computation gates on its slot's
    # continuation.
    RELAY = 2


class CallFlag(IntFlag):
    """Bits of :attr:`CallField.FLAGS`."""

    TOKEN_OUTPUT = 1 << 0
    TRANSITION_OUTPUT = 1 << 1
    COMPLETION_OUTPUT = 1 << 2
    WRITES_CONTEXT = 1 << 3
    WRITES_VISUAL_STATE = 1 << 4
    # The step JSON holds the call's sampling state, vision features or
    # readout.
    SAMPLING_STATE = 1 << 5
    FEATURES = 1 << 6
    READOUT = 1 << 7
    # The call evaluates tokens its request's cache already holds, from the
    # state checkpoint at its first token, to fill its draft cache: it
    # writes no K/V of the target and keeps no per-request state.
    DRAFT_REFILL = 1 << 8


class ForwardField(IntEnum):
    """The first rows of the ``forward`` columns, one field per row."""

    CALL_INDEX = 0
    REQUEST_POOL_INDEX = 1
    SEQ_LEN = 2
    QUERY_LEN = 3
    WRITE_CACHE = 4


FORWARD_FIELDS = len(ForwardField)


def page_start(group: int) -> int:
    """Return the ``forward`` row of cache group ``group``'s first page."""
    return FORWARD_FIELDS + 2 * group


def page_count(group: int) -> int:
    """Return the ``forward`` row of cache group ``group``'s page count."""
    return FORWARD_FIELDS + 2 * group + 1


def forward_fields(groups: int) -> int:
    """Return the ``forward`` rows given ``groups`` cache groups."""
    return FORWARD_FIELDS + 2 * groups


class GraphMode(IntEnum):
    """How a batch's forwards ran, as the header reports it.

    The numbering orders the modes by precedence: a batch whose microbatches
    ran differently reports the greatest (:func:`combine_modes`).
    """

    NONE = -1
    EAGER = 0
    REPLAY = 1
    CAPTURE = 2


def combine_modes(modes: Iterable[int]) -> GraphMode:
    """Return a batch's mode: any capture, else any replay, else eager.

    A batch without a forward reports :attr:`GraphMode.NONE`.
    """
    return GraphMode(max(modes, default=GraphMode.NONE))


class HeaderField(IntEnum):
    """Words of the records' header."""

    # :data:`RECORD_MAGIC`, which carries the layout version.
    MAGIC = 0
    # The batch's calls, every call of the plan.
    CALLS = 1
    # The segments following the call records.
    SEGMENTS = 2
    # The output words the batch wrote, at most those lent.
    WORDS_WRITTEN = 3
    # The batch's :class:`GraphMode`.
    MODE = 4
    # Statistics: live and padding tokens, forward rows and draft tokens.
    LIVE_TOKENS = 5
    PADDED_TOKENS = 6
    ROWS = 7
    DRAFT_TOKENS = 8
    # The routed-expert token capacity the batch's step agreed, 0 for none.
    EXPERT_CAPACITY = 9


HEADER_WORDS = 12


class RecordField(IntEnum):
    """Words of one call's record."""

    # :class:`RecordFlag` bits.
    FLAGS = 0
    # The call's progress after the batch.
    LOGICAL = 1
    RNG = 2
    VISIBLE = 3
    COMPUTED = 4
    # The completion block of the call's selection: output word offset,
    # extent (``COMPLETION_FIELDS`` words per task) and the call's task
    # column.
    SAMPLING_OFFSET = 5
    SAMPLING_EXTENT = 6
    SAMPLING_ROW = 7
    # The call's sampled logprobs: segment index and packed row. A
    # verification's rows for its drafts' positions follow that row, one per
    # position it may commit.
    LOGPROB_SEGMENT = 8
    LOGPROB_ROW = 9
    # The call's candidate or canvas words: output word offset and count.
    RANGE_OFFSET = 10
    RANGE_COUNT = 11
    # The call's tail: record word offset and words.
    TAIL_OFFSET = 12
    TAIL_WORDS = 13


CALL_WORDS = 16

# Words of one task's sampling completion, one section per field: valid,
# active, token, accepted drafts and finite (1 while every source logit of
# the task was finite; 0 marks a numerical fault of the forward, which no
# request constraint causes). A completion block of ``n`` tasks takes
# ``COMPLETION_FIELDS * n`` words, field-major.
COMPLETION_FIELDS = 5


class RecordFlag(IntFlag):
    """Bits of :attr:`RecordField.FLAGS`; an inactive call's flags are 0."""

    RAN = 1 << 0
    PROGRESS = 1 << 1
    SAMPLING = 1 << 2
    LOGPROB = 1 << 3
    PROMPT = 1 << 4
    CANDIDATES = 1 << 5
    CANVAS = 1 << 6
    SPECULATION = 1 << 7
    PROMPT_LOGITS_READY = 1 << 8
    # The range words hold the call's proposed draft tokens.
    PROPOSALS = 1 << 9


# "TKRC" in the high word, the record layout version in the low word.
RECORD_VERSION = 3
RECORD_MAGIC = (0x544B_5243 << 32) | RECORD_VERSION


def record_capacity(calls: int, tokens: int) -> int:
    """Return the record words the worker allocates for a batch.

    ``calls`` counts every call of the plan and ``tokens`` the batch's
    input tokens. Beside the header, a call takes at most 27 words of its
    own: its record (16), its tail's three counts, one sampling task's
    segment row (2), one prompt chunk's segment header (3) and one
    sampled-logprob segment header (3; a microbatch's segment has a task,
    so there are no more segments than calls). A token takes at most 7
    more: a scored prompt token its segment row (2), its tail entry (2) and
    another chunk's header (3); a verified position a task's segment row (2)
    and its draft (1).
    """
    return HEADER_WORDS + 27 * calls + 7 * tokens


class RecordWriter:
    """Writes one batch's completion records into the worker's host column.

    The header and every call record start zeroed. Call records are set
    column-wise over call indices (:meth:`ran`, :meth:`progress`,
    :meth:`sampling`, :meth:`logprobs`, :meth:`ranges`). Segments follow
    the call records in the order :meth:`segment` adds them, and tails
    follow every segment (:meth:`tail`). :meth:`finish` writes the header.

    Args:
        records: The worker's ``int64 [K]`` record column.
        calls: The batch's calls, every call of the plan.

    Raises:
        ValueError: The column is not a flat int64 column.
        AssertionError: The column cannot hold the header and call records.
    """

    def __init__(self, records: np.ndarray, calls: int) -> None:
        if records.dtype != np.int64 or records.ndim != 1:
            raise ValueError("completion records are a flat int64 column")
        self._records = records
        self._calls = calls
        self._cursor = HEADER_WORDS + CALL_WORDS * calls
        self._require(0)
        records[: self._cursor] = 0
        self._segments = 0
        self._tails = False
        # [C, CALL_WORDS] view of the call records.
        self.calls = records[HEADER_WORDS : self._cursor].reshape(
            calls, CALL_WORDS
        )

    @property
    def segments(self) -> int:
        """Return the segments written so far."""
        return self._segments

    def _require(self, words: int) -> None:
        if self._cursor + words > len(self._records):
            raise AssertionError(
                "completion records exceed the worker's record column: "
                f"{self._cursor + words} words of {len(self._records)}"
            )

    def _flag(self, calls: np.ndarray, flag: int) -> None:
        self.calls[calls, RecordField.FLAGS] |= int(flag)

    def ran(self, calls: Sequence[int] | np.ndarray) -> None:
        """Mark ``calls`` as run; every active call is."""
        self._flag(np.asarray(calls, dtype=np.int64), RecordFlag.RAN)

    def progress(
        self,
        calls: Sequence[int] | np.ndarray,
        logical: np.ndarray,
        rng: np.ndarray,
        visible: np.ndarray,
        computed: np.ndarray,
        prompt_ready: np.ndarray,
    ) -> None:
        """Set ``calls``' progress after the batch.

        ``prompt_ready`` is each call's prompt-logits readiness after the
        batch, which the worker installs as it is.
        """
        index = np.asarray(calls, dtype=np.int64)
        self._flag(index, RecordFlag.PROGRESS)
        self.calls[index, RecordField.LOGICAL] = logical
        self.calls[index, RecordField.RNG] = rng
        self.calls[index, RecordField.VISIBLE] = visible
        self.calls[index, RecordField.COMPUTED] = computed
        ready = index[np.asarray(prompt_ready, dtype=bool)]
        self._flag(ready, RecordFlag.PROMPT_LOGITS_READY)

    def sampling(
        self,
        calls: Sequence[int] | np.ndarray,
        offset: int,
        extent: int,
        rows: np.ndarray,
    ) -> None:
        """Set the completion block ``calls`` read their selection from.

        Args:
            calls: The sampled calls of one invocation.
            offset: The block's first output word.
            extent: The block's words: :data:`COMPLETION_FIELDS` per
                task.
            rows: Each call's task column in the block.
        """
        index = np.asarray(calls, dtype=np.int64)
        self._flag(index, RecordFlag.SAMPLING)
        self.calls[index, RecordField.SAMPLING_OFFSET] = offset
        self.calls[index, RecordField.SAMPLING_EXTENT] = extent
        self.calls[index, RecordField.SAMPLING_ROW] = rows

    def logprobs(
        self,
        calls: Sequence[int] | np.ndarray,
        segment: int,
        rows: np.ndarray,
    ) -> None:
        """Set the segment and packed rows of ``calls``' sampled logprobs."""
        index = np.asarray(calls, dtype=np.int64)
        if not 0 <= segment < self._segments:
            raise AssertionError("a call names a segment the batch lacks")
        self._flag(index, RecordFlag.LOGPROB)
        self.calls[index, RecordField.LOGPROB_SEGMENT] = segment
        self.calls[index, RecordField.LOGPROB_ROW] = rows

    def ranges(
        self,
        calls: Sequence[int] | np.ndarray,
        offset: np.ndarray,
        count: np.ndarray,
        canvas: bool = False,
        proposals: bool = False,
    ) -> None:
        """Set ``calls``' candidate (or ``canvas``, or proposed draft)
        output words.
        """  # noqa: D205
        index = np.asarray(calls, dtype=np.int64)
        self._flag(
            index,
            RecordFlag.PROPOSALS
            if proposals
            else RecordFlag.CANVAS
            if canvas
            else RecordFlag.CANDIDATES,
        )
        self.calls[index, RecordField.RANGE_OFFSET] = offset
        self.calls[index, RecordField.RANGE_COUNT] = count

    def segment(
        self,
        offset: int,
        count: int,
        rows: Sequence[int] | np.ndarray,
        counts: Sequence[int] | np.ndarray,
    ) -> int:
        """Append one logprob segment and return its index.

        Args:
            offset: The segment's first output word.
            count: The segment's output words.
            rows: The layout's reported row of each packed row.
            counts: The layout's top count of each packed row.

        Raises:
            AssertionError: A tail was written already, the layout is
                malformed, or the column is full.
        """
        if self._tails:
            raise AssertionError("record segments precede every tail")
        n = len(rows)
        if len(counts) != n:
            raise AssertionError("a segment's rows and counts disagree")
        self._require(3 + 2 * n)
        start = self._cursor
        self._records[start : start + 3] = (offset, count, n)
        self._records[start + 3 : start + 3 + n] = rows
        self._records[start + 3 + n : start + 3 + 2 * n] = counts
        self._cursor += 3 + 2 * n
        self._segments += 1
        return self._segments - 1

    def tail(
        self,
        call: int,
        prompt: Sequence[tuple[int, int]] | None = None,
        drafts: Sequence[int] | None = None,
        terminal: int | None = None,
    ) -> None:
        """Write call ``call``'s tail.

        Args:
            call: The call index.
            prompt: The call's prompt logprob ranges as ``(segment, row)``
                pairs in prompt order; ``None`` for a call without prompt
                chunks. A call with chunks of no scored row passes an empty
                list, and the worker sets its ranges empty.
            drafts: A verification's draft tokens; ``None`` without
                speculation.
            terminal: The one-based terminal draft prefix, if any.

        Raises:
            AssertionError: A pair names a segment the batch lacks, or the
                column is full.
        """
        pairs = list(prompt or ())
        tokens = list(drafts or ())
        words = 3 + 2 * len(pairs) + len(tokens)
        self._require(words)
        if any(not 0 <= segment < self._segments for segment, _ in pairs):
            raise AssertionError("a prompt range names a missing segment")
        start = self._cursor
        tail = np.empty(words, dtype=np.int64)
        tail[0] = len(pairs)
        tail[1 : 1 + 2 * len(pairs)] = np.asarray(
            pairs, dtype=np.int64
        ).reshape(-1)
        tail[1 + 2 * len(pairs)] = len(tokens)
        tail[2 + 2 * len(pairs) : -1] = tokens
        tail[-1] = -1 if terminal is None else terminal
        self._records[start : start + words] = tail
        self._cursor += words
        self._tails = True

        record = self.calls[call]
        record[RecordField.TAIL_OFFSET] = start
        record[RecordField.TAIL_WORDS] = words
        if prompt is not None:
            record[RecordField.FLAGS] |= int(RecordFlag.PROMPT)
        if drafts is not None:
            record[RecordField.FLAGS] |= int(RecordFlag.SPECULATION)

    def finish(
        self,
        *,
        words_written: int,
        lent: int,
        mode: int,
        live_tokens: int,
        padded_tokens: int,
        rows: int,
        draft_tokens: int,
        expert_capacity: int = 0,
    ) -> int:
        """Write the header and return the record words written.

        Raises:
            AssertionError: More output words were written than lent.
        """
        if not 0 <= words_written <= lent:
            raise AssertionError("a batch wrote more output words than lent")
        header = self._records[:HEADER_WORDS]
        header[:] = 0
        header[HeaderField.MAGIC] = RECORD_MAGIC
        header[HeaderField.CALLS] = self._calls
        header[HeaderField.SEGMENTS] = self._segments
        header[HeaderField.WORDS_WRITTEN] = words_written
        header[HeaderField.MODE] = mode
        header[HeaderField.LIVE_TOKENS] = live_tokens
        header[HeaderField.PADDED_TOKENS] = padded_tokens
        header[HeaderField.ROWS] = rows
        header[HeaderField.DRAFT_TOKENS] = draft_tokens
        header[HeaderField.EXPERT_CAPACITY] = expert_capacity
        return self._cursor
