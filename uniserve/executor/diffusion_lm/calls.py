"""Token-denoising calls of a diffusion LM lane: validation, packing and
completion records.

A token-denoising call carries either a readout or a denoising step, and a
batch's active calls carry one form (:func:`batch_form`). Every forward row
of a call reads its request's unchanged cached prefix and writes no prefix
cache: the row's prefix is the call's ``VISIBLE`` length and its first
position the call's ``LOGICAL`` position. A row's ``seq_len - query_len`` is
attention geometry the engine builds from the same request state, so the
two are not compared.

**Readouts.** A readout's rows partition its tokens, back to back, and every
row reads at least one answer slot. The worker appends each readout's slot
tokens, candidate offsets and candidate ids (``ipc::Readout``) to the flat
``tokens`` column, after the calls' input tokens, and the step JSON locates
the three lists as ``[start, count]`` ranges of that column
(:class:`ReadoutSpans`)::

    "readout": {"slot_tokens": [start, slots],
                "candidate_offsets": [start, slots + 1],
                "candidate_ids": [start, candidates]}

:func:`pack_readouts` packs the active readouts into one forward
(:class:`ReadoutPass`); its readout launches (:meth:`ReadoutPass.windows`)
each read one slot chunk's window of candidates.

**Steps.** A denoising step is one row of the block length over its
request's prefix, stepping the block its request keeps resident with the
served sampling and the request's seed (:class:`Seeds`, :func:`pack_steps`).

**Records.** :func:`record_readouts` and :func:`record_steps` fill the
completion records of a batch's active calls: their progress (unchanged,
with the computed length at the visible one) and their candidate or canvas
words. An inactive call's record stays zero.

Every check runs on the host before any device work or collective, so every
rank of a group refuses the same calls.
"""  # noqa: D205

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from uniserve.executor.text.protocol import (
    CallField,
    CallKind,
    ForwardField,
    RecordWriter,
)

from .bounds import ServedSampling
from .shapes import CANDIDATES, SLOT_BUCKETS, slot_chunks

__all__ = [
    "ReadoutPass",
    "ReadoutSpans",
    "Seeds",
    "Steps",
    "Window",
    "batch_form",
    "pack_readouts",
    "pack_steps",
    "readout_spans",
    "record_readouts",
    "record_steps",
    "seed_bits",
]

# A block prefix is a 32-bit token count in the attention metadata.
_PREFIX_LIMIT = (1 << 32) - 1


def batch_form(calls: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a batch's active calls and whether they are denoising steps.

    A call is a denoising step when it carries canvas coordinates
    (``CANVAS_BLOCK`` is not -1), otherwise a readout. A batch without an
    active call reports readouts.

    Args:
        calls: The batch's ``int64 [CALL_FIELDS, C]`` call columns.

    Raises:
        ValueError: A call is not token-denoising, or the active calls mix
            readouts and denoising steps.
    """
    if np.any(calls[CallField.KIND] != CallKind.TOKEN_DENOISING):
        raise ValueError("a diffusion LM batch carries token-denoising calls")
    active = np.flatnonzero(calls[CallField.ACTIVE] != 0)
    canvas = calls[CallField.CANVAS_BLOCK, active] != -1
    if np.any(canvas) and not np.all(canvas):
        raise ValueError(
            "a batch carries readouts or denoising steps, not both"
        )
    return active, bool(np.any(canvas))


def _ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Return the indices ``starts[i] + arange(counts[i])``, concatenated."""
    ends = np.cumsum(counts)
    total = int(ends[-1]) if len(ends) else 0
    # Each index's offset within its range is its packed position less the
    # packed start of its range.
    return np.arange(total, dtype=np.int64) + np.repeat(
        starts - (ends - counts), counts
    )


def _check_prefix(visible: np.ndarray) -> None:
    if np.any((visible < 0) | (visible > _PREFIX_LIMIT)):
        raise ValueError("a block prefix exceeds the token range")


def _read_only(
    calls: np.ndarray, forward: np.ndarray, active: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the forward rows of ``active`` and the call of each row.

    The calls' rows must read their request's prefix without writing the
    prefix cache: each call has one or more rows, each of its slot, of at
    least one query token and not writing.

    Returns:
        The rows' indices into the forward columns, in call and row order,
        and each row's position in ``active``.

    Raises:
        ValueError: A call has no row, a row lies outside the forward
            columns or belongs to another call, writes, reads another slot
            or has no query token.
    """
    starts = calls[CallField.ROW_START, active]
    counts = calls[CallField.ROW_COUNT, active]
    if np.any(counts < 1):
        raise ValueError("block rows must read the request's unchanged prefix")
    if np.any(starts < 0) or np.any(starts + counts > forward.shape[1]):
        raise ValueError("a call's rows lie outside the forward columns")
    rows = _ranges(starts, counts)
    owner = np.repeat(np.arange(len(active)), counts)
    if np.any(forward[ForwardField.CALL_INDEX, rows] != active[owner]):
        raise ValueError("a call's rows belong to another call")
    slots = calls[CallField.SLOT, active]
    if (
        np.any(forward[ForwardField.WRITE_CACHE, rows] != 0)
        or np.any(
            forward[ForwardField.REQUEST_POOL_INDEX, rows] != slots[owner]
        )
        or np.any(forward[ForwardField.QUERY_LEN, rows] <= 0)
    ):
        raise ValueError("block rows must read the request's unchanged prefix")
    return rows, owner


class ReadoutSpans(NamedTuple):
    """Where a readout's lists lie in the batch's flat ``tokens`` column.

    Each is a ``(start, count)`` range: the slot tokens (each slot's index
    in the call's concatenated tokens, increasing), the ``slots + 1``
    offsets of each slot's candidates in the candidate ids, and the
    candidate ids of every slot, slot after slot.
    """

    slot_tokens: tuple[int, int]
    candidate_offsets: tuple[int, int]
    candidate_ids: tuple[int, int]

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> ReadoutSpans:
        """Decode one call's ``readout`` entry of the step JSON."""

        def span(name: str) -> tuple[int, int]:
            start, count = value[name]
            return int(start), int(count)

        return cls(
            span("slot_tokens"),
            span("candidate_offsets"),
            span("candidate_ids"),
        )


def readout_spans(calls: Mapping[str, Any]) -> dict[int, ReadoutSpans]:
    """Return the readouts of a step JSON's ``calls`` entries, by call."""
    return {
        int(call): ReadoutSpans.from_json(entry["readout"])
        for call, entry in calls.items()
        if "readout" in entry
    }


class Window(NamedTuple):
    """One ``readout`` launch: a slot chunk and a window of its candidates.

    Attributes:
        start: The chunk's first slot of the pass.
        live: The chunk's slots.
        bucket: The ``readout`` slot bucket holding them.
        first: The window's first candidate of the pass.
        end: The candidate after the window's last.
    """

    start: int
    live: int
    bucket: int
    first: int
    end: int


@dataclass(frozen=True, slots=True)
class ReadoutPass:
    """A readout forward: the sequences, slots and candidates of its calls,
    back to back.

    Per-sequence columns have one entry per packed sequence, in call and row
    order; candidates are in sequence, slot and candidate order, so each
    call's candidates are contiguous.

    Attributes:
        calls: ``[n]`` the active calls, ascending.
        rows: ``[r]`` each sequence's index into the forward columns, which
            locates its selected pages.
        slot: ``[r]`` each sequence's request slot.
        query: ``[r]`` each sequence's block tokens.
        prefix: ``[r]`` the cached prefix tokens each sequence reads: its
            call's visible prefix.
        position: ``[r]`` the position of each sequence's first token: its
            call's logical position.
        tokens: ``[t]`` the packed block tokens of every sequence.
        slots: ``[s]`` the packed token index of every answer slot,
            strictly increasing.
        slot_candidates: ``[s + 1]`` offsets of each slot's candidates in
            ``candidates``.
        candidates: ``[m]`` candidate token ids.
        index: ``[m]`` each candidate's flat index into the ``[bucket,
            vocab]`` log-probabilities of its slot chunk: ``(slot - chunk
            start) * vocab + id``.
        candidate_start: ``[n]`` each call's first candidate.
        candidate_count: ``[n]`` each call's candidates.
    """  # noqa: D205

    calls: np.ndarray
    rows: np.ndarray
    slot: np.ndarray
    query: np.ndarray
    prefix: np.ndarray
    position: np.ndarray
    tokens: np.ndarray
    slots: np.ndarray
    slot_candidates: np.ndarray
    candidates: np.ndarray
    index: np.ndarray
    candidate_start: np.ndarray
    candidate_count: np.ndarray

    @property
    def longest(self) -> int:
        """Return the tokens of the longest block."""
        return int(self.query.max(initial=0))

    def windows(self) -> Iterator[Window]:
        """Yield the pass's ``readout`` launches, in slot order.

        Each slot chunk (:func:`slot_chunks`) reads its candidates in
        windows of at most :data:`CANDIDATES`; a chunk without candidates
        launches none.
        """
        for chunk in slot_chunks(len(self.slots)):
            first = int(self.slot_candidates[chunk.start])
            last = int(self.slot_candidates[chunk.start + chunk.live])
            for window in range(first, last, CANDIDATES):
                yield Window(
                    chunk.start,
                    chunk.live,
                    chunk.bucket,
                    window,
                    min(window + CANDIDATES, last),
                )


def _spans(
    active: np.ndarray,
    readouts: Mapping[int, ReadoutSpans],
    tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the starts and counts of the active readouts' lists.

    Both are ``int64 [3, n]``: the slot tokens, the candidate offsets and
    the candidate ids of each call.

    Raises:
        ValueError: A call has no readout, or a list lies outside the
            ``tokens`` column.
    """
    try:
        spans = np.asarray(
            [readouts[int(call)] for call in active], dtype=np.int64
        ).reshape(len(active), 3, 2)
    except KeyError as error:
        raise ValueError("a readout forward requires a readout") from error
    starts, counts = spans[:, :, 0].T, spans[:, :, 1].T
    if (
        np.any(starts < 0)
        or np.any(counts < 0)
        or np.any(starts + counts > tokens)
    ):
        raise ValueError("a readout's lists lie outside the token column")
    return starts, counts


def pack_readouts(
    calls: np.ndarray,
    forward: np.ndarray,
    tokens: np.ndarray,
    active: np.ndarray,
    readouts: Mapping[int, ReadoutSpans],
    *,
    vocab: int,
    capacity: int,
) -> ReadoutPass:
    """Pack a batch's active readouts into one forward.

    Args:
        calls: The batch's ``int64 [CALL_FIELDS, C]`` call columns.
        forward: The batch's ``int64 [FORWARD_FIELDS + 2G, R]`` forward
            columns.
        tokens: The batch's flat ``int64`` token column: the calls' input
            tokens and the readouts' lists.
        active: The active calls, ascending, every one a readout.
        readouts: Each call's readout lists (:func:`readout_spans`).
        vocab: The model's vocabulary entries.
        capacity: The tokens one forward holds, which bounds its slots.

    Raises:
        ValueError: A call has no readout, its rows do not read its
            prefix read-only or do not partition its tokens, a row reads no
            slot, a slot lies outside the call's rows or out of order, the
            candidate offsets do not partition the candidates, a candidate
            is outside the vocabulary, or the pass reads no slot or more
            than ``capacity``.
    """
    active = np.asarray(active, dtype=np.int64)
    n = len(active)
    if n == 0:
        raise ValueError("a readout forward reads no call")
    rows, owner = _read_only(calls, forward, active)
    query = forward[ForwardField.QUERY_LEN, rows]

    # The rows of each call partition its input tokens, so the packed rows
    # partition the packed tokens.
    token_start = calls[CallField.TOKEN_START, active]
    token_count = calls[CallField.TOKEN_COUNT, active]
    if np.any(token_start < 0) or np.any(
        token_start + token_count > len(tokens)
    ):
        raise ValueError("a call's tokens lie outside the token column")
    row_count = calls[CallField.ROW_COUNT, active]
    first_row = np.cumsum(row_count) - row_count
    if np.any(np.add.reduceat(query, first_row) != token_count):
        raise ValueError(
            "block rows must partition the call's tokens over its prefix"
        )
    visible = calls[CallField.VISIBLE, active]
    _check_prefix(visible)
    packed = tokens[_ranges(token_start, token_count)]
    token_base = np.cumsum(token_count) - token_count

    # Slots address each call's own tokens in increasing order, and every
    # row reads at least one.
    starts, counts = _spans(active, readouts, len(tokens))
    slot_count, offset_count, candidate_count = counts
    slot_tokens = tokens[_ranges(starts[0], slot_count)]
    slot_owner = np.repeat(np.arange(n), slot_count)
    if np.any(slot_tokens < 0) or np.any(
        slot_tokens >= token_count[slot_owner]
    ):
        raise ValueError("a readout slot lies outside its block rows")
    same = slot_owner[1:] == slot_owner[:-1]
    if np.any(np.diff(slot_tokens)[same] <= 0):
        raise ValueError("readout slots are not in the order of their tokens")
    slots = token_base[slot_owner] + slot_tokens
    row_of = np.searchsorted(np.cumsum(query), slots, side="right")
    if np.any(np.bincount(row_of, minlength=len(rows)) == 0):
        raise ValueError("every block row must read a slot")
    if len(slots) == 0 or len(slots) > capacity:
        raise ValueError(
            "a readout reads between one slot and its token capacity"
        )

    # Each call's offsets start at zero, never decrease and end at its
    # candidate count; the pass's offsets continue across calls.
    if np.any(offset_count != slot_count + 1):
        raise ValueError("readout candidate offsets do not bound every slot")
    offsets = tokens[_ranges(starts[1], offset_count)]
    first_offset = np.cumsum(offset_count) - offset_count
    last_offset = first_offset + offset_count - 1
    offset_owner = np.repeat(np.arange(n), offset_count)
    within = offset_owner[1:] == offset_owner[:-1]
    if (
        np.any(offsets[first_offset] != 0)
        or np.any(offsets[last_offset] != candidate_count)
        or np.any(np.diff(offsets)[within] < 0)
    ):
        raise ValueError(
            "readout candidate offsets do not partition the candidates"
        )
    candidate_base = np.cumsum(candidate_count) - candidate_count
    later = np.ones(len(offsets), dtype=bool)
    later[first_offset] = False
    slot_candidates = np.concatenate(
        ([0], candidate_base[offset_owner[later]] + offsets[later])
    ).astype(np.int64)
    candidates = tokens[_ranges(starts[2], candidate_count)]
    if np.any((candidates < 0) | (candidates >= vocab)):
        raise ValueError(
            f"a readout candidate is outside the {vocab}-token vocabulary"
        )

    # A readout launch reads the log-probabilities of one slot chunk; a
    # candidate's row is its slot's place in the chunk.
    candidate_slot = np.repeat(
        np.arange(len(slots), dtype=np.int64), np.diff(slot_candidates)
    )
    index = (candidate_slot % SLOT_BUCKETS[-1]) * vocab + candidates

    return ReadoutPass(
        calls=active,
        rows=rows,
        slot=calls[CallField.SLOT, active][owner],
        query=query,
        prefix=visible[owner],
        position=calls[CallField.LOGICAL, active][owner],
        tokens=packed,
        slots=slots,
        slot_candidates=slot_candidates,
        candidates=candidates,
        index=index,
        candidate_start=candidate_base,
        candidate_count=candidate_count,
    )


def seed_bits(seed: int) -> int:
    """Return the int64 column value of a u64 sampling seed: its
    two's-complement bits.

    Raises:
        ValueError: The seed is not a 64-bit unsigned integer.
    """  # noqa: D205
    if not 0 <= seed < 1 << 64:
        raise ValueError("a sampling seed is a 64-bit unsigned integer")
    return seed - (1 << 64) if seed >= 1 << 63 else seed


class Seeds:
    """The request and sampling seed each request slot of a lane holds.

    A request slot ``s`` in ``1..=slots`` holds one started request; slot 0
    is the padding slot, so every array has ``slots + 1`` rows.

    Attributes:
        request_id: ``int64 [S]`` the slot's request identifier.
        epoch: ``int64 [S]`` the slot's admission epoch.
        seeded: ``bool [S]`` the slot's request generates with a seed.
        seed: ``int64 [S]`` the seed's two's-complement bits
            (:func:`seed_bits`), 0 when unseeded.
    """

    def __init__(self, slots: int) -> None:
        if slots <= 0:
            raise ValueError("a lane serves at least one request slot")
        rows = slots + 1
        self.request_id = np.zeros(rows, dtype=np.int64)
        self.epoch = np.zeros(rows, dtype=np.int64)
        self.seeded = np.zeros(rows, dtype=bool)
        self.seed = np.zeros(rows, dtype=np.int64)

    @property
    def slots(self) -> int:
        """Return the request slots, the padding slot excluded."""
        return len(self.seeded) - 1

    def start(
        self, slot: int, request_id: int, epoch: int, seed: int | None
    ) -> None:
        """Record a started request in its slot.

        Args:
            slot: The request's slot.
            request_id: The request key's identifier.
            epoch: The request key's admission epoch.
            seed: The request's sampling seed; ``None`` for a request that
                generates no block or is unseeded.

        Raises:
            ValueError: The slot is the padding slot or outside the lane, or
                the seed is not a 64-bit unsigned integer.
        """
        if not 0 < slot <= self.slots:
            raise ValueError("request slot is outside the lane")
        bits = 0 if seed is None else seed_bits(seed)
        self.request_id[slot] = request_id
        self.epoch[slot] = epoch
        self.seeded[slot] = seed is not None
        self.seed[slot] = bits

    def release(self, slots: Sequence[int] | np.ndarray) -> None:
        """Clear ``slots``."""
        index = np.asarray(slots, dtype=np.int64)
        self.request_id[index] = 0
        self.epoch[index] = 0
        self.seeded[index] = False
        self.seed[index] = 0


@dataclass(frozen=True, slots=True)
class Steps:
    """One denoising step of resident blocks: one row per active call.

    Attributes:
        calls: ``[n]`` the active calls, ascending.
        rows: ``[n]`` each call's index into the forward columns.
        slot: ``[n]`` the request slot whose resident block each row steps.
        seed: ``[n]`` the request's seed bits.
        block: ``[n]`` the blocks committed before the row's block.
        step: ``[n]`` the denoising steps the block already ran.
        prefix: ``[n]`` the cached prefix tokens each row reads.
        position: ``[n]`` the position of each block's first token.
    """

    calls: np.ndarray
    rows: np.ndarray
    slot: np.ndarray
    seed: np.ndarray
    block: np.ndarray
    step: np.ndarray
    prefix: np.ndarray
    position: np.ndarray

    @property
    def first(self) -> bool:
        """Whether every row is at step zero (no self-conditioning)."""
        return bool(np.all(self.step == 0))

    def columns(self) -> np.ndarray:
        """Return ``int64 [4, n]``: the denoising columns' live rows.

        Slot, seed bits, block and step of each row. The lane zeroes the
        rows that pad a step to its bucket: blocks of the sentinel slot,
        whose commit is skipped.
        """
        return np.stack((self.slot, self.seed, self.block, self.step))


def pack_steps(
    calls: np.ndarray,
    forward: np.ndarray,
    active: np.ndarray,
    seeds: Seeds,
    sampling: ServedSampling | None,
) -> Steps:
    """Validate a batch's active denoising steps and return their rows.

    Args:
        calls: The batch's ``int64 [CALL_FIELDS, C]`` call columns.
        forward: The batch's ``int64 [FORWARD_FIELDS + 2G, R]`` forward
            columns.
        active: The active calls, ascending, every one a step.
        seeds: The requests the lane's slots hold.
        sampling: The sampling the lane steps blocks with; ``None`` on a
            rank that serves readouts only.

    Raises:
        ValueError: The batch has no step, a step has no canvas
            coordinates, names a request its slot does not hold or an
            unseeded one, the rank generates no block, its rows are not one
            read-only block row of the canvas length over its prefix, or its
            step reaches the step limit.
    """
    active = np.asarray(active, dtype=np.int64)
    if len(active) == 0:
        raise ValueError("a denoising step carries no row")
    block = calls[CallField.CANVAS_BLOCK, active]
    step = calls[CallField.CANVAS_STEP, active]
    if np.any(block < 0) or np.any(step < 0):
        raise ValueError("a block step requires its coordinates")

    slot = calls[CallField.SLOT, active]
    if np.any(slot <= 0) or np.any(slot > seeds.slots):
        raise ValueError("request slot is outside the lane")
    if np.any(
        seeds.request_id[slot] != calls[CallField.REQUEST_ID, active]
    ) or np.any(seeds.epoch[slot] != calls[CallField.EPOCH, active]):
        raise ValueError("a call names a request its slot does not hold")
    if not np.all(seeds.seeded[slot]):
        raise ValueError("a block step requires a seeded request")
    if sampling is None:
        raise ValueError("this worker keeps no generating blocks")

    rows, _ = _read_only(calls, forward, active)
    length = sampling.canvas_length
    if (
        len(rows) != len(active)
        or np.any(calls[CallField.MAX_TOKENS, active] != length)
        or np.any(forward[ForwardField.QUERY_LEN, rows] != length)
    ):
        raise ValueError(
            "a block step is one read-only block row over its prefix"
        )
    if np.any(step >= sampling.max_steps):
        raise ValueError("the block step exceeds its step limit")
    visible = calls[CallField.VISIBLE, active]
    _check_prefix(visible)

    return Steps(
        calls=active,
        rows=rows,
        slot=slot,
        seed=seeds.seed[slot],
        block=block,
        step=step,
        prefix=visible,
        position=calls[CallField.LOGICAL, active],
    )


def _progress(
    writer: RecordWriter, calls: np.ndarray, active: np.ndarray
) -> None:
    # A token-denoising call leaves its request's progress as it was, with
    # the computed cache length at the visible one, and echoes the prompt
    # logits readiness, which the worker installs as reported.
    visible = calls[CallField.VISIBLE, active]
    writer.ran(active)
    writer.progress(
        active,
        logical=calls[CallField.LOGICAL, active],
        rng=calls[CallField.RNG, active],
        visible=visible,
        computed=visible,
        prompt_ready=calls[CallField.PROMPT_READY, active] != 0,
    )


def _require_words(end: int, lent: int) -> None:
    if end > lent:
        raise AssertionError("a batch's output words exceed the words lent")


def record_readouts(
    writer: RecordWriter,
    calls: np.ndarray,
    readout: ReadoutPass,
    *,
    lent: int,
) -> int:
    """Fill the records of a readout batch's active calls.

    Candidate ``i`` of the pass is output word ``i``: each call's candidate
    values are the words from its first candidate, one per candidate, and a
    call without candidates names an empty range there.

    Args:
        writer: The batch's record writer.
        calls: The batch's call columns.
        readout: The batch's packed readouts.
        lent: The output words lent to the batch.

    Returns:
        The output words the batch writes.

    Raises:
        AssertionError: The candidate values exceed the lent words; the
            caller records before any device work.
    """
    end = len(readout.candidates)
    _require_words(end, lent)
    _progress(writer, calls, readout.calls)
    writer.ranges(
        readout.calls, readout.candidate_start, readout.candidate_count
    )
    return end


def record_steps(
    writer: RecordWriter,
    calls: np.ndarray,
    steps: Steps,
    *,
    length: int,
    lent: int,
) -> int:
    """Fill the records of a step batch's active calls.

    Row ``i``'s result, its outcome and its block of ``length`` tokens, is
    the ``1 + length`` words from ``i * (1 + length)``.

    Args:
        writer: The batch's record writer.
        calls: The batch's call columns.
        steps: The batch's denoising steps.
        length: Tokens of one block.
        lent: The output words lent to the batch.

    Returns:
        The output words the batch writes.

    Raises:
        AssertionError: The results exceed the lent words; the caller
            records before any device work.
    """
    width = 1 + length
    live = len(steps.calls)
    end = live * width
    _require_words(end, lent)
    _progress(writer, calls, steps.calls)
    writer.ranges(
        steps.calls,
        width * np.arange(live, dtype=np.int64),
        np.full(live, width, dtype=np.int64),
        canvas=True,
    )
    return end
