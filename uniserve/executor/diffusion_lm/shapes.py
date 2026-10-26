"""Graph shapes of a diffusion LM lane: row, token and slot buckets and
their padding.

An ``attend`` graph holds ``tokens`` tokens in up to ``2 * rows`` sequences
of at most a block's tokens each: the live blocks, then padding sequences of
a block's tokens that make up the tokens the live blocks leave (the last one
shorter), then empty sequences (:func:`attend_padding`). Padding sequences
read no prefix. A row bucket's token buckets are its rows times each readout
length bucket, up to the batch's token bound, so a pass of short blocks pads
few tokens, and every pass the engine forms (at most its rows and its token
budget, each block at most the model's) fits a bucket.

A ``denoise`` graph holds ``rows`` full blocks, one sequence each; a padding
row starts a block in the sentinel slot zero. A ``finish`` graph completes
the final layer at a bucket of slot rows, once per forward. A ``readout``
graph projects up to :data:`SLOT_BUCKETS`' largest finished rows and reads
up to :data:`CANDIDATES` of their candidates; a forward with more slots
launches one per slot chunk (:func:`slot_chunks`), and a chunk with more
candidates one per candidate window.

Graph selection depends only on call metadata, so every rank of a group
selects the same graph.
"""  # noqa: D205

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from uniserve.executor.text.shapes import bucket

__all__ = [
    "CANDIDATES",
    "ROW_BUCKETS",
    "SLOT_BUCKETS",
    "AttendKey",
    "DenoiseKey",
    "FinishKey",
    "ReadoutKey",
    "Shapes",
    "SlotChunk",
    "attend_padding",
    "chunk_rows",
    "slot_chunks",
]

# Row buckets. After four rows, each step adds at most half the previous
# count; the capacity itself is added too, so padding never exceeds one
# third of a bucket's rows.
ROW_BUCKETS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)

# Slot rows of the ``readout`` graphs; forwards are chunked at the largest.
SLOT_BUCKETS = (64, 128, 256)

# Candidates one ``readout`` graph selects: the size of the candidate
# columns. A slot chunk with more candidates reads them in windows of this
# many.
CANDIDATES = 1 << 16

# The shortest readout length bucket; buckets double up to the block.
_SHORTEST_LENGTH = 16

# The bytes of transient sampler logits and weights one step chunk may
# hold.
_STEP_CHUNK_BYTES = 512 << 20

# Transient sampler bytes per vocabulary entry and block position: FP32
# logits and BF16 weights.
_ENTRY_BYTES = 4 + 2


@dataclass(frozen=True, slots=True)
class AttendKey:
    """``attend`` over up to ``2 * rows`` sequences holding ``tokens``."""

    rows: int
    tokens: int


@dataclass(frozen=True, slots=True)
class DenoiseKey:
    """``denoise`` of ``rows`` blocks.

    ``first`` when every row is at step zero, which skips the
    self-conditioning signal.
    """

    rows: int
    first: bool


@dataclass(frozen=True, slots=True)
class FinishKey:
    """``finish`` of ``slots`` slot rows."""

    slots: int


@dataclass(frozen=True, slots=True)
class ReadoutKey:
    """``readout`` of ``slots`` finished rows."""

    slots: int


def chunk_rows(sequence_length: int, vocab_size: int, max_rows: int) -> int:
    """Return the blocks a sampler chunk projects and steps at once.

    As many as keep the chunk's transient logits and weights within
    :data:`_STEP_CHUNK_BYTES`, at least one, at most ``max_rows``.
    """
    per_row = sequence_length * vocab_size * _ENTRY_BYTES
    return max(min(_STEP_CHUNK_BYTES // per_row, max_rows), 1)


def _buckets_to(catalog: Sequence[int], limit: int) -> list[int]:
    """Return the buckets of ``catalog`` below ``limit``, then ``limit``."""
    return [size for size in catalog if size < limit] + [limit]


class Shapes:
    """The graph buckets of one diffusion LM lane.

    Args:
        sequence_length: Tokens of one block (``C``).
        capacity: Blocks one denoising step may hold.
        readout_rows: Blocks one readout may hold.
        tokens: Tokens one readout may hold.
        slots: The request slots of a generating lane, which never steps
            more blocks than it has slots; ``None`` on a lane that serves
            readouts only.

    Raises:
        ValueError: A block length, capacity, readout bound or token bound
            is not positive.
    """

    def __init__(
        self,
        sequence_length: int,
        capacity: int,
        readout_rows: int,
        tokens: int,
        slots: int | None = None,
    ) -> None:
        if min(sequence_length, capacity, readout_rows, tokens) <= 0:
            raise ValueError(
                "diffusion LM graphs need a positive block length, capacity "
                "and token bound"
            )
        limit = capacity if slots is None else min(capacity, slots)
        self.sequence_length = sequence_length
        self._capacity = limit
        self._rows = tuple(_buckets_to(ROW_BUCKETS, limit))

        # Readout length buckets: powers of two from the shortest, then the
        # block itself.
        lengths = []
        length = _SHORTEST_LENGTH
        while length < sequence_length:
            lengths.append(length)
            length *= 2
        lengths.append(sequence_length)

        # A row bucket holds its rows times each length bucket, within the
        # token bound; when its full blocks exceed the bound, the bound
        # itself is its largest bucket.
        attend = []
        for rows in _buckets_to(ROW_BUCKETS, min(readout_rows, tokens)):
            sizes = [rows * length for length in lengths]
            sizes = [size for size in sizes if size <= tokens]
            if rows * sequence_length > tokens and sizes[-1:] != [tokens]:
                sizes.append(tokens)
            attend.append((rows, tuple(sizes)))
        self._attend = tuple(attend)

        # A forward's slots are some of its tokens: slot buckets double from
        # the smallest readout bucket up to the token bound.
        finish = []
        size = SLOT_BUCKETS[0]
        while size < tokens:
            finish.append(size)
            size *= 2
        finish.append(tokens)
        self._finish = tuple(finish)

    def capacity(self) -> int:
        """Return the blocks one denoising step may hold."""
        return self._capacity

    def rows(self) -> tuple[int, ...]:
        """Return the ``denoise`` row buckets, ascending."""
        return self._rows

    def attend_buckets(self) -> tuple[tuple[int, int], ...]:
        """Return the ``attend`` buckets ``(rows, tokens)``, ascending."""
        return tuple(
            (rows, tokens) for rows, sizes in self._attend for tokens in sizes
        )

    def finish_buckets(self) -> tuple[int, ...]:
        """Return the ``finish`` slot buckets, ascending."""
        return self._finish

    def attend(self, rows: int, tokens: int, longest: int) -> tuple[int, int]:
        """Return the ``attend`` bucket ``(rows, tokens)`` of a readout.

        The readout holds ``rows`` blocks of ``tokens`` tokens, the longest
        ``longest``. Its bucket is the smallest row bucket holding the rows,
        then that bucket's smallest token bucket holding the tokens.

        Raises:
            ValueError: A block is longer than the model's, or the pass
                exceeds the readout bounds the buckets were built for.
        """
        if longest > self.sequence_length:
            raise ValueError(
                f"a block of {longest} tokens exceeds the model's "
                f"{self.sequence_length} tokens"
            )
        for bucket_rows, sizes in self._attend:
            if bucket_rows >= rows:
                size = bucket(sizes, tokens)
                if size is not None:
                    return bucket_rows, size
                break
        raise ValueError(
            f"no captured readout bucket holds {rows} blocks of {tokens} tokens"
        )

    def denoise(self, rows: int) -> int:
        """Return the ``denoise`` bucket of ``rows`` blocks.

        Raises:
            ValueError: A step holds more blocks than the capacity.
        """
        size = bucket(self._rows, rows)
        if size is None:
            raise ValueError(
                f"a denoising step of {rows} blocks exceeds the "
                f"{self._capacity} a step holds"
            )
        return size

    def finish(self, slots: int) -> int:
        """Return the ``finish`` bucket of ``slots`` slot rows.

        Raises:
            ValueError: A readout reads more slots than a forward holds.
        """
        size = bucket(self._finish, slots)
        if size is None:
            raise ValueError(
                f"a readout of {slots} slots exceeds the "
                f"{self._finish[-1]} a forward holds"
            )
        return size


def attend_padding(
    live: int, tokens: int, rows: int, total: int, length: int
) -> list[int]:
    """Return the query tokens of a readout's padding sequences.

    The padding widens a readout of ``live`` blocks holding ``tokens``
    tokens to a bucket of ``rows`` sequences and ``total`` tokens: sequences
    of ``length`` tokens, then the remainder, then empty sequences up to
    ``2 * rows`` sequences in all.

    Raises:
        AssertionError: The readout exceeds its bucket, or the bucket holds
            more tokens than ``rows`` blocks of ``length``; the bucket was
            not selected by :meth:`Shapes.attend`.
    """
    if live > rows or tokens > total or total > rows * length:
        raise AssertionError(
            f"a readout of {live} blocks and {tokens} tokens exceeds its "
            f"bucket of {rows} blocks and {total} tokens"
        )
    remaining = total - tokens
    padding = [length] * (remaining // length)
    if remaining % length:
        padding.append(remaining % length)
    # At most ``rows - tokens // length`` sequences carry tokens, and
    # ``2 * rows - live`` is at least ``rows``: empty sequences only extend.
    padding.extend([0] * (2 * rows - live - len(padding)))
    return padding


class SlotChunk(NamedTuple):
    """One ``readout`` launch's slots.

    Attributes:
        start: The chunk's first slot of the forward.
        live: The chunk's slots.
        bucket: The ``readout`` slot bucket holding them.
    """

    start: int
    live: int
    bucket: int


def slot_chunks(slots: int) -> list[SlotChunk]:
    """Return the slot chunks of a readout of ``slots`` slots.

    Chunks hold the largest slot bucket's slots each, the last the rest.
    """
    largest = SLOT_BUCKETS[-1]
    chunks = []
    for start in range(0, slots, largest):
        live = min(slots - start, largest)
        size = bucket(SLOT_BUCKETS, live) or largest
        chunks.append(SlotChunk(start, live, size))
    return chunks
