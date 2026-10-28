"""Text graph shapes: the buckets startup captures and serving uses.

A captured text graph holds a fixed numerical shape. Decode graphs hold a
row bucket of one-token rows; prefill graphs hold a row and a token bucket of
one numerical variant (:class:`PrefillVariant`), with outputs or writing the
prefix cache only (:class:`PrefillShape`). :class:`TextShapes` selects the
smallest captured bucket that holds a forward (:class:`TextForward`).

Every bucket has room for padding. A decode bucket pads with inert
one-token rows of slot 0. A prefill bucket adds one sequence holding every
padding token, then empty sequences, so a row bucket of ``n`` rows holds
forwards of fewer than ``n`` rows (:func:`prefill_row_buckets`).

The catalogs hold every batch the worker admits, so a computation with
graphs refuses a forward no captured bucket holds; graph selection depends
only on call metadata, so every rank of a group selects the same graph.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import ClassVar, NamedTuple, TypeVar

import numpy as np

__all__ = [
    "PrefillShape",
    "PrefillVariant",
    "Selected",
    "TextForward",
    "TextShapes",
    "bucket",
    "capture_order",
    "prefill_row_buckets",
    "prefill_shapes",
    "prefill_units",
    "task_buckets",
]

T = TypeVar("T")

# A cache group's pages as startup sizes them: (tokens per page, pool units
# per page).
Pages = Sequence[tuple[int, int]]


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def bucket(sizes: Iterable[int], count: int) -> int | None:
    """Return the smallest configured size holding ``count``, if any.

    ``sizes`` need not be sorted.
    """
    return min((size for size in sizes if size >= count), default=None)


def capture_order(
    shapes: Iterable[T], footprint: Callable[[T], tuple[int, int]]
) -> list[T]:
    """Order shapes for capture: largest ``footprint`` first.

    Duplicates are removed and equal footprints keep their given order, so
    pool blocks a large capture frees are reused by the smaller ones that
    follow.
    """
    ordered: list[T] = []
    for shape in shapes:
        if shape not in ordered:
            ordered.append(shape)
    # Python's sort is stable: equal footprints keep the configured order.
    ordered.sort(key=footprint, reverse=True)
    return ordered


def task_buckets(rows: int) -> list[int]:
    """Return the task buckets of selections over a binding of ``rows`` rows.

    Powers of two below ``rows``, then ``rows`` itself: a bucket at most
    doubles its tasks.
    """
    buckets = []
    size = 1
    while size < rows:
        buckets.append(size)
        size *= 2
    buckets.append(rows)
    return buckets


def prefill_row_buckets(max_rows: int) -> list[int]:
    """Return the prefill row buckets for at most ``max_rows`` rows a batch.

    Powers of two from 8, then ``max_rows + 1``, so that every batch the
    engine forms fits one: a bucket of ``n`` rows holds batches of fewer
    rows, the remaining row being an inert padding sequence. The last power
    of two below ``max_rows + 1`` merges into it rather than leaving a
    bucket that holds a single row count.
    """
    last = max_rows + 1
    buckets = []
    size = 8
    while 2 * size <= last:
        buckets.append(size)
        size *= 2
    buckets.append(last)
    return buckets


@dataclass(frozen=True, slots=True)
class PrefillVariant:
    """The numerical semantics a prefill graph is captured for.

    Attributes:
        causal: The rows' shared causality; ``None`` reads each row's
            causality from the device column, so that the graph serves rows
            that differ.
        embeddings: Vision features replace the embeddings of the tokens the
            embedding mask marks.
        axes: Rotary position axes the rows carry: 1, or 3 for prompts
            placing vision blocks at spatial positions.
    """

    causal: bool | None
    embeddings: bool
    axes: int

    # Causal text at one position axis.
    TEXT: ClassVar[PrefillVariant]

    @classmethod
    def media(cls, axes: int) -> PrefillVariant:
        """Return the variant of prompts placing vision blocks on ``axes``.

        Rows of either causality, features replacing the placeholder
        embeddings.
        """
        return cls(causal=None, embeddings=True, axes=axes)


PrefillVariant.TEXT = PrefillVariant(causal=True, embeddings=False, axes=1)


@dataclass(frozen=True, slots=True)
class PrefillShape:
    """One prefill capture, including its padding row and attention mode.

    Attributes:
        token_bucket: Flat tokens the captured numerical input holds.
        row_bucket: Rows the captured input holds, one of them padding.
        live_rows: The least rows the bucket serves: one, or the preceding
            row bucket. Startup sizes the capture's scratch prefix cache by
            it.
        variant: The numerical variant.
        outputs: False captures cache writes alone, without hidden outputs.
    """

    token_bucket: int
    row_bucket: int
    live_rows: int
    variant: PrefillVariant
    outputs: bool

    def scratch_lengths(
        self, row_tokens: int, pages: Pages, units: int | None
    ) -> list[int]:
        """Fit the capture's rows into a scratch cache pool.

        Returns the fullest numerical batch, one length per row, whose whole
        pages fit every cache group within ``units`` pool units and whose
        rows stay within a row's token limit; tokens that do not fit become
        graph padding.

        Args:
            row_tokens: The most tokens one row (request) holds.
            pages: Each cache group's ``(tokens per page, units per page)``.
                Block sizes are positive powers of two.
            units: The scratch pool's units; ``None`` lays out the full
                capture before capacity planning.
        """
        row_units = sum(count for _, count in pages)
        rows = max(
            min(
                max(self.live_rows, _ceil_div(self.token_bucket, row_tokens)),
                max(self.row_bucket - 1, 0),
                self.row_bucket if units is None else units // row_units,
            ),
            1,
        )
        fitted = [1] * rows
        low = rows
        high = min(self.token_bucket, rows * row_tokens)

        # One token in every row fits. Search for the fullest numerical
        # batch whose whole pages fit all groups, leaving only graph padding
        # unused.
        while low < high:
            tokens = low + _ceil_div(high - low, 2)
            lengths = _spread(rows, tokens, row_tokens, pages)
            if lengths is not None and (
                units is None or _pool_units(lengths, pages) <= units
            ):
                low = tokens
                fitted = lengths
            else:
                high = tokens - 1
        return fitted


def _pool_units(lengths: Sequence[int], pages: Pages) -> int:
    # The pool units rows of ``lengths`` tokens take in whole pages.
    return sum(
        count * _ceil_div(length, page)
        for length in lengths
        for page, count in pages
    )


def _spread(
    rows: int, tokens: int, row_tokens: int, pages: Pages
) -> list[int] | None:
    # Distribute tokens over rows using the fewest whole pages in every
    # cache group. Overfilled groups (more pages than rows) have smaller
    # pages, and their largest page divides the others: rows take whole
    # such pages, the final row the remainder. No row may exceed the
    # smallest page of the other groups or the row limit.
    step = max(
        (page for page, _ in pages if _ceil_div(tokens, page) > rows),
        default=1,
    )
    cap = min(
        min(
            (page for page, _ in pages if _ceil_div(tokens, page) <= rows),
            default=row_tokens,
        ),
        row_tokens,
    )
    if cap < step:
        return None

    steps = tokens // step
    base, extra = divmod(steps, rows)
    lengths = [
        (base + int(index < extra)) * step
        + (tokens % step if index + 1 == rows else 0)
        for index in range(rows)
    ]
    if all(0 < length <= cap for length in lengths):
        return lengths
    return None


def prefill_units(pages: Pages, rows: int, tokens: int) -> int:
    """Return the fewest pool units ``rows`` rows of ``tokens`` tokens hold.

    Each independent row holds at least one page of every cache group;
    ``pages`` holds each group's ``(tokens per page, units per page)``.
    """
    return sum(
        units * max(rows, _ceil_div(tokens, page)) for page, units in pages
    )


def prefill_shapes(
    token_sizes: Iterable[int],
    row_sizes: Iterable[int],
    max_rows: int,
    max_tokens: int,
    variants: Sequence[PrefillVariant],
    outputs: bool,
    pool: tuple[Pages, int] | None = None,
) -> list[PrefillShape]:
    """Build the configured prefill captures that can serve an admitted batch.

    Token buckets are the configured sizes within ``max_tokens``, then
    ``max_tokens``. Row buckets above one are taken in increasing order. A
    row bucket needs an extra sequence for padding: it serves from one row
    (the first bucket) or one past the preceding bucket. Its first token
    bucket serves one token per live row; later token buckets begin above
    the previous size. With a scratch ``pool`` of ``(pages, units)``, a
    token bucket whose least batch cannot fit the pool is not captured, nor
    are the larger ones of its row bucket. Row buckets serving more than
    ``max_rows`` rows are not captured.

    Returns:
        One shape per variant of every captured ``(row, token)`` bucket.
    """
    tokens = sorted(
        {size for size in token_sizes if size <= max_tokens} | {max_tokens}
    )
    rows = sorted({size for size in row_sizes if size > 1})
    shapes: list[PrefillShape] = []
    live_rows = 1
    for row_bucket in rows:
        if live_rows > max_rows:
            break
        minimum = 1 if live_rows == 1 else live_rows + 1
        least = live_rows
        for token_bucket in (size for size in tokens if size >= minimum):
            if pool is not None:
                pages, units = pool
                if prefill_units(pages, live_rows, least) > units:
                    break
            shapes.extend(
                PrefillShape(
                    token_bucket=token_bucket,
                    row_bucket=row_bucket,
                    live_rows=live_rows,
                    variant=variant,
                    outputs=outputs,
                )
                for variant in variants
            )
            least = token_bucket + 1
        live_rows = row_bucket
    return shapes


@dataclass(frozen=True, slots=True)
class TextForward:
    """The forward a text batch asks a graph for.

    Attributes:
        queries: Query tokens of each row.
        tokens: Query tokens of the batch.
        decode: The batch is a decode step.
        variant: The numerical variant the rows need.
        last_logits: Only each row's last logits are computed.
        cache_only: The forward only writes the prefix cache.
    """

    queries: Sequence[int] | np.ndarray
    tokens: int
    decode: bool
    variant: PrefillVariant
    last_logits: bool
    cache_only: bool


class Selected(NamedTuple):
    """A captured bucket holding a forward.

    Attributes:
        rows: The bucket's rows.
        tokens: The bucket's tokens (its rows for a decode bucket).
        decode: The bucket is a decode graph.
        outputs: The prefill graph computes hidden outputs (true for a
            decode graph).
    """

    rows: int
    tokens: int
    decode: bool
    outputs: bool


class TextShapes:
    """Immutable capture sizes of one text computation.

    Args:
        decode: The decode row buckets, in configured order.
        prefill: The prefill captures.
    """

    def __init__(
        self, decode: Sequence[int], prefill: Sequence[PrefillShape]
    ) -> None:
        self.decode = tuple(decode)
        self.prefill = tuple(prefill)
        self._cache_only = any(not shape.outputs for shape in self.prefill)
        self._outputs = any(shape.outputs for shape in self.prefill)

    def select(self, forward: TextForward) -> Selected | None:
        """Return the smallest captured bucket holding ``forward``, if any.

        A decode of one-token rows takes the first configured decode size
        that holds its rows, in configured order, and never a prefill
        bucket; it needs the text variant and last logits. A prefill takes
        the least ``(row bucket, token bucket)`` of its variant with a
        padding row to spare and room for its tokens. A cache-only prefill
        uses a graph computing outputs only when no cache-only family was
        captured.

        Raises:
            ValueError: The forward has no row.
        """
        queries = np.asarray(forward.queries, dtype=np.int64)
        rows = len(queries)
        if rows == 0:
            raise ValueError("text graphs require a nonempty query batch")
        if forward.decode and np.all(queries == 1):
            if (
                forward.variant != PrefillVariant.TEXT
                or not forward.last_logits
            ):
                return None
            size = next((size for size in self.decode if size >= rows), None)
            return None if size is None else Selected(size, size, True, True)

        outputs = not (forward.cache_only and self._cache_only)
        if (outputs and not self._outputs) or np.any(queries == 0):
            return None
        held = [
            shape
            for shape in self.prefill
            if shape.outputs == outputs
            and shape.variant == forward.variant
            and shape.row_bucket > rows
            and shape.token_bucket >= forward.tokens
        ]
        if not held:
            return None
        shape = min(held, key=lambda held: (held.row_bucket, held.token_bucket))
        return Selected(shape.row_bucket, shape.token_bucket, False, outputs)
