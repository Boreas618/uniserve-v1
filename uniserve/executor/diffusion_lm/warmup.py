"""Diffusion LM warmup: every forward shape serving can select, over scratch
prefix caches, in one phase per lane.

Scratch tables belong to the worker, and the executor never calls back into
it, so a lane's warmup is a phase the worker drives (:class:`Phase`): it
installs scratch tables of one token for slots ``1..n``, ``n`` the blocks a
denoising step holds (one page of every cache group each), selects the
attention pages of each of the phase's synthetic forwards (rows ``[slot,
prefix, query]``; row ``r`` of a forward reads the table of slot ``r % n +
1`` over a one-token prefix, as a serving readout's rows of one request
share its slot) and calls the phase with each forward's page columns. The
layout (:func:`layout`) depends only on configuration, pool facts and model
facts, so every rank of a group reaches the same collectives in the same
order.

With a graph pool, every graph serving can select is captured, largest
first, each right after its method ran eagerly:

1. every slot bucket's ``readout`` alone (:class:`Alone`);
2. on the final stage of a model whose final layer exchanges no tokens,
   every ``finish`` slot bucket alone;
3. every ``attend`` bucket in a readout (:class:`Readout`) of its blocks
   (:func:`spread`): with a local ``finish``, one slot per block, at every
   expert capacity holding the bucket; with a ``finish`` that exchanges
   tokens, and so completes the readout's expert step, once per ``finish``
   slot bucket its slots can select, at every capacity the step can agree
   on, which captures those ``finish`` buckets too
   (:func:`exchanging_readouts`);
4. on a generating lane, every ``denoise`` row bucket, starting blocks and
   stepping them (:class:`Denoise`, :func:`synthetic_steps`), at every
   expert capacity holding it.

Without a pool, a generating lane runs one eager step of every sampler
chunk shape: the first self-conditioning product of a shape selects its
algorithm and synchronizes the device, so serving never does. Denoising
steps run in slots ``1..``; their synthetic state is harmless because a
request's first step starts its block before reading any of it.
"""  # noqa: D205

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

from .calls import ReadoutPass, Steps
from .shapes import SLOT_BUCKETS, AttendKey, FinishKey, Shapes

if TYPE_CHECKING:
    from .lane import DiffusionLMLane

__all__ = [
    "Alone",
    "Capacities",
    "Denoise",
    "Phase",
    "Readout",
    "WarmStep",
    "exchanging_readouts",
    "layout",
    "phases",
    "scratch_slots",
    "spread",
    "synthetic_readout",
    "synthetic_steps",
]

# The expert capacities a forward of some tokens is captured at, largest
# first; ``[None]`` without an expert union.
Capacities = Callable[[int], Sequence[int | None]]

# The cached prefix every synthetic row reads: one token of its scratch
# table.
_PREFIX = 1


def scratch_slots(rows: int, installed: int) -> np.ndarray:
    """Return the slot whose scratch table each synthetic row reads.

    Row ``r`` reads slot ``r % installed + 1`` of the ``installed`` slots,
    as a serving readout's rows of one request share its slot.
    """
    return np.arange(rows, dtype=np.int64) % installed + 1


def _rows(queries: Sequence[int], installed: int) -> list[list[int]]:
    """Return the attention rows ``[slot, prefix, query]`` of ``queries``."""
    slots = scratch_slots(len(queries), installed)
    return [
        [int(slot), _PREFIX, int(query)]
        for slot, query in zip(slots, queries, strict=True)
    ]


@dataclass(frozen=True, slots=True)
class Alone:
    """The ``readout`` or ``finish`` graph of a slot bucket, captured alone.

    Attributes:
        method: ``"readout"`` or ``"finish"``.
        slots: The slot bucket.
    """

    method: str
    slots: int


@dataclass(frozen=True, slots=True)
class Readout:
    """A synthetic readout: blocks of ``queries`` tokens, one per row.

    Attributes:
        queries: Each block's tokens.
        slot_tokens: Each answer slot's index in the packed tokens.
        capacity: The expert capacity its step is forced to; ``None`` runs
            it at the smallest capacity holding it, or without an expert
            step.
    """

    queries: tuple[int, ...]
    slot_tokens: tuple[int, ...]
    capacity: int | None = None

    def attention(self, installed: int) -> list[list[int]]:
        """Return the readout's attention rows ``[slot, prefix, query]``."""
        return _rows(self.queries, installed)


@dataclass(frozen=True, slots=True)
class Denoise:
    """A synthetic denoising step of ``rows`` blocks.

    Attributes:
        rows: The blocks stepped.
        first: Every block starts at step zero.
        length: Tokens of one block.
        capacity: As :attr:`Readout.capacity`.
    """

    rows: int
    first: bool
    length: int
    capacity: int | None = None

    def attention(self, installed: int) -> list[list[int]]:
        """Return the step's attention rows ``[slot, prefix, query]``."""
        return _rows([self.length] * self.rows, installed)


WarmStep = Alone | Readout | Denoise


def spread(tokens: int, rows: int) -> list[int]:
    """Return ``tokens`` tokens over ``rows`` blocks, the first ``tokens %
    rows`` one token longer than the others.
    """  # noqa: D205
    each, longer = divmod(int(tokens), int(rows))
    return [each + int(row < longer) for row in range(rows)]


def synthetic_steps(count: int, first: bool) -> Steps:
    """Return denoising steps of slots ``1..=count`` over one-token
    prefixes, at step zero (``first``) or one.

    Row ``r`` steps slot ``r + 1`` with seed ``r + 1`` in block zero, its
    block's first token right after the prefix.
    """  # noqa: D205
    rows = np.arange(count, dtype=np.int64)
    prefix = np.full(count, _PREFIX, dtype=np.int64)
    return Steps(
        calls=rows,
        rows=rows,
        slot=rows + 1,
        seed=rows + 1,
        block=np.zeros(count, dtype=np.int64),
        step=np.full(count, 0 if first else 1, dtype=np.int64),
        prefix=prefix,
        position=prefix,
    )


def synthetic_readout(
    step: Readout, installed: int, *, vocab: int
) -> ReadoutPass:
    """Return a synthetic readout's pass: zero tokens over one-token
    prefixes, one candidate (token zero) per answer slot.

    Its rows read the scratch tables :func:`scratch_slots` names.
    """  # noqa: D205
    query = np.asarray(step.queries, dtype=np.int64)
    rows = len(query)
    slots = np.asarray(step.slot_tokens, dtype=np.int64)
    count = len(slots)
    # One candidate per slot, its flat index in the slot's chunk.
    within = np.arange(count, dtype=np.int64) % SLOT_BUCKETS[-1]
    return ReadoutPass(
        calls=np.zeros(0, dtype=np.int64),
        rows=np.arange(rows, dtype=np.int64),
        slot=scratch_slots(rows, installed),
        query=query,
        prefix=np.full(rows, _PREFIX, dtype=np.int64),
        position=np.full(rows, _PREFIX, dtype=np.int64),
        tokens=np.zeros(int(query.sum()), dtype=np.int64),
        slots=slots,
        slot_candidates=np.arange(count + 1, dtype=np.int64),
        candidates=np.zeros(count, dtype=np.int64),
        index=within * vocab,
        candidate_start=np.zeros(0, dtype=np.int64),
        candidate_count=np.zeros(0, dtype=np.int64),
    )


def exchanging_readouts(
    shapes: Shapes, capacities: Capacities
) -> list[Readout]:
    """Lay out the readouts that capture every ``attend`` bucket and every
    ``finish`` slot bucket of a model whose final layer exchanges tokens.

    Such a ``finish`` completes its readout's expert step, which sends
    ``max(tokens, finish(slots))`` tokens: the ``attend`` bucket's tokens
    and the ``finish`` bucket of its slots. A readout's slots are distinct
    positions among its tokens, so they select a ``finish`` bucket of at
    most ``finish(tokens)``, and a bucket ``(rows, tokens)`` reaches the
    extents ``max(tokens, f)`` of every ``finish`` bucket ``f`` up to
    ``finish(tokens)``. Each bucket is captured, with each such ``finish``
    bucket, at every capacity those extents can agree on; a readout whose
    ``attend`` and ``finish`` graphs at its capacity are both listed
    already is left out. Per bucket the ``finish`` buckets and capacities
    run largest first, and each readout reads the fewest slots of its
    ``finish`` bucket, at its first token positions.
    """  # noqa: D205
    finish = shapes.finish_buckets()
    listed: set[tuple[Hashable, int | None]] = set()
    readouts: list[Readout] = []
    for rows, tokens in reversed(shapes.attend_buckets()):
        queries = tuple(spread(tokens, rows))
        reached = shapes.finish(tokens)
        for index in reversed(range(len(finish))):
            bucket = finish[index]
            if bucket > reached:
                continue
            # The fewest slots selecting the bucket; at most ``tokens``,
            # because the bucket before it holds fewer than ``tokens``.
            slots = 1 if index == 0 else finish[index - 1] + 1
            for capacity in capacities(max(tokens, bucket)):
                graphs = {
                    (AttendKey(rows, tokens), capacity),
                    (FinishKey(bucket), capacity),
                }
                if graphs <= listed:
                    continue
                listed |= graphs
                readouts.append(Readout(queries, tuple(range(slots)), capacity))
    return readouts


def layout(
    shapes: Shapes,
    *,
    pooled: bool,
    generating: bool,
    local: bool,
    last: bool,
    chunk_rows: int,
    capacities: Capacities,
) -> list[WarmStep]:
    """Lay out a lane's warmup forwards in the order every rank runs them.

    Args:
        shapes: The lane's graph buckets.
        pooled: The lane captures graphs into a pool; otherwise it runs
            every forward eagerly.
        generating: The lane denoises resident blocks.
        local: ``finish`` exchanges no tokens across ranks.
        last: The rank holds the final pipeline stage.
        chunk_rows: Blocks a sampler chunk steps at once.
        capacities: The expert capacities a forward of some tokens is
            captured at, largest first.

    Returns:
        The forwards, in order; none for an eager lane that serves readouts
        only.
    """
    length = shapes.sequence_length
    if not pooled:
        if not generating:
            return []
        return [
            Denoise(rows, True, length) for rows in range(1, chunk_rows + 1)
        ]

    steps: list[WarmStep] = [
        Alone("readout", slots) for slots in reversed(SLOT_BUCKETS)
    ]
    if local and last:
        steps += [
            Alone("finish", slots)
            for slots in reversed(shapes.finish_buckets())
        ]

    if local:
        # Every ``attend`` bucket, filled by blocks of one slot each: the
        # slot reads its block's first token. The step sends the bucket's
        # tokens.
        for rows, tokens in reversed(shapes.attend_buckets()):
            queries = spread(tokens, rows)
            starts = np.cumsum([0, *queries[:-1]]).tolist()
            steps += [
                Readout(tuple(queries), tuple(starts), capacity)
                for capacity in capacities(tokens)
            ]
    else:
        steps += exchanging_readouts(shapes, capacities)

    if generating:
        for rows in reversed(shapes.rows()):
            padded = shapes.denoise(rows)
            for first in (True, False):
                steps += [
                    Denoise(rows, first, length, capacity)
                    for capacity in capacities(padded * length)
                ]
    return steps


@dataclass(slots=True)
class Phase:
    """One lane's warmup phase.

    Attributes:
        lane: The lane's index among the executor's lanes.
        lengths: The scratch tables' lengths of slots ``1..``.
        steps: The forwards, in order.
        run: Runs the phase given the page columns (``int64 [2 * G,
            rows]``) of each forward that attends, in order.
    """

    lane: int
    lengths: tuple[int, ...]
    steps: tuple[WarmStep, ...]
    run: Callable[[Sequence[np.ndarray]], None]

    def attention(self) -> list[list[list[int]]]:
        """Return each attending forward's rows ``[slot, prefix, query]``.

        A slot bucket's graph captured alone attends to no prefix cache.
        """
        installed = len(self.lengths)
        return [
            step.attention(installed)
            for step in self.steps
            if not isinstance(step, Alone)
        ]

    def describe(self) -> dict[str, Any]:
        """Return what the worker reads of the phase."""
        return {
            "lane": self.lane,
            "lengths": list(self.lengths),
            "steps": self.attention(),
        }


def phases(lanes: Sequence[DiffusionLMLane]) -> list[Phase]:
    """Lay out every lane's warmup phase in the order the worker runs them.

    A lane without warmup forwards has no phase.
    """
    laid = []
    for lane in lanes:
        steps = tuple(lane.warmup())
        if not steps:
            continue
        # One one-token scratch table per block a denoising step holds.
        installed = lane.shapes.capacity()
        laid.append(
            Phase(
                lane.index, (1,) * installed, steps, partial(lane.warm, steps)
            )
        )
    return laid
