"""Text warmup: graph capture over scratch prefix caches, in phases.

Scratch tables belong to the worker, and the executor never calls back into
it, so warmup runs in phases the executor lays out (:func:`phases`) and the
worker drives: per phase it installs scratch tables of the phase's lengths
for slots ``1..``, selects the attention pages of each of the phase's steps
(``[slot, prefix, query]`` rows), calls the phase with each step's page
columns and releases the tables. Then :func:`finish` captures what needs no
scratch table. The layout (:class:`Layout`) depends only on configuration,
pool facts and model facts, so every rank of a group reaches the same
collectives in the same order.

Per lane and runner (one per expert microbatch):

1. **Prefill buckets** (:func:`prefill_layouts`), largest footprint first
   (:func:`~.shapes.capture_order` by tokens times rows, then tokens),
   each over the fullest batch its scratch pool holds
   (:meth:`~.shapes.PrefillShape.scratch_lengths`) and at every expert
   capacity its forward can run at, largest first. The media variant's
   graphs read each row's causality and the embedding columns on the
   device, so text rows at explicit positions capture them.
2. **Decode buckets** (:func:`decode_layouts`), largest first, over tables
   of two tokens per slot: one real token per row is prefilled in batches
   no larger than the scheduler admits and committed as each slot's cache
   (capturing the bucket's ``commit`` graph); then, with every row's
   continuation flag set, the greedy and the sampling decode are captured
   at every capacity, over neutral parameter rows (the sampler's kernels do
   not depend on the values); the flags are cleared before the phase
   returns.
3. **Context** (:func:`context_layout`), image models only: the two
   compositions a vision block is written in (behind prompt text, and
   alone) run once over zero features, replaying the media variant's
   buckets under graphs and selecting and compiling their kernels
   otherwise.

:func:`finish` then captures, per runner with a graph pool, the ``commit``
graph of its rows where no decode bucket holds them, and its ``select``
graphs (per task bucket and set of samplers), ``score`` graphs (per bucket
of reporting tasks and top width a forward may report) and ``score_prompt``
graphs (per top width a prompt row may report), each alone over padding
inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from uniserve.sampling import state as slot_commit

from .commit import COMMIT_PADDING, CommitWork
from .context import (
    Injection,
    block_row,
    host_rows,
    next_position,
    placed_rows,
    text_row,
)
from .graphs import PromptKey, ScoreKey, SelectKey
from .lane import TextLane
from .protocol import CallKind
from .rows import Rows, TokenSelection
from .sampling import RowSampling, least_top, logprob_words
from .shapes import TextShapes, capture_order, task_buckets

__all__ = [
    "SAMPLERS",
    "Layout",
    "Phase",
    "Step",
    "context_layout",
    "decode_layouts",
    "finish",
    "least",
    "phases",
    "prefill_layouts",
    "top_buckets",
]

# The samplers a selection graph may serve: the top-k one, and the general
# one with whether it sorts whole vocabulary rows.
SAMPLERS: tuple[tuple[bool, bool | None], ...] = (
    (True, True),
    (False, True),
    (True, False),
    (False, False),
    (True, None),
    (False, None),
)

# Features a context warmup block holds, in a 2 x 2 patch grid: kernel
# choice depends on the attention input class, not on lengths.
CONTEXT_FEATURES = 4
CONTEXT_GRID = (2, 2)

# The expert capacities a forward of some tokens is captured at, largest
# first; ``[None]`` without experts.
Capacities = Callable[[int], Sequence[int | None]]


def _synchronize(stream: torch.cuda.Stream | None) -> None:
    """Wait for a lane's stream; work on the host already completed."""
    if stream is not None:
        stream.synchronize()


def top_buckets(vocab: int) -> list[int]:
    """Return the top widths of reported logprobs over ``vocab`` tokens.

    Zero, the powers of two below the vocabulary, then the vocabulary.
    """
    buckets = [0]
    top = 1
    while top < vocab:
        buckets.append(top)
        top *= 2
    buckets.append(vocab)
    return buckets


def least(buckets: Sequence[int], position: int) -> int:
    """Return the least count bucket ``position`` of ``buckets`` holds.

    ``buckets`` ascend. The least count is one past the bucket before, or
    for the first bucket at most one.
    """
    if position == 0:
        return min(buckets[0], 1)
    return buckets[position - 1] + 1


@dataclass(frozen=True, slots=True)
class Step:
    """One forward of a warmup phase.

    Attributes:
        rows: The forward's rows.
        kind: Its call kind.
        capacity: The expert capacity it runs at, ``None`` without experts.
        sampled: A decode samples its rows from their top-k candidates.
        features: A context step's first feature token, ``None`` for a
            forward without vision features.
    """

    rows: Rows
    kind: int
    capacity: int | None = None
    sampled: bool = False
    features: int | None = None

    def attention(self) -> list[list[int]]:
        """Return the step's attention rows ``[slot, prefix, query]``."""
        return [
            [int(slot), int(prefix), int(query)]
            for slot, prefix, query in zip(
                self.rows.slot, self.rows.prefix, self.rows.query
            )
        ]


@dataclass(frozen=True, slots=True)
class Layout:
    """What one warmup phase runs.

    Attributes:
        lengths: The scratch tables' lengths of slots ``1..``.
        steps: The forwards, in order.
        primed: For a decode phase, its bucket's rows: after its prefill
            steps, slots ``1..primed`` commit their one token as their
            cache and continue, until the phase ends.
    """

    lengths: tuple[int, ...]
    steps: tuple[Step, ...]
    primed: int = 0


def prefill_layouts(
    shapes: TextShapes,
    *,
    pages: Sequence[tuple[int, int]],
    row_tokens: int,
    units: int,
    capacities: Capacities,
) -> list[Layout]:
    """Lay out the prefill buckets' phases, largest footprint first.

    Args:
        shapes: The runner's graph catalogs.
        pages: Each cache group's ``(tokens per page, units per page)``.
        row_tokens: The most tokens one capture row holds.
        units: The scratch pool's units, sentinel excluded.
        capacities: The capacities a forward is captured at.
    """
    layouts = []
    ordered = capture_order(
        list(shapes.prefill),
        lambda shape: (
            shape.token_bucket * shape.row_bucket,
            shape.token_bucket,
        ),
    )
    for shape in ordered:
        lengths = shape.scratch_lengths(row_tokens, pages, units)
        selection = (
            TokenSelection.LAST_LOGITS
            if shape.outputs
            else TokenSelection.CACHE
        )
        slots = range(1, len(lengths) + 1)
        if shape.variant.embeddings:
            rows = placed_rows(
                [
                    text_row(
                        slot,
                        np.zeros(length, dtype=np.int64),
                        0,
                        0,
                        selection,
                        shape.variant.axes == 3,
                    )
                    for slot, length in zip(slots, lengths)
                ]
            )
        else:
            rows = host_rows(
                list(slots),
                [np.zeros(length, dtype=np.int64) for length in lengths],
                prefix=0,
                selection=selection,
            )
        layouts.append(
            Layout(
                lengths=tuple(lengths),
                steps=tuple(
                    Step(rows, CallKind.PREFILL, capacity)
                    for capacity in capacities(shape.token_bucket)
                ),
            )
        )
    return layouts


def decode_layouts(
    shapes: TextShapes, *, prefill_outputs: bool, capacities: Capacities
) -> list[Layout]:
    """Lay out the decode buckets' phases, largest first.

    Each primes its rows with one prefilled token each, in batches of at
    most the largest prefill bucket's live rows, then decodes one token
    per row greedily and sampled at every capacity.
    """
    layouts = []
    prime = max(
        (shape.row_bucket - 1 for shape in shapes.prefill), default=None
    )
    selection = (
        TokenSelection.LAST_LOGITS if prefill_outputs else TokenSelection.CACHE
    )
    for size in capture_order(list(shapes.decode), lambda rows: (rows, rows)):
        chunk = max(prime, 1) if prime is not None else size
        steps = [
            Step(
                host_rows(
                    list(
                        range(first + 1, first + 1 + min(chunk, size - first))
                    ),
                    [np.zeros(1, dtype=np.int64)] * min(chunk, size - first),
                    prefix=0,
                    selection=selection,
                ),
                CallKind.PREFILL,
            )
            for first in range(0, size, chunk)
        ]
        rows = host_rows(
            list(range(1, size + 1)),
            [np.zeros(1, dtype=np.int64)] * size,
            prefix=1,
            selection=TokenSelection.LAST_LOGITS,
        )
        steps.extend(
            Step(rows, CallKind.DECODE, capacity, sampled)
            for capacity in capacities(size)
            for sampled in (False, True)
        )
        layouts.append(
            Layout(lengths=(2,) * size, steps=tuple(steps), primed=size)
        )
    return layouts


def context_layout(
    injection: Injection, *, spatial: bool, prefill_outputs: bool
) -> Layout:
    """Lay out the context warmup of a model placing vision features.

    A prompt token at position 0, then a block from position 1, in one
    forward; then a block alone where the first one's positions end. A
    block ends its prompt, so its row selects as the lane's prefills do.
    """
    selection = (
        TokenSelection.LAST_LOGITS if prefill_outputs else TokenSelection.CACHE
    )
    first = injection.block(CONTEXT_FEATURES, 1, False, CONTEXT_GRID)
    text = text_row(1, [0], 0, 0, TokenSelection.CACHE, spatial)
    mixed = block_row(first, 1, 1, spatial, selection)
    second = injection.block(
        CONTEXT_FEATURES, next_position(mixed), False, CONTEXT_GRID
    )
    alone = block_row(second, 1, 1 + len(first), spatial, selection)
    return Layout(
        lengths=(1 + len(first) + len(second),),
        steps=(
            Step(
                placed_rows([text, mixed]),
                CallKind.PREFILL,
                features=1 + first.features.start,
            ),
            Step(
                placed_rows([alone]),
                CallKind.PREFILL,
                features=second.features.start,
            ),
        ),
    )


@dataclass(slots=True)
class Phase:
    """One warmup phase on one lane's runner.

    Attributes:
        lane: The lane's index among the rank's token lanes.
        microbatch: The runner the phase captures on.
        layout: What the phase runs.
        run: Runs the phase given each step's page columns (``int64 [2 * G,
            rows]``).
    """

    lane: int
    microbatch: int
    layout: Layout
    run: Callable[[Sequence[np.ndarray]], None]

    def describe(self) -> dict[str, Any]:
        """Return what the worker reads of the phase."""
        return {
            "lane": self.lane,
            "lengths": list(self.layout.lengths),
            "steps": [step.attention() for step in self.layout.steps],
        }


def phases(
    lanes: Sequence[TextLane], *, row_tokens: int, units: int
) -> list[Phase]:
    """Lay out every lane's warmup phases in the order the worker runs them.

    Every lane's prefill and decode phases, runner by runner, then the
    context phases of lanes placing vision features.

    Args:
        lanes: The rank's text lanes, in lane order.
        row_tokens: The most tokens one capture row holds: a request's
            length bound within the batch token bound.
        units: The scratch pool's units, sentinel excluded.
    """
    laid: list[Phase] = []

    def add(lane: TextLane, microbatch: int, layout: Layout) -> None:
        def run(pages: Sequence[np.ndarray]) -> None:
            _run(lane, microbatch, layout, pages)

        laid.append(Phase(lane.index, microbatch, layout, run))

    for lane in lanes:
        capacities = _capacities(lane)
        for microbatch, runner in enumerate(lane.runners):
            for layout in prefill_layouts(
                runner.shapes,
                pages=lane.bounds.pages,
                row_tokens=row_tokens,
                units=units,
                capacities=capacities,
            ):
                add(lane, microbatch, layout)
            for layout in decode_layouts(
                runner.shapes,
                prefill_outputs=lane.prefill_outputs,
                capacities=capacities,
            ):
                add(lane, microbatch, layout)
    for lane in lanes:
        facts = lane.facts
        injection = None if facts is None else facts.injection
        if facts is None or injection is None:
            continue
        for microbatch, runner in enumerate(lane.runners):
            if runner.dims.embeddings:
                add(
                    lane,
                    microbatch,
                    context_layout(
                        injection,
                        spatial=facts.spatial_positions(),
                        prefill_outputs=lane.prefill_outputs,
                    ),
                )
    return laid


def _capacities(lane: TextLane) -> Capacities:
    # Every capacity a forward of ``tokens`` tokens can run at, largest
    # first; one forward without experts.
    union = lane.union
    if union is None:
        return lambda tokens: [None]
    microbatches = len(lane.runners)
    return lambda tokens: [
        int(capacity)
        for capacity in reversed(union.capacities_for(tokens, microbatches))
    ]


def _run(
    lane: TextLane,
    microbatch: int,
    layout: Layout,
    pages: Sequence[np.ndarray],
) -> None:
    """Run ``layout``'s steps on ``microbatch``, then wait for the stream.

    A phase returns once its device work completed: the worker releases
    the scratch tables it read right after.
    """
    if len(pages) != len(layout.steps):
        raise ValueError(
            f"a warmup phase of {len(layout.steps)} steps was given "
            f"{len(pages)} page columns"
        )
    runner = lane.runners[microbatch]
    stream = runner.stream
    # Zero features stand for an encoder's: their values do not choose
    # kernels. They are allocated for the first step placing features and
    # stay referenced until the stream synchronized.
    features: torch.Tensor | None = None
    slots = range(1, layout.primed + 1)
    primed = False
    try:
        for step, columns in zip(layout.steps, pages, strict=True):
            if layout.primed and step.kind == CallKind.DECODE and not primed:
                _prime(lane, microbatch, layout.primed)
                primed = True
            embeddings: list[tuple[int, torch.Tensor]] = []
            if step.features is not None:
                if features is None:
                    with torch.cuda.stream(stream):
                        features = torch.zeros(
                            (CONTEXT_FEATURES, runner.dims.hidden),
                            dtype=runner.dims.dtype,
                            device=runner.buffers.arena.device,
                        )
                embeddings.append((step.features, features))
            lane.warm_step(
                microbatch,
                step.rows,
                step.kind,
                columns,
                row_sampling=(
                    RowSampling.neutral(len(step.rows))
                    if step.sampled
                    else None
                ),
                embeddings=embeddings,
                capacity=step.capacity,
            )
    finally:
        if primed:
            with torch.cuda.stream(stream):
                lane.state.set_predicates(slots, False)
        _synchronize(stream)


def _prime(lane: TextLane, microbatch: int, size: int) -> None:
    # Each slot's one prefilled token becomes its cache, which captures the
    # bucket's commit graph; the slots then continue.
    runner = lane.runners[microbatch]
    rows = np.tile(COMMIT_PADDING, (size, 1))
    rows[:, slot_commit.SLOT] = np.arange(1, size + 1)
    rows[:, slot_commit.CACHE_LENGTH] = 1
    runner.commit(CommitWork(rows=rows, tails=np.zeros((0, 2), dtype=np.int64)))
    with torch.cuda.stream(runner.stream):
        lane.state.set_predicates(range(1, size + 1), True)


def finish(lanes: Sequence[TextLane]) -> None:
    """Capture the graphs that follow every phase, then wait for them.

    Per runner with a graph pool: the ``commit`` graph of its rows when no
    decode bucket holds them, then its selection graphs, each alone over
    padding inputs, largest first. A ``score`` graph is captured when the
    least reporting tasks and top tokens its bucket holds fit the words a
    forward may report, and, when the lane's prefills keep the hidden
    states prompt scoring projects, a ``score_prompt`` graph when one row
    of its bucket's least top tokens fits them.
    """
    for lane in lanes:
        for runner in lane.runners:
            if runner.pool is None:
                continue
            dims = runner.dims
            if dims.rows not in runner.shapes.decode:
                padding = np.tile(COMMIT_PADDING, (dims.rows, 1))
                runner.commit(
                    CommitWork(
                        rows=padding, tails=np.zeros((0, 2), dtype=np.int64)
                    )
                )
            runner.reset_selection()
            buckets = task_buckets(dims.rows)
            for tasks in reversed(buckets):
                for top_k, categorical in SAMPLERS:
                    runner.capture_alone(SelectKey(tasks, top_k, categorical))
            words = dims.logprob_words()
            tops = top_buckets(dims.vocab)
            for place in reversed(range(len(buckets))):
                for position in reversed(range(len(tops))):
                    least_words = logprob_words(
                        least(buckets, place), least(tops, position)
                    )
                    if least_words <= words:
                        runner.capture_alone(
                            ScoreKey(buckets[place], tops[position])
                        )
            for top in reversed(tops):
                if (
                    lane.prefill_outputs
                    and logprob_words(1, least_top(top)) <= words
                ):
                    runner.capture_alone(PromptKey(dims.prompt_block(top), top))
            _synchronize(runner.stream)
