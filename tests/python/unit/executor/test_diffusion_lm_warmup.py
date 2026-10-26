"""The warmup forwards of a diffusion LM lane and the phase the worker
drives (``uniserve.executor.diffusion_lm.warmup``).

Every rank of a group runs the same forwards in the same order: the order
below is the protocol its collectives follow.
"""  # noqa: D205

from __future__ import annotations

import numpy as np
import pytest

from uniserve.executor.diffusion_lm.shapes import Shapes
from uniserve.executor.diffusion_lm.warmup import (
    Alone,
    Denoise,
    Phase,
    Readout,
    layout,
    spread,
    synthetic_readout,
    synthetic_steps,
)

pytestmark = pytest.mark.unit

LENGTH = 32


def _shapes() -> Shapes:
    # Two-block steps of 32-token blocks, readouts of two blocks and 64
    # tokens, four slots: ``attend`` buckets (1, 16), (1, 32), (2, 32) and
    # (2, 64), one ``finish`` bucket of 64 slots, ``denoise`` rows 1 and 2.
    return Shapes(LENGTH, 2, 2, 64, slots=4)


def _layout(**overrides):
    arguments = {
        "pooled": True,
        "generating": True,
        "local": True,
        "last": True,
        "chunk_rows": 1,
        "capacities": lambda tokens: [None],
    }
    arguments.update(overrides)
    return layout(_shapes(), **arguments)


def test_spread_puts_the_remainder_on_the_first_blocks():
    assert spread(10, 4) == [3, 3, 2, 2]
    assert spread(64, 2) == [32, 32]
    assert spread(5, 1) == [5]


def test_a_pooled_lane_captures_alone_graphs_then_readouts_then_steps():
    readouts = [
        Readout((32, 32), (0, 32)),
        Readout((16, 16), (0, 16)),
        Readout((32,), (0,)),
        Readout((16,), (0,)),
    ]
    steps = [
        Denoise(rows, first, LENGTH)
        for rows in (2, 1)
        for first in (True, False)
    ]
    assert _layout() == [
        Alone("readout", 256),
        Alone("readout", 128),
        Alone("readout", 64),
        Alone("finish", 64),
        *readouts,
        *steps,
    ]
    # Off the final stage no ``finish`` runs alone; a readout-only lane
    # steps no block.
    assert _layout(last=False, generating=False) == [
        Alone("readout", 256),
        Alone("readout", 128),
        Alone("readout", 64),
        *readouts,
    ]


def test_an_exchanging_finish_is_captured_in_readouts_at_every_capacity():
    # Fake capacities, largest first, that reveal each forward's extent.
    steps = _layout(local=False, capacities=lambda tokens: [2 * tokens, tokens])
    assert not any(
        isinstance(step, Alone) and step.method == "finish" for step in steps
    )
    readouts = [step for step in steps if isinstance(step, Readout)]
    # Every ``attend`` bucket's step also sends its slots' 64-slot finish
    # bucket; each readout reads the fewest slots of that bucket.
    assert [step.queries for step in readouts] == [
        (32, 32),
        (32, 32),
        (16, 16),
        (16, 16),
        (32,),
        (32,),
        (16,),
        (16,),
    ]
    assert [step.capacity for step in readouts] == [128, 64] * 4
    assert {step.slot_tokens for step in readouts} == {(0,)}
    denoising = [step for step in steps if isinstance(step, Denoise)]
    assert [(step.rows, step.first, step.capacity) for step in denoising] == [
        (2, True, 128),
        (2, True, 64),
        (2, False, 128),
        (2, False, 64),
        (1, True, 64),
        (1, True, 32),
        (1, False, 64),
        (1, False, 32),
    ]


class _Union:
    """An expert union of one source group: a step runs at the smallest
    capacity holding the tokens it sends.
    """  # noqa: D205

    CAPACITIES = (48, 64, 96, 128, 256)

    def capacities_for(self, tokens: int) -> tuple[int, ...]:
        return tuple(
            capacity for capacity in self.CAPACITIES if capacity >= tokens
        )[:1]


def test_exchanging_readouts_capture_every_capacity_a_bucket_can_agree_on():
    # ``attend`` buckets (1, 16), (1, 32), (2, 32), (2, 64), (3, 48),
    # (3, 96), (4, 64) and (4, 128); ``finish`` buckets 64, 128 and 192.
    shapes = Shapes(LENGTH, 4, 4, 192, slots=4)
    union = _Union()
    steps = layout(
        shapes,
        pooled=True,
        generating=False,
        local=False,
        last=True,
        chunk_rows=1,
        capacities=lambda tokens: list(reversed(union.capacities_for(tokens))),
    )
    readouts = [step for step in steps if isinstance(step, Readout)]
    attend, finish = set(), set()
    for step in readouts:
        rows, tokens = shapes.attend(
            len(step.queries), sum(step.queries), max(step.queries)
        )
        slots = len(step.slot_tokens)
        # A readout's slots are distinct positions among its tokens.
        assert slots <= tokens and max(step.slot_tokens) < tokens
        attend.add((rows, tokens, step.capacity))
        finish.add((shapes.finish(slots), step.capacity))

    # A readout of 96 tokens sends 96 tokens with at most 64 slots and 128
    # with more: its ``attend`` graph is captured at both capacities. Small
    # buckets always send their 64 finish slots.
    assert attend == {
        (1, 16, 64),
        (1, 32, 64),
        (2, 32, 64),
        (2, 64, 64),
        (3, 48, 64),
        (3, 96, 96),
        (3, 96, 128),
        (4, 64, 64),
        (4, 128, 128),
    }
    # Each ``finish`` bucket at the capacity of every extent it completes;
    # no readout's slots reach the 192-slot bucket.
    assert finish == {(64, 64), (64, 96), (64, 128), (128, 128)}
    # Largest buckets first, each bucket's capacities largest first.
    buckets = [
        shapes.attend(len(step.queries), sum(step.queries), max(step.queries))
        for step in readouts
    ]
    assert buckets == sorted(buckets, reverse=True)
    for bucket in set(buckets):
        capacities = [
            step.capacity
            for step, held in zip(readouts, buckets, strict=True)
            if held == bucket
        ]
        assert capacities == sorted(capacities, reverse=True)


def test_an_eager_lane_runs_one_first_step_per_sampler_chunk_shape():
    assert _layout(pooled=False, chunk_rows=3) == [
        Denoise(1, True, LENGTH),
        Denoise(2, True, LENGTH),
        Denoise(3, True, LENGTH),
    ]
    assert _layout(pooled=False, generating=False) == []


def test_the_phase_lists_the_rows_of_every_forward_that_attends():
    steps = (
        Alone("readout", 64),
        Readout((16, 16, 16), (0, 16, 32)),
        Denoise(3, True, LENGTH),
    )
    phase = Phase(0, (1, 1), steps, lambda pages: None)
    # Row ``r`` reads the one-token table of slot ``r % 2 + 1``.
    assert phase.describe() == {
        "lane": 0,
        "lengths": [1, 1],
        "steps": [
            [[1, 1, 16], [2, 1, 16], [1, 1, 16]],
            [[1, 1, LENGTH], [2, 1, LENGTH], [1, 1, LENGTH]],
        ],
    }


def test_synthetic_steps_start_or_continue_blocks_of_distinct_slots():
    started = synthetic_steps(3, True)
    assert started.first
    assert started.columns().tolist() == [
        [1, 2, 3],
        [1, 2, 3],
        [0, 0, 0],
        [0, 0, 0],
    ]
    continued = synthetic_steps(2, False)
    assert not continued.first
    assert continued.step.tolist() == [1, 1]
    assert continued.prefix.tolist() == [1, 1]
    assert continued.position.tolist() == [1, 1]


def test_synthetic_readouts_read_one_candidate_per_slot_of_each_chunk():
    vocab = 10
    readout = synthetic_readout(
        Readout((16,), tuple(slot % 16 for slot in range(300))),
        installed=2,
        vocab=vocab,
    )
    assert readout.slot.tolist() == [1]
    assert len(readout.tokens) == 16
    windows = list(readout.windows())
    assert [(w.start, w.live, w.bucket) for w in windows] == [
        (0, 256, 256),
        (256, 44, 64),
    ]
    assert [(w.first, w.end) for w in windows] == [(0, 256), (256, 300)]
    # A candidate's flat index addresses its slot's row within its chunk.
    np.testing.assert_array_equal(readout.index[256:], np.arange(44) * vocab)
