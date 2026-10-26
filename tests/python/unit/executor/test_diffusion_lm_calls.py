"""Validation, packing and completion records of token-denoising calls
(``uniserve.executor.diffusion_lm.calls``).

Each batch here is built as the worker builds it: call columns, forward rows
grouped by call, the flat token column with every readout's lists appended
after the calls' input tokens, and the step JSON locating those lists.
"""  # noqa: D205

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from uniserve.executor.diffusion_lm.bounds import ServedSampling
from uniserve.executor.diffusion_lm.calls import (
    ReadoutPass,
    Seeds,
    Window,
    batch_form,
    pack_readouts,
    pack_steps,
    readout_spans,
    record_readouts,
    record_steps,
    seed_bits,
)
from uniserve.executor.diffusion_lm.shapes import CANDIDATES
from uniserve.executor.text.protocol import (
    CALL_FIELDS,
    CALL_WORDS,
    HEADER_WORDS,
    CallField,
    CallFlag,
    CallKind,
    RecordField,
    RecordFlag,
    RecordWriter,
    forward_fields,
    record_capacity,
)

pytestmark = pytest.mark.unit

VOCAB = 1 << 10
# The request every call of these batches names.
REQUEST = (7, 1)


@dataclass
class _Call:
    slot: int
    visible: int
    logical: int
    queries: tuple[int, ...]
    tokens: tuple[int, ...]
    readout: tuple[list[int], list[int], list[int]] | None
    canvas: tuple[int, int] | None
    max_tokens: int
    active: bool
    prompt_ready: bool
    rng: int
    row_slot: int
    writes: bool
    request: tuple[int, int]


@dataclass
class _Batch:
    """The columns of one token-denoising batch, call after call."""

    calls: list[_Call] = field(default_factory=list)

    def add(
        self,
        slot: int,
        visible: int,
        logical: int,
        queries,
        *,
        tokens=None,
        readout=None,
        canvas=None,
        max_tokens=None,
        active=True,
        prompt_ready=False,
        rng=0,
        row_slot=None,
        writes=False,
        request=REQUEST,
    ) -> _Batch:
        """Add a call whose rows of ``queries`` tokens read ``slot``.

        A readout's input tokens default to ``100, 101, ...``.
        """
        if tokens is None:
            tokens = range(100, 100 + sum(queries)) if readout else ()
        self.calls.append(
            _Call(
                slot,
                visible,
                logical,
                tuple(queries),
                tuple(tokens),
                readout,
                canvas,
                len(tokens) if max_tokens is None else max_tokens,
                active,
                prompt_ready,
                rng,
                slot if row_slot is None else row_slot,
                writes,
                request,
            )
        )
        return self

    def build(self):
        """Return the call and forward columns, the tokens and the step
        JSON's call entries.
        """  # noqa: D205
        columns = np.zeros((CALL_FIELDS, len(self.calls)), dtype=np.int64)
        rows: list[list[int]] = []
        tokens = [token for call in self.calls for token in call.tokens]
        entries = {}
        start = 0
        for index, call in enumerate(self.calls):
            column = columns[:, index]
            column[CallField.SLOT] = call.slot
            column[CallField.KIND] = CallKind.TOKEN_DENOISING
            column[CallField.ACTIVE] = call.active
            column[CallField.LOGICAL] = call.logical
            column[CallField.VISIBLE] = call.visible
            column[CallField.COMPUTED] = call.visible
            column[CallField.RNG] = call.rng
            column[CallField.PROMPT_READY] = call.prompt_ready
            column[CallField.ALLOCATED] = call.visible
            column[CallField.ROW_START] = len(rows)
            column[CallField.ROW_COUNT] = len(call.queries)
            column[CallField.TOKEN_START] = start
            column[CallField.TOKEN_COUNT] = len(call.tokens)
            column[CallField.MAX_TOKENS] = call.max_tokens
            column[CallField.CANVAS_BLOCK] = -1
            column[CallField.CANVAS_STEP] = -1
            column[CallField.REQUEST_ID], column[CallField.EPOCH] = call.request
            start += len(call.tokens)
            for query in call.queries:
                rows.append(
                    [
                        index,
                        call.row_slot,
                        call.visible + query,
                        query,
                        call.writes,
                    ]
                )
            if call.canvas is not None:
                block, step = call.canvas
                column[CallField.CANVAS_BLOCK] = block
                column[CallField.CANVAS_STEP] = step
            if call.readout is not None:
                column[CallField.FLAGS] = CallFlag.READOUT
                spans = {}
                for name, values in zip(
                    ("slot_tokens", "candidate_offsets", "candidate_ids"),
                    call.readout,
                    strict=True,
                ):
                    spans[name] = [len(tokens), len(values)]
                    tokens.extend(values)
                entries[str(index)] = {"readout": spans}

        # One cache group, whose page columns these checks do not read.
        forward = np.zeros((forward_fields(1), len(rows)), dtype=np.int64)
        forward[:5] = np.asarray(rows, dtype=np.int64).T
        return columns, forward, np.asarray(tokens, dtype=np.int64), entries


def _pack(batch: _Batch, *, capacity: int = 4096) -> ReadoutPass:
    calls, forward, tokens, entries = batch.build()
    active, steps = batch_form(calls)
    assert not steps
    return pack_readouts(
        calls,
        forward,
        tokens,
        active,
        readout_spans(entries),
        vocab=VOCAB,
        capacity=capacity,
    )


def test_readouts_pack_rows_slots_and_candidates_of_every_call():
    batch = _Batch()
    # Two rows of 3 and 2 tokens; each reads a slot.
    batch.add(
        2, 6, 40, [3, 2], readout=([1, 3, 4], [0, 2, 3, 5], [7, 8, 9, 10, 11])
    )
    batch.add(3, 1, 9, [2], readout=([0], [0, 1], [12]))
    readout = _pack(batch)

    assert readout.tokens.tolist() == [100, 101, 102, 103, 104, 100, 101]
    # Slots address the forward's packed tokens.
    assert readout.slots.tolist() == [1, 3, 4, 5]
    assert readout.slot_candidates.tolist() == [0, 2, 3, 5, 6]
    assert readout.candidates.tolist() == [7, 8, 9, 10, 11, 12]
    # Each call's range of the forward's candidates.
    assert readout.candidate_start.tolist() == [0, 5]
    assert readout.candidate_count.tolist() == [5, 1]
    # Every row reads its call's visible prefix from its logical position.
    assert readout.query.tolist() == [3, 2, 2]
    assert readout.prefix.tolist() == [6, 6, 1]
    assert readout.position.tolist() == [40, 40, 9]
    assert readout.slot.tolist() == [2, 2, 3]
    assert readout.rows.tolist() == [0, 1, 2]
    assert readout.longest == 3


def test_readouts_refuse_rows_that_write_skip_slots_or_misread_the_prefix():
    readout = ([4], [0, 1], [1])

    def refused(**call) -> bool:
        batch = _Batch().add(
            2, 6, 0, call.pop("queries"), readout=readout, **call
        )
        try:
            _pack(batch)
        except ValueError:
            return True
        return False

    # The first row reads no slot.
    assert refused(queries=[3, 2])
    # Another slot, a write, or rows not partitioning the tokens.
    assert refused(queries=[5], row_slot=3)
    assert refused(queries=[5], writes=True)
    assert refused(queries=[4], tokens=range(5))
    assert not refused(queries=[5])


def test_readouts_refuse_slots_and_candidates_outside_their_bounds():
    def refused(readout, *, capacity: int = 4096) -> bool:
        batch = _Batch().add(2, 6, 0, [3, 2], readout=readout)
        try:
            _pack(batch, capacity=capacity)
        except ValueError:
            return True
        return False

    assert not refused(([1, 4], [0, 1, 2], [5, VOCAB - 1]))
    # A slot beyond the call's tokens or out of order.
    assert refused(([1, 5], [0, 1, 2], [5, 6]))
    assert refused(([4, 1], [0, 1, 2], [5, 6]))
    # Offsets that do not bound every slot or partition the candidates.
    assert refused(([1, 4], [0, 2], [5, 6]))
    assert refused(([1, 4], [0, 2, 1], [5, 6]))
    assert refused(([1, 4], [1, 1, 2], [5, 6]))
    # A candidate outside the vocabulary.
    assert refused(([1, 4], [0, 1, 2], [5, VOCAB]))
    # More slots than a forward's tokens.
    assert refused(([1, 4], [0, 1, 2], [5, 6]), capacity=1)


def test_readout_launches_cover_every_candidate_in_slot_chunks():
    # One call of 300 slots: the first reads more candidates than a
    # launch selects, the others one each.
    slots = 300
    counts = [CANDIDATES + 4464] + [1] * (slots - 1)
    offsets = np.concatenate(([0], np.cumsum(counts))).tolist()
    ids = (np.arange(offsets[-1]) % VOCAB).tolist()
    batch = _Batch().add(
        1, 0, 0, [slots], readout=(list(range(slots)), offsets, ids)
    )
    readout = _pack(batch)

    first = CANDIDATES + 4464 + 255
    assert list(readout.windows()) == [
        Window(0, 256, 256, 0, CANDIDATES),
        Window(0, 256, 256, CANDIDATES, first),
        Window(256, 44, 64, first, first + 44),
    ]
    # Each candidate selects its slot's row of its chunk's log-probabilities.
    candidate_slot = np.repeat(np.arange(slots), counts)
    expected = (candidate_slot - 256 * (candidate_slot >= 256)) * VOCAB + ids
    assert readout.index.tolist() == expected.tolist()
    assert readout.index[first] == 0 * VOCAB + ids[first]
    assert readout.index[first + 1] == 1 * VOCAB + ids[first + 1]

    # A chunk without candidates launches no readout.
    batch = _Batch().add(1, 0, 0, [2], readout=([0, 1], [0, 0, 0], []))
    assert list(_pack(batch).windows()) == []


def test_batches_carry_readouts_or_steps_of_active_calls():
    batch = _Batch()
    batch.add(1, 4, 4, [2], readout=([0], [0, 1], [3]))
    batch.add(2, 4, 4, [4], canvas=(0, 0), active=False)
    batch.add(3, 4, 4, [2], readout=([1], [0, 1], [3]))
    calls, *_ = batch.build()
    active, steps = batch_form(calls)
    assert active.tolist() == [0, 2] and not steps

    calls[CallField.ACTIVE, 1] = 1
    with pytest.raises(ValueError):
        batch_form(calls)
    calls[CallField.ACTIVE, [0, 2]] = 0
    active, steps = batch_form(calls)
    assert active.tolist() == [1] and steps

    calls[CallField.KIND, 0] = CallKind.DECODE
    with pytest.raises(ValueError):
        batch_form(calls)


def _sampling(canvas_length: int = 4) -> ServedSampling:
    return ServedSampling(
        canvas_length=canvas_length,
        max_steps=3,
        entropy_bound=0.5,
        t_min=0.4,
        t_max=0.8,
        confidence_threshold=0.2,
        stability_threshold=1,
    )


def test_steps_require_the_served_sampling_and_a_seeded_request():
    # A block step runs under the sampling this rank was launched with and
    # the request's seed, carried as its int64 bits.
    served = _sampling()
    seed = (1 << 64) - 2
    seeds = Seeds(3)
    seeds.start(2, *REQUEST, seed)

    def steps(seeds: Seeds, served, **call):
        batch = _Batch().add(2, 10, 12, call.pop("queries", [4]), **call)
        calls, forward, _, _ = batch.build()
        active, stepping = batch_form(calls)
        assert stepping
        return pack_steps(calls, forward, active, seeds, served)

    row = steps(seeds, served, canvas=(2, 1), max_tokens=4)
    assert row.calls.tolist() == [0]
    assert row.columns().tolist() == [[2], [-2], [2], [1]]
    assert (row.prefix.tolist(), row.position.tolist()) == ([10], [12])
    assert not row.first

    def refused(seeds: Seeds, served, **call) -> bool:
        try:
            steps(seeds, served, **call)
        except ValueError:
            return True
        return False

    seeded = Seeds(3)
    seeded.start(2, *REQUEST, 1)
    unseeded = Seeds(3)
    unseeded.start(2, *REQUEST, None)
    assert not refused(seeded, served, canvas=(2, 1), max_tokens=4)
    assert refused(unseeded, served, canvas=(2, 1), max_tokens=4)
    assert refused(seeded, None, canvas=(2, 1), max_tokens=4)
    # The last step of a block is the step limit's predecessor.
    assert refused(seeded, served, canvas=(0, 3), max_tokens=4)
    # A step is exactly one block-length row.
    assert refused(seeded, served, canvas=(2, 1), max_tokens=3)
    assert refused(seeded, served, canvas=(2, 1), max_tokens=4, queries=[3])
    assert refused(seeded, served, canvas=(2, 1), max_tokens=4, queries=[2, 2])
    # The slot holds another request, or none since its release.
    assert refused(seeded, served, canvas=(2, 1), max_tokens=4, request=(7, 2))
    seeded.release([2])
    assert refused(seeded, served, canvas=(2, 1), max_tokens=4)


def test_seeds_are_carried_as_their_int64_bits():
    assert seed_bits(0) == 0
    assert seed_bits((1 << 63) - 1) == (1 << 63) - 1
    assert seed_bits(1 << 63) == -(1 << 63)
    assert seed_bits((1 << 64) - 1) == -1
    with pytest.raises(ValueError):
        seed_bits(1 << 64)
    with pytest.raises(ValueError):
        seed_bits(-1)


def _writer(calls: int, tokens: int) -> tuple[np.ndarray, RecordWriter]:
    records = np.full(record_capacity(calls, tokens), -7, dtype=np.int64)
    return records, RecordWriter(records, calls)


def test_readout_records_report_unchanged_progress_and_candidate_words():
    batch = _Batch()
    batch.add(1, 6, 40, [3], readout=([0, 2], [0, 2, 3], [7, 8, 9]), rng=5)
    batch.add(2, 3, 9, [2], readout=([1], [0, 1], [4]), active=False)
    # A readout whose slot reads no candidate, ready prompt logits.
    batch.add(3, 1, 8, [1], readout=([0], [0, 0], []), prompt_ready=True)
    batch.add(1, 6, 40, [2], readout=([1], [0, 2], [5, 6]))
    calls, _, tokens, _ = batch.build()
    readout = _pack(batch)
    records, writer = _writer(calls.shape[1], len(tokens))

    written = record_readouts(writer, calls, readout, lent=5)
    assert written == 5
    call = records[HEADER_WORDS : HEADER_WORDS + 4 * CALL_WORDS].reshape(
        4, CALL_WORDS
    )

    ran = RecordFlag.RAN | RecordFlag.PROGRESS | RecordFlag.CANDIDATES
    assert call[:, RecordField.FLAGS].tolist() == [
        ran,
        0,
        ran | RecordFlag.PROMPT_LOGITS_READY,
        ran,
    ]
    progress = call[:, RecordField.LOGICAL : RecordField.COMPUTED + 1]
    assert progress.tolist() == [
        [40, 5, 6, 6],
        [0] * 4,
        [8, 0, 1, 1],
        [40, 0, 6, 6],
    ]
    ranges = call[:, RecordField.RANGE_OFFSET : RecordField.RANGE_COUNT + 1]
    # The empty range lies within the written words.
    assert ranges.tolist() == [[0, 3], [0, 0], [3, 0], [3, 2]]
    # An inactive call's record is zero.
    assert not call[1].any()

    with pytest.raises(AssertionError):
        record_readouts(writer, calls, readout, lent=4)


def test_step_records_name_each_row_result():
    length = 4
    seeds = Seeds(4)
    for slot in (1, 3, 4):
        seeds.start(slot, *REQUEST, slot)
    batch = _Batch()
    for slot, active in ((1, True), (2, False), (3, True), (4, True)):
        batch.add(
            slot,
            8,
            20,
            [length],
            canvas=(1, 0),
            max_tokens=length,
            active=active,
            prompt_ready=slot == 3,
        )
    calls, forward, tokens, _ = batch.build()
    active, _ = batch_form(calls)
    steps = pack_steps(calls, forward, active, seeds, _sampling(length))
    assert steps.first
    records, writer = _writer(calls.shape[1], len(tokens))

    written = record_steps(writer, calls, steps, length=length, lent=15)
    assert written == 3 * (1 + length)
    call = records[HEADER_WORDS : HEADER_WORDS + 4 * CALL_WORDS].reshape(
        4, CALL_WORDS
    )
    ran = RecordFlag.RAN | RecordFlag.PROGRESS | RecordFlag.CANVAS
    assert call[:, RecordField.FLAGS].tolist() == [
        ran,
        0,
        ran | RecordFlag.PROMPT_LOGITS_READY,
        ran,
    ]
    ranges = call[:, RecordField.RANGE_OFFSET : RecordField.RANGE_COUNT + 1]
    assert ranges.tolist() == [[0, 5], [0, 0], [5, 5], [10, 5]]
    assert call[[0, 2, 3], RecordField.VISIBLE].tolist() == [8, 8, 8]
    assert call[[0, 2, 3], RecordField.COMPUTED].tolist() == [8, 8, 8]

    with pytest.raises(AssertionError):
        record_steps(writer, calls, steps, length=length, lent=14)
