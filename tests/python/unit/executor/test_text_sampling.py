"""Sampling inputs of a text forward (``uniserve.executor.text.sampling``)."""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

from uniserve.errors import OutOfMemory
from uniserve.executor.text.philox import DrawWindow, sampling_uniform
from uniserve.executor.text.sampling import (
    NEUTRAL,
    OwnSelection,
    RowSampling,
    SampleTask,
    SamplingControls,
    SamplingMetadata,
    SamplingParams,
    SamplingState,
    controls_for_call,
    least_top,
    logprob_shape,
    logprob_words,
    own_selection,
    pad_selection,
    padding_selection,
    prompt_block,
    sampling_paths,
    task_draws,
    top_bucket,
)
from uniserve.executor.text.state import NewRequest, RequestTable
from uniserve.sampling.select import CATEGORICAL, GREEDY, TOP_K_PATH

pytestmark = pytest.mark.unit


def _task(
    params: SamplingParams, row: int, draws: list[float] | None
) -> SampleTask:
    return SampleTask(
        params=params,
        controls=SamplingControls(allowed=[None]),
        draws=None if draws is None else np.asarray(draws, np.float32),
        slot=3,
        row=row,
        gated=True,
    )


def _bits(words: np.ndarray, vocab: int) -> list[int]:
    return [
        token
        for token in range(vocab)
        if int(words[token // 32]) & (1 << (token % 32))
    ]


def test_tasks_keep_their_order_and_report_logprobs_in_a_padded_shape():
    # Tasks keep their input order, each with its sampler, slot, gate and
    # finish set; reporting tasks pack their logprobs into a padded shape.
    greedy = SamplingParams()
    stochastic = SamplingParams(temperature=0.7, n_logprobs=3)
    first = _task(stochastic, 0, [0.25])
    first.controls.finish_token_ids = (2, 40, 70)
    tasks = [first, _task(greedy, 1, None), _task(stochastic, 2, [0.5])]
    metadata = SamplingMetadata.build(tasks, 50, 8, 1000)
    assert (metadata.tasks, metadata.width) == (3, 1)
    assert metadata.paths.tolist() == [CATEGORICAL, GREEDY, CATEGORICAL]
    assert (metadata.top_k, metadata.categorical, metadata.sorts) == (
        False,
        True,
        False,
    )
    assert metadata.rows.tolist() == [0, 1, 2]
    assert metadata.draws.tolist() == [0.25, 0.0, 0.5]
    assert metadata.params.shape == (3, 8)
    assert metadata.slots.tolist() == [3, 3, 3]
    assert metadata.mask_rows.tolist() == [-1, -1, -1]
    # Token 70 is outside the 50-token vocabulary.
    assert _bits(metadata.finish[0], 50) == [2, 40]
    assert not metadata.finish[1:].any()

    # Two reporting tasks of three top tokens, packed compactly and scored
    # at a top width of four over a bucket of two tasks.
    assert metadata.logprob_tasks.tolist() == [0, 2]
    assert metadata.logprobs is not None
    assert metadata.logprobs.counts.tolist() == [3, 3]
    assert metadata.logprob_bucket == (2, 4)
    assert metadata.logprob_end() == 2 * (3 + 3 * 3)

    # The packed words bound the binding; a task without draws for its
    # sampler is refused.
    with pytest.raises(OutOfMemory):
        SamplingMetadata.build(tasks, 50, 8, 23)
    with pytest.raises(ValueError):
        SamplingMetadata.build([_task(stochastic, 0, None)], 50, 8, 1000)


def test_rows_of_one_request_share_their_token_set():
    # A request's rows share one token set: its allowed tokens less the
    # suppressed ones, or every token less the suppressed ones.
    stochastic = SamplingParams(temperature=1.0)
    allowed = _task(stochastic, 0, [0.5])
    allowed.controls.allowed = [(1, 5, 9)]
    allowed.controls.suppress = (5,)
    suppressed = _task(stochastic, 1, [0.5])
    suppressed.controls.suppress = (0, 39)
    metadata = SamplingMetadata.build([allowed, suppressed], 40, 8, 1000)
    assert metadata.mask_rows.tolist() == [0, 1]
    assert _bits(metadata.masks[0], 40) == [1, 9]
    assert _bits(metadata.masks[1], 40) == list(range(1, 39))
    # Bits past the vocabulary stay clear.
    assert int(metadata.masks[1, -1].view(np.uint32)) >> 8 == 0

    # A verification forcing a token at each position has a set per row;
    # more sets than the binding's rows are refused.
    forced = _task(stochastic, 0, [0.5] * 3)
    forced.controls.allowed = [(1,), (2,), (3,)]
    forced.controls.draft_token_ids = (1, 2)
    metadata = SamplingMetadata.build([forced], 40, 3, 1000)
    assert metadata.mask_rows.tolist() == [0, 1, 2]
    with pytest.raises(OutOfMemory):
        SamplingMetadata.build([forced], 40, 2, 1000)


def test_verification_tasks_pad_to_the_widest_task():
    # A verification's tasks hold its widest task's rows: a task's own
    # rows, then copies of its first, and its drafts padded with -1.
    stochastic = SamplingParams(temperature=1.0)
    wide = _task(stochastic, 0, [0.1, 0.2, 0.3])
    wide.controls.allowed = [None] * 3
    wide.controls.draft_token_ids = (7, 8)
    wide.controls.finish_token_ids = (8,)
    wide.controls.terminal_draft_prefix = 2
    narrow = _task(stochastic, 3, [0.4, 0.5])
    narrow.controls.allowed = [None] * 2
    narrow.controls.draft_token_ids = (6,)
    metadata = SamplingMetadata.build([wide, narrow], 20, 8, 1000)
    assert (metadata.tasks, metadata.width) == (2, 3)
    assert metadata.rows.tolist() == [0, 1, 2, 3, 4, 3]
    assert np.array_equal(
        metadata.draws,
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.4], dtype=np.float32),
    )
    assert metadata.drafts.reshape(-1).tolist() == [7, 8, 6, -1]
    assert metadata.terminal.tolist() == [2, 0]


def test_a_top_bucket_holds_its_least_top_through_itself():
    # Over 100 tokens: 0, 1, 2, 4, ..., 64, then 100.
    for top in range(101):
        held = top_bucket(top, 100)
        assert least_top(held) <= top <= held
    assert least_top(100) == 65
    # A vocabulary of three tokens makes three a bucket of itself.
    assert least_top(top_bucket(3, 3)) == 3


def test_logprob_shapes_double_at_most():
    assert logprob_shape(3, 5, 100, 40) == (4, 8)
    assert logprob_shape(1, 0, 100, 40) == (1, 0)
    assert logprob_shape(5, 90, 100, 40) == (8, 100)
    # A bucket never exceeds the binding's rows.
    assert logprob_shape(33, 1, 100, 40) == (40, 1)
    for rows in range(1, 41):
        for top in range(121):
            stride, width = logprob_shape(rows, top, 100, 40)
            assert rows <= stride <= 40 and width >= min(top, 100)
            assert logprob_words(stride, width) <= 4 * logprob_words(
                rows, min(top, 100)
            )


def test_a_prompt_scoring_call_holds_rows_within_every_bound():
    # Transient bytes bound a large vocabulary's block: 512 MiB over 24
    # bytes per vocabulary entry.
    assert prompt_block(0, 152_064, 8192) == 147
    # The binding's logprob words bound wide tops: 16 tokens of 63 words
    # report nine rows of the bucket's least top (33 tokens).
    assert prompt_block(64, 100, 16) == 9
    # A call projects at most the binding's tokens, which its operators
    # are prepared for.
    assert prompt_block(0, 100, 16) == 16


def test_a_call_forces_its_tokens_by_position_and_ends_drafts_at_finish():
    params = SamplingParams(
        temperature=0.5, forced_token_ids=(11,), allowed_token_ids=(4, 5)
    )
    state = SamplingState(finish_token_ids=(9, 3))
    controls, draws = controls_for_call(
        params, state, [3, 2], [7, 8, 9], [6, 3], key=5, return_transition=True
    )
    # The first position is forced; the others take the request's set.
    assert controls.allowed == [(11,), (4, 5), (4, 5)]
    # Admitted and branch finish tokens merge into their sorted union.
    assert controls.finish_token_ids == (2, 3, 9)
    # The second draft is a finish token.
    assert controls.terminal_draft_prefix == 2
    assert np.array_equal(draws, sampling_uniform(5, [7, 8, 9]))

    # The unshaped greedy path draws nothing; a zero-temperature sampler
    # reads zero draws.
    _, draws = controls_for_call(SamplingParams(), None, [], [4], [], key=5)
    assert draws is None
    _, draws = controls_for_call(
        SamplingParams(top_k=5), None, [], [4], [], key=5
    )
    assert draws.tolist() == [0.0]


def _decode_case(**changes: object) -> dict[str, np.ndarray]:
    # Two sampled decode calls that select in their own forward.
    case = {
        "sampled": np.array([True, True]),
        "decode": np.array([True, True]),
        "tagged": np.array([True, True]),
        "token_output": np.array([True, True]),
        "transition_output": np.array([False, False]),
        "transitions": np.array([False, False]),
        "finish_count": np.array([1, 8]),
        "paths": np.array([GREEDY, GREEDY]),
        "suppressed": np.array([False, False]),
    }
    for name, value in changes.items():
        case[name] = np.asarray(value)
    return case


def test_a_decode_selects_itself_only_when_every_call_qualifies():
    assert own_selection(**_decode_case()) is OwnSelection.GREEDY
    assert (
        own_selection(**_decode_case(paths=[GREEDY, TOP_K_PATH]))
        is OwnSelection.TOP_K
    )
    for disqualified in [
        {"paths": [GREEDY, CATEGORICAL]},
        {"suppressed": [True, False]},
        {"sampled": [True, False]},
        {"decode": [True, False]},
        {"tagged": [False, True]},
        {"token_output": [True, False]},
        {"transition_output": [False, True]},
        {"transitions": [True, False]},
        {"finish_count": [1, 9]},
    ]:
        assert own_selection(**_decode_case(**disqualified)) is None


def _start(table: RequestTable, slot: int, finish, **sampling) -> None:
    table.start(
        NewRequest(
            slot=slot,
            engine_id=1,
            request_id=slot,
            epoch=0,
            sampling=SamplingParams(**sampling),
            finish_token_ids=tuple(finish),
        )
    )


def test_slot_constants_lay_out_the_same_metadata_as_each_calls_controls():
    vocab = 300
    table = RequestTable(slots=8, vocab=vocab)
    _start(table, 1, [2])
    _start(table, 2, [2, 5], temperature=0.7, top_k=20, seed=3)
    _start(table, 3, [], temperature=1.0, n_logprobs=4, top_p=0.5)
    _start(table, 4, [2], temperature=0.9, allowed_token_ids=(1, 2, 3))
    _start(table, 5, [2], forced_token_ids=(17, 18))
    _start(table, 6, range(250, 262), temperature=0.3, typical_p=0.9)
    _start(table, 7, [299], logit_bias=((4, 1.5),), return_logprobs=True)
    _start(table, 8, [2], temperature=0.6, top_k=200)
    slots = np.array([5, 1, 2, 3, 4, 6, 7, 8, 4])
    positions = np.arange(len(slots)) * 7 + 3
    rows = np.arange(len(slots)) + 10
    gates = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1])

    tasks = []
    for slot, position, row, gate in zip(slots, positions, rows, gates):
        controls, draws = controls_for_call(
            table.sampling[slot],
            None,
            table.finish_tokens[slot],
            [position],
            [],
            int(table.key[slot]),
        )
        tasks.append(
            SampleTask(table.sampling[slot], controls, draws, slot, row, gate)
        )
    built = SamplingMetadata.build(tasks, vocab, 16, 10_000)

    draws = task_draws(table, DrawWindow(8), slots, positions)
    gathered = SamplingMetadata.gather(
        table, slots, rows, gates, draws, 16, 10_000
    )
    for field in fields(SamplingMetadata):
        expected = getattr(built, field.name)
        actual = getattr(gathered, field.name)
        if field.name == "logprobs":
            assert np.array_equal(expected.rows, actual.rows)
            assert np.array_equal(expected.counts, actual.counts)
        elif isinstance(expected, np.ndarray):
            assert np.array_equal(expected, actual), field.name
            assert expected.shape == actual.shape, field.name
        else:
            assert expected == actual, field.name
    assert sampling_paths(table, slots).tolist() == built.paths.tolist()
    assert built.top_k and built.categorical and built.sorts


def test_padding_tasks_select_greedily_in_slot_zero_and_report_nothing():
    stochastic = SamplingParams(temperature=1.0, n_logprobs=2)
    tasks = [_task(stochastic, row, [0.5]) for row in range(3)]
    metadata = SamplingMetadata.build(tasks, 50, 8, 1000)
    padded = pad_selection(metadata, 8)
    # Three tasks pad to a bucket of four.
    assert padded.tasks == 4
    assert padded.rows.tolist() == [0, 1, 2, 0]
    assert padded.paths.tolist() == [CATEGORICAL] * 3 + [GREEDY]
    assert padded.slots.tolist() == [3, 3, 3, 0]
    assert padded.mask_rows.tolist() == [-1] * 4
    assert padded.params[3].tolist() == NEUTRAL.tolist()
    assert padded.draws.tolist() == [0.5, 0.5, 0.5, 0.0]
    # The reporting tasks pad to their bucket with task 0 reporting none.
    assert padded.logprob_tasks.tolist() == [0, 1, 2, 0]
    assert padded.logprob_counts.tolist() == [2, 2, 2, -1]

    # A verification selects eagerly at its own tasks.
    wide = _task(stochastic, 0, [0.1, 0.2])
    wide.controls.allowed = [None] * 2
    wide.controls.draft_token_ids = (4,)
    verification = SamplingMetadata.build([wide] * 3, 50, 8, 1000)
    assert pad_selection(verification, 8).tasks == 3

    idle = padding_selection(4)
    assert idle.paths.tolist() == [GREEDY] * 4
    assert idle.logprob_counts.tolist() == [-1] * 4
    assert idle.params.tolist() == [NEUTRAL.tolist()] * 4


def test_a_decode_samples_its_rows_and_pads_with_neutral_rows():
    top_k = SamplingParams(temperature=0.5, top_k=10)
    rows = RowSampling.build(
        [_task(top_k, 0, [0.25]), _task(SamplingParams(), 1, None)], 1000
    )
    draws, params = rows.padded(4)
    assert draws.tolist() == [0.25, 0.0, 0.0, 0.0]
    assert params[0, 7] == 10.0
    assert params[1:].tolist() == [NEUTRAL.tolist()] * 3

    # A decode forward samples only top-k and unsuppressed greedy rows.
    categorical = replace(top_k, typical_p=0.5)
    with pytest.raises(AssertionError):
        RowSampling.build([_task(categorical, 0, [0.25])], 1000)
    suppressed = _task(SamplingParams(), 0, None)
    suppressed.controls.suppress = (1,)
    with pytest.raises(AssertionError):
        RowSampling.build([suppressed], 1000)
