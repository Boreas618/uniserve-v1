"""Token selection, prompt scoring and slot-state commit at the library API."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from uniserve.sampling.logprobs import logprob_details, scatter_logprobs
from uniserve.sampling.select import (
    CATEGORICAL,
    GREEDY,
    TOKEN_CONTINUATION_BIT,
    TOP_K_PATH,
    Controls,
    select,
)
from uniserve.sampling.state import (
    COMMIT_FIELDS,
    DECODE,
    EXPLICIT,
    VERIFY,
    commit,
)
from uniserve.sampling.top_k import (
    FREQUENCY,
    MIN_P,
    PARAMETER_COLUMNS,
    PRESENCE,
    REPETITION,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    TYPICAL_P,
)

pytestmark = pytest.mark.unit


def _sets(tokens: list[set[int]], vocab: int) -> torch.Tensor:
    """``[n, words]`` int32 bitmasks whose row ``i`` holds ``tokens[i]``."""
    words = torch.zeros((len(tokens), (vocab + 31) // 32), dtype=torch.int64)
    for row, members in enumerate(tokens):
        for token in members:
            words[row, token // 32] |= 1 << (token % 32)
    # Bit 31 of a word is the int32 sign bit.
    return torch.where(words >= 1 << 31, words - (1 << 32), words).to(
        torch.int32
    )


def _controls(
    rows: list[list[int]],
    params: torch.Tensor,
    vocab: int,
    *,
    paths: list[int] | None = None,
    draws: torch.Tensor | None = None,
    slots: list[int] | None = None,
    allowed: dict[int, set[int]] | None = None,
    biased: list[bool] | None = None,
    force: list[bool] | None = None,
    finish: list[set[int]] | None = None,
    transitions: list[set[int]] | None = None,
    drafts: list[list[int]] | None = None,
    terminal: list[int] | None = None,
) -> Controls:
    """The controls of tasks whose candidate rows read ``rows`` of the logits.

    Unless given, every task is categorical, active, unbiased and unforced
    in slot zero, without finish or transition tokens or drafts, and its
    candidate rows draw zero and may select any token. ``allowed`` maps a
    candidate row, counted task by task, to the tokens it may select.
    """
    tasks, width = len(rows), len(rows[0])
    allowed = allowed or {}
    masked = list(allowed)
    return Controls(
        rows=torch.tensor(rows),
        paths=torch.tensor(paths or [CATEGORICAL] * tasks),
        params=params,
        draws=torch.zeros(tasks * width) if draws is None else draws,
        slots=torch.tensor(slots or [0] * tasks),
        active=torch.ones(tasks, dtype=torch.bool),
        masks=_sets(list(allowed.values()) or [set()], vocab),
        mask_rows=torch.tensor(
            [
                masked.index(row) if row in allowed else -1
                for row in range(tasks * width)
            ]
        ),
        biased=torch.tensor(biased or [False] * tasks),
        force=torch.tensor(force or [False] * tasks),
        finish=_sets(finish or [set()] * tasks, vocab),
        transitions=_sets(transitions or [set()] * tasks, vocab),
        drafts=torch.tensor(
            drafts or [[-1] * (width - 1)] * tasks, dtype=torch.int64
        ).reshape(tasks, width - 1),
        terminal=torch.tensor(terminal or [0] * tasks),
    )


def _on(controls: Controls, device: torch.device) -> Controls:
    return Controls(
        **{
            field.name: getattr(controls, field.name).to(device)
            for field in dataclasses.fields(Controls)
        }
    )


def _select(
    logits: torch.Tensor,
    controls: Controls,
    *,
    top_k: bool = False,
    categorical: bool = True,
    sorts: bool = False,
    counts: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
):
    """Select over every candidate, with no counts or biases unless given."""
    slots, vocab = int(controls.slots.max()) + 1, int(logits.shape[1])
    return select(
        logits,
        controls,
        candidates=vocab,
        top_k=top_k,
        categorical=categorical,
        sorts=sorts,
        counts=(
            torch.zeros((slots, vocab), dtype=torch.int32)
            if counts is None
            else counts
        ),
        bias=torch.zeros((slots, vocab)) if bias is None else bias,
    )


def _params(rows: int, temperature: float = 0.0) -> torch.Tensor:
    """Per-row sampling parameters with neutral cutoffs and penalties."""
    params = torch.zeros((rows, PARAMETER_COLUMNS), dtype=torch.float32)
    params[:, TEMPERATURE] = temperature
    params[:, TOP_P] = 1.0
    params[:, TYPICAL_P] = 1.0
    params[:, REPETITION] = 1.0
    return params


@pytest.mark.parametrize(
    ("logits", "count", "column", "penalty", "selected"),
    [
        # Repetition divides a positive logit: 1.0 -> 0.5 < 0.6.
        ((1.0, 0.6, -2.0), 1, REPETITION, 2.0, 1),
        # Repetition multiplies a negative logit: -0.5 -> -1.0 < -0.8.
        ((-0.5, -0.8, -3.0), 1, REPETITION, 2.0, 1),
        # Frequency scales with the count: 2.0 - 3 * 0.2 = 1.4 < 1.5.
        ((2.0, 1.5, -1.0), 3, FREQUENCY, 0.2, 1),
        ((2.0, 1.5, -1.0), 1, FREQUENCY, 0.2, 0),
        # Presence is flat in the count: 2.0 - 0.6 = 1.4 < 1.5.
        ((2.0, 1.5, -1.0), 5, PRESENCE, 0.6, 1),
    ],
)
def test_penalties_shape_the_seen_tokens_before_selection(
    logits: tuple[float, ...],
    count: int,
    column: int,
    penalty: float,
    selected: int,
) -> None:
    params = _params(1)
    params[0, column] = penalty
    counts = torch.zeros((1, len(logits)), dtype=torch.int32)
    counts[0, 0] = count

    selection = _select(
        torch.tensor([logits]),
        _controls([[0]], params, len(logits)),
        counts=counts,
    )

    assert selection.tokens.tolist() == [selected]
    assert selection.valid.tolist() == [True]


def test_a_task_selects_what_its_sampler_selects_alone() -> None:
    # Every sampler runs over every task of a call; each task keeps its own
    # sampler's token, as a call of that sampler alone selects it.
    generator = torch.Generator().manual_seed(11)
    tasks, vocab = 12, 64
    logits = torch.randn((tasks, vocab), generator=generator) * 2
    draws = torch.rand(tasks, generator=generator)
    params = _params(tasks, temperature=0.8)
    params[:, TOP_K] = 8.0
    params[:, TOP_P] = 0.9
    paths = [GREEDY, TOP_K_PATH, CATEGORICAL] * (tasks // 3)
    rows = [[row] for row in range(tasks)]

    mixed = _select(
        logits,
        _controls(rows, params, vocab, paths=paths, draws=draws),
        top_k=True,
    )
    for path, top_k, categorical in (
        (GREEDY, False, False),
        (TOP_K_PATH, True, False),
        (CATEGORICAL, False, True),
    ):
        alone = _select(
            logits,
            _controls(rows, params, vocab, paths=[path] * tasks, draws=draws),
            top_k=top_k,
            categorical=categorical,
        )
        chosen = torch.tensor(paths) == path
        assert torch.equal(mixed.tokens[chosen], alone.tokens[chosen])
        assert torch.equal(mixed.valid[chosen], alone.valid[chosen])


@pytest.mark.parametrize("path", (GREEDY, CATEGORICAL))
def test_a_masked_row_selects_only_tokens_its_set_holds(path: int) -> None:
    # Token 95 is bit 31 of the third word, the int32 sign bit. Row 0 may
    # select tokens 37 and 95, row 1 any token.
    vocab = 100
    logits = torch.zeros((2, vocab))
    logits[:, 5] = 4.0
    logits[:, 37] = 1.0
    logits[:, 95] = 2.0

    selection = _select(
        logits,
        _controls(
            [[0], [1]],
            _params(2),
            vocab,
            paths=[path, path],
            allowed={0: {37, 95}},
        ),
        categorical=path == CATEGORICAL,
    )

    assert selection.tokens.tolist() == [95, 5]
    assert selection.valid.tolist() == [True, True]


def test_finish_and_transition_sets_decide_each_task() -> None:
    # Each task selects token 3. Task 0 finishes on it, task 1 transitions
    # on it, task 2 is forced to finish and task 3 continues.
    logits = torch.zeros((4, 40))
    logits[:, 3] = 1.0

    selection = _select(
        logits,
        _controls(
            [[0], [1], [2], [3]],
            _params(4),
            40,
            finish=[{3, 33}, {33}, set(), set()],
            transitions=[set(), {3}, set(), set()],
            force=[False, False, True, False],
        ),
    )

    assert selection.tokens.tolist() == [3, 3, 3, 3]
    assert selection.finish.tolist() == [True, False, True, False]
    assert selection.continuation.tolist() == [False, True, False, True]
    assert selection.transition.tolist() == [False, True, False, False]
    assert selection.tagged.tolist() == [
        3,
        3 | TOKEN_CONTINUATION_BIT,
        3,
        3 | TOKEN_CONTINUATION_BIT,
    ]


def test_a_biased_task_adds_its_slots_bias_row() -> None:
    # Both tasks read the same logits; slot 1's bias row lifts token 2
    # above token 0, and only the biased task adds it.
    logits = torch.tensor([[1.0, 0.0, 0.5], [1.0, 0.0, 0.5]])
    bias = torch.zeros((2, 3))
    bias[1, 2] = 1.0

    selection = _select(
        logits,
        _controls(
            [[0], [1]], _params(2), 3, slots=[1, 1], biased=[True, False]
        ),
        bias=bias,
    )

    assert selection.tokens.tolist() == [2, 0]


def _verify(
    drafts: tuple[int, ...],
    *,
    finish_ids: tuple[int, ...] = (),
    terminal_prefix: int = 0,
):
    # Greedy targets per verification row: 3, 1, then the bonus row's 4.
    logits = torch.full((3, 5), -1.0)
    for row, token in enumerate((3, 1, 4)):
        logits[row, token] = 2.0
    return _select(
        logits,
        _controls(
            [[0, 1, 2]],
            _params(3),
            5,
            drafts=[list(drafts)],
            terminal=[terminal_prefix],
            finish=[set(finish_ids)],
        ),
    )


def test_verification_emits_the_correction_after_the_matching_prefix() -> None:
    selection = _verify((3, 2))

    assert selection.accepted.tolist() == [1]
    assert selection.points.tolist() == [2]
    assert selection.tokens.tolist() == [1]
    assert selection.continuation.tolist() == [True]
    assert selection.tagged.tolist() == [1 | TOKEN_CONTINUATION_BIT]


def test_verification_emits_the_bonus_token_after_a_full_match() -> None:
    selection = _verify((3, 1))

    assert selection.accepted.tolist() == [2]
    assert selection.points.tolist() == [3]
    assert selection.tokens.tolist() == [4]


def test_an_accepted_terminal_draft_finishes_without_a_bonus_token() -> None:
    selection = _verify((3, 1), finish_ids=(3,), terminal_prefix=1)

    assert selection.accepted.tolist() == [1]
    assert selection.points.tolist() == [1]
    assert selection.tokens.tolist() == [3]
    assert selection.finish.tolist() == [True]
    assert selection.continuation.tolist() == [False]
    assert selection.tagged.tolist() == [3]


def _targets(tokens: tuple[int, ...]) -> torch.Tensor:
    """Logits rows whose greedy selection is ``tokens``.

    Each row's target logit is distinct, so a reported score names its row.
    """
    logits = torch.full((len(tokens), 5), -1.0)
    for row, token in enumerate(tokens):
        logits[row, token] = 2.0 + 0.5 * row
    return logits


def test_verifications_of_different_draft_counts_share_one_call() -> None:
    # Greedy targets: task 0 (rows 0-2) 3, 1, bonus 4; task 1 (row 3) 2;
    # task 2 (rows 4-5) 0, bonus 4; task 3 (rows 6-7) 1, bonus 3. Each task
    # pads to three candidate rows by repeating its first row.
    logits = _targets((3, 1, 4, 2, 0, 4, 1, 3))

    selection = _select(
        logits,
        _controls(
            [[0, 1, 2], [3, 3, 3], [4, 5, 4], [6, 7, 6]],
            _params(12),
            5,
            # Task 0 mismatches its second draft, task 2 matches its draft
            # and task 3 matches its draft, which is its finish token.
            drafts=[[3, 2], [-1, -1], [0, -1], [1, -1]],
            terminal=[0, 0, 0, 1],
            finish=[set(), set(), set(), {1}],
        ),
    )

    assert selection.accepted.tolist() == [1, 0, 1, 1]
    assert selection.points.tolist() == [2, 1, 2, 1]
    assert selection.tokens.tolist() == [1, 2, 4, 1]
    assert selection.finish.tolist() == [False, False, False, True]
    assert selection.valid.tolist() == [True] * 4
    # Task 0's correction comes from row 1, task 2's bonus token from row 5
    # and task 3's finish draft from row 6, the row before its output row.
    assert selection.scored.tolist() == [1, 3, 5, 6]
    reporting = torch.tensor([0, 2, 3])
    packed = logprob_details(
        logits,
        selection.scored[reporting],
        selection.tokens[reporting],
        0,
    )
    expected = [
        torch.log_softmax(logits[row], dim=-1)[token].item()
        for row, token in ((1, 1), (5, 4), (6, 1))
    ]
    assert _selected_logprobs(packed, 3) == pytest.approx(expected, rel=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_selections_queue_behind_the_forward_without_synchronizing() -> None:
    # The selection runs behind the forward that produced its logits; a
    # synchronizing operation would hold the host until that forward
    # finished and leave the device idle until the next step is queued.
    device = torch.device("cuda")
    logits = _targets((3, 1, 4, 2, 0, 4)).to(device)
    # Verification tasks over rows 0-2, row 3 and rows 4-5, every shaping
    # stage on, a token set on task 1 and task 2, and task 2 biased.
    params = _params(9, temperature=0.7)
    params[:, TOP_P] = 0.9
    params[:, MIN_P] = 0.05
    params[:, TYPICAL_P] = 0.95
    params[:3, TOP_K] = 2.0
    general = _on(
        _controls(
            [[0, 1, 2], [3, 3, 3], [4, 5, 4]],
            params,
            5,
            draws=torch.rand(9),
            slots=[0, 1, 2],
            allowed={
                **dict.fromkeys((3, 4, 5), {2, 4}),
                **dict.fromkeys((6, 7, 8), {0, 2, 3, 4}),
            },
            biased=[False, False, True],
            drafts=[[3, 2], [-1, -1], [0, -1]],
        ),
        device,
    )
    # Two greedy tasks, one masked and finishing on its token, one forced to
    # finish, then two top-k tasks over the same rows.
    sampled = _params(4, temperature=0.7)
    sampled[:, TOP_K] = 2.0
    sampled[:, TOP_P] = 0.9
    decode = _on(
        _controls(
            [[0], [1], [0], [1]],
            sampled,
            5,
            paths=[GREEDY, GREEDY, TOP_K_PATH, TOP_K_PATH],
            draws=torch.rand(4),
            slots=[0, 1, 0, 1],
            allowed={0: {0, 1, 2, 4}},
            force=[False, True, False, False],
            finish=[{0}, set(), set(), set()],
        ),
        device,
    )
    counts = torch.zeros((3, 5), dtype=torch.int32, device=device)
    counts[:, 1] = 1
    bias = torch.zeros((3, 5), device=device)
    bias[2, 4] = 0.5
    reporting = torch.tensor([0, 2], device=device)

    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        verified = select(
            logits,
            general,
            candidates=4,
            top_k=False,
            categorical=True,
            sorts=True,
            counts=counts,
            bias=bias,
        )
        packed = logprob_details(
            logits.float(),
            verified.scored.index_select(0, reporting),
            verified.tokens.index_select(0, reporting),
            2,
        )
        decoded = select(
            logits,
            decode,
            candidates=4,
            top_k=True,
            categorical=False,
            sorts=False,
            counts=counts,
            bias=bias,
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")

    # The selections are complete once the stream reaches them.
    torch.cuda.synchronize()
    assert verified.valid.all()
    assert int(packed.numel()) == 2 * (3 + 3 * 2)
    assert decoded.tokens[:2].tolist() == [0, 1]
    assert decoded.finish[:2].tolist() == [True, True]
    assert decoded.valid.all()


def test_reported_logprobs_score_the_unshaped_distribution() -> None:
    # A top-k cutoff of two at temperature one half selects among tokens 0
    # and 1; the four reported tokens keep the model's log probabilities.
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    params = _params(1, temperature=0.5)
    params[0, TOP_K] = 2.0

    selection = _select(logits, _controls([[0]], params, 4))

    assert selection.tokens.tolist() == [0]
    packed = logprob_details(logits, selection.scored, selection.tokens, 4)
    # Selected token, score and rank, then four top ids, scores and ranks.
    expected = torch.log_softmax(logits[0], dim=-1).tolist()
    assert packed[3:7].tolist() == [0, 1, 2, 3]
    scores = packed[7:11].to(torch.int32).view(torch.float32).tolist()
    assert scores == pytest.approx(expected, rel=1e-6)
    assert _selected_logprobs(packed, 1) == pytest.approx(
        expected[:1], rel=1e-6
    )


def _selected_logprobs(packed: torch.Tensor, rows: int) -> list[float]:
    """Selected-token scores: the packed column's second section."""
    bits = packed[rows : 2 * rows].to(torch.int32)
    return bits.view(torch.float32).tolist()


def _layout(*fields: list[int]) -> tuple[torch.Tensor, ...]:
    """The ``bases, index, rows, prefix, total, counts`` of rows."""
    return tuple(torch.tensor(values) for values in fields)


def _fields(packed: torch.Tensor, rows: int, top: int) -> list[torch.Tensor]:
    """The fields of ``logprob_details`` words, in packed order.

    The selected tokens, scores and ranks, then the top ids, scores and
    ranks, ``[rows, top]`` each.
    """
    selected = [packed[field * rows : (field + 1) * rows] for field in range(3)]
    tops = packed[3 * rows :].view(3, rows, top)
    return [*selected, *tops]


def test_reporting_tasks_pack_their_own_top_counts_compactly() -> None:
    # Three tasks report two, no and one top tokens; a padding row reports
    # nothing. The block holds the three rows' selected fields, then each
    # row's own top tokens in turn; the padding row's words go to the dump.
    torch.manual_seed(1)
    vocab, top = 6, 2
    logits = torch.randn(4, vocab)
    targets = torch.tensor([1, 2, 3, 4])
    rows = torch.arange(4)
    dump = 18
    words = torch.full((dump + 1,), -7, dtype=torch.long)

    scatter_logprobs(
        logits,
        rows,
        targets,
        _layout(
            [0] * 4, [0, 1, 2, 3], [3] * 4, [0, 2, 2, 3], [3] * 4, [2, 0, 1, -1]
        ),
        top,
        words,
        dump,
    )

    fields = _fields(logprob_details(logits, rows, targets, top), 4, top)
    selected = [field[:3] for field in fields[:3]]
    tops = [torch.cat((field[0, :2], field[2, :1])) for field in fields[3:]]
    assert words[:dump].tolist() == torch.cat(selected + tops).tolist()


def test_prompt_rows_score_into_their_chunks_words() -> None:
    # Rows 0-1 score chunk A, two top tokens each, from word 0; rows 2-3
    # score chunk B, one top token each, from word 18; row 4 has no target
    # and scores into the dump. Each chunk's words are its rows' compact
    # words at its count.
    torch.manual_seed(0)
    vocab, top = 6, 2
    logits = torch.randn(5, vocab)
    targets = torch.tensor([2, 3, 0, 5, 1])
    dump = 30
    words = torch.full((dump + 1,), -7, dtype=torch.long)

    scatter_logprobs(
        logits,
        torch.arange(5),
        targets,
        _layout(
            [0, 0, 18, 18, dump],
            [0, 1, 0, 1, 0],
            [2, 2, 2, 2, 1],
            [0, 2, 0, 1, 0],
            [4, 4, 2, 2, 0],
            [2, 2, 1, 1, -1],
        ),
        top,
        words,
        dump,
    )

    rows = torch.arange(2)
    first = logprob_details(logits[:2], rows, targets[:2], 2)
    second = logprob_details(logits[2:4], rows, targets[2:4], 1)
    assert words[:18].tolist() == first.tolist()
    assert words[18:dump].tolist() == second.tolist()
    assert _selected_logprobs(words[:18], 2) == pytest.approx(
        [
            torch.log_softmax(logits[row], dim=-1)[token].item()
            for row, token in ((0, 2), (1, 3))
        ],
        rel=1e-6,
    )


def test_commit_advances_decode_explicit_and_verify_slots() -> None:
    slots, vocab = 4, 8
    state = {
        "future_input_tokens": torch.zeros((slots, 2), dtype=torch.int64),
        "predicates": torch.zeros(slots, dtype=torch.uint8),
        "logical_lengths": torch.tensor([0, 10, 0, 0], dtype=torch.int64),
        "sampling_positions": torch.tensor([0, 4, 0, 0], dtype=torch.int64),
        "verified_lengths": torch.tensor([0, 9, 0, 0], dtype=torch.int32),
        "penalty_counts": torch.zeros((slots, vocab), dtype=torch.int32),
    }
    # Tasks: decode continues with 5; the prefill finishes with 6; the
    # verification emits 7 after three points.
    selection = {
        "tokens": torch.tensor([5 | TOKEN_CONTINUATION_BIT, 6, 7]),
        "valid": torch.ones(3, dtype=torch.bool),
        "active": torch.ones(3, dtype=torch.bool),
        "continuation": torch.tensor([True, False, True]),
        "points": torch.tensor([1, 1, 3]),
    }
    columns = torch.zeros((COMMIT_FIELDS, 3), dtype=torch.int64)
    # Rows: [slot, mode, task, cache, logical, sampling, visible, penalties].
    columns[:, 0] = torch.tensor([1, DECODE, 0, 0, 0, 0, 0, 1])
    columns[:, 1] = torch.tensor([2, EXPLICIT, 1, 12, 12, 0, 0, 0])
    columns[:, 2] = torch.tensor([3, VERIFY, 2, 0, 20, 2, 19, 1])

    commit(columns, 3, **selection, **state)

    assert state["future_input_tokens"][1:, 0].tolist() == [5, 6, 7]
    assert state["predicates"][1:].tolist() == [1, 0, 1]
    assert state["logical_lengths"][1:].tolist() == [11, 12, 23]
    assert state["sampling_positions"][1:].tolist() == [5, 0, 5]
    assert state["verified_lengths"][1:].tolist() == [10, 12, 22]
    counted = state["penalty_counts"]
    assert counted[1, 5].item() == 1
    assert counted[2].sum().item() == 0
    assert counted[3, 7].item() == 1
    assert counted.sum().item() == 2
