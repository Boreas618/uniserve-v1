"""Token selection, prompt scoring and slot-state commit at the library API."""

from __future__ import annotations

import pytest
import torch

from uniserve_models.sampling.logprobs import LogprobRequest, prompt_logprobs
from uniserve_models.sampling.select import (
    FREQUENCY,
    PARAMETER_COLUMNS,
    PRESENCE,
    REPETITION,
    TEMPERATURE,
    TOKEN_CONTINUATION_BIT,
    TOP_P,
    TYPICAL_P,
    Task,
    select_categorical,
)
from uniserve_models.sampling.state import (
    COMMIT_FIELDS,
    DECODE,
    EXPLICIT,
    VERIFY,
    commit,
)

pytestmark = pytest.mark.unit


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
    counts = torch.zeros(len(logits), dtype=torch.int32)
    counts[0] = count

    selection = select_categorical(
        [Task(row=0, rows=1, slot=1, penalties=True)],
        torch.tensor([logits]),
        torch.zeros(1),
        params,
        torch.ones(1, dtype=torch.bool),
        [counts],
    )

    assert selection.tokens.tolist() == [selected]
    assert selection.valid.tolist() == [True]


def _verify(
    drafts: tuple[int, ...],
    *,
    finish_ids: tuple[int, ...] = (),
    terminal_prefix: int | None = None,
):
    # Greedy targets per verification row: 3, 1, then the bonus row's 4.
    logits = torch.full((3, 5), -1.0)
    for row, token in enumerate((3, 1, 4)):
        logits[row, token] = 2.0
    task = Task(
        row=0,
        rows=3,
        slot=1,
        drafts=drafts,
        finish_ids=finish_ids,
        terminal_prefix=terminal_prefix,
        allowed=(None, None, None),
    )
    return select_categorical(
        [task],
        logits,
        torch.zeros(3),
        _params(3),
        torch.ones(1, dtype=torch.bool),
        [None, None, None],
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


def _selected_logprobs(packed: torch.Tensor, rows: int) -> list[float]:
    """Selected-token scores: the packed column's second section."""
    bits = packed[rows : 2 * rows].to(torch.int32)
    return bits.view(torch.float32).tolist()


def test_continued_prompt_chunk_scores_first_token_from_retained_logits() -> (
    None
):
    torch.manual_seed(0)
    tokens = torch.tensor([1, 2, 3, 0])
    logits = torch.randn(4, 6)
    request = LogprobRequest(count=0, token_ids=())

    # A first chunk cannot score its first token.
    retained, packed = prompt_logprobs(tokens[:2], logits[:2], None, request)
    assert packed is not None
    expected = torch.log_softmax(logits[0], dim=-1)[2].item()
    assert _selected_logprobs(packed, 1) == pytest.approx([expected], rel=1e-6)

    # The continuation's first token is predicted by the retained row.
    _, packed = prompt_logprobs(tokens[2:], logits[2:], retained, request)
    assert packed is not None
    expected = [
        torch.log_softmax(logits[1], dim=-1)[3].item(),
        torch.log_softmax(logits[2], dim=-1)[0].item(),
    ]
    assert _selected_logprobs(packed, 2) == pytest.approx(expected, rel=1e-6)


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
