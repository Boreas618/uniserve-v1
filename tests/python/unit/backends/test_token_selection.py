"""Token selection, prompt scoring and slot-state commit at the library API."""

from __future__ import annotations

import pytest
import torch

from uniserve.sampling.logprobs import LogprobRequest, prompt_logprobs
from uniserve.sampling.select import (
    TOKEN_CONTINUATION_BIT,
    select_categorical,
    select_greedy,
    select_top_k,
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


def _select(logits, params, **controls):
    """Select over ``logits`` rows with neutral host controls.

    The rows are one task's unless ``row_counts`` divides them into tasks.
    """
    rows = int(logits.shape[0])
    tasks = len(controls.get("row_counts", [rows]))
    neutral = {
        "row_counts": [rows],
        "drafts": [],
        "terminal_prefixes": [0] * tasks,
        "candidates": int(logits.shape[1]),
        "top_ks": [0] * rows,
        "top_p_rows": [],
        "min_p_rows": [],
        "typical_p_rows": [],
        "allowed_rows": [],
        "allowed": [],
        "suppressed": [],
        "penalty_rows": [],
        "penalty_counts": None,
        "bias": [],
        "bias_values": [],
        "logprob_tasks": [],
        "logprob_requests": [],
        "force_finish": [0] * tasks,
        "finish_ids": [-1] * tasks,
        "transition_ids": [-1] * tasks,
    }
    return select_categorical(
        logits,
        torch.zeros(rows, device=logits.device),
        params,
        torch.ones(tasks, dtype=torch.bool, device=logits.device),
        **{**neutral, **controls},
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
    counts = torch.zeros(len(logits), dtype=torch.int32)
    counts[0] = count

    selection = _select(
        torch.tensor([logits]),
        params,
        penalty_rows=[0],
        penalty_counts=counts.unsqueeze(0),
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
    return _select(
        logits,
        _params(3),
        drafts=list(drafts),
        terminal_prefixes=[terminal_prefix or 0],
        finish_ids=list(finish_ids) or [-1],
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
    # task 2 (rows 4-5) 0, bonus 4; task 3 (rows 6-7) 1, bonus 3.
    logits = _targets((3, 1, 4, 2, 0, 4, 1, 3))

    selection = _select(
        logits,
        _params(8),
        row_counts=[3, 1, 2, 2],
        # Task 0 mismatches its second draft, task 2 matches its draft and
        # task 3 matches its draft, which is its finish token.
        drafts=[3, 2, 0, 1],
        terminal_prefixes=[0, 0, 0, 1],
        finish_ids=[-1, -1, -1, 1],
        logprob_tasks=[0, 2, 3],
        logprob_requests=[LogprobRequest(count=0, token_ids=())] * 3,
    )

    assert selection.accepted.tolist() == [1, 0, 1, 1]
    assert selection.points.tolist() == [2, 1, 2, 1]
    assert selection.tokens.tolist() == [1, 2, 4, 1]
    assert selection.finish.tolist() == [False, False, False, True]
    assert selection.valid.tolist() == [True] * 4
    # Task 0's correction comes from row 1, task 2's bonus token from row 5
    # and task 3's finish draft from row 6, the row before its output row.
    packed = selection.logprobs
    assert packed is not None
    expected = [
        torch.log_softmax(logits[row], dim=-1)[token].item()
        for row, token in ((1, 1), (5, 4), (6, 1))
    ]
    assert _selected_logprobs(packed, 3) == pytest.approx(expected, rel=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_selections_queue_behind_the_forward_without_synchronizing() -> None:
    # The eager selection runs behind the forward that produced its logits;
    # a synchronizing operation would hold the host until that forward
    # finished and leave the device idle until the next step is queued.
    device = torch.device("cuda")
    logits = _targets((3, 1, 4, 2, 0, 4)).to(device)
    params = _params(6, temperature=0.7).to(device)
    params[:, TOP_P] = 0.9
    params[:, MIN_P] = 0.05
    params[:, TYPICAL_P] = 0.95
    params[:3, TOP_K] = 2.0
    counts = torch.zeros((2, 5), dtype=torch.int32, device=device)
    counts[:, 1] = 1
    tasks = torch.ones(2, dtype=torch.bool, device=device)

    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        general = _select(
            logits,
            params,
            row_counts=[3, 1, 2],
            drafts=[3, 2, 0],
            terminal_prefixes=[0, 0, 0],
            top_ks=[2, 2, 2, 0, 0, 0],
            top_p_rows=[3, 4, 5],
            min_p_rows=[0, 3],
            typical_p_rows=[0, 1, 2, 3, 4, 5],
            allowed_rows=[3],
            allowed=[3 * 5 + 2, 3 * 5 + 4],
            suppressed=[4 * 5 + 1],
            penalty_rows=[0, 4],
            penalty_counts=counts,
            bias=[5 * 5 + 4],
            bias_values=[0.5],
            logprob_tasks=[0, 2],
            logprob_requests=[
                LogprobRequest(count=2, token_ids=(1, 3)),
                LogprobRequest(count=2, token_ids=(0,)),
            ],
        )
        greedy = select_greedy(
            logits[:2],
            tasks,
            suppressed=[0 * 5 + 3],
            force_finish=[0, 1],
            finish_ids=[0, -1],
            transition_ids=[-1, -1],
        )
        top_k = select_top_k(
            logits[:2],
            torch.zeros(2, device=device),
            params[:2],
            tasks,
            4,
            counts=counts,
            slots=torch.arange(2, device=device),
            force_finish=[0, 0],
            finish_ids=[-1, -1],
            transition_ids=[-1, -1],
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")

    # The selections are complete once the stream reaches them.
    torch.cuda.synchronize()
    assert general.logprobs is not None
    assert general.valid.all()
    assert greedy.tokens.tolist() == [0, 1]
    assert greedy.finish.tolist() == [True, True]
    assert top_k.valid.all()


def test_reported_logprobs_score_the_unshaped_distribution() -> None:
    # A top-k cutoff of two at temperature one half selects among tokens 0
    # and 1; the four reported tokens keep the model's log probabilities.
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    params = _params(1, temperature=0.5)

    selection = _select(
        logits,
        params,
        top_ks=[2],
        logprob_tasks=[0],
        logprob_requests=[LogprobRequest(count=4, token_ids=())],
    )

    assert selection.tokens.tolist() == [0]
    packed = selection.logprobs
    assert packed is not None
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
