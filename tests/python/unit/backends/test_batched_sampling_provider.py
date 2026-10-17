"""Exactness and capture checks of the top-k sampler."""

from __future__ import annotations

import pytest
import torch

from uniserve.sampling import sample_top_k
from uniserve.sampling.select import select_categorical
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
    shape_rows,
)

pytestmark = pytest.mark.unit

CUDA = pytest.param(
    "cuda",
    marks=(
        pytest.mark.gpu,
        pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is required"
        ),
    ),
)


_COLUMNS = {
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "min_p": MIN_P,
    "repetition": REPETITION,
    "frequency": FREQUENCY,
    "presence": PRESENCE,
    "top_k": TOP_K,
}


def _params(rows: int, **columns: list[float]) -> torch.Tensor:
    """Parameter rows with neutral cutoffs and penalties, then ``columns``."""
    params = torch.zeros((rows, PARAMETER_COLUMNS), dtype=torch.float32)
    params[:, TOP_P] = 1.0
    params[:, TYPICAL_P] = 1.0
    params[:, REPETITION] = 1.0
    for name, values in columns.items():
        params[:, _COLUMNS[name]] = torch.tensor(values)
    return params


def _inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    logits = torch.tensor(
        [
            [0.2, 1.4, -0.6, 2.1, 0.7, 1.0],
            [1.6, -0.4, 0.8, 1.2, 0.1, 2.0],
            [0.2, 1.4, -0.6, 2.1, 0.7, 1.0],
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    draws = torch.tensor([0.25, 0.76, 0.9], dtype=torch.float32, device=device)
    # Row 0 samples its top 4, row 1 selects greedily, row 2 its top 2.
    params = _params(
        3,
        temperature=[0.7, 0.0, 1.0],
        top_p=[0.84, 0.92, 1.0],
        min_p=[0.08, 0.0, 0.0],
        top_k=[4.0, 0.0, 2.0],
    ).to(device)
    return logits, draws, params


def _no_counts(rows: int, vocab: int, device) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros((1, vocab), dtype=torch.int32, device=device),
        torch.zeros(rows, dtype=torch.int64, device=device),
    )


@pytest.mark.parametrize("device", ("cpu", CUDA))
def test_each_row_samples_from_its_own_top_k_candidates(device) -> None:
    logits, draws, params = _inputs(torch.device(device))

    tokens, valid = sample_top_k(
        logits, draws, params, 4, *_no_counts(3, 6, device)
    )

    assert valid.tolist() == [True, True, True]
    # Row 0 retains IDs 1, 3, 5 with probabilities about .234, .635, .131:
    # draw .25 selects ID 3. Row 1 selects its maximum at zero temperature.
    # Row 2 keeps IDs 1 and 3 with probabilities about .332 and .668 in
    # vocabulary order: draw .9 selects ID 3.
    assert tokens.tolist() == [3, 5, 3]


@pytest.mark.parametrize("device", ("cpu", CUDA))
@pytest.mark.parametrize(
    "columns",
    (
        pytest.param({"top_p": [0.9]}, id="top-p"),
        pytest.param({"min_p": [0.5]}, id="min-p"),
    ),
)
def test_top_k_zero_draw_skips_candidates_removed_by_filters(
    device, columns
) -> None:
    # The top-3 candidates are IDs 1, 2, and 0 with probabilities about .517,
    # .468, and .016. Top-p .9 and min-p .5 each drop ID 0, the lowest-ID
    # candidate, so a draw of exactly 0.0 must select ID 1.
    logits = torch.tensor(
        [[0.5, 4.0, 3.9, 0.0, -1.0, -2.0]],
        dtype=torch.float32,
        device=device,
    )
    draws = torch.zeros((1,), dtype=torch.float32, device=device)
    params = _params(1, temperature=[1.0], top_k=[3.0], **columns)

    tokens, valid = sample_top_k(
        logits, draws, params.to(device), 3, *_no_counts(1, 6, device)
    )

    assert valid.tolist() == [True]
    assert tokens.tolist() == [1]


def test_penalties_read_each_rows_slot_counts() -> None:
    # Both rows sample greedily over the same logits; only row 1's slot has
    # seen token 0, whose repetition penalty halves 1.0 to 0.5 < 0.6.
    logits = torch.tensor([[1.0, 0.6, -2.0], [1.0, 0.6, -2.0]])
    counts = torch.zeros((3, 3), dtype=torch.int32)
    counts[2, 0] = 1
    params = _params(2, repetition=[2.0, 2.0])

    tokens, _ = sample_top_k(
        logits,
        torch.zeros(2),
        params,
        3,
        counts,
        torch.tensor([1, 2]),
    )

    assert tokens.tolist() == [0, 1]


@pytest.mark.parametrize(
    "spread",
    (
        pytest.param(None, id="distinct-logits"),
        # Logits on a coarse grid, as low-precision logits are: equal values
        # straddle the top-k and nucleus cutoffs.
        pytest.param(0.5, id="tied-logits"),
    ),
)
def test_the_top_k_sampler_selects_what_the_general_path_selects(
    spread,
) -> None:
    # Random rows under every parameter the top-k sampler reads, sampled by
    # both selectors from the same draws.
    generator = torch.Generator().manual_seed(7)
    rows, vocab, width = 64, 300, 32
    logits = torch.randn((rows, vocab), generator=generator) * 3
    if spread is not None:
        logits = torch.round(logits / spread) * spread
    counts = torch.randint(0, 3, (rows, vocab), generator=generator)
    counts = counts.to(torch.int32)
    draws = torch.rand(rows, generator=generator)
    top_ks = torch.randint(1, width + 1, (rows,), generator=generator)
    params = _params(
        rows,
        temperature=(torch.rand(rows, generator=generator) * 1.5).tolist(),
        top_p=(0.5 + torch.rand(rows, generator=generator) / 2).tolist(),
        min_p=(torch.rand(rows, generator=generator) / 10).tolist(),
        repetition=(1 + torch.rand(rows, generator=generator)).tolist(),
        frequency=(torch.rand(rows, generator=generator) / 2).tolist(),
        presence=(torch.rand(rows, generator=generator) / 2).tolist(),
        top_k=top_ks.float().tolist(),
    )
    params[::4, TEMPERATURE] = 0.0

    tokens, valid = sample_top_k(
        logits, draws, params, width, counts, torch.arange(rows)
    )
    general = select_categorical(
        logits,
        draws,
        params,
        torch.ones(rows, dtype=torch.bool),
        row_counts=[1] * rows,
        drafts=[],
        terminal_prefixes=[0] * rows,
        candidates=width,
        top_ks=top_ks.tolist(),
        top_p_rows=[],
        min_p_rows=list(range(rows)),
        typical_p_rows=[],
        allowed_rows=[],
        allowed=[],
        suppressed=[],
        penalty_rows=list(range(rows)),
        penalty_counts=counts,
        bias=[],
        bias_values=[],
        logprob_tasks=[],
        logprob_requests=[],
        force_finish=[0] * rows,
        finish_ids=[-1] * rows,
        transition_ids=[-1] * rows,
    )

    assert valid.all()
    assert torch.equal(tokens, general.tokens)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_device_shaping_rounds_like_the_tensor_operations() -> None:
    # The fused CUDA pass and the separately rounded tensor operations of
    # the host produce identical bits, penalized or not.
    generator = torch.Generator().manual_seed(3)
    rows, vocab, slots = 16, 5000, 6
    logits = (torch.randn((rows, vocab), generator=generator) * 8).to(
        torch.bfloat16
    )
    logits[0, :7] = float("-inf")
    counts = torch.randint(0, 4, (slots, vocab), generator=generator)
    counts = counts.to(torch.int32)
    rows_slots = torch.randint(0, slots, (rows,), generator=generator)
    params = _params(
        rows,
        temperature=(torch.rand(rows, generator=generator) * 2).tolist(),
        repetition=(0.5 + torch.rand(rows, generator=generator)).tolist(),
        frequency=torch.randn(rows, generator=generator).tolist(),
        presence=torch.randn(rows, generator=generator).tolist(),
    )
    params[1::3, TEMPERATURE] = 0.0
    params[2::3, REPETITION] = 1.0
    params[2::3, FREQUENCY] = 0.0
    params[2::3, PRESENCE] = 0.0

    host = shape_rows(logits, params, counts, rows_slots)
    device = shape_rows(
        logits.cuda(), params.cuda(), counts.cuda(), rows_slots.cuda()
    )

    assert torch.equal(device.cpu(), host)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_a_captured_sampler_follows_its_parameter_tensors() -> None:
    logits, draws, params = _inputs(torch.device("cuda"))
    counts, slots = _no_counts(3, 6, "cuda")
    expected, _ = sample_top_k(logits, draws, params, 4, counts, slots)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        captured, valid = sample_top_k(logits, draws, params, 4, counts, slots)
    graph.replay()
    torch.cuda.synchronize()
    assert bool(valid.all())
    assert torch.equal(captured, expected)

    # New contents of the same tensors: every row greedy, then row 0 at a
    # temperature high enough that its draw selects the lowest candidate.
    params[:, TEMPERATURE] = 0.0
    graph.replay()
    torch.cuda.synchronize()
    assert captured.tolist() == [3, 5, 3]
    params[0, TEMPERATURE] = 50.0
    params[0, TOP_P] = 1.0
    params[0, MIN_P] = 0.0
    draws[0] = 0.0
    graph.replay()
    torch.cuda.synchronize()
    assert captured.tolist() == [1, 5, 3]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_top_k_sampler_accepts_every_serving_wave_row_count() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(42)

    for row_count in range(1, 17):
        logits = torch.randn(
            (row_count, 257),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        draws = torch.rand(
            (row_count,),
            dtype=torch.float32,
            device=device,
            generator=generator,
        ).clamp_(1e-7, 1.0 - 1e-7)
        params = _params(row_count, top_k=[1.0] * row_count).to(device)

        tokens, valid = sample_top_k(
            logits, draws, params, 128, *_no_counts(row_count, 257, device)
        )

        assert bool(valid.all())
        assert torch.equal(tokens, torch.argmax(logits, dim=-1))
