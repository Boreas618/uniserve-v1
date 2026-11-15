"""The logprob_gather program reads one token's log-probability and rank.

Per row, the program reads the chosen token's log-probability out of the FP32
row ``log_softmax`` produced, and its competition rank: one plus the entries
strictly above it, tied entries sharing a rank. Values move and ranks count,
so the comparison is exact; a negative token (a prompt row the batch does not
score) scores nothing, with rank zero.

The cases: the vocabulary row with tokens spread across it, at its first and
last column and the same token in two rows; a dropped candidate's ``-inf``,
gathered as it stands rather than repaired; tied entries sharing a rank, in a
row the step does not divide; and an unscored row.

A module compiled for a dependent launch reads only what the grid before it
wrote (:mod:`uniserve_testing.kernels.chain`); that launch needs a device.
"""

import pytest
import torch
from uniserve_testing.kernels import chain

pytestmark = pytest.mark.unit

#: The fixture's vocabulary row.
COLUMNS = 2048

#: The cases: rows, columns, the tokens, whether the edges are taken, a
#: masked entry, ties and an unscored row, the step one program counts the
#: rank in, and the seed.
CASES = {
    "the vocabulary row, tokens spread across it": (
        4,
        COLUMNS,
        (0, COLUMNS // 2, 1, 3),
        {"edges"},
        1024,
        1,
    ),
    "a dropped candidate's -inf, gathered as it stands": (
        2,
        64,
        (63, 5),
        {"masked"},
        64,
        2,
    ),
    "tied entries share a rank, a row of a step the width does not divide": (
        3,
        100,
        (7, 40, 99),
        {"ties"},
        32,
        3,
    ),
    "an unscored row reports nothing": (2, 64, (3, 9), {"unscored"}, 64, 4),
}


def launch(kernels, logprobabilities, tokens, out, ranks, step, pdl=False):
    """One launch: a program per row; compiled for a dependent launch when
    `pdl` is.
    """  # noqa: D205
    rows, columns = logprobabilities.shape
    kernels._logprob_gather_kernel[(rows,)](
        logprobabilities,
        tokens,
        out,
        ranks,
        rows,
        columns,
        STEP=step,
        PDL=pdl,
        launch_pdl=pdl,
    )


@pytest.mark.parametrize("case", CASES)
def test_every_row_reports_its_token_and_rank(
    case, torch_device, kernel_source
):
    rows, columns, chosen, shape, step, seed = CASES[case]
    generator = torch.Generator().manual_seed(seed)
    logprobabilities = -torch.rand(rows, columns, generator=generator)
    if "masked" in shape:
        logprobabilities[0, columns - 1] = float("-inf")
    tokens = torch.tensor(chosen[:rows], dtype=torch.int32)
    if "edges" in shape:
        # The first and the last column, and one token in two rows.
        tokens[0] = 0
        tokens[-1] = columns - 1
        if rows > 2:
            tokens[1] = columns - 1
    if "ties" in shape:
        # The token's value repeated beside it and once above it: tied
        # entries share the rank of the first, so only the larger counts.
        logprobabilities[0, 1] = logprobabilities[0, tokens[0]]
        logprobabilities[0, 2] = 0.0
    if "unscored" in shape:
        tokens[-1] = -1
    out = torch.full((rows,), float("nan"), device=torch_device)
    ranks = torch.full((rows,), -7, dtype=torch.int32, device=torch_device)

    launch(
        kernel_source("logprob_gather"),
        logprobabilities.to(torch_device),
        tokens.to(torch_device),
        out,
        ranks,
        step,
    )

    scored = tokens >= 0
    expected = torch.where(
        scored,
        logprobabilities[torch.arange(rows), tokens.clamp(min=0).long()],
        0.0,
    )
    expected_ranks = torch.where(
        scored,
        1 + (logprobabilities > expected[:, None]).sum(dim=1).to(torch.int32),
        0,
    ).to(torch.int32)
    value_differences = int((out.cpu() != expected).sum())
    rank_differences = int((ranks.cpu() != expected_ranks).sum())
    assert value_differences + rank_differences == 0, (
        f"{case}: {rows} rows of {columns}, tokens {tokens.tolist()}: "
        f"{value_differences} log-probabilities and {rank_differences} ranks "
        f"differ; got {out.cpu().tolist()} ranked {ranks.cpu().tolist()}, the "
        f"reference {expected.tolist()} ranked {expected_ranks.tolist()}"
    )


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    torch.manual_seed(0)
    kernels = kernel_source("logprob_gather")
    rows, columns = 8, 1024

    def launched(inputs, outputs, pdl):
        launch(
            kernels,
            inputs["logprobabilities"],
            inputs["tokens"],
            outputs["out"],
            outputs["ranks"],
            1024,
            pdl,
        )
        return [outputs["out"], outputs["ranks"]]

    case = chain.Case(
        "logprob_gather",
        {
            "logprobabilities": torch.log_softmax(
                torch.randn(rows, columns), dim=-1
            ),
            "tokens": torch.randint(1, columns, (rows,), dtype=torch.int64),
        },
        {
            "out": torch.full((rows,), float("nan")),
            "ranks": torch.full((rows,), -1, dtype=torch.int32),
        },
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"logprob_gather launched as a dependent: {failures} of "
        f"{chain.ROUNDS} rounds differ from its launch after a "
        "synchronization"
    )
