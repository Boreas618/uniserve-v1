"""The filter_candidates program drops candidates by the sampling thresholds.

The program applies a row's top-k width, its nucleus (top-p) and its min-p
threshold to candidates ordered by value, dropping a candidate by writing
``-inf`` over it, and raises the row's validity flag when a finite candidate
survives. These are decisions, so both outputs are compared exactly. The
thresholds of every case sit more than ``1e-4`` from any candidate's
cumulative boundary, where an FP32 prefix sum and an FP64 one may legitimately
disagree about a tie.

The cases: a width of four with a nucleus and a min-p; a row far wider than a
tile, walked in five of them; a dropped ``-inf`` tail, which the thresholds
must leave dropped while the nucleus is measured over what is left; and a row
with neither a width nor a min-p, so only the nucleus removes anything, with a
tie at the top.

Every test needs a device. The program combines a scalar comparison with an
array comparison by ``|`` (``(min_p <= 0) | (kept >= threshold)``), which
Triton's interpreter evaluates as ``numpy.bool_ | ndarray`` and its bitwise
ufunc refuses. A module compiled for a dependent launch reads only what the
grid before it wrote (:mod:`uniserve_testing.kernels.chain`).
"""

import pytest
import torch
from uniserve_testing.kernels import chain

pytestmark = [pytest.mark.unit, pytest.mark.gpu]

#: The fixture's candidates per row, and its tile.
WIDTH = 2048
TILE = 2048

#: The cases: rows, width, tile, top_k, top_p, min_p, whether the row's tail
#: is already dropped, whether the top two candidates tie, and the seed.
CASES = {
    "a top-k width of four, a nucleus of 0.9 and a min-p of 0.1": (
        3,
        WIDTH,
        TILE,
        4.0,
        0.9,
        0.1,
        False,
        False,
        1,
    ),
    "a row far wider than one tile, walked in five of them": (
        2,
        20480,
        4096,
        0.0,
        0.9,
        0.05,
        False,
        False,
        4,
    ),
    "a dropped tail, which the thresholds must leave dropped": (
        2,
        WIDTH,
        TILE,
        0.0,
        0.95,
        0.0,
        True,
        False,
        2,
    ),
    "no width and no min-p, so only the nucleus removes anything": (
        2,
        WIDTH,
        TILE,
        0.0,
        0.5,
        -1.0,
        False,
        True,
        3,
    ),
}


def launch(kernels, values, ids, parameters, valid, tile, pdl=False):
    """One launch: a program per row, filtering its values in place; compiled
    for a dependent launch when `pdl` is.
    """  # noqa: D205
    rows, width = values.shape
    kernels._filter_candidates_kernel[(rows,)](
        values,
        ids,
        parameters,
        valid,
        rows,
        width,
        BLOCK=tile,
        PDL=pdl,
        launch_pdl=pdl,
    )


@pytest.mark.parametrize("case", CASES)
def test_every_threshold_drops_exactly_its_candidates(
    case, torch_device, kernel_source
):
    rows, width, tile, top_k, top_p, min_p, tail_dropped, tie, seed = CASES[
        case
    ]
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(rows, width, generator=generator)
    if tail_dropped:
        # The previous stage leaves dropped candidates as `-inf`.
        values[:, width // 2 :] = float("-inf")
    if tie:
        values[:, 0] = 5.0
        values[:, 1] = 5.0
    ids = torch.arange(1000, 1000 + width, dtype=torch.int32).repeat(rows, 1)
    parameters = torch.zeros(rows, 8, dtype=torch.float32)
    parameters[:, 1] = top_k
    parameters[:, 2] = top_p
    parameters[:, 3] = min_p
    filtered = values.clone().to(torch_device)
    valid = torch.full((rows,), -1, dtype=torch.int32, device=torch_device)

    launch(
        kernel_source("filter_candidates"),
        filtered,
        ids.to(torch_device),
        parameters.to(torch_device),
        valid,
        tile,
    )

    dropped = torch.tensor(float("-inf"), dtype=torch.float64)
    exact = values.double()
    limit = min(int(top_k), width) if top_k > 0.0 else width
    kept = torch.where(torch.arange(width) < limit, exact, dropped)
    maximum = kept.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(kept - maximum)
    probabilities = exponentials / exponentials.sum(dim=-1, keepdim=True)
    cumulative = torch.cumsum(probabilities, dim=-1)
    kept = torch.where((cumulative - probabilities) > top_p, dropped, kept)
    if min_p > 0.0:
        threshold = kept.max(dim=-1, keepdim=True).values + torch.log(
            torch.tensor(min_p, dtype=torch.float64)
        )
        kept = torch.where(kept >= threshold, kept, dropped)
    usable = ((kept != kept) | (kept == float("inf"))).sum(dim=-1) == 0
    usable = usable & ((kept != float("-inf")).sum(dim=-1) > 0)

    value_differences = int((filtered.cpu().double() != kept).sum())
    flag_differences = int((valid.cpu() != usable.to(torch.int32)).sum())
    assert value_differences + flag_differences == 0, (
        f"{case}: {rows} x {width} in tiles of {tile}, top_k {top_k:g}, "
        f"top_p {top_p:g}, min_p {min_p:g}: {value_differences} of "
        f"{kept.numel()} values and {flag_differences} of {rows} flags "
        "differ from the reference"
    )


def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    torch.manual_seed(0)
    kernels = kernel_source("filter_candidates")
    rows, width = 4, 1024
    parameters = torch.zeros(rows, 8)
    parameters[:, 1], parameters[:, 2], parameters[:, 3] = 64.0, 0.9, 0.01

    def launched(inputs, outputs, pdl):
        # The values are filtered in place: an input and the result.
        launch(
            kernels,
            inputs["values"],
            inputs["ids"],
            inputs["parameters"],
            outputs["valid"],
            width,
            pdl,
        )
        return [inputs["values"], outputs["valid"]]

    case = chain.Case(
        "filter_candidates",
        {
            "values": torch.randn(rows, width)
            .sort(dim=-1, descending=True)
            .values
            * 3.0,
            "ids": torch.randperm(width).to(torch.int32).repeat(rows, 1),
            "parameters": parameters,
        },
        {"valid": torch.full((rows,), -1, dtype=torch.int32)},
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"filter_candidates launched as a dependent: {failures} of "
        f"{chain.ROUNDS} rounds differ from its launch after a "
        "synchronization"
    )
