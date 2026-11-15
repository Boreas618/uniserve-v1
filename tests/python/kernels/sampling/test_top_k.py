"""The top_k program keeps a row's largest logits with their ids, exactly.

The program keeps the ``width`` largest logits of a row, sorted descending,
with their vocabulary ids; a tie keeps the **lower id first**. A selection, so
the comparison is exact, and the case it turns on is the tie: a radix select
that breaks ties by arrival order passes every row without one. The values
are written in FP32 whatever the row's element type, and the ordering passes
use a scratch of two key planes, two id planes and two nibble tables per row.

The destinations are filled with the operator's own sentinels, ``nan`` and
``-1``, which no candidate can be: a row's ``width`` slots are the operator's
whole output, so a sentinel that survives anywhere is a slot nothing wrote,
a defect whether or not the comparison of values would have caught it.

The cases: the fixture's width out of the vocabulary; three tied logits,
which come out in ascending id order; the deployment's default bound out of
the same vocabulary, whose equal keys' quota is far smaller than the row; a
row keeping its whole width; and a masked ``-inf`` tail, never kept over a
real candidate.

A module compiled for a dependent launch reads only what the grid before it
wrote (:mod:`tests.python.kernels.chain`); that launch needs a device.
"""

import pytest
import torch

from tests.python.kernels import chain

pytestmark = pytest.mark.unit

#: The bins of one ordering pass, a nibble of the key
#: (`ops::sampler::TOP_K_BINS`).
BINS = 16

#: The lanes one pass reads, the fixture's descriptor `tile`.
TILE = 2048

#: The fixture's vocabulary.
VOCABULARY = 151936

#: The element types the operator's `dtype` takes, as the check's `--dtype`
#: offered them.
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}

#: The cases: rows, vocabulary, width kept, whether logits tie, whether the
#: row's tail is masked, and the seed.
CASES = {
    "the fixture's width out of the vocabulary": (
        2,
        VOCABULARY,
        2048,
        False,
        False,
        1,
    ),
    "three tied logits, which must come out in ascending id order": (
        2,
        64,
        8,
        True,
        False,
        2,
    ),
    # The quota the equal keys hold is far smaller than the row: the case a
    # rank computed per tile corrupts from the low slots up.
    "the deployment's default bound out of the same vocabulary": (
        2,
        VOCABULARY,
        128,
        False,
        False,
        4,
    ),
    # A row that keeps everything it has: the quota is the row's width and
    # every slot is written.
    "a row keeping its whole width": (2, 128, 128, False, False, 5),
    "a masked -inf tail, which is never kept over a real candidate": (
        2,
        100,
        8,
        False,
        True,
        3,
    ),
}


def launch(kernels, logits, values, ids, workspace, tile=TILE, pdl=False):
    """One launch: a program per row, its passes reading `tile` lanes;
    compiled for a dependent launch when `pdl` is.
    """  # noqa: D205
    rows, vocabulary = logits.shape
    kernels._top_k_kernel[(rows,)](
        logits,
        values,
        ids,
        workspace,
        rows,
        vocabulary,
        values.shape[1],
        TILE=tile,
        PDL=pdl,
        launch_pdl=pdl,
    )


def scratch(rows, width, device):
    """The ordering passes' scratch: two key planes, two id planes and the
    two nibble tables per row.
    """  # noqa: D205
    return torch.zeros(
        rows, 4 * width + 2 * BINS, dtype=torch.int32, device=device
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES)
def test_every_row_keeps_its_largest_logits_lower_id_first(
    case, dtype, torch_device, kernel_source
):
    rows, vocabulary, width, ties, masked_tail, seed = CASES[case]
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, vocabulary, generator=generator).to(
        DTYPES[dtype]
    )
    if ties:
        # Three columns of one value and one above them all: the kept ids are
        # the three tied ones in ascending order.
        logits[0, 5] = 2.0
        logits[0, 9] = 2.0
        logits[0, 1] = 2.0
        logits[0, 40] = 5.0
    if masked_tail:
        # A `-inf` tail, as `filter_candidates` leaves it.
        logits[:, vocabulary // 2 :] = float("-inf")
    values = torch.full(
        (rows, width), float("nan"), dtype=torch.float32, device=torch_device
    )
    ids = torch.full((rows, width), -1, dtype=torch.int32, device=torch_device)

    launch(
        kernel_source("top_k"),
        logits.to(torch_device),
        values,
        ids,
        scratch(rows, width, torch_device),
    )
    values, ids = values.cpu(), ids.cpu()

    exact = logits.double()
    faults = []
    for row in range(rows):
        # Descending by value, ascending by id among equals.
        order = torch.argsort(exact[row], descending=True, stable=True)
        wanted_values = exact[row][order][:width]
        wanted_ids = order[:width].to(torch.int32)
        # A masked tail compares only the kept real candidates.
        keep = (
            wanted_values != float("-inf")
            if masked_tail
            else torch.ones(width, dtype=torch.bool)
        )
        value_faults = int(
            (values[row].double()[keep] != wanted_values[keep]).sum()
        )
        id_faults = int((ids[row][keep] != wanted_ids[keep]).sum())
        holes = int((ids[row] < 0).sum()) + int(torch.isnan(values[row]).sum())
        if value_faults or id_faults or holes:
            faults.append(
                f"row {row}: {value_faults} values and {id_faults} ids differ, "
                f"{holes} slots still hold the pre-fill; wanted ids "
                f"{wanted_ids.tolist()[:6]}, got {ids[row].tolist()[:6]}"
            )
    assert not faults, (
        f"{case}, {dtype}: {rows} x {vocabulary} keeping {width}: "
        + "; ".join(faults)
    )


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    if not chain.supported():
        pytest.skip("this device compiles no dependent launch")
    torch.manual_seed(0)
    kernels = kernel_source("top_k")
    rows, vocabulary, width = 4, 8192, 64

    def launched(inputs, outputs, pdl):
        launch(
            kernels,
            inputs["logits"],
            outputs["values"],
            outputs["ids"],
            outputs["workspace"],
            4096,
            pdl,
        )
        return [outputs["values"], outputs["ids"]]

    case = chain.Case(
        "top_k",
        {"logits": torch.randn(rows, vocabulary)},
        {
            "values": torch.full((rows, width), float("nan")),
            "ids": torch.full((rows, width), -1, dtype=torch.int32),
            "workspace": scratch(rows, width, "cpu"),
        },
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"top_k launched as a dependent: {failures} of {chain.ROUNDS} rounds "
        "differ from its launch after a synchronization"
    )
