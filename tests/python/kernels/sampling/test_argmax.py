"""The argmax program takes each row's greedy token, exactly.

The greedy token is the index of the row's largest value, the lowest index
among ties. The program walks a compiled width in tiles of ``BLOCK`` lanes, so
a row wider than one tile exercises the walk rather than one load (the width
this operator sees in service is 151,936). The cases: the fixture's width; and
ties, a repeated maximum and one at the last column.

A module compiled for a dependent launch reads only what the grid before it
wrote (:mod:`uniserve_testing.kernels.chain`); that launch needs a device.
"""

import pytest
import torch
from uniserve_testing.kernels import chain

pytestmark = pytest.mark.unit

#: The fixture's width.
WIDTH = 2048

#: The lanes one step of the walk reads, the package's `BLOCK`.
BLOCK = 1024

#: The element types the operator's `dtype` takes, as the check's `--dtype`
#: offered them.
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}

#: The cases: rows, whether the rows carry ties, and the seed.
CASES = {
    "the fixture's width": (2, False, 1),
    "ties, and a maximum at the last column": (2, True, 2),
}


def launch(kernels, values, out, pdl=False):
    """One launch: a program per row; compiled for a dependent launch when
    `pdl` is.
    """  # noqa: D205
    rows, width = values.shape
    kernels._argmax_kernel[(rows,)](
        values,
        out,
        rows,
        WIDTH=width,
        BLOCK=BLOCK,
        PDL=pdl,
        launch_pdl=pdl,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES)
def test_every_row_takes_its_largest_value_lowest_index_first(
    case, dtype, torch_device, kernel_source
):
    rows, ties, seed = CASES[case]
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(rows, WIDTH, generator=generator).to(DTYPES[dtype])
    if ties:
        # A repeated maximum, and one at the last column: the tie rule is the
        # lowest index.
        values[0, :] = 1.0
        values[1, WIDTH - 1] = 100.0
    out = torch.full((rows,), -1, dtype=torch.int32, device=torch_device)

    launch(kernel_source("argmax"), values.to(torch_device), out)

    wanted = (values == values.max(dim=-1, keepdim=True).values).int()
    wanted = wanted.argmax(dim=-1).to(torch.int32)
    got = out.cpu()
    assert torch.equal(got, wanted), (
        f"{case}, {dtype}: {rows} x {WIDTH}: took {got.tolist()}, the "
        f"reference takes {wanted.tolist()}"
    )


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    torch.manual_seed(0)
    kernels = kernel_source("argmax")
    rows, width = 4, 4096

    def launched(inputs, outputs, pdl):
        launch(kernels, inputs["values"], outputs["out"], pdl)
        return [outputs["out"]]

    case = chain.Case(
        "argmax",
        {"values": torch.randn(rows, width)},
        {"out": torch.full((rows,), -7, dtype=torch.int32)},
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"argmax launched as a dependent: {failures} of {chain.ROUNDS} rounds "
        "differ from its launch after a synchronization"
    )
