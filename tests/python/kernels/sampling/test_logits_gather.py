"""The logits_gather program copies the rows lm_head computes, exactly.

The first step of the sampling path copies each request's last position out
of the batch's hidden state. Values move, so the comparison is exact. The
cases: the fixture's shape with rows out of order and one repeated, which a
copy assuming identity order or reading ``count`` rows from the top fails;
and a copy the block does not divide, whose last program is a tail.

A module compiled for a dependent launch reads only what the grid before it
wrote (:mod:`tests.python.kernels.chain`); that launch needs a device.
"""

import pytest
import torch

from tests.python.kernels import chain

pytestmark = pytest.mark.unit

#: The fixture's descriptor: `dtype_bf16_columns_2048_block_1024`.
COLUMNS = 2048
BLOCK = 1024

#: The element types the operator's `dtype` takes, as the check's `--dtype`
#: offered them.
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}

#: The cases: the hidden state's rows, the program's block, whether the rows
#: come out of order with one repeated, and the seed.
CASES = {
    "the fixture's shape, rows out of order and one repeated": (
        8,
        BLOCK,
        True,
        1,
    ),
    "a copy the block does not divide, so the last program is a tail": (
        3,
        2 * COLUMNS,
        False,
        2,
    ),
}


def launch(kernels, hidden, rows, out, block, pdl=False):
    """One launch over every copied element, compiled for a dependent launch
    when `pdl` is.
    """  # noqa: D205
    count, columns = out.shape
    kernels._logits_gather_kernel[(-(-count * columns // block),)](
        hidden,
        rows,
        out,
        count,
        columns,
        BLOCK=block,
        PDL=pdl,
        launch_pdl=pdl,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES)
def test_every_selected_row_is_copied(case, dtype, torch_device, kernel_source):
    tokens, block, orders_rows, seed = CASES[case]
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(tokens, COLUMNS, generator=generator).to(DTYPES[dtype])
    if orders_rows:
        rows = torch.tensor(
            [tokens - 1, 7 % tokens, 0, 7 % tokens], dtype=torch.int32
        )
    else:
        rows = torch.arange(min(tokens, 4), dtype=torch.int32)
    out = torch.full(
        (rows.numel(), COLUMNS),
        float("nan"),
        dtype=DTYPES[dtype],
        device=torch_device,
    )

    launch(
        kernel_source("logits_gather"),
        hidden.to(torch_device),
        rows.to(torch_device),
        out,
        block,
    )

    expected = hidden[rows.long()]
    differing = int((out.cpu() != expected).sum())
    assert differing == 0, (
        f"{case}, {dtype}: {rows.numel()} of {tokens} rows, {COLUMNS} "
        f"columns, block {block}: {differing} of {expected.numel()} differ "
        "from the reference"
    )


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    if not chain.supported():
        pytest.skip("this device compiles no dependent launch")
    torch.manual_seed(0)
    kernels = kernel_source("logits_gather")
    count, columns = 8, 1024

    def launched(inputs, outputs, pdl):
        launch(
            kernels, inputs["hidden"], inputs["rows"], outputs["out"], 1024, pdl
        )
        return [outputs["out"]]

    case = chain.Case(
        "logits_gather",
        {
            "hidden": torch.randn(64, columns).to(torch.bfloat16),
            "rows": torch.randperm(64)[:count].to(torch.int32),
        },
        {"out": torch.zeros(count, columns, dtype=torch.bfloat16)},
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"logits_gather launched as a dependent: {failures} of "
        f"{chain.ROUNDS} rounds differ from its launch after a "
        "synchronization"
    )
