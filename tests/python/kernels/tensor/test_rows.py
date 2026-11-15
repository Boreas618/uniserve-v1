"""The rows program in each of its modes, against every destination row.

The operator places or accumulates the rows a call picks out by index::

    replace: destination[indices[j]] = source[j]
    add:     destination[indices[j]] += source[j]   one rounding into its type
    place:   destination[indices[j]] = source[j]    for indices[j] >= 0 only

The cases are what a row scatter can get wrong: a row narrower than the
program's block of columns and one of several blocks, rows written out of
order, rows no index names (which keep their values), and, for ``place``,
negative indices among the rows, whose source rows must reach no destination
at all. The indices of a case are distinct: the operator orders nothing
between two source rows that name one destination.

``replace`` and ``place`` copy and are exact. ``add`` sums in FP32 and rounds
once on the store, so it is held to two units of the destination type's
rounding, as ``Rows::tolerance`` states; a narrow store under Triton's
interpreter truncates where a device rounds, which the bound's second unit
covers.
"""

import pytest
import torch

pytestmark = pytest.mark.unit

UNIT_ROUNDOFF = {torch.bfloat16: 2.0**-8, torch.float32: 2.0**-24}

#: The cases: the mode, the element type, the destination's rows and columns,
#: the program's block of columns, the indices, and the seed.
CASES = {
    "replace, bf16, a row narrower than the block": (
        "replace",
        torch.bfloat16,
        9,
        100,
        128,
        [4, 0, 7],
        1,
    ),
    "replace, f32, rows of several blocks": (
        "replace",
        torch.float32,
        6,
        300,
        128,
        [5, 2],
        2,
    ),
    "add, bf16": ("add", torch.bfloat16, 9, 100, 128, [1, 8, 3, 0], 3),
    # A negative index names no row: its source row reaches no destination.
    "place, bf16, negative indices among the rows": (
        "place",
        torch.bfloat16,
        9,
        128,
        128,
        [4, -1, 7, -1, 0],
        4,
    ),
    "place, bf16, every index negative": (
        "place",
        torch.bfloat16,
        5,
        300,
        128,
        [-1, -1],
        5,
    ),
    "place, i32, a block wider than the row": (
        "place",
        torch.int32,
        8,
        3,
        4,
        [6, -1, 1],
        6,
    ),
}


def launch(kernels, source, destination, indices, mode, columns, block):
    """One launch: a program per source row and block of columns."""
    count = source.shape[0]
    kernels._rows_kernel[(count, -(-columns // block))](
        source,
        destination,
        indices,
        count,
        columns,
        block,
        0,
        COLUMNS=columns,
        REPLACE=mode != "add",
        BLOCK=block,
        PDL=False,
        SKIP=mode == "place",
        num_warps=4,
    )


@pytest.mark.parametrize("case", CASES)
def test_every_destination_row_is_placed_or_kept(
    case, torch_device, kernel_source
):
    mode, dtype, rows, columns, block, indices, seed = CASES[case]
    torch.manual_seed(seed)
    count = len(indices)
    if dtype == torch.int32:
        source = torch.randint(-999, 999, (count, columns)).to(dtype)
        before = torch.randint(-999, 999, (rows, columns)).to(dtype)
    else:
        source = torch.randn(count, columns).to(dtype)
        before = torch.randn(rows, columns).to(dtype)
    destination = before.clone().to(torch_device)

    launch(
        kernel_source("rows"),
        source.to(torch_device),
        destination,
        torch.tensor(indices, dtype=torch.int32, device=torch_device),
        mode,
        columns,
        block,
    )

    expected = before.double()
    for row, index in enumerate(indices):
        if index < 0:
            continue
        if mode == "add":
            expected[index] += source[row].double()
        else:
            expected[index] = source[row].double()
    error = (destination.cpu().double() - expected).abs()
    bound = (
        2 * UNIT_ROUNDOFF[dtype] * expected.abs()
        if mode == "add"
        else torch.zeros_like(expected)
    )
    outside = int((~(error <= bound)).sum())
    assert outside == 0, (
        f"{mode}, {count} rows into {rows} x {columns} of block {block}, "
        f"{dtype}: {outside} outside the bound, max abs "
        f"{float(error.max()):.3e}"
    )
