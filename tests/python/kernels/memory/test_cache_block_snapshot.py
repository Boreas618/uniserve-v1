"""The cache_block_snapshot program reads each named block into its row.

The first half of the prefix cache's block copy:
``snapshot[row, :] = field[blocks[row], ...]``, so the second half can write
the same blocks elsewhere without a host read between the two launches. It
reads a block-major bank at the field's own strides
(:mod:`tests.python.kernels.memory.banks`), the arithmetic that can be wrong
without a build noticing: a bank whose blocks are 2048 elements apart while a
block's content covers 1088 comes back as a shifted row and no error.

So the bank holds distinct values at its own strides and zeros in its gaps,
and every snapshot row must be the block its index names, element for
element. A block index of ``-1`` and one past the bank copy **nothing**: those
rows keep the sentinel the caller left in them. The bank is only read, so it
must come back unchanged. A move is exact: the comparison is element for
element.
"""

import pytest
import torch

from tests.python.kernels.memory import banks

pytestmark = pytest.mark.unit


def launch(kernels, field, blocks, snapshot, case):
    """One launch: a program per snapshot row and block of its columns."""
    kernels._snapshot_kernel[(len(case["blocks"]), banks.tiles(case))](
        field,
        blocks,
        snapshot,
        1,
        case["shape"],
        case["strides"],
        case["width"],
        case["block"],
    )


def bank_of(case) -> torch.Tensor:
    """The buffer of a bank whose every element holds its own value, with
    zeros in the gaps the strides leave.
    """  # noqa: D205
    buffer = torch.zeros(banks.span(case), dtype=case["dtype"])
    field = torch.as_strided(buffer, case["shape"], case["strides"])
    count = field.numel()
    if case["dtype"] == torch.uint8:
        values = torch.arange(count, dtype=torch.int64).remainder(256)
    else:
        values = torch.arange(count, dtype=torch.float32).remainder(997)
    field.copy_(values.to(case["dtype"]).view(case["shape"]))
    return buffer


@pytest.mark.parametrize("case", banks.CASES)
def test_every_row_is_its_block_and_nothing_else_moves(
    case, torch_device, kernel_source
):
    case = banks.CASES[case]
    buffer = bank_of(case)
    rows = len(case["blocks"])
    snapshot = torch.full(
        (rows, case["width"]),
        banks.SENTINELS[case["dtype"]],
        dtype=case["dtype"],
    )

    # The rows torch's own strided read gives, not the program's formula.
    expected = snapshot.clone()
    read = 0
    for row, block in enumerate(case["blocks"]):
        if banks.holds(case, block):
            expected[row] = banks.block_of(buffer, case, block).reshape(-1)
            read += 1

    launched = buffer.clone().to(torch_device)
    launched_snapshot = snapshot.clone().to(torch_device)
    launch(
        kernel_source("cache_block_snapshot"),
        torch.as_strided(launched, case["shape"], case["strides"]),
        torch.tensor(case["blocks"], dtype=torch.int64, device=torch_device),
        launched_snapshot,
        case,
    )
    launched_snapshot = launched_snapshot.cpu()

    where = (
        f"{rows} rows from blocks {case['blocks']} of a "
        f"{case['shape'][0]}-block bank at strides {case['strides']}"
    )
    assert torch.equal(launched.cpu(), buffer), (
        f"{where}: the bank changed; this program only reads it"
    )
    wrong = (launched_snapshot != expected).reshape(-1)
    first = int(wrong.nonzero()[0]) if wrong.any() else None
    assert first is None, (
        f"{where}: {int(wrong.sum())} of {launched_snapshot.numel()} snapshot "
        f"elements are not the bank's ({read} of {rows} rows name a block of "
        f"it, the rest keep what they held); first at {first}: "
        f"{launched_snapshot.reshape(-1)[first].item()!r} vs "
        f"{expected.reshape(-1)[first].item()!r}"
    )
