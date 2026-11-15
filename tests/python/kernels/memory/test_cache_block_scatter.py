"""The cache_block_scatter program writes each snapshot row into its block.

The second half of the prefix cache's block copy:
``field[blocks[row], ...] = snapshot[row, :]`` through the bank's own strides
(:mod:`tests.python.kernels.memory.banks`). The address arithmetic is what can
be wrong, and wrong invisibly: a dense-looking bank holds the right *count* of
right values in the wrong places, and every later read is a shifted tensor.

So the bank is a strided view of one buffer primed with a sentinel: the gaps
between blocks and inside a block keep it, and every element no row names must
still hold it afterwards. The rows are distinct in every element, so a shifted
write cannot pass. A block index of ``-1`` and one past the bank write
nothing, and the snapshot is only read, so it must come back unchanged. A move
is exact: the comparison is element for element.
"""

import pytest
import torch

from tests.python.kernels.memory import banks

pytestmark = pytest.mark.unit


def launch(kernels, field, blocks, snapshot, case):
    """One launch: a program per snapshot row and block of its columns."""
    kernels._scatter_kernel[(len(case["blocks"]), banks.tiles(case))](
        field,
        blocks,
        snapshot,
        1,
        case["shape"],
        case["strides"],
        case["width"],
        case["block"],
    )


def rows_of(case) -> torch.Tensor:
    """The rows to scatter, distinct in every element."""
    shape = (len(case["blocks"]), case["width"])
    generator = torch.Generator().manual_seed(7 + shape[0])
    if case["dtype"] == torch.uint8:
        return torch.randint(
            0, 256, shape, generator=generator, dtype=torch.uint8
        )
    values = torch.randn(shape, generator=generator, dtype=torch.float32)
    return values.to(case["dtype"])


@pytest.mark.parametrize("case", banks.CASES)
def test_every_row_lands_in_its_block_and_nothing_else_moves(
    case, torch_device, kernel_source
):
    case = banks.CASES[case]
    buffer = torch.full(
        (banks.span(case),), banks.SENTINELS[case["dtype"]], dtype=case["dtype"]
    )
    snapshot = rows_of(case)

    # The bank torch's own strided copy writes, not the program's formula.
    expected = buffer.clone()
    written = 0
    for row, block in enumerate(case["blocks"]):
        if banks.holds(case, block):
            banks.block_of(expected, case, block).copy_(
                snapshot[row].view(case["shape"][1:])
            )
            written += 1

    launched = buffer.clone().to(torch_device)
    launched_snapshot = snapshot.clone().to(torch_device)
    launch(
        kernel_source("cache_block_scatter"),
        torch.as_strided(launched, case["shape"], case["strides"]),
        torch.tensor(case["blocks"], dtype=torch.int64, device=torch_device),
        launched_snapshot,
        case,
    )
    launched = launched.cpu()

    where = (
        f"{len(case['blocks'])} rows into blocks {case['blocks']} of a "
        f"{case['shape'][0]}-block bank at strides {case['strides']}"
    )
    assert torch.equal(launched_snapshot.cpu(), snapshot), (
        f"{where}: the snapshot changed; this program only reads it"
    )
    wrong = (launched != expected).reshape(-1)
    first = int(wrong.nonzero()[0]) if wrong.any() else None
    assert first is None, (
        f"{where}: {int(wrong.sum())} of {launched.numel()} elements of the "
        f"bank differ from what the rows write ({written} of them land in "
        f"the bank); first at {first}: {launched[first].item()!r} vs "
        f"{expected[first].item()!r}"
    )
