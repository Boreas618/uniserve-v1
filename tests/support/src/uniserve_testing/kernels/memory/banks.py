"""The block banks the prefix cache's block copy reads and writes.

A state bank stores each field block-major: axis 0 selects a block and the
trailing axes are the block's own, **at the field's own strides**. One
descriptor's blocks start 2048 elements apart while a block's content covers
1088; another's are packed. ``cache_block_snapshot`` reads blocks out of such
a bank into a snapshot of flattened rows and ``cache_block_scatter`` writes the
rows back into blocks, so both turn a column of a flattened block into an
offset through those axes, and both checks build their bank as a strided view
of one buffer over the same descriptors (:data:`CASES`).

A block index of ``-1`` (an inactive pair's placeholder) and one past the bank
copy nothing, which keeps an unvalidated device vector inside its allocation;
the cases name both.
"""

import torch

#: The sentinel a buffer is primed with, per element type: a byte no small
#: integer of these cases produces and a half no value of them is, so a
#: program writing where it should not shows wherever it lands.
SENTINELS = {torch.uint8: 0xA5, torch.bfloat16: 1234.5}

#: One descriptor per case: ``shape`` is the bank's block-major shape,
#: ``strides`` its element strides, ``width`` the elements of one flattened
#: block and ``block`` the columns one program covers, as the descriptor
#: compiles them; ``blocks`` the block index of each snapshot row.
CASES = {
    "a packed block bank": {
        "dtype": torch.uint8,
        "shape": (2048, 64, 1, 128),
        "strides": (8192, 128, 128, 1),
        "width": 8192,
        "block": 1024,
        "blocks": [3, 0, 7, 1],
    },
    # Blocks 2048 elements apart whose content covers 1088: the gaps inside a
    # block are the check.
    "a bank whose blocks hold 2048 elements in 1088": {
        "dtype": torch.bfloat16,
        "shape": (1024, 16, 2, 64),
        "strides": (2048, 64, 64, 1),
        "width": 2048,
        "block": 1024,
        "blocks": [0, 5, 2],
    },
    # The placeholders: an inactive pair and an index past the bank copy
    # nothing, while the rows between them still copy their own blocks. The
    # bank's last block and the first index past it sit either side of the
    # bound, so a bound off by one, or short by any number, changes an answer.
    "a placeholding index": {
        "dtype": torch.uint8,
        "shape": (1024, 16, 2, 64),
        "strides": (2048, 64, 64, 1),
        "width": 2048,
        "block": 1024,
        "blocks": [-1, 4, 1019, 1023, 1024, 9],
    },
}


def span(case) -> int:
    """The elements of the smallest buffer a case's bank is a view of."""
    return 1 + sum(
        (count - 1) * stride
        for count, stride in zip(case["shape"], case["strides"], strict=True)
    )


def holds(case, block: int) -> bool:
    """Whether ``block`` names a block of the bank: one past it, and the
    placeholder ``-1``, name none.
    """  # noqa: D205
    return 0 <= block < case["shape"][0]


def block_of(buffer: torch.Tensor, case, block: int) -> torch.Tensor:
    """Block ``block`` of a bank over ``buffer``, as torch's own strided view.

    A reference built through this view walks the block's axes in their
    logical order, so it does not restate the address arithmetic under test.
    """
    return torch.as_strided(
        buffer,
        case["shape"][1:],
        case["strides"][1:],
        storage_offset=block * case["strides"][0],
    )


def tiles(case) -> int:
    """The programs one snapshot row takes: its width in blocks of columns."""
    return -(-case["width"] // case["block"])
