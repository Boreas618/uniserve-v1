"""The fill program: what it stores, and what it leaves.

Each case launches the module of its descriptor key as ``Fill::launch`` does
(the destination merged into the axes a program traverses, its extents and
strides packed right-aligned, the element as its bits) and compares the
**whole buffer** the destination is a block of, bit for bit, with torch's
assignment of that element to that block. An element stored outside the block
fails the case as surely as one left unset inside it.

What is covered:

- **every element type**, with the patterns a conversion would lose: zero, a
  negative zero, a NaN with a payload, an infinity, the largest and the
  smallest integer, and all bits set. An FP8 (``f8e4m3``) element is stored
  as the byte it is;
- **a whole tensor** (``zeros``, a constant) of several programs and a masked
  tail;
- **a block of a larger tensor** (``const_set`` on a slice) of rank 1 to 5,
  whose strides the destination keeps, and **a stepped run**, which sets
  every other element and leaves the rest.
"""

import pytest
import torch
from uniserve_testing.kernels.tensor import operands

pytestmark = pytest.mark.unit

DTYPES = operands.FLOATS + operands.INTEGERS + tuple(operands.MOVED)

#: Bit patterns to store, by the element's width: zero, the sign bit alone (a
#: negative zero, or the smallest signed integer), all bits set (a NaN with a
#: payload, or minus one), a pattern with no symmetry, and the largest signed
#: integer.
PATTERNS = {
    8: [0x00, 0x80, 0xFF, 0x5A, 0x7F],
    16: [0x0000, 0x8000, 0xFFFF, 0x3C5A, 0x7FFF, 0x7F80, 0x7C00],
    32: [
        0x00000000,
        0x80000000,
        0xFFFFFFFF,
        0x3FC00000,
        0x7FFFFFFF,
        0x7F800000,
    ],
    64: [
        0x0,
        0x8000000000000000,
        0xFFFFFFFFFFFFFFFF,
        0x3FF8000000000000,
        0x7FFFFFFFFFFFFFFF,
        0x7FF0000000000000,
    ],
}

#: Every element type with every pattern of its width, and the seed of the
#: tensor it fills.
WHOLE = [
    pytest.param(dtype, pattern, seed, id=f"{dtype}-{pattern:#x}")
    for seed, dtype in enumerate(DTYPES)
    for pattern in PATTERNS[operands.width(dtype)]
]

#: Blocks of a larger tensor, of rank 1 to 5, and a stepped run: the tensor's
#: shape and the block's index.
BLOCKS = {
    "a run inside a longer one": ((40,), (slice(7, 29),)),
    "every other element": ((40,), (slice(3, None, 2),)),
    "a block of rank 2": ((5, 9), (slice(1, 4), slice(2, 6))),
    "whole rows of rank 2": ((5, 9), (slice(1, 4), slice(None))),
    "a block of rank 3": ((4, 6, 8), (slice(1, 3), slice(2, 5), slice(3, 7))),
    "a block of rank 4": (
        (3, 5, 4, 6),
        (slice(1, 3), slice(0, 3), slice(2, 4), slice(3, 6)),
    ),
    "a block of rank 5": (
        (3, 4, 5, 3, 4),
        (slice(0, 2), slice(1, 3), slice(2, 5), slice(1, 3), slice(2, 4)),
    ),
    "one element": ((3, 4), (slice(1, 2), slice(2, 3))),
}

#: The element types a block is filled with, each with its seed.
BLOCK_DTYPES = [
    pytest.param(dtype, seed, id=dtype)
    for seed, dtype in enumerate(("bf16", "f32", "i64", "u8", "f64", "f8e4m3"))
]


def element_of(primitives, dtype, pattern):
    """The one-element tensor whose bits are `pattern`."""
    bits = operands.width(dtype)
    signed = {
        8: torch.uint8,
        16: torch.int16,
        32: torch.int32,
        64: torch.int64,
    }[bits]
    # Torch's integers of these widths are signed (the byte aside): the
    # pattern is stated as the two's complement number that has it.
    number = (
        pattern
        if bits == 8 or pattern < 1 << (bits - 1)
        else pattern - (1 << bits)
    )
    stored = torch.tensor([number], dtype=signed).view(operands.element(dtype))
    return stored.to(primitives.device)


def fill(primitives, dtype, pattern, buffer, block, name):
    """Fill `buffer[block]` with `pattern` and hold the whole buffer to
    torch's assignment.
    """  # noqa: D205
    destination = buffer[block]
    expected = buffer.clone()
    expected[block] = element_of(primitives, dtype, pattern)

    extents, (strides,) = operands.merged([destination])
    elements = destination.numel()
    # `Descriptor`'s `Display`.
    key = f"dtype_{dtype}_rank_{len(extents)}_block_{operands.BLOCK}_pdl_false"
    primitives.launch(
        "fill",
        key,
        operands.grid(elements),
        destination,
        elements,
        *operands.extent_words(extents),
        *operands.stride_words(strides),
        pattern,
    )
    label = f"{dtype} {pattern:#x} {name} {tuple(destination.shape)}"
    operands.assert_same(label, buffer, expected)


@pytest.mark.parametrize(("dtype", "pattern", "seed"), WHOLE)
def test_a_pattern_fills_a_whole_tensor(dtype, pattern, seed, primitives):
    buffer = primitives.values(dtype, (2 * operands.BLOCK + 9,), seed)
    fill(primitives, dtype, pattern, buffer, (slice(None),), "a whole tensor")


@pytest.mark.parametrize("name", BLOCKS)
@pytest.mark.parametrize(("dtype", "seed"), BLOCK_DTYPES)
def test_a_block_is_filled_and_the_rest_left(dtype, seed, name, primitives):
    shape, block = BLOCKS[name]
    buffer = primitives.values(dtype, shape, seed + 50)
    pattern = PATTERNS[operands.width(dtype)][3]
    fill(primitives, dtype, pattern, buffer, block, name)
