#!/usr/bin/env python3
"""Run `fill`'s kernel and check what it stores, and what it leaves, on the CPU.

    python fill_check.py             # Triton's interpreter, no device
    python fill_check.py --device    # compiled for a device

Each case launches the module of its descriptor key as `Fill::launch` does — the destination merged
into the axes a program traverses, its extents and strides packed right-aligned, the element as its
bits — and compares the **whole buffer** the destination is a block of, bit for bit, with torch's
assignment of that element to that block. An element stored outside the block fails the case as
surely as one left unset inside it.

What is covered:

- **every element type**, with the patterns a conversion would lose: zero, a negative zero, a NaN
  with a payload, an infinity, the largest and the smallest integer, and all bits set;
- **a whole tensor** (`zeros`, a constant) of several programs and a masked tail;
- **a block of a larger tensor** (`const_set` on a slice) of rank 1 to 5, whose strides the
  destination keeps, and **a stepped run**, which sets every other element and leaves the rest.
"""

import sys

# `operands` chooses between the interpreter and a device, and is imported before Triton.
import operands
import torch

DTYPES = operands.FLOATS + operands.INTEGERS

#: The name `ops::DType` prints for each torch element type.
NAMES = {element: name for name, element in operands.ELEMENTS.items()}

#: Bit patterns to store, by the element's width: zero, the sign bit alone (a negative zero, or the
#: smallest signed integer), all bits set (a NaN with a payload, or minus one), a pattern with no
#: symmetry, and the largest signed integer.
PATTERNS = {
    8: [0x00, 0x80, 0xFF, 0x5A, 0x7F],
    16: [0x0000, 0x8000, 0xFFFF, 0x3C5A, 0x7FFF, 0x7F80, 0x7C00],
    32: [0x00000000, 0x80000000, 0xFFFFFFFF, 0x3FC00000, 0x7FFFFFFF, 0x7F800000],
    64: [0x0, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF, 0x3FF8000000000000, 0x7FFFFFFFFFFFFFFF, 0x7FF0000000000000],
}  # fmt: skip


def element_of(dtype, pattern):
    """The one-element tensor whose bits are `pattern`."""
    element = operands.ELEMENTS[dtype]
    width = torch.empty((), dtype=element).element_size() * 8
    signed = {
        8: torch.uint8,
        16: torch.int16,
        32: torch.int32,
        64: torch.int64,
    }[width]
    # Torch's integers of these widths are signed (the byte aside): the pattern is stated as the
    # two's complement number that has it.
    number = (
        pattern
        if width == 8 or pattern < 1 << (width - 1)
        else pattern - (1 << width)
    )
    return (
        torch.tensor([number], dtype=signed).view(element).to(operands.TARGET)
    )


def check(name, dtype, pattern, buffer, block):
    """Fill `buffer[block]` with `pattern` and hold the whole buffer to torch's assignment."""
    destination = buffer[block]
    expected = buffer.clone()
    expected[block] = element_of(dtype, pattern)

    extents, (strides,) = operands.merged([destination])
    elements = destination.numel()

    # `Descriptor`'s `Display`.
    key = f"dtype_{dtype}_rank_{len(extents)}_block_{operands.BLOCK}_pdl_false"
    operands.launch(
        "fill",
        key,
        operands.grid(elements),
        destination,
        elements,
        *operands.extent_words(extents),
        *operands.stride_words(strides),
        pattern,
        0,
    )
    label = f"{dtype} {pattern:#x} {name} {tuple(destination.shape)}"
    return operands.same(label, buffer, expected)


def main():
    if not operands.ready():
        return 0

    checked = 0
    everything = (slice(None),)

    # Every element type and every pattern, over a whole tensor of several programs.
    patterns = 0
    for seed, dtype in enumerate(DTYPES):
        width = (
            torch.empty((), dtype=operands.ELEMENTS[dtype]).element_size() * 8
        )
        for pattern in PATTERNS[width]:
            buffer = operands.values(dtype, (2 * operands.BLOCK + 9,), seed)
            checked += check(
                "a whole tensor", dtype, pattern, buffer, everything
            )
            patterns += 1

    # A block of a larger tensor, of rank 1 to 5, and a stepped run.
    blocks = {
        "a run inside a longer one": ((40,), (slice(7, 29),)),
        "every other element": ((40,), (slice(3, None, 2),)),
        "a block of rank 2": ((5, 9), (slice(1, 4), slice(2, 6))),
        "whole rows of rank 2": ((5, 9), (slice(1, 4), slice(None))),
        "a block of rank 3": ((4, 6, 8), (slice(1, 3), slice(2, 5), slice(3, 7))),
        "a block of rank 4": ((3, 5, 4, 6), (slice(1, 3), slice(0, 3), slice(2, 4), slice(3, 6))),
        "a block of rank 5": ((3, 4, 5, 3, 4), (slice(0, 2), slice(1, 3), slice(2, 5), slice(1, 3), slice(2, 4))),
        "one element": ((3, 4), (slice(1, 2), slice(2, 3))),
    }  # fmt: skip
    for seed, dtype in enumerate(("bf16", "f32", "i64", "u8", "f64")):
        width = (
            torch.empty((), dtype=operands.ELEMENTS[dtype]).element_size() * 8
        )
        for name, (shape, block) in blocks.items():
            buffer = operands.values(dtype, shape, seed + 50)
            checked += check(name, dtype, PATTERNS[width][3], buffer, block)

    print(
        f"ok: {checked} buffer elements bit for bit: {patterns} patterns over {len(DTYPES)} element "
        f"types on a whole tensor, and {len(blocks)} blocks and steps of rank 1 to 5, on "
        f"{operands.where()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
