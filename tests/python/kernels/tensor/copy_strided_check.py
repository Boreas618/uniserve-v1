#!/usr/bin/env python3
"""Run `copy_strided`'s kernel and check every copy a tensor library makes, on the CPU.

    python copy_strided_check.py             # Triton's interpreter, no device
    python copy_strided_check.py --device    # compiled for a device

Each case launches the module of its descriptor key as `CopyStrided::launch` does — the source and
the destination merged into the axes a program traverses, the extents and both stride vectors packed
right-aligned — and compares **bit for bit**, because a copy converts nothing. The destination is a
block of a larger buffer that starts as a sentinel, and the whole buffer is compared, so an element
written outside the block fails the case as surely as one not written inside it.

What is covered, by the backend primitive it serves:

- **`contiguous` and `copy_strided_src`**: every layout of `operands.views` — rank 1 to 6, windows,
  steps, permutations, zero-stride broadcasts, several programs, a single element — into a
  contiguous run that begins at an element offset inside a longer buffer;
- **`copy2d`**: rows whose stride differs on the two sides, read from an offset and written at an
  offset, which is how a concatenation along the last axis places each operand;
- **`cat`, `pad` and `slice_set`**: a source of rank 2 to 5 written into an interior block of a
  larger tensor, whose strides the destination keeps;
- **every element type**, an FP8 (`f8e4m3`) tensor as the bytes its elements are, and among the floats a NaN, an infinity and a negative zero.
"""

import sys

# `operands` chooses between the interpreter and a device, and is imported before Triton.
import operands
import torch

#: Every element type a copy moves: an FP8 (`f8e4m3`) tensor is held as its bytes.
DTYPES = operands.FLOATS + operands.INTEGERS + tuple(operands.MOVED)


def copy(dtype, source, destination):
    """One launch of `source`, of `dtype` elements, into the view `destination`."""
    extents, (source_strides, destination_strides) = operands.merged(
        [source, destination]
    )
    elements = destination.numel()

    # `Descriptor`'s `Display`.
    key = f"dtype_{dtype}_rank_{len(extents)}_block_{operands.BLOCK}_pdl_false"
    operands.launch(
        "copy_strided",
        key,
        operands.grid(elements),
        source,
        destination,
        elements,
        *operands.extent_words(extents),
        *operands.stride_words(source_strides),
        *operands.stride_words(destination_strides),
        0,
    )


def check(name, dtype, source, buffer, block):
    """Copy `source` into `buffer[block]` and hold the whole buffer to torch's assignment."""
    expected = buffer.clone()
    expected[block] = source.contiguous()

    copy(dtype, source, buffer[block])
    label = f"{dtype} {name} {tuple(source.shape)}"
    return operands.same(label, buffer, expected)


def sentinel(dtype, shape, seed):
    """A buffer of values no case writes: what a destination holds before a copy."""
    return operands.values(dtype, shape, seed + 9000)


def main():
    if not operands.ready():
        return 0

    checked = 0

    # A strided source made contiguous, at an element offset of a longer buffer.
    layouts = 0
    for dtype in ("bf16", "f32", "i64", "u8", "f8e4m3"):
        for name, view in operands.views(dtype, 11):
            offset = 5
            buffer = sentinel(dtype, (view.numel() + 11,), layouts)
            run = slice(offset, offset + view.numel())
            # The destination is the run reshaped to the source's shape: contiguous, so the call
            # is the one `copy_strided_src` makes.
            expected = buffer.clone()
            expected[run] = view.contiguous().reshape(-1)
            copy(dtype, view, buffer[run].view(view.shape))
            checked += operands.same(
                f"{dtype} contiguous at an offset, {name}", buffer, expected
            )
            layouts += 1

    # Every element type: a permuted source, and a broadcast one.
    for seed, dtype in enumerate(DTYPES):
        permuted = operands.values(dtype, (3, 4, 5), seed).permute(2, 0, 1)
        buffer = sentinel(dtype, (5, 3, 4), seed)
        checked += check(
            "permuted", dtype, permuted, buffer, (slice(None),) * 3
        )

        repeated = operands.values(dtype, (1, 6), seed + 20).expand(4, 6)
        buffer = sentinel(dtype, (4, 6), seed + 20)
        checked += check(
            "broadcast", dtype, repeated, buffer, (slice(None),) * 2
        )

    # The two-dimensional copy: rows read from an offset of a wider source and written at an
    # offset of a wider destination, the two row strides differing.
    for seed, dtype in enumerate(DTYPES):
        wide = operands.values(dtype, (6, 9), seed + 40)
        source = wide[1:4, 2:7]
        buffer = sentinel(dtype, (5, 12), seed + 40)
        checked += check(
            "rows of differing strides",
            dtype,
            source,
            buffer,
            (slice(2, 5), slice(4, 9)),
        )

    # A concatenation along the last axis: each operand lands in its columns of the result.
    for dtype in ("bf16", "i32", "f8e4m3"):
        left = operands.values(dtype, (3, 5), 61)
        right = operands.values(dtype, (3, 7), 62).flip(0).contiguous()
        result = sentinel(dtype, (3, 12), 63)
        copy(dtype, left, result[:, :5])
        copy(dtype, right, result[:, 5:])
        expected = torch.cat([left, right], dim=1)
        checked += operands.same(f"{dtype} concatenation", result, expected)

    # A padding or a slice assignment: a source written into an interior block.
    for dtype, shape, outer, block in (
        ("f32", (3, 4), (5, 9), (slice(1, 4), slice(2, 6))),
        ("bf16", (2, 3, 4), (4, 6, 8), (slice(1, 3), slice(2, 5), slice(3, 7))),
        (
            "u32",
            (2, 3, 2, 3),
            (3, 5, 4, 6),
            (slice(1, 3), slice(0, 3), slice(2, 4), slice(3, 6)),
        ),
        (
            "f64",
            (2, 2, 3, 2, 2),
            (3, 4, 5, 3, 4),
            (slice(0, 2), slice(1, 3), slice(2, 5), slice(1, 3), slice(2, 4)),
        ),
    ):
        source = (
            operands.values(dtype, shape, 71)
            .transpose(0, -1)
            .contiguous()
            .transpose(0, -1)
        )
        buffer = sentinel(dtype, outer, 72)
        checked += check("an interior block", dtype, source, buffer, block)

    # Values a bitwise comparison tells apart.
    for dtype in operands.FLOATS:
        special = operands.tensor(
            [float("nan"), -0.0, 0.0, float("inf"), -float("inf"), 1.0], dtype
        )
        buffer = sentinel(dtype, (2, 6), 81)
        checked += check(
            "special values", dtype, special.flip(0), buffer, (1, slice(None))
        )

    print(
        f"ok: {checked} buffer elements bit for bit over {len(DTYPES)} element types: {layouts} "
        f"strided sources made contiguous at an offset, rows of differing strides, a "
        f"concatenation and interior blocks of rank 2 to 5, on {operands.where()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
