#!/usr/bin/env python3
"""Run `reduce`'s kernel and check every function against numpy, on the CPU.

    python reduce_check.py             # Triton's interpreter, no device
    python reduce_check.py --device    # compiled for a device

Each case launches the module of its descriptor key as `Reduce::launch` does — the input split into
the axes the call keeps and the axes it reduces, each merged into the axes a program traverses,
their extents and strides packed right-aligned — and compares the output with numpy on the same
values: a floating sum inside the bound of its class (`Reduce::tolerance`), everything else exactly.

What is covered:

- **every function on every element type**, over a reduction shorter than a tile, where one program
  folds several outputs, and one of several tiles with a masked tail, where it folds one;
- **every choice of axes** of a rank-4 input: the last, the first, a middle one that splits the kept
  axes in two, two that are not adjacent, and all of them; an index along each single axis;
- **every layout** of `operands.views` — rank 1 to 6, windows, steps, permutations, zero-stride
  broadcasts — reduced along its last axis, its first axis and all of its axes;
- **ties**: the first of equal values wins an index, within a tile and across two tiles;
- **an integer sum that wraps**, and **a sum along an axis of no element**, which is zero.
"""

import math
import sys

# `operands` chooses between the interpreter and a device, and is imported before Triton.
import numpy as np
import operands
import torch

DTYPES = operands.FLOATS + operands.INTEGERS
FUNCTIONS = ("sum", "min", "max", "argmin", "argmax")
INDEXES = ("argmin", "argmax")

#: The name `ops::DType` prints for each torch element type.
NAMES = {element: name for name, element in operands.ELEMENTS.items()}


def reduction_class(length):
    """`reduce::reduction_class`: the least power of two that holds `length`."""
    return 1 << max(length - 1, 0).bit_length()


def expected_of(function, x, axes):
    """The reduction in numpy: float64 for a floating sum, exact values otherwise."""
    wide = operands.real(x)
    if function == "sum":
        folded = wide.sum(axis=tuple(axes), keepdims=True)
        if x.dtype.is_floating_point:
            return folded
        # An integer sum wraps at the element's own width.
        numpy_dtype = np.dtype(str(x.dtype).removeprefix("torch."))
        return folded.astype(numpy_dtype).astype(np.int64)
    if function == "min":
        return wide.min(axis=tuple(axes), keepdims=True)
    if function == "max":
        return wide.max(axis=tuple(axes), keepdims=True)
    (axis,) = axes
    # Numpy returns the first of equal values, as the operator does.
    found = (
        wide.argmin(axis=axis)
        if function == "argmin"
        else wide.argmax(axis=axis)
    )
    return np.expand_dims(found, axis)


def check(function, name, x, axes):
    """One launch held to numpy and the operator's bound."""
    dtype = NAMES[x.dtype]
    kept_axes = [axis for axis in range(x.dim()) if axis not in axes]
    reduced_axes = [axis for axis in range(x.dim()) if axis in axes]
    kept, (kept_strides,) = operands.merge(
        [x.shape[axis] for axis in kept_axes],
        [[x.stride(axis) for axis in kept_axes]],
    )
    reduced, (reduced_strides,) = operands.merge(
        [x.shape[axis] for axis in reduced_axes],
        [[x.stride(axis) for axis in reduced_axes]],
    )
    outputs = math.prod(kept)
    length = math.prod(reduced)

    folded = [
        1 if axis in axes else extent for axis, extent in enumerate(x.shape)
    ]
    out = operands.empty(
        folded, torch.uint32 if function in INDEXES else x.dtype
    )

    # `Descriptor`'s `Display`: the tile is the class, capped at the block.
    tile = min(reduction_class(length), operands.BLOCK)
    key = (
        f"function_{function}_dtype_{dtype}_kept_{len(kept)}_reduced_{len(reduced)}"
        f"_tile_{tile}_block_{operands.BLOCK}_pdl_false"
    )
    operands.launch(
        "reduce",
        key,
        operands.grid(outputs, operands.BLOCK // tile),
        x,
        out,
        outputs,
        length,
        *operands.extent_words(kept),
        *operands.stride_words(kept_strides),
        *operands.extent_words(reduced),
        *operands.stride_words(reduced_strides),
        0,
    )

    # `Reduce::tolerance`.
    bound = operands.EXACT
    if function == "sum" and dtype in operands.FLOATS:
        bound = operands.dot(reduction_class(length), dtype)
    label = f"{function} {dtype} {name} {tuple(x.shape)} along {list(axes)}"
    return operands.compare(label, out, expected_of(function, x, axes), bound)


def main():
    if not operands.ready():
        return 0

    checked = 0

    # Every function on every element type: a short reduction and one of several tiles.
    for dtype in DTYPES:
        for seed, function in enumerate(FUNCTIONS):
            short = operands.values(dtype, (37, 5), seed)
            long = operands.values(
                dtype, (3, 2 * operands.BLOCK + 22), seed + 10
            )
            checked += check(function, "shorter than a tile", short, [1])
            checked += check(function, "several tiles", long, [1])

    # Every choice of axes of a rank-4 input.
    choices = (
        [3],
        [0],
        [1],
        [2, 3],
        [0, 1],
        [0, 3],
        [1, 2],
        [3, 0],
        [0, 1, 2, 3],
    )
    for dtype in ("f32", "bf16", "i32"):
        x = operands.values(dtype, (2, 3, 4, 5), 20)
        for axes in choices:
            for function in ("sum", "min", "max"):
                checked += check(function, "rank 4", x, axes)
        for axis in range(4):
            for function in INDEXES:
                checked += check(function, "rank 4", x, [axis])

    # Every layout, along its last axis, its first axis and all of its axes.
    layouts = 0
    for dtype in ("f32", "i64", "bf16"):
        for name, view in operands.views(dtype, 30):
            last = view.dim() - 1
            everything = list(range(view.dim()))
            checked += check("sum", name, view, [last])
            checked += check("max", name, view, [0])
            checked += check("min", name, view, everything)
            checked += check("sum", name, view, everything)
            checked += check("argmax", name, view, [last])
            checked += check("argmin", name, view, [0])
            layouts += 6

    # Ties: the first of equal values, within one tile and across two.
    across = [0] * (operands.BLOCK + 40)
    across[10] = across[operands.BLOCK + 5] = 7
    for dtype in ("f32", "i32", "u8"):
        within = operands.tensor(
            [[1, 3, 3, 1], [2, 2, 2, 2], [0, 5, 0, 5]], dtype
        )
        spread = operands.tensor(
            [across, [-value for value in across] if dtype != "u8" else across],
            dtype,
        )
        for function in INDEXES + ("min", "max"):
            checked += check(function, "ties within a tile", within, [1])
            checked += check(function, "ties across tiles", spread, [1])
            checked += check(function, "ties down a column", within, [0])

    # An infinity is an element like any other, and a row of one value returns its first index.
    for dtype in operands.FLOATS:
        edges = operands.tensor(
            [
                [float("-inf")] * 4,
                [1.0, float("inf"), float("inf"), 0.0],
                [-2.0, float("-inf"), 3.0, 3.0],
            ],
            dtype,
        )
        for function in ("min", "max") + INDEXES:
            checked += check(function, "infinities", edges, [1])

    # An integer sum that passes its element's range wraps.
    for dtype, fill in (
        ("u8", 200),
        ("i16", 30000),
        ("i32", 2**30),
        ("u32", 2**31),
        ("i64", 2**62),
    ):
        x = operands.tensor(np.full((3, 9), fill), dtype)
        checked += check("sum", "a sum that wraps", x, [1])

    # A sum along an axis of no element is zero, for every output.
    for dtype in ("f32", "bf16", "i64"):
        checked += check(
            "sum", "no element", operands.values(dtype, (4, 0), 1), [1]
        )
        checked += check(
            "sum", "no element", operands.values(dtype, (3, 0, 5), 2), [1]
        )

    print(
        f"ok: {checked} outputs over {len(FUNCTIONS)} functions on {len(DTYPES)} element types, "
        f"{len(choices)} choices of axes, {layouts} strided launches of rank 1 to 6, ties and empty "
        f"axes, on {operands.where()}: floating sums within dot(class), the rest exact"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
