#!/usr/bin/env python3
"""Run `elementwise_binary`'s kernel and check every function against numpy, on the CPU.

    python elementwise_binary_check.py             # Triton's interpreter, no device
    python elementwise_binary_check.py --device    # compiled for a device

Each case launches the package's kernel as `ElementwiseBinary::launch` does — the two inputs and the
output merged into the axes a program traverses, the extents and both stride vectors packed
right-aligned — and compares the output with numpy on the same values: arithmetic inside the
operator's bound (`ElementwiseBinary::tolerance`), everything else exactly.

What is covered:

- **every function on every element type**, over two contiguous inputs and over a permuted input
  against a broadcast one;
- **integer arithmetic** in the element type itself: sums and products that wrap, and quotients of
  either sign, which truncate toward zero where a floor would differ;
- **the functions of the integer element types alone** on every one of them: `remainder` over
  dividends and divisors of both signs, where it takes the divisor's sign and the smallest value by
  minus one is zero; the three `bitwise_` combinations over every pair of the type's extreme bit
  patterns; and the two shifts by every count below the element's width, a right shift repeating
  the sign bit of a signed element. A floating element type has no module for them;
- **the comparisons and the extrema on a NaN, on equal values and on signed zeros**: a comparison
  writes `u8`, and `maximum`/`minimum` return the left operand unless the right is strictly beyond;
- **every layout** of `operands.views` as the left operand, against a right operand that is
  contiguous, one value repeated everywhere (every stride zero), and one row repeated over the
  leading axes — the three broadcasts a tensor library's arithmetic makes.
"""

import sys

# `operands` chooses between the interpreter and a device, and is imported before Triton.
import numpy as np
import operands
import torch

ARITHMETIC = ("add", "sub", "mul", "div")
EXTREMA = ("maximum", "minimum")
COMPARISONS = ("eq", "ne", "lt", "le", "gt", "ge")
FUNCTIONS = ARITHMETIC + EXTREMA + COMPARISONS

#: The functions that exist on the integer element types alone (`BinaryFunction::integral`).
BITWISE = ("bitwise_and", "bitwise_or", "bitwise_xor")
SHIFTS = ("bitwise_left_shift", "bitwise_right_shift")
INTEGRAL = ("remainder",) + BITWISE + SHIFTS

#: The width in bits of each integer element type.
WIDTH = {"u8": 8, "u32": 32, "i16": 16, "i32": 32, "i64": 64}


def expected_of(function, left, right):
    """`left o right` in numpy: float64 for a floating type, the element type for an integer."""
    with np.errstate(all="ignore"):
        if function == "div" and left.dtype.kind in "iu":
            # Truncation toward zero, from magnitudes: numpy's own `//` floors.
            quotient = np.abs(left.astype(np.int64)) // np.abs(
                right.astype(np.int64)
            )
            negative = (left < 0) != (right < 0)
            return np.where(negative, -quotient, quotient).astype(left.dtype)
        return {
            "add": lambda: left + right,
            "sub": lambda: left - right,
            "mul": lambda: left * right,
            "div": lambda: left / right,
            "maximum": lambda: np.where(left < right, right, left),
            "minimum": lambda: np.where(left > right, right, left),
            "eq": lambda: left == right,
            "ne": lambda: left != right,
            "lt": lambda: left < right,
            "le": lambda: left <= right,
            "gt": lambda: left > right,
            "ge": lambda: left >= right,
            # numpy's remainder is the floored quotient's, as `torch.remainder` is, and its
            # right shift repeats a signed element's sign bit.
            "remainder": lambda: np.remainder(left, right),
            "bitwise_and": lambda: left & right,
            "bitwise_or": lambda: left | right,
            "bitwise_xor": lambda: left ^ right,
            "bitwise_left_shift": lambda: np.left_shift(left, right),
            "bitwise_right_shift": lambda: np.right_shift(left, right),
        }[function]()


def numpy_values(tensor):
    """A tensor's values for the reference: float64, or its own integer type."""
    wide = operands.real(tensor)
    if tensor.dtype.is_floating_point:
        return wide
    return wide.astype(np.dtype(str(tensor.dtype).removeprefix("torch.")))


def check(function, dtype, name, left, right):
    """One launch held to numpy and the operator's bound."""
    compares = function in COMPARISONS
    out = operands.empty(left.shape, torch.uint8 if compares else left.dtype)
    extents, (left_strides, right_strides, _) = operands.merged(
        [left, right, out]
    )
    elements = out.numel()

    # `BinaryDescriptor`'s `Display`.
    key = (
        f"function_{function}_dtype_{dtype}_rank_{len(extents)}"
        f"_block_{operands.BLOCK}_pdl_false"
    )
    operands.launch(
        "elementwise_binary",
        key,
        operands.grid(elements),
        left,
        right,
        out,
        elements,
        *operands.extent_words(extents),
        *operands.stride_words(left_strides),
        *operands.stride_words(right_strides),
        0,
    )

    expected = expected_of(function, numpy_values(left), numpy_values(right))
    bound = operands.EXACT
    if function in ARITHMETIC and dtype in operands.FLOATS:
        # `ElementwiseBinary::tolerance`: one operation, rounded once into the element type.
        bound = operands.rounding(1, dtype)
    label = f"{function} {dtype} {name} {tuple(left.shape)}"
    return operands.compare(label, out, expected, bound)


def nonzero(tensor):
    """An integer divisor with its zeros replaced, of either sign where the type has one."""
    return torch.where(tensor == 0, torch.full_like(tensor, 3), tensor)


def operand_pair(function, dtype, left, right):
    """The two operands of a case, the right one held to what the function defines.

    An integer quotient and a remainder never divide by zero, and a shift count is below the
    element's width.
    """
    if function in ("div", "remainder") and dtype in operands.INTEGERS:
        right = nonzero(right)
    if function in SHIFTS:
        right = operands.tensor(
            operands.real(right) % WIDTH[dtype], dtype
        ).view(right.shape)
    return left, right


def integers(dtype, numbers):
    """`numbers` as a tensor of the integer element type `dtype`, wrapped into its range."""
    element = np.dtype(str(operands.ELEMENTS[dtype]).removeprefix("torch."))
    wrapped = np.array(numbers, dtype=object) % (1 << WIDTH[dtype])
    return operands.tensor(wrapped.astype(np.uint64).astype(element), dtype)


def refused(function, dtype):
    """Whether the package has no module of `function` for `dtype`."""
    key = f"function_{function}_dtype_{dtype}_rank_1_block_{operands.BLOCK}_pdl_false"
    try:
        operands.described("elementwise_binary", key)
    except SystemExit as stated:
        return "integer element types alone" in str(stated)
    return False


def main():
    if not operands.ready():
        return 0

    checked = 0

    # Every function on every element type: contiguous against contiguous, and a permuted left
    # operand against one row repeated over its leading axes.
    for dtype in operands.FLOATS + operands.INTEGERS:
        for seed, function in enumerate(FUNCTIONS):
            left = operands.values(dtype, (5, 7), seed)
            right = operands.values(dtype, (5, 7), seed + 50)
            checked += check(
                function,
                dtype,
                "contiguous",
                *operand_pair(function, dtype, left, right),
            )

            left = operands.values(dtype, (3, 4, 5), seed + 100).permute(
                2, 0, 1
            )
            right = operands.values(dtype, (1, 1, 4), seed + 150).expand(
                5, 3, 4
            )
            checked += check(
                function,
                dtype,
                "permuted against a row",
                *operand_pair(function, dtype, left, right),
            )

    # Decided values: a NaN on either side, equal values, signed zeros, an infinity.
    special = [
        float("nan"),
        1.0,
        1.0,
        0.0,
        -0.0,
        float("inf"),
        -2.0,
        3.0,
        float("nan"),
        -1.0,
    ]
    against = [
        1.0,
        float("nan"),
        1.0,
        -0.0,
        0.0,
        float("inf"),
        3.0,
        -2.0,
        float("nan"),
        -1.0,
    ]
    for dtype in operands.FLOATS:
        left = operands.tensor(
            torch.tensor(special, dtype=torch.float64), dtype
        )
        right = operands.tensor(
            torch.tensor(against, dtype=torch.float64), dtype
        )
        for function in EXTREMA + COMPARISONS:
            checked += check(function, dtype, "decided values", left, right)

    # Integer arithmetic at the element type's own edges: a sum and a product that wrap, and a
    # quotient whose operands differ in sign.
    for dtype, numbers, others in (
        ("u8", [250, 255, 16, 7, 0], [10, 255, 16, 2, 9]),
        ("u32", [4294967295, 65536, 7, 0, 123456789], [2, 65536, 2, 5, 1000]),
        ("i16", [32767, -32768, -7, 7, -7], [1, 2, -2, -2, 2]),
        ("i32", [2147483647, -2147483647, -7, 7, -7], [2, 3, -2, -2, 2]),
        (
            "i64",
            [9223372036854775807, -9223372036854775807, -7, 7, -7],
            [2, 3, -2, -2, 2],
        ),
    ):
        element = np.dtype(str(operands.ELEMENTS[dtype]).removeprefix("torch."))
        left = operands.tensor(np.array(numbers, dtype=element), dtype)
        right = operands.tensor(np.array(others, dtype=element), dtype)
        for function in ARITHMETIC:
            checked += check(function, dtype, "the type's edges", left, right)

    # The functions of the integer element types alone, on every one of them: contiguous
    # operands, and a permuted left operand against one row repeated over its leading axes.
    for dtype in operands.INTEGERS:
        for seed, function in enumerate(INTEGRAL):
            left = operands.values(dtype, (5, 7), seed + 200)
            right = operands.values(dtype, (5, 7), seed + 250)
            checked += check(
                function,
                dtype,
                "contiguous",
                *operand_pair(function, dtype, left, right),
            )

            left = operands.values(dtype, (3, 4, 5), seed + 300).permute(
                2, 0, 1
            )
            right = operands.values(dtype, (1, 1, 4), seed + 350)
            left, right = operand_pair(function, dtype, left, right)
            checked += check(
                function,
                dtype,
                "permuted against a row",
                left,
                right.expand(5, 3, 4),
            )

    for dtype in operands.INTEGERS:
        width = WIDTH[dtype]
        signed = dtype in ("i16", "i32", "i64")
        low = -(1 << (width - 1)) if signed else 0
        high = (1 << (width - 1)) - 1 if signed else (1 << width) - 1

        # A remainder takes the divisor's sign: both signs on both sides, an exact multiple, a
        # zero dividend, and the type's two ends. The smallest value by minus one is zero.
        if signed:
            dividends = [7, -7, 7, -7, 6, -6, 0, low, low, high, high, low + 1]
            divisors = [3, 3, -3, -3, 3, -3, 5, -1, 7, -1, low, high]
        else:
            dividends = [7, 6, 0, high, high, 10, high - 1]
            divisors = [3, 3, 5, 7, high, high, high]
        checked += check(
            "remainder",
            dtype,
            "both signs and the type's ends",
            integers(dtype, dividends),
            integers(dtype, divisors),
        )

        # Every pair of the type's extreme bit patterns: each one down a column against each
        # one along a row.
        patterns = [0, 1, -1, low, high, 0x5555555555555555, 0xAAAAAAAAAAAAAAAA]
        count = len(patterns)
        column = integers(dtype, patterns).view(count, 1).expand(count, count)
        row = integers(dtype, patterns).view(1, count).expand(count, count)
        for function in BITWISE:
            checked += check(
                function, dtype, "extreme bit patterns", column, row
            )

        # Each pattern shifted by every count the element's width admits.
        column = integers(dtype, patterns).view(count, 1).expand(count, width)
        row = (
            integers(dtype, list(range(width)))
            .view(1, width)
            .expand(count, width)
        )
        for function in SHIFTS:
            checked += check(
                function, dtype, "every count below the width", column, row
            )

    # A floating element type has no module for an integer function.
    for dtype in operands.FLOATS:
        for function in INTEGRAL:
            if not refused(function, dtype):
                raise SystemExit(
                    f"{function} on {dtype}: the package states a module for a function of "
                    f"the integer element types alone"
                )

    # Every layout as the left operand, against the three broadcasts of a right operand.
    layouts = 0
    for dtype, function in (
        ("f32", "add"),
        ("bf16", "mul"),
        ("i32", "sub"),
        ("f64", "lt"),
    ):
        for name, left in operands.views(dtype, 61):
            shape = tuple(left.shape)
            ones = (1,) * len(shape)
            rights = {
                "contiguous": operands.values(dtype, shape, 71),
                "one value": operands.values(dtype, ones, 72).expand(shape),
                "one row": operands.values(
                    dtype, ones[:-1] + shape[-1:], 73
                ).expand(shape),
            }
            for kind, right in rights.items():
                checked += check(
                    function, dtype, f"{name} against {kind}", left, right
                )
                layouts += 1

    print(
        f"ok: {checked} elements over {len(FUNCTIONS)} functions on "
        f"{len(operands.FLOATS + operands.INTEGERS)} element types, {len(INTEGRAL)} integer "
        f"functions on {len(operands.INTEGERS)} and {layouts} strided and broadcast launches of "
        f"rank 1 to 6 on {operands.where()}, float arithmetic within rounding(1) and the rest "
        f"exact"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
