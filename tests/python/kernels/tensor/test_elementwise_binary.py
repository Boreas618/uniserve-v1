"""The elementwise_binary program against numpy, for every function.

Each case launches the package's kernel as ``ElementwiseBinary::launch`` does
(the two inputs and the output merged into the axes a program traverses, the
extents and both stride vectors packed right-aligned) and compares the output
with numpy on the same values: arithmetic inside the operator's bound
(``ElementwiseBinary::tolerance``), everything else exactly.

What is covered:

- **every function on every element type**, over two contiguous inputs and
  over a permuted input against a broadcast one;
- **integer arithmetic** in the element type itself: sums and products that
  wrap, and quotients of either sign, which truncate toward zero where a floor
  would differ;
- **the functions of the integer element types alone** on every one of them:
  ``remainder`` over dividends and divisors of both signs, where it takes the
  divisor's sign and the smallest value by minus one is zero; the three
  ``bitwise_`` combinations over every pair of the type's extreme bit
  patterns; and the two shifts by every count below the element's width, a
  right shift repeating the sign bit of a signed element. A floating element
  type has no module for them, which the package states;
- **the comparisons and the extrema on a NaN, on equal values and on signed
  zeros**: a comparison writes ``u8``, and ``maximum``/``minimum`` return the
  left operand unless the right is strictly beyond;
- **every layout** of :data:`~uniserve_testing.kernels.tensor.operands.LAYOUTS`
  as the left operand, against a right operand that is contiguous, one value
  repeated everywhere (every stride zero), and one row repeated over the
  leading axes: the three broadcasts a tensor library's arithmetic makes.
"""

import numpy as np
import pytest
import torch
from uniserve_testing.kernels.packages import RefusedLaunchError
from uniserve_testing.kernels.tensor import operands

pytestmark = pytest.mark.unit

ARITHMETIC = ("add", "sub", "mul", "div")
EXTREMA = ("maximum", "minimum")
COMPARISONS = ("eq", "ne", "lt", "le", "gt", "ge")
FUNCTIONS = ARITHMETIC + EXTREMA + COMPARISONS

#: The functions that exist on the integer element types alone
#: (`BinaryFunction::integral`).
BITWISE = ("bitwise_and", "bitwise_or", "bitwise_xor")
SHIFTS = ("bitwise_left_shift", "bitwise_right_shift")
INTEGRAL = ("remainder",) + BITWISE + SHIFTS

#: The width in bits of each integer element type.
WIDTH = {"u8": 8, "u32": 32, "i16": 16, "i32": 32, "i64": 64}

#: Decided values, a NaN on either side, equal values, signed zeros and an
#: infinity, against their partners.
SPECIAL = [
    float("nan"), 1.0, 1.0, 0.0, -0.0, float("inf"), -2.0, 3.0, float("nan"),
    -1.0,
]  # fmt: skip
AGAINST = [
    1.0, float("nan"), 1.0, -0.0, 0.0, float("inf"), 3.0, -2.0, float("nan"),
    -1.0,
]  # fmt: skip

#: Integer arithmetic at each element type's own edges: a sum and a product
#: that wrap, and a quotient whose operands differ in sign.
EDGES = {
    "u8": ([250, 255, 16, 7, 0], [10, 255, 16, 2, 9]),
    "u32": ([4294967295, 65536, 7, 0, 123456789], [2, 65536, 2, 5, 1000]),
    "i16": ([32767, -32768, -7, 7, -7], [1, 2, -2, -2, 2]),
    "i32": ([2147483647, -2147483647, -7, 7, -7], [2, 3, -2, -2, 2]),
    "i64": (
        [9223372036854775807, -9223372036854775807, -7, 7, -7],
        [2, 3, -2, -2, 2],
    ),
}

#: The functions every layout is the left operand of.
LAYOUT_FUNCTIONS = (
    ("f32", "add"),
    ("bf16", "mul"),
    ("i32", "sub"),
    ("f64", "lt"),
)

#: The right operands every layout is combined with.
BROADCASTS = ("contiguous", "one value", "one row")


def expected_of(function, left, right):
    """`left o right` in numpy: float64 for a floating type, the element type
    for an integer.
    """  # noqa: D205
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
            # numpy's remainder is the floored quotient's, as
            # `torch.remainder` is, and its right shift repeats a signed
            # element's sign bit.
            "remainder": lambda: np.remainder(left, right),
            "bitwise_and": lambda: left & right,
            "bitwise_or": lambda: left | right,
            "bitwise_xor": lambda: left ^ right,
            "bitwise_left_shift": lambda: np.left_shift(left, right),
            "bitwise_right_shift": lambda: np.right_shift(left, right),
        }[function]()


def numpy_values(tensor):
    """A tensor's values for the reference: float64, or its own integer
    type.
    """  # noqa: D205
    wide = operands.real(tensor)
    if tensor.dtype.is_floating_point:
        return wide
    return wide.astype(np.dtype(str(tensor.dtype).removeprefix("torch.")))


def key(function, dtype, rank):
    """`BinaryDescriptor`'s `Display`."""
    return (
        f"function_{function}_dtype_{dtype}_rank_{rank}"
        f"_block_{operands.BLOCK}_pdl_false"
    )


def check(primitives, function, dtype, name, left, right):
    """One launch held to numpy and the operator's bound."""
    compares = function in COMPARISONS
    out = primitives.empty(left.shape, torch.uint8 if compares else left.dtype)
    extents, (left_strides, right_strides, _) = operands.merged(
        [left, right, out]
    )
    elements = out.numel()
    primitives.launch(
        "elementwise_binary",
        key(function, dtype, len(extents)),
        operands.grid(elements),
        left,
        right,
        out,
        elements,
        *operands.extent_words(extents),
        *operands.stride_words(left_strides),
        *operands.stride_words(right_strides),
    )

    expected = expected_of(function, numpy_values(left), numpy_values(right))
    bound = operands.EXACT
    if function in ARITHMETIC and dtype in operands.FLOATS:
        # `ElementwiseBinary::tolerance`: one operation, rounded once into the
        # element type.
        bound = operands.rounding(1, dtype)
    label = f"{function} {dtype} {name} {tuple(left.shape)}"
    primitives.assert_close(label, out, expected, bound)


def nonzero(tensor):
    """An integer divisor with its zeros replaced, of either sign where the
    type has one.
    """  # noqa: D205
    return torch.where(tensor == 0, torch.full_like(tensor, 3), tensor)


def operand_pair(primitives, function, dtype, left, right):
    """The two operands of a case, the right one held to what the function
    defines.

    An integer quotient and a remainder never divide by zero, and a shift count
    is below the element's width.
    """  # noqa: D205
    if function in ("div", "remainder") and dtype in operands.INTEGERS:
        right = nonzero(right)
    if function in SHIFTS:
        counts = operands.real(right) % WIDTH[dtype]
        right = primitives.tensor(counts, dtype).view(right.shape)
    return left, right


def integers(primitives, dtype, numbers):
    """`numbers` as a tensor of the integer element type `dtype`, wrapped into
    its range.
    """  # noqa: D205
    element = np.dtype(str(operands.ELEMENTS[dtype]).removeprefix("torch."))
    wrapped = np.array(numbers, dtype=object) % (1 << WIDTH[dtype])
    return primitives.tensor(wrapped.astype(np.uint64).astype(element), dtype)


def extremes(dtype):
    """The type's extreme bit patterns, and its smallest and largest value."""
    width = WIDTH[dtype]
    signed = dtype in ("i16", "i32", "i64")
    low = -(1 << (width - 1)) if signed else 0
    high = (1 << (width - 1)) - 1 if signed else (1 << width) - 1
    patterns = [0, 1, -1, low, high, 0x5555555555555555, 0xAAAAAAAAAAAAAAAA]
    return patterns, low, high


@pytest.mark.parametrize(("seed", "function"), list(enumerate(FUNCTIONS)))
@pytest.mark.parametrize("dtype", operands.FLOATS + operands.INTEGERS)
def test_a_function_matches_numpy_on_every_element_type(
    dtype, seed, function, primitives
):
    """Contiguous against contiguous, and a permuted left operand against one
    row repeated over its leading axes.
    """  # noqa: D205
    left = primitives.values(dtype, (5, 7), seed)
    right = primitives.values(dtype, (5, 7), seed + 50)
    check(
        primitives,
        function,
        dtype,
        "contiguous",
        *operand_pair(primitives, function, dtype, left, right),
    )
    left = primitives.values(dtype, (3, 4, 5), seed + 100).permute(2, 0, 1)
    right = primitives.values(dtype, (1, 1, 4), seed + 150).expand(5, 3, 4)
    check(
        primitives,
        function,
        dtype,
        "permuted against a row",
        *operand_pair(primitives, function, dtype, left, right),
    )


@pytest.mark.parametrize("function", EXTREMA + COMPARISONS)
@pytest.mark.parametrize("dtype", operands.FLOATS)
def test_a_comparison_decides_nans_equal_values_and_signed_zeros(
    dtype, function, primitives
):
    left = primitives.tensor(torch.tensor(SPECIAL, dtype=torch.float64), dtype)
    right = primitives.tensor(torch.tensor(AGAINST, dtype=torch.float64), dtype)
    check(primitives, function, dtype, "decided values", left, right)


@pytest.mark.parametrize("function", ARITHMETIC)
@pytest.mark.parametrize("dtype", list(EDGES))
def test_integer_arithmetic_wraps_at_the_types_edges(
    dtype, function, primitives
):
    numbers, others = EDGES[dtype]
    element = np.dtype(str(operands.ELEMENTS[dtype]).removeprefix("torch."))
    left = primitives.tensor(np.array(numbers, dtype=element), dtype)
    right = primitives.tensor(np.array(others, dtype=element), dtype)
    check(primitives, function, dtype, "the type's edges", left, right)


@pytest.mark.parametrize(("seed", "function"), list(enumerate(INTEGRAL)))
@pytest.mark.parametrize("dtype", operands.INTEGERS)
def test_an_integer_function_matches_numpy(dtype, seed, function, primitives):
    """Contiguous operands, and a permuted left operand against one row
    repeated over its leading axes.
    """  # noqa: D205
    left = primitives.values(dtype, (5, 7), seed + 200)
    right = primitives.values(dtype, (5, 7), seed + 250)
    check(
        primitives,
        function,
        dtype,
        "contiguous",
        *operand_pair(primitives, function, dtype, left, right),
    )
    left = primitives.values(dtype, (3, 4, 5), seed + 300).permute(2, 0, 1)
    right = primitives.values(dtype, (1, 1, 4), seed + 350)
    left, right = operand_pair(primitives, function, dtype, left, right)
    check(
        primitives,
        function,
        dtype,
        "permuted against a row",
        left,
        right.expand(5, 3, 4),
    )


@pytest.mark.parametrize("dtype", operands.INTEGERS)
def test_a_remainder_takes_the_divisors_sign(dtype, primitives):
    """Both signs on both sides, an exact multiple, a zero dividend, and the
    type's two ends. The smallest value by minus one is zero.
    """  # noqa: D205
    _, low, high = extremes(dtype)
    if dtype in ("i16", "i32", "i64"):
        dividends = [7, -7, 7, -7, 6, -6, 0, low, low, high, high, low + 1]
        divisors = [3, 3, -3, -3, 3, -3, 5, -1, 7, -1, low, high]
    else:
        dividends = [7, 6, 0, high, high, 10, high - 1]
        divisors = [3, 3, 5, 7, high, high, high]
    check(
        primitives,
        "remainder",
        dtype,
        "both signs and the type's ends",
        integers(primitives, dtype, dividends),
        integers(primitives, dtype, divisors),
    )


@pytest.mark.parametrize("function", BITWISE)
@pytest.mark.parametrize("dtype", operands.INTEGERS)
def test_a_bitwise_combination_of_every_pair_of_extreme_patterns(
    dtype, function, primitives
):
    """Each pattern down a column against each one along a row."""
    patterns, _, _ = extremes(dtype)
    count = len(patterns)
    column = integers(primitives, dtype, patterns).view(count, 1)
    row = integers(primitives, dtype, patterns).view(1, count)
    check(
        primitives,
        function,
        dtype,
        "extreme bit patterns",
        column.expand(count, count),
        row.expand(count, count),
    )


@pytest.mark.parametrize("function", SHIFTS)
@pytest.mark.parametrize("dtype", operands.INTEGERS)
def test_a_shift_by_every_count_below_the_width(dtype, function, primitives):
    patterns, _, _ = extremes(dtype)
    count, width = len(patterns), WIDTH[dtype]
    column = integers(primitives, dtype, patterns).view(count, 1)
    row = integers(primitives, dtype, list(range(width))).view(1, width)
    check(
        primitives,
        function,
        dtype,
        "every count below the width",
        column.expand(count, width),
        row.expand(count, width),
    )


@pytest.mark.parametrize("function", INTEGRAL)
@pytest.mark.parametrize("dtype", operands.FLOATS)
def test_a_floating_type_has_no_module_of_an_integer_function(
    dtype, function, primitives
):
    with pytest.raises(RefusedLaunchError, match="integer element types alone"):
        primitives.described("elementwise_binary", key(function, dtype, 1))


@pytest.mark.parametrize("broadcast", BROADCASTS)
@pytest.mark.parametrize("layout", operands.LAYOUTS)
@pytest.mark.parametrize(("dtype", "function"), LAYOUT_FUNCTIONS)
def test_every_layout_against_every_broadcast(
    dtype, function, layout, broadcast, primitives
):
    left = primitives.view(layout, dtype, 61)
    shape = tuple(left.shape)
    ones = (1,) * len(shape)
    if broadcast == "contiguous":
        right = primitives.values(dtype, shape, 71)
    elif broadcast == "one value":
        right = primitives.values(dtype, ones, 72).expand(shape)
    else:
        right = primitives.values(dtype, ones[:-1] + shape[-1:], 73)
        right = right.expand(shape)
    check(
        primitives,
        function,
        dtype,
        f"{layout} against {broadcast}",
        left,
        right,
    )
