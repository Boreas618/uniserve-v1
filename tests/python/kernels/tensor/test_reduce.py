"""The reduce program against numpy, for every function.

Each case launches the module of its descriptor key as ``Reduce::launch`` does
(the input split into the axes the call keeps and the axes it reduces, each
merged into the axes a program traverses, their extents and strides packed
right-aligned) and compares the output with numpy on the same values: a
floating sum inside the bound of its class (``Reduce::tolerance``), everything
else exactly.

What is covered:

- **every function on every element type**, over a reduction shorter than a
  tile, where one program folds several outputs, and one of several tiles
  with a masked tail, where it folds one;
- **a block of 1024 elements**, the one the backend launches a primitive with,
  where one program folds several hundred outputs and each of its threads
  several of them: every function over two and three elements per output,
  with the outputs adjacent in memory and with the reduced elements adjacent;
- **every choice of axes** of a rank-4 input: the last, the first, a middle
  one that splits the kept axes in two, two that are not adjacent, and all of
  them; an index along each single axis;
- **every layout** of :data:`~tests.python.kernels.tensor.operands.LAYOUTS`
  reduced along its last axis, its first axis and all of its axes;
- **the largest magnitude** of every floating element type, over the same
  lengths and layouts, and of infinities of either sign;
- **ties**: the first of equal values wins an index, within a tile and across
  two tiles;
- **an integer sum that wraps**, and **a sum along an axis of no element**,
  which is zero.
"""

import math

import numpy as np
import pytest
import torch

from tests.python.kernels.tensor import operands

pytestmark = pytest.mark.unit

DTYPES = operands.FLOATS + operands.INTEGERS
FUNCTIONS = ("sum", "min", "max", "argmin", "argmax")
INDEXES = ("argmin", "argmax")

#: The block of the cases that fold several hundred outputs in one program:
#: the backend's (`BLOCK` of `crates/ops/src/primitives.rs`).
WIDE_BLOCK = 1024

#: Every choice of axes of a rank-4 input.
CHOICES = (
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

#: What every layout is reduced by: a function and the axes, by the view's
#: rank.
LAYOUT_REDUCTIONS = {
    "sum along the last axis": ("sum", lambda rank: [rank - 1]),
    "max along the first axis": ("max", lambda rank: [0]),
    "min along every axis": ("min", lambda rank: list(range(rank))),
    "sum along every axis": ("sum", lambda rank: list(range(rank))),
    "argmax along the last axis": ("argmax", lambda rank: [rank - 1]),
    "argmin along the first axis": ("argmin", lambda rank: [0]),
}

#: The integer sums that pass their element's range: the type and the value
#: summed nine times.
WRAPPING = (
    ("u8", 200),
    ("i16", 30000),
    ("i32", 2**30),
    ("u32", 2**31),
    ("i64", 2**62),
)


def reduction_class(length):
    """`reduce::reduction_class`: the least power of two that holds
    `length`.
    """  # noqa: D205
    return 1 << max(length - 1, 0).bit_length()


def expected_of(function, x, axes):
    """The reduction in numpy: float64 for a floating sum, exact values
    otherwise.
    """  # noqa: D205
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
    if function == "absmax":
        return np.abs(wide).max(axis=tuple(axes), keepdims=True)
    (axis,) = axes
    # Numpy returns the first of equal values, as the operator does.
    found = (
        wide.argmin(axis=axis)
        if function == "argmin"
        else wide.argmax(axis=axis)
    )
    return np.expand_dims(found, axis)


def check(primitives, function, name, x, axes, block=operands.BLOCK):
    """One launch of a module of `block` elements, held to numpy and the
    operator's bound.
    """  # noqa: D205
    dtype = operands.NAMES[x.dtype]
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
    out = primitives.empty(
        folded, torch.uint32 if function in INDEXES else x.dtype
    )
    # `Descriptor`'s `Display`: the tile is the class, capped at the block.
    tile = min(reduction_class(length), block)
    key = (
        f"function_{function}_dtype_{dtype}_kept_{len(kept)}"
        f"_reduced_{len(reduced)}_tile_{tile}_block_{block}_pdl_false"
    )
    primitives.launch(
        "reduce",
        key,
        operands.grid(outputs, block // tile),
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
    primitives.assert_close(label, out, expected_of(function, x, axes), bound)


@pytest.mark.parametrize(("seed", "function"), list(enumerate(FUNCTIONS)))
@pytest.mark.parametrize("dtype", DTYPES)
def test_a_function_folds_a_short_reduction_and_several_tiles(
    dtype, seed, function, primitives
):
    short = primitives.values(dtype, (37, 5), seed)
    long = primitives.values(dtype, (3, 2 * operands.BLOCK + 22), seed + 10)
    check(primitives, function, "shorter than a tile", short, [1])
    check(primitives, function, "several tiles", long, [1])


@pytest.mark.parametrize(("seed", "dtype"), list(enumerate(operands.FLOATS)))
def test_the_largest_magnitude_of_a_floating_type(seed, dtype, primitives):
    """A short reduction, several tiles, a wide block and infinities of either
    sign.
    """  # noqa: D205
    short = primitives.values(dtype, (37, 5), seed + 60)
    long = primitives.values(dtype, (3, 2 * operands.BLOCK + 22), seed + 70)
    wide = primitives.values(dtype, (300, 3), seed + 80)
    edges = primitives.tensor(
        [[float("-inf"), 1.0, 0.0, 2.0], [-3.0, 2.0, 1.0, -0.5]], dtype
    )
    check(primitives, "absmax", "shorter than a tile", short, [1])
    check(primitives, "absmax", "several tiles", long, [1])
    check(primitives, "absmax", "a wide block", wide, [1], WIDE_BLOCK)
    check(primitives, "absmax", "infinities", edges, [1])


@pytest.mark.parametrize("layout", operands.LAYOUTS)
def test_the_largest_magnitude_of_every_layout(layout, primitives):
    view = primitives.view(layout, "bf16", 31)
    check(primitives, "absmax", layout, view, [view.dim() - 1])
    check(primitives, "absmax", layout, view, list(range(view.dim())))


@pytest.mark.parametrize(("seed", "function"), list(enumerate(FUNCTIONS)))
@pytest.mark.parametrize("dtype", ["f32", "f64", "bf16", "i32"])
def test_a_program_folds_several_hundred_outputs(
    dtype, seed, function, primitives
):
    """Each thread of it folding several: the outputs adjacent in memory and
    the reduced elements a row apart, then the reduced elements adjacent.
    """  # noqa: D205
    apart = primitives.values(dtype, (2, 300), seed + 40)
    adjacent = primitives.values(dtype, (300, 3), seed + 50)
    check(primitives, function, "a wide block", apart, [0], WIDE_BLOCK)
    check(primitives, function, "a wide block", adjacent, [1], WIDE_BLOCK)


@pytest.mark.parametrize("dtype", ["f32", "bf16", "i32"])
def test_every_choice_of_axes_of_a_rank_4_input(dtype, primitives):
    x = primitives.values(dtype, (2, 3, 4, 5), 20)
    for axes in CHOICES:
        for function in ("sum", "min", "max"):
            check(primitives, function, "rank 4", x, axes)
    for axis in range(4):
        for function in INDEXES:
            check(primitives, function, "rank 4", x, [axis])


@pytest.mark.parametrize("reduction", LAYOUT_REDUCTIONS)
@pytest.mark.parametrize("layout", operands.LAYOUTS)
@pytest.mark.parametrize("dtype", ["f32", "i64", "bf16"])
def test_every_layout_is_reduced(dtype, layout, reduction, primitives):
    view = primitives.view(layout, dtype, 30)
    function, axes = LAYOUT_REDUCTIONS[reduction]
    check(primitives, function, layout, view, axes(view.dim()))


@pytest.mark.parametrize("function", INDEXES + ("min", "max"))
@pytest.mark.parametrize("dtype", ["f32", "i32", "u8"])
def test_the_first_of_equal_values_wins(dtype, function, primitives):
    """Within one tile, down a column, and across two tiles."""
    across = [0] * (operands.BLOCK + 40)
    across[10] = across[operands.BLOCK + 5] = 7
    within = primitives.tensor(
        [[1, 3, 3, 1], [2, 2, 2, 2], [0, 5, 0, 5]], dtype
    )
    spread = primitives.tensor(
        [across, [-value for value in across] if dtype != "u8" else across],
        dtype,
    )
    check(primitives, function, "ties within a tile", within, [1])
    check(primitives, function, "ties across tiles", spread, [1])
    check(primitives, function, "ties down a column", within, [0])


@pytest.mark.parametrize("function", ("min", "max") + INDEXES)
@pytest.mark.parametrize("dtype", operands.FLOATS)
def test_an_infinity_is_an_element_like_any_other(dtype, function, primitives):
    """And a row of one value returns its first index."""
    edges = primitives.tensor(
        [
            [float("-inf")] * 4,
            [1.0, float("inf"), float("inf"), 0.0],
            [-2.0, float("-inf"), 3.0, 3.0],
        ],
        dtype,
    )
    check(primitives, function, "infinities", edges, [1])


@pytest.mark.parametrize(("dtype", "fill"), WRAPPING)
def test_an_integer_sum_wraps(dtype, fill, primitives):
    x = primitives.tensor(np.full((3, 9), fill), dtype)
    check(primitives, "sum", "a sum that wraps", x, [1])


@pytest.mark.parametrize("dtype", ["f32", "bf16", "i64"])
def test_a_sum_along_an_axis_of_no_element_is_zero(dtype, primitives):
    check(
        primitives,
        "sum",
        "no element",
        primitives.values(dtype, (4, 0), 1),
        [1],
    )
    check(
        primitives,
        "sum",
        "no element",
        primitives.values(dtype, (3, 0, 5), 2),
        [1],
    )
