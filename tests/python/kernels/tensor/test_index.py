"""The index package's two programs against torch: five backend primitives.

Each case launches the module of its descriptor key as ``Index::launch`` does.
A gather traverses the destination: the destination, the source row that
index zero selects and the indices are merged into the axes a program
traverses, and the source's step along the indexed axis travels as a word. A
scatter traverses columns: the three operands along every axis but the
indexed one are merged, and the three steps along it travel as words. Every
result is compared **bit for bit**: the operator's tolerance is exact for all
three functions, a scatter's order and arithmetic being its own.

What is covered, by the backend primitive:

- **``gather``** (an index per destination element) and **``index_select``**
  (an index vector, repeated off the axis by a zero stride), against
  ``torch.gather`` and ``torch.index_select``;
- **``scatter_set``**, **``scatter_add_set``** and **``index_add``**, against
  the operator's statement traversed along the axis in order, so **a repeated
  index** leaves the last source for a set and adds every source for an add,
  in axis order, each addition rounded once into the element type;
- **an index of its type's largest value**, which names no element: a gather
  stores zero for it and a scatter skips it;
- **every element type and every index type**, along the first, a middle and
  the last axis of operands of rank 1 to 5, with a permuted or stepped source,
  permuted indices, columns that span several programs, and a destination that
  stays as it was wherever no index names it;
- **an FP8 (``f8e4m3``) tensor** gathered and set as the bytes its elements
  are, over the same shapes; an addition computes, so the package has no
  ``scatter_add`` module for it.

A BF16 sum is rounded to nearest on a device and truncated by the
interpreter; the reference makes the store the run makes (:func:`stored`), so
the comparison stays exact in both.
"""

import functools
import math

import pytest
import torch

from tests.python.kernels.packages import RefusedLaunchError
from tests.python.kernels.tensor import operands

pytestmark = pytest.mark.unit

DTYPES = operands.FLOATS + operands.INTEGERS
INDICES = ("u8", "u32", "i64")

#: The index that names no element, by the indices' type: its largest value.
ABSENT = {"u8": 2**8 - 1, "u32": 2**32 - 1, "i64": 2**63 - 1}

#: The shapes a case indexes, as (source shape, axis, extent of the other
#: side along the axis).
SHAPES = (
    ((23,), 0, 9),
    ((6, 7), 0, 11),
    ((6, 7), 1, 4),
    ((3, 5, 4), 1, 6),
    ((2, 3, 4, 5), 2, 7),
    ((2, 3, 2, 3, 4), 4, 5),
    ((3, 2 * operands.BLOCK + 9), 0, 4),
)

#: Every shape with every element type, and the index type each takes.
TYPED_SHAPES = [
    pytest.param(
        seed,
        offset,
        dtype,
        INDICES[(seed + offset) % len(INDICES)],
        id=f"{SHAPES[seed][0]}-axis-{SHAPES[seed][1]}-{dtype}",
    )
    for seed in range(len(SHAPES))
    for offset, dtype in enumerate(DTYPES)
]

#: The element and index types of the strided operands' cases.
STRIDED_TYPES = (("bf16", "u32"), ("f32", "i64"), ("i32", "u8"))


def shape_id(seed):
    """A readable id of `SHAPES[seed]`."""
    shape, axis, _ = SHAPES[seed]
    return f"{shape}-axis-{axis}"


def positions(primitives, dtype, shape, seed, extent):
    """Indices below `extent`, repeated where the shape allows, with one in
    eight absent.
    """  # noqa: D205
    generator = torch.Generator().manual_seed(seed)
    drawn = torch.randint(0, extent, shape, generator=generator)
    absent = torch.randint(0, 8, shape, generator=generator) == 0
    return primitives.tensor(torch.where(absent, ABSENT[dtype], drawn), dtype)


def chosen_of(indices):
    """The indices as int64, with a mask of the ones that name an element."""
    chosen = indices.contiguous().to(torch.int64)
    return chosen, chosen != ABSENT[operands.NAMES[indices.dtype]]


def key(function, dtype, indices, rank):
    """`Descriptor`'s `Display`."""
    return (
        f"function_{function}_dtype_{dtype}_indices_{indices}"
        f"_rank_{rank}_block_{operands.BLOCK}_pdl_false"
    )


def launch_gather(primitives, source, indices, destination, axis, dtype):
    """One gather launch, with the words `Index::launch` packs for it.

    `dtype` names the element type where the tensors hold its bytes.
    """
    # The source row that index zero selects, repeated along the axis.
    row = list(source.stride())
    step = row[axis]
    row[axis] = 0
    extents, (_, row, index_strides) = operands.merge(
        list(destination.shape),
        [list(destination.stride()), row, list(indices.stride())],
    )
    elements = destination.numel()
    primitives.launch(
        "index",
        key("gather", dtype, operands.NAMES[indices.dtype], len(extents)),
        operands.grid(elements),
        source,
        indices,
        destination,
        elements,
        *operands.extent_words(extents),
        *operands.stride_words(row),
        *operands.stride_words(index_strides),
        step,
        0,
    )


def launch_scatter(
    primitives, function, source, indices, destination, axis, dtype
):
    """One scatter launch, with the words `Index::launch` packs for it.

    `dtype` names the element type where the tensors hold its bytes.
    """
    off_axis = [other for other in range(source.dim()) if other != axis]
    extents, (source_strides, index_strides, destination_strides) = (
        operands.merge(
            [source.shape[other] for other in off_axis],
            [
                [operand.stride(other) for other in off_axis]
                for operand in (source, indices, destination)
            ],
        )
    )
    columns = math.prod(extents)
    primitives.launch(
        "index",
        key(function, dtype, operands.NAMES[indices.dtype], len(extents)),
        operands.grid(columns),
        source,
        indices,
        destination,
        columns,
        source.shape[axis],
        *operands.extent_words(extents),
        *operands.stride_words(source_strides),
        *operands.stride_words(index_strides),
        *operands.stride_words(destination_strides),
        source.stride(axis),
        indices.stride(axis),
        destination.stride(axis),
        0,
    )


def check_gather(primitives, name, source, indices, axis, dtype=None):
    """A gather held to torch's, with zero wherever the index names no
    element.
    """  # noqa: D205
    dtype = dtype or operands.NAMES[source.dtype]
    destination = primitives.empty(indices.shape, source.dtype)
    launch_gather(primitives, source, indices, destination, axis, dtype)
    chosen, present = chosen_of(indices)
    moved = torch.gather(
        operands.bits(source), axis, torch.where(present, chosen, 0)
    )
    expected = torch.where(present, moved, 0).view(source.dtype)
    label = (
        f"gather {dtype} by {operands.NAMES[indices.dtype]} {name} axis "
        f"{axis} {tuple(indices.shape)}"
    )
    operands.assert_same(label, destination, expected)


def stored(total, element, interpreted):
    """An FP32 sum as the run stores it into `element`."""
    if element == torch.bfloat16 and interpreted:
        # The interpreter keeps the upper sixteen bits where a device rounds to
        # nearest.
        return (
            (total.view(torch.int32) >> 16).to(torch.int16).view(torch.bfloat16)
        )
    return total.to(element)


def added(held, value, interpreted):
    """`held + value` in the operator's arithmetic: one addition, rounded
    once.
    """  # noqa: D205
    element = held.dtype
    if element in (torch.float16, torch.bfloat16, torch.float32):
        total = held.to(torch.float32) + value.to(torch.float32)
        return stored(total, element, interpreted)
    if element == torch.float64:
        return held + value
    # An integer wraps at its own width.
    return (held.to(torch.int64) + value.to(torch.int64)).to(element)


def check_scatter(
    primitives, function, name, source, indices, destination, axis, dtype=None
):
    """A scatter held to the operator's statement, traversed along the axis in
    order.
    """  # noqa: D205
    dtype = dtype or operands.NAMES[source.dtype]
    expected = destination.clone()
    pattern = operands.bits(expected)
    values = source.contiguous()
    chosen, present = chosen_of(indices)
    for along in range(source.shape[axis]):
        at = chosen.select(axis, along).unsqueeze(axis)
        named = present.select(axis, along).unsqueeze(axis)
        # A column whose index names no element rewrites what it holds at
        # position zero.
        target = torch.where(named, at, 0)
        held = torch.gather(pattern, axis, target)
        value = values.select(axis, along).unsqueeze(axis)
        if function == "scatter_add":
            value = added(
                held.view(source.dtype), value, primitives.interpreted
            )
        pattern.scatter_(
            axis, target, torch.where(named, operands.bits(value), held)
        )

    launch_scatter(
        primitives, function, source, indices, destination, axis, dtype
    )
    label = (
        f"{function} {dtype} by {operands.NAMES[indices.dtype]} {name} axis "
        f"{axis} {tuple(source.shape)} into {tuple(destination.shape)}"
    )
    operands.assert_same(label, destination, expected)


def repeated(vector, shape, axis):
    """An index vector along `axis` as indices of `shape`: a zero stride off
    the axis.
    """  # noqa: D205
    lifted = [1] * len(shape)
    lifted[axis] = shape[axis]
    return vector.view(lifted).expand(shape)


def permuted(make, dtype, shape, seed, **arguments):
    """A tensor of `shape` drawn by `make` whose axes lie reversed in
    memory.
    """  # noqa: D205
    order = list(range(len(shape)))[::-1]
    reversed_shape = [shape[axis] for axis in order]
    return make(dtype, reversed_shape, seed, **arguments).permute(order)


def traversed_shape(shape, axis, other):
    """The other side's shape: `shape` with `other` elements along `axis`."""
    traversed = list(shape)
    traversed[axis] = other
    return traversed


@pytest.mark.parametrize(
    ("seed", "offset", "dtype", "index_dtype"), TYPED_SHAPES
)
def test_every_primitive_on_every_element_and_index_type(
    seed, offset, dtype, index_dtype, primitives
):
    shape, axis, other = SHAPES[seed]
    traversed = traversed_shape(shape, axis, other)
    base = 100 * seed + offset
    source = primitives.values(dtype, shape, base)

    # `gather`: an index per destination element.
    indices = positions(primitives, index_dtype, traversed, base, shape[axis])
    check_gather(primitives, "contiguous", source, indices, axis)
    # `index_select`: an index vector, repeated off the axis.
    vector = positions(primitives, index_dtype, (other,), base + 1, shape[axis])
    check_gather(
        primitives,
        "an index vector",
        source,
        repeated(vector, traversed, axis),
        axis,
    )
    # `scatter_set` and `scatter_add_set`: an index per source element, into a
    # longer axis.
    for function in ("scatter_set", "scatter_add"):
        destination = primitives.values(dtype, traversed, base + 2)
        indices = positions(primitives, index_dtype, shape, base + 3, other)
        check_scatter(
            primitives,
            function,
            "contiguous",
            source,
            indices,
            destination,
            axis,
        )
    # `index_add`: an index vector along the source's axis, repeated off it.
    destination = primitives.values(dtype, traversed, base + 4)
    vector = positions(primitives, index_dtype, (shape[axis],), base + 5, other)
    check_scatter(
        primitives,
        "scatter_add",
        "an index vector",
        source,
        repeated(vector, shape, axis),
        destination,
        axis,
    )


@pytest.mark.parametrize("dtype", list(operands.MOVED))
@pytest.mark.parametrize("seed", range(len(SHAPES)), ids=shape_id)
def test_an_element_that_only_moves_is_gathered_and_set(
    seed, dtype, primitives
):
    """As the bytes its elements are."""
    shape, axis, other = SHAPES[seed]
    traversed = traversed_shape(shape, axis, other)
    index_dtype = INDICES[seed % len(INDICES)]
    base = 500 + 10 * seed
    source = primitives.values(dtype, shape, base)
    indices = positions(
        primitives, index_dtype, traversed, base + 1, shape[axis]
    )
    check_gather(primitives, "contiguous", source, indices, axis, dtype)
    vector = positions(primitives, index_dtype, (other,), base + 2, shape[axis])
    check_gather(
        primitives,
        "an index vector",
        source,
        repeated(vector, traversed, axis),
        axis,
        dtype,
    )
    reversed_source = permuted(primitives.values, dtype, shape, base + 3)
    check_gather(
        primitives, "reversed in memory", reversed_source, indices, axis, dtype
    )
    destination = primitives.values(dtype, traversed, base + 4)
    indices = positions(primitives, index_dtype, shape, base + 5, other)
    check_scatter(
        primitives,
        "scatter_set",
        "contiguous",
        source,
        indices,
        destination,
        axis,
        dtype,
    )


@pytest.mark.parametrize("dtype", list(operands.MOVED))
def test_an_element_that_only_moves_has_no_addition(dtype, primitives):
    """An addition computes, and no primitive computes with such an element."""
    with pytest.raises(RefusedLaunchError, match="move as their bytes"):
        primitives.described("index", key("scatter_add", dtype, "u32", 1))


@pytest.mark.parametrize(("dtype", "index_dtype"), STRIDED_TYPES)
@pytest.mark.parametrize("seed", range(5), ids=lambda seed: shape_id(seed + 1))
def test_strided_operands_are_traversed(seed, dtype, index_dtype, primitives):
    """A source and indices whose axes lie reversed in memory, and a stepped
    source.
    """  # noqa: D205
    shape, axis, other = SHAPES[seed + 1]
    traversed = traversed_shape(shape, axis, other)
    make_positions = functools.partial(positions, primitives)
    source = permuted(primitives.values, dtype, shape, 700 + seed)
    indices = permuted(
        make_positions, index_dtype, traversed, 710 + seed, extent=shape[axis]
    )
    check_gather(primitives, "reversed in memory", source, indices, axis)
    stepped = primitives.values(
        dtype, [2 * extent for extent in shape], 720 + seed
    )
    stepped = stepped[tuple(slice(1, None, 2) for _ in shape)]
    check_gather(primitives, "stepped", stepped, indices, axis)
    for function in ("scatter_set", "scatter_add"):
        destination = primitives.values(dtype, traversed, 730 + seed)
        indices = permuted(
            make_positions, index_dtype, shape, 740 + seed, extent=other
        )
        check_scatter(
            primitives,
            function,
            "reversed in memory",
            source,
            indices,
            destination,
            axis,
        )


@pytest.mark.parametrize("function", ["scatter_set", "scatter_add"])
@pytest.mark.parametrize("dtype", ["bf16", "f16", "f32", "f64", "i16", "u8"])
def test_one_index_repeated_along_the_whole_axis(dtype, function, primitives):
    """A set leaves the last source, an add sums them all."""
    source = primitives.values(dtype, (40, 3), 800)
    destination = primitives.values(dtype, (5, 3), 801)
    indices = primitives.tensor(torch.full((40, 3), 2), "u32")
    check_scatter(
        primitives,
        function,
        "one index repeated",
        source,
        indices,
        destination,
        0,
    )
