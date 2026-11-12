#!/usr/bin/env python3
"""Run `index`'s two kernels and check the five backend primitives against torch, on the CPU.

    python index_check.py             # Triton's interpreter, no device
    python index_check.py --device    # compiled for a device

Each case launches the module of its descriptor key as `Index::launch` does. A gather traverses the
destination: the destination, the source row that index zero selects and the indices are merged into
the axes a program traverses, and the source's step along the indexed axis travels as a word. A
scatter traverses columns: the three operands along every axis but the indexed one are merged, and
the three steps along it travel as words. Every result is compared **bit for bit**: the operator's
tolerance is exact for all three functions, a scatter's order and arithmetic being its own.

What is covered, by the backend primitive:

- **`gather`** (an index per destination element) and **`index_select`** (an index vector, repeated
  off the axis by a zero stride), against `torch.gather` and `torch.index_select`;
- **`scatter_set`**, **`scatter_add_set`** and **`index_add`**, against the operator's statement
  traversed along the axis in order — so **a repeated index** leaves the last source for a set and
  adds every source for an add, in axis order, each addition rounded once into the element type;
- **an index of its type's largest value**, which names no element: a gather stores zero for it and
  a scatter skips it;
- **every element type and every index type**, along the first, a middle and the last axis of
  operands of rank 1 to 5, with a permuted or stepped source, permuted indices, columns that span
  several programs, and a destination that stays as it was wherever no index names it;
- **an FP8 (`f8e4m3`) tensor** gathered and set as the bytes its elements are, over the same
  shapes; an addition computes, so the package has no `scatter_add` module for it.

A BF16 sum is rounded to nearest on a device and truncated by the interpreter; the reference makes
the store the run makes (`stored`), so the comparison stays exact in both.
"""

import math
import sys

# `operands` chooses between the interpreter and a device, and is imported before Triton.
import operands
import torch

DTYPES = operands.FLOATS + operands.INTEGERS
INDICES = ("u8", "u32", "i64")

#: The index that names no element, by the indices' type: its largest value.
ABSENT = {"u8": 2**8 - 1, "u32": 2**32 - 1, "i64": 2**63 - 1}

#: The name `ops::DType` prints for each torch element type.
NAMES = {element: name for name, element in operands.ELEMENTS.items()}


def positions(dtype, shape, seed, extent):
    """Indices below `extent`, repeated where the shape allows, with one in eight absent."""
    generator = torch.Generator().manual_seed(seed)
    drawn = torch.randint(0, extent, shape, generator=generator)
    absent = torch.randint(0, 8, shape, generator=generator) == 0
    return operands.tensor(torch.where(absent, ABSENT[dtype], drawn), dtype)


def chosen_of(indices):
    """The indices as int64, with a mask of the ones that name an element."""
    chosen = indices.contiguous().to(torch.int64)
    return chosen, chosen != ABSENT[NAMES[indices.dtype]]


def launch_gather(source, indices, destination, axis, dtype=None):
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

    # `Descriptor`'s `Display`.
    key = (
        f"function_gather_dtype_{dtype or NAMES[source.dtype]}_indices_{NAMES[indices.dtype]}"
        f"_rank_{len(extents)}_block_{operands.BLOCK}_pdl_false"
    )
    operands.launch(
        "index",
        key,
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


def launch_scatter(function, source, indices, destination, axis, dtype=None):
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

    # `Descriptor`'s `Display`.
    key = (
        f"function_{function}_dtype_{dtype or NAMES[source.dtype]}_indices_{NAMES[indices.dtype]}"
        f"_rank_{len(extents)}_block_{operands.BLOCK}_pdl_false"
    )
    operands.launch(
        "index",
        key,
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


def check_gather(name, source, indices, axis, dtype=None):
    """A gather held to torch's, with zero wherever the index names no element."""
    destination = operands.empty(indices.shape, source.dtype)
    launch_gather(source, indices, destination, axis, dtype)

    chosen, present = chosen_of(indices)
    moved = torch.gather(
        operands.bits(source), axis, torch.where(present, chosen, 0)
    )
    expected = torch.where(present, moved, 0).view(source.dtype)
    label = (
        f"gather {dtype or NAMES[source.dtype]} by {NAMES[indices.dtype]} {name} axis {axis} "
        f"{tuple(indices.shape)}"
    )
    return operands.same(label, destination, expected)


def stored(total, element):
    """An FP32 sum as the run stores it into `element`."""
    if element == torch.bfloat16 and not operands.DEVICE:
        # The interpreter keeps the upper sixteen bits where a device rounds to nearest.
        return (
            (total.view(torch.int32) >> 16).to(torch.int16).view(torch.bfloat16)
        )
    return total.to(element)


def added(held, value):
    """`held + value` in the operator's arithmetic: one addition, rounded once."""
    element = held.dtype
    if element in (torch.float16, torch.bfloat16, torch.float32):
        return stored(held.to(torch.float32) + value.to(torch.float32), element)
    if element == torch.float64:
        return held + value
    # An integer wraps at its own width.
    return (held.to(torch.int64) + value.to(torch.int64)).to(element)


def check_scatter(
    function, name, source, indices, destination, axis, dtype=None
):
    """A scatter held to the operator's statement, traversed along the axis in order."""
    expected = destination.clone()
    pattern = operands.bits(expected)
    values = source.contiguous()
    chosen, present = chosen_of(indices)

    for along in range(source.shape[axis]):
        at = chosen.select(axis, along).unsqueeze(axis)
        named = present.select(axis, along).unsqueeze(axis)
        # A column whose index names no element rewrites what it holds at position zero.
        target = torch.where(named, at, 0)
        held = torch.gather(pattern, axis, target)
        value = values.select(axis, along).unsqueeze(axis)
        if function == "scatter_add":
            value = added(held.view(source.dtype), value)
        pattern.scatter_(
            axis, target, torch.where(named, operands.bits(value), held)
        )

    launch_scatter(function, source, indices, destination, axis, dtype)
    label = (
        f"{function} {dtype or NAMES[source.dtype]} by {NAMES[indices.dtype]} {name} axis {axis} "
        f"{tuple(source.shape)} into {tuple(destination.shape)}"
    )
    return operands.same(label, destination, expected)


def repeated(vector, shape, axis):
    """An index vector along `axis` as indices of `shape`: a zero stride off the axis."""
    lifted = [1] * len(shape)
    lifted[axis] = shape[axis]
    return vector.view(lifted).expand(shape)


def permuted(dtype, shape, seed, make=operands.values, **arguments):
    """A tensor of `shape` whose axes lie reversed in memory."""
    order = list(range(len(shape)))[::-1]
    reversed_shape = [shape[axis] for axis in order]
    return make(dtype, reversed_shape, seed, **arguments).permute(order)


#: The shapes a case indexes, as (source shape, axis, extent of the other side along the axis).
SHAPES = (
    ((23,), 0, 9),
    ((6, 7), 0, 11),
    ((6, 7), 1, 4),
    ((3, 5, 4), 1, 6),
    ((2, 3, 4, 5), 2, 7),
    ((2, 3, 2, 3, 4), 4, 5),
    ((3, 2 * operands.BLOCK + 9), 0, 4),
)


def main():
    if not operands.ready():
        return 0

    checked = 0
    cases = 0

    for seed, (shape, axis, other) in enumerate(SHAPES):
        traversed = list(shape)
        traversed[axis] = other

        for offset, dtype in enumerate(DTYPES):
            index_dtype = INDICES[(seed + offset) % len(INDICES)]
            source = operands.values(dtype, shape, 100 * seed + offset)

            # `gather`: an index per destination element.
            indices = positions(
                index_dtype, traversed, 100 * seed + offset, shape[axis]
            )
            checked += check_gather("contiguous", source, indices, axis)

            # `index_select`: an index vector, repeated off the axis.
            vector = positions(
                index_dtype, (other,), 100 * seed + offset + 1, shape[axis]
            )
            checked += check_gather(
                "an index vector",
                source,
                repeated(vector, traversed, axis),
                axis,
            )

            # `scatter_set` and `scatter_add_set`: an index per source element, into a longer axis.
            for function in ("scatter_set", "scatter_add"):
                destination = operands.values(
                    dtype, traversed, 100 * seed + offset + 2
                )
                indices = positions(
                    index_dtype, shape, 100 * seed + offset + 3, other
                )
                checked += check_scatter(
                    function, "contiguous", source, indices, destination, axis
                )

            # `index_add`: an index vector along the source's axis, repeated off it.
            destination = operands.values(
                dtype, traversed, 100 * seed + offset + 4
            )
            vector = positions(
                index_dtype, (shape[axis],), 100 * seed + offset + 5, other
            )
            indices = repeated(vector, shape, axis)
            checked += check_scatter(
                "scatter_add",
                "an index vector",
                source,
                indices,
                destination,
                axis,
            )
            cases += 5

    # An FP8 tensor is gathered and set as the bytes its elements are, over the same shapes.
    for seed, (shape, axis, other) in enumerate(SHAPES):
        traversed = list(shape)
        traversed[axis] = other

        for dtype in operands.MOVED:
            index_dtype = INDICES[seed % len(INDICES)]
            source = operands.values(dtype, shape, 500 + 10 * seed)

            indices = positions(
                index_dtype, traversed, 501 + 10 * seed, shape[axis]
            )
            checked += check_gather("contiguous", source, indices, axis, dtype)

            vector = positions(
                index_dtype, (other,), 502 + 10 * seed, shape[axis]
            )
            checked += check_gather(
                "an index vector",
                source,
                repeated(vector, traversed, axis),
                axis,
                dtype,
            )

            reversed_source = permuted(dtype, shape, 503 + 10 * seed)
            checked += check_gather(
                "reversed in memory", reversed_source, indices, axis, dtype
            )

            destination = operands.values(dtype, traversed, 504 + 10 * seed)
            indices = positions(index_dtype, shape, 505 + 10 * seed, other)
            checked += check_scatter(
                "scatter_set",
                "contiguous",
                source,
                indices,
                destination,
                axis,
                dtype,
            )
            cases += 4

    # An addition computes, and no primitive computes with an element that only moves.
    for dtype in operands.MOVED:
        key = f"function_scatter_add_dtype_{dtype}_indices_u32_rank_1_block_{operands.BLOCK}_pdl_false"
        try:
            operands.described("index", key)
        except SystemExit as stated:
            if "move as their bytes" not in str(stated):
                raise
        else:
            raise SystemExit(
                f"{key}: the package states a module that adds {dtype} elements"
            )

    # Strided operands: a source and indices whose axes lie reversed in memory, and a stepped one.
    for seed, (shape, axis, other) in enumerate(SHAPES[1:6]):
        traversed = list(shape)
        traversed[axis] = other
        for dtype, index_dtype in (
            ("bf16", "u32"),
            ("f32", "i64"),
            ("i32", "u8"),
        ):
            source = permuted(dtype, shape, 700 + seed)
            indices = permuted(
                index_dtype,
                traversed,
                710 + seed,
                make=positions,
                extent=shape[axis],
            )
            checked += check_gather("reversed in memory", source, indices, axis)

            stepped = operands.values(
                dtype, [2 * extent for extent in shape], 720 + seed
            )
            stepped = stepped[tuple(slice(1, None, 2) for _ in shape)]
            checked += check_gather("stepped", stepped, indices, axis)

            for function in ("scatter_set", "scatter_add"):
                destination = operands.values(dtype, traversed, 730 + seed)
                indices = permuted(
                    index_dtype, shape, 740 + seed, make=positions, extent=other
                )
                checked += check_scatter(
                    function,
                    "reversed in memory",
                    source,
                    indices,
                    destination,
                    axis,
                )
            cases += 4

    # One index repeated along the whole axis: a set leaves the last source, an add sums them all.
    for dtype in ("bf16", "f16", "f32", "f64", "i16", "u8"):
        source = operands.values(dtype, (40, 3), 800)
        for function in ("scatter_set", "scatter_add"):
            destination = operands.values(dtype, (5, 3), 801)
            indices = operands.tensor(torch.full((40, 3), 2), "u32")
            checked += check_scatter(
                function, "one index repeated", source, indices, destination, 0
            )
            cases += 1

    print(
        f"ok: {checked} elements bit for bit over {cases} launches: gather, index_select, "
        f"scatter_set, scatter_add_set and index_add on {len(DTYPES)} element types and "
        f"{len(INDICES)} index types, gather and scatter_set on {len(operands.MOVED)} that only "
        f"moves, along every kind of axis of rank 1 to 5, on {operands.where()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
