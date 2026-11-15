"""The copy_strided program: every copy a tensor library makes.

Each case launches the module of its descriptor key as ``CopyStrided::launch``
does (the source and the destination merged into the axes a program
traverses, the extents and both stride vectors packed right-aligned) and
compares **bit for bit**, because a copy converts nothing. The destination is
a block of a larger buffer that starts as a sentinel, and the whole buffer is
compared, so an element written outside the block fails the case as surely as
one not written inside it.

What is covered, by the backend primitive it serves:

- **``contiguous`` and ``copy_strided_src``**: every layout of
  :data:`~uniserve_testing.kernels.tensor.operands.LAYOUTS` into a contiguous
  run that begins at an element offset inside a longer buffer;
- **``copy2d``**: rows whose stride differs on the two sides, read from an
  offset and written at an offset, which is how a concatenation along the last
  axis places each operand;
- **``cat``, ``pad`` and ``slice_set``**: a source of rank 2 to 5 written into
  an interior block of a larger tensor, whose strides the destination keeps;
- **every element type**, an FP8 (``f8e4m3``) tensor as the bytes its
  elements are, and among the floats a NaN, an infinity and a negative zero.
"""

import pytest
import torch
from uniserve_testing.kernels.tensor import operands

pytestmark = pytest.mark.unit

#: Every element type a copy moves: an FP8 (`f8e4m3`) tensor is held as its
#: bytes.
DTYPES = operands.FLOATS + operands.INTEGERS + tuple(operands.MOVED)

#: The element types a strided source of every layout is copied in.
LAYOUT_DTYPES = ("bf16", "f32", "i64", "u8", "f8e4m3")

#: Every layout in each of those types, with the seed of its sentinel.
LAYOUT_CASES = [
    pytest.param(dtype, layout, index, id=f"{dtype}-{layout}")
    for index, (dtype, layout) in enumerate(
        (dtype, layout)
        for dtype in LAYOUT_DTYPES
        for layout in operands.LAYOUTS
    )
]

#: Sources written into an interior block of a larger tensor: the element
#: type, the source's shape, the larger tensor's and the block.
INTERIOR = {
    "rank 2, f32": ("f32", (3, 4), (5, 9), (slice(1, 4), slice(2, 6))),
    "rank 3, bf16": (
        "bf16",
        (2, 3, 4),
        (4, 6, 8),
        (slice(1, 3), slice(2, 5), slice(3, 7)),
    ),
    "rank 4, u32": (
        "u32",
        (2, 3, 2, 3),
        (3, 5, 4, 6),
        (slice(1, 3), slice(0, 3), slice(2, 4), slice(3, 6)),
    ),
    "rank 5, f64": (
        "f64",
        (2, 2, 3, 2, 2),
        (3, 4, 5, 3, 4),
        (slice(0, 2), slice(1, 3), slice(2, 5), slice(1, 3), slice(2, 4)),
    ),
}


def copy(primitives, dtype, source, destination):
    """One launch of `source`, of `dtype` elements, into the view
    `destination`.
    """  # noqa: D205
    extents, (source_strides, destination_strides) = operands.merged(
        [source, destination]
    )
    elements = destination.numel()
    # `Descriptor`'s `Display`.
    key = f"dtype_{dtype}_rank_{len(extents)}_block_{operands.BLOCK}_pdl_false"
    primitives.launch(
        "copy_strided",
        key,
        operands.grid(elements),
        source,
        destination,
        elements,
        *operands.extent_words(extents),
        *operands.stride_words(source_strides),
        *operands.stride_words(destination_strides),
    )


def check(primitives, name, dtype, source, buffer, block):
    """Copy `source` into `buffer[block]` and hold the whole buffer to torch's
    assignment.
    """  # noqa: D205
    expected = buffer.clone()
    expected[block] = source.contiguous()
    copy(primitives, dtype, source, buffer[block])
    label = f"{dtype} {name} {tuple(source.shape)}"
    operands.assert_same(label, buffer, expected)


def sentinel(primitives, dtype, shape, seed):
    """A buffer of values no case writes: what a destination holds before a
    copy.
    """  # noqa: D205
    return primitives.values(dtype, shape, seed + 9000)


@pytest.mark.parametrize(("dtype", "layout", "index"), LAYOUT_CASES)
def test_a_strided_source_is_made_contiguous_at_an_offset(
    dtype, layout, index, primitives
):
    view = primitives.view(layout, dtype, 11)
    offset = 5
    buffer = sentinel(primitives, dtype, (view.numel() + 11,), index)
    run = slice(offset, offset + view.numel())
    # The destination is the run reshaped to the source's shape: contiguous,
    # so the call is the one `copy_strided_src` makes.
    expected = buffer.clone()
    expected[run] = view.contiguous().reshape(-1)
    copy(primitives, dtype, view, buffer[run].view(view.shape))
    operands.assert_same(
        f"{dtype} contiguous at an offset, {layout}", buffer, expected
    )


@pytest.mark.parametrize(("seed", "dtype"), list(enumerate(DTYPES)))
def test_a_permuted_source_is_copied(seed, dtype, primitives):
    permuted = primitives.values(dtype, (3, 4, 5), seed).permute(2, 0, 1)
    buffer = sentinel(primitives, dtype, (5, 3, 4), seed)
    check(primitives, "permuted", dtype, permuted, buffer, (slice(None),) * 3)


@pytest.mark.parametrize(("seed", "dtype"), list(enumerate(DTYPES)))
def test_a_broadcast_source_is_copied(seed, dtype, primitives):
    repeated = primitives.values(dtype, (1, 6), seed + 20).expand(4, 6)
    buffer = sentinel(primitives, dtype, (4, 6), seed + 20)
    check(primitives, "broadcast", dtype, repeated, buffer, (slice(None),) * 2)


@pytest.mark.parametrize(("seed", "dtype"), list(enumerate(DTYPES)))
def test_rows_of_differing_strides_are_copied(seed, dtype, primitives):
    """Rows read from an offset of a wider source and written at an offset of
    a wider destination, the two row strides differing.
    """  # noqa: D205
    wide = primitives.values(dtype, (6, 9), seed + 40)
    source = wide[1:4, 2:7]
    buffer = sentinel(primitives, dtype, (5, 12), seed + 40)
    check(
        primitives,
        "rows of differing strides",
        dtype,
        source,
        buffer,
        (slice(2, 5), slice(4, 9)),
    )


@pytest.mark.parametrize("dtype", ["bf16", "i32", "f8e4m3"])
def test_a_concatenation_places_each_operand(dtype, primitives):
    """Along the last axis, each operand lands in its columns of the result."""
    left = primitives.values(dtype, (3, 5), 61)
    right = primitives.values(dtype, (3, 7), 62).flip(0).contiguous()
    result = sentinel(primitives, dtype, (3, 12), 63)
    copy(primitives, dtype, left, result[:, :5])
    copy(primitives, dtype, right, result[:, 5:])
    expected = torch.cat([left, right], dim=1)
    operands.assert_same(f"{dtype} concatenation", result, expected)


@pytest.mark.parametrize("name", INTERIOR)
def test_a_source_is_written_into_an_interior_block(name, primitives):
    """A padding or a slice assignment."""
    dtype, shape, outer, block = INTERIOR[name]
    source = (
        primitives.values(dtype, shape, 71)
        .transpose(0, -1)
        .contiguous()
        .transpose(0, -1)
    )
    buffer = sentinel(primitives, dtype, outer, 72)
    check(primitives, "an interior block", dtype, source, buffer, block)


@pytest.mark.parametrize("dtype", operands.FLOATS)
def test_values_a_bitwise_comparison_tells_apart_are_copied(dtype, primitives):
    special = primitives.tensor(
        [float("nan"), -0.0, 0.0, float("inf"), -float("inf"), 1.0], dtype
    )
    buffer = sentinel(primitives, dtype, (2, 6), 81)
    check(
        primitives,
        "special values",
        dtype,
        special.flip(0),
        buffer,
        (1, slice(None)),
    )
