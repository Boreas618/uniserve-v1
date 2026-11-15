"""The copy_block program moves every element of a strided block.

The operator's tolerance is exact: it moves elements, so a correct program
stores the source's bits and the comparison is of values, not a bound. The
program performs no arithmetic beyond index decomposition and addresses, so a
wrong stride shows as a wrong value.

Both sides are views with explicit strides (``torch.as_strided``), which makes
the cases launches of one module rather than separate programs:

- a contiguous copy;
- a permutation: the video decoder's channel-first to channel-last reorder,
  the source's strides stated in the destination's axis order;
- a destination placed inside a larger raster, its strides wider than its
  extents;
- a destination whose innermost axis is not unit stride, which the operator
  allows (it has no unit-stride rule);
- a last axis longer than one program, which the general module walks a row
  at a time;
- the wide module (``wide = true``), which moves aligned unit-stride rows as
  vectors.

The grid of each launch is the operator's (``crates/ops/src/copy_block.rs``):
rows times blocks per row for the row walk and the wide module, the elements
covered otherwise.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.unit

#: The elements one program moves: `COPY` for the general module, `WIDE_COPY`
#: for a wide one (``crates/ops/src/copy_block.rs``).
COPY = 1024
WIDE_COPY = 2048

DTYPES = {"f32": torch.float32, "f16": torch.float16}


def launch(kernels, source, out, wide):
    """One launch over a rank-4 block, with the operator's grid."""
    shape = list(source.shape)
    total = math.prod(shape)
    block = WIDE_COPY if wide else COPY
    # A wide copy, or a last axis one program wide or more, is walked a row at
    # a time, so the grid counts each row's blocks; otherwise it covers the
    # elements.
    if wide or shape[3] >= block:
        grid = math.prod(shape[:3]) * -(-shape[3] // block)
    else:
        grid = -(-total // block)
    kernels._copy_block_kernel[(grid,)](
        source,
        out,
        *shape,
        *source.stride(),
        *out.stride(),
        total,
        block,
        0,
        BLOCK=block,
        WIDE=wide,
    )


def padded(shape, innermost):
    """Row-major strides of `shape` whose innermost stride is `innermost`.

    A view's strides must not overlap: each outer stride is at least the
    extent of the axis it skips times the stride inside it. Overlapping
    strides would let two elements of one block share a byte, and the
    comparison would then be about which write landed last.
    """
    strides = [0] * len(shape)
    running = innermost
    for axis in reversed(range(len(shape))):
        strides[axis] = running
        running *= shape[axis]
    return strides


def row_major(shape, order):
    """Row-major strides of `shape`, permuted to `order`."""
    strides = [0] * len(shape)
    running = 1
    for axis in reversed(order):
        strides[axis] = running
        running *= shape[axis]
    return strides


SHAPE = (2, 3, 4, 5)
PLAIN = row_major(SHAPE, (3, 2, 1, 0))
ROWS = (1, 2, 2, COPY + 3)
WIDE = (1, 2, 2, WIDE_COPY + 16)


def wide_raster():
    """Unit-stride rows of whole 16-element runs placed in a wider raster."""
    placed = padded(WIDE, 1)
    placed[2] = WIDE_COPY + 48
    placed[1] = placed[2] * WIDE[2]
    placed[0] = placed[1] * WIDE[1]
    return placed


#: The copies: the block's shape, the source's and the destination's strides,
#: and whether the wide module runs.
CASES = {
    "contiguous": (SHAPE, PLAIN, PLAIN, False),
    # The video decoder's reorder: [channels, frames, h, w] to
    # [frames, h, w, channels].
    "permuted": (
        SHAPE,
        PLAIN,
        [1, SHAPE[0], SHAPE[0] * SHAPE[1], SHAPE[0] * SHAPE[1] * SHAPE[2]],
        False,
    ),
    # Placed inside a larger raster: the innermost stride is eight for a run
    # of five, so the destination leaves the gaps a placed block has.
    "placed": (SHAPE, PLAIN, padded(SHAPE, 8), False),
    # An innermost axis that is not unit stride, which the operator permits.
    "inner stride two": (SHAPE, PLAIN, padded(SHAPE, 2), False),
    # A last axis longer than one program: the general module walks it a row
    # at a time, with a partial last block and the destination's inner stride
    # above one.
    "row walk": (ROWS, padded(ROWS, 1), padded(ROWS, 2), False),
    # The wide module, with a partial last block of the row.
    "wide": (WIDE, padded(WIDE, 1), wide_raster(), True),
}


def strided_view(shape, strides, dtype, device):
    """A view of `shape` at `strides` over the smallest buffer holding it."""
    span = 1 + sum(
        (extent - 1) * stride
        for extent, stride in zip(shape, strides, strict=True)
    )
    base = torch.zeros(span, dtype=dtype, device=device)
    return base.as_strided(shape, strides), base


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES)
def test_every_element_of_the_block_moves(
    case, dtype, torch_device, kernel_source
):
    shape, source_strides, out_strides, wide = CASES[case]
    element = DTYPES[dtype]
    source, _ = strided_view(shape, source_strides, element, torch_device)
    values = torch.arange(math.prod(shape), dtype=torch.float64)
    source.copy_(values.reshape(shape).to(element))
    out, base = strided_view(shape, out_strides, element, torch_device)
    base.fill_(-1.0)

    launch(kernel_source("copy_block"), source, out, wide)

    got, expected = out.cpu(), source.cpu()
    wrong = (got != expected).nonzero()
    at = tuple(int(axis) for axis in wrong[0]) if wrong.numel() else None
    assert at is None, (
        f"{case} {dtype}: {wrong.size(0)} of {got.numel()} elements differ; "
        f"first at {at}: {got[at].item()}, expected {expected[at].item()}"
    )
