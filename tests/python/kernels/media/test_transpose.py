"""The transpose program against the source's transpose, bit for bit.

A transpose performs no arithmetic, so the operator's tolerance is exact and
any difference is a wrong element rather than a rounding. Every case is
compared as raw bit patterns, so a value that is not a number can never
compare equal to itself, against a destination prefilled with a value no
source holds.

The shapes are the ones a tiled transpose breaks on: an exact grid of tiles, a
partial tile at the right edge, one at the bottom edge, both at once, a matrix
smaller than one tile, the degenerate one-element tile, and a sliver whose
destination is one long row per source column, the worst case for the
coalescing the tiling exists for. The elements are the operator's three types
drawn as normal values, and 32-bit integers drawn over their whole range,
whose bit patterns include the ones a float draw never holds.
"""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit

#: The tile the program moves.
BLOCK = 32

#: The shapes, as (rows, columns) of the source.
SHAPES = {
    "an exact grid of tiles": (64, 64),
    "a partial tile in the destination's rows": (64, 96),
    "a partial tile in the destination's columns": (96, 64),
    "both edges partial": (70, 77),
    "smaller than one tile": (5, 7),
    "the degenerate tile": (1, 1),
    "a sliver of three columns": (1000, 3),
}

#: The element types moved; `i32` holds raw 32-bit patterns.
ELEMENTS = {
    "f32": torch.float32,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "i32": torch.int32,
}


def launch(kernels, source, out, block):
    """One launch over every tile of `source`, writing its transpose."""
    rows, columns = source.shape
    kernels._transpose_kernel[(-(-columns // block), -(-rows // block))](
        source,
        out,
        rows,
        columns,
        block,
        BLOCK=block,
        PDL=False,
    )


def drawn(rows, columns, element, seed):
    """The source matrix and the destination's fill value no element of it
    holds.
    """  # noqa: D205
    if element == torch.int32:
        generator = np.random.default_rng(seed)
        values = generator.integers(
            -(2**31), 2**31 - 1, size=(rows, columns), dtype=np.int64
        )
        return torch.from_numpy(values.astype(np.int32)), -7
    torch.manual_seed(seed)
    values = torch.randn(rows, columns, dtype=torch.float32).to(element)
    return values, float("nan")


def bits(tensor):
    """The tensor's raw bit patterns, so equality is bit equality."""
    return tensor.view({4: torch.int32, 2: torch.int16}[tensor.element_size()])


@pytest.mark.parametrize("dtype", list(ELEMENTS))
@pytest.mark.parametrize("shape", SHAPES)
def test_the_destination_is_the_transpose(
    shape, dtype, torch_device, kernel_source
):
    rows, columns = SHAPES[shape]
    element = ELEMENTS[dtype]
    source, fill = drawn(rows, columns, element, rows * 1000 + columns)
    out = torch.full((columns, rows), fill, dtype=element, device=torch_device)

    launch(kernel_source("transpose"), source.to(torch_device), out, BLOCK)

    got = out.cpu()
    expected = source.T.contiguous()
    differing = (bits(got) != bits(expected)).nonzero()
    if differing.numel():
        at = tuple(int(value) for value in differing[0])
        pytest.fail(
            f"{dtype} {rows} x {columns} written as {columns} x {rows}, tile "
            f"{BLOCK}: {differing.size(0)} of {got.numel()} differ; first at "
            f"{at}: got {got[at].item()}, expected {expected[at].item()}"
        )
