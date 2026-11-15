"""A contiguous matrix written as its transposed matrix: the Triton program.

One program moves one `block × block` tile: it reads the tile coalesced along the
source's columns and writes it coalesced along the destination's columns, which Triton
does by transposing the tile through shared memory. The naive program — one thread per
element, reading a column and writing a row — writes one element per cache line at both
ends, which is why the operator is specified as a tiled transpose rather than an index
swap.

The two extents are words of the launch, not of the key: the frame count of an audio
clip changes per request while the arithmetic does not, so a module serves a matrix of
any shape. `block` arrives twice on purpose — the pack carries it as the operator's
argument, and the module was compiled with it, because the grid a launch computes has
to agree with the tile a program moves.
"""

import triton
import triton.language as tl
from uniserve_kernels import dependent_launch


@triton.jit
def _transpose_kernel(
    input,
    out,
    rows,
    columns,
    block,
    BLOCK: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Write one tile of the source where its transpose belongs.

    `out[i, o] = input[o, i]`: a source row becomes a destination column, so the tile is
    read along the source's rows and written along the destination's columns. Tiles at
    the right and bottom edges are clipped, and a clipped element is neither read nor
    written — the destination's shape is the operator's promise, so a tile that fell
    outside it would be a wrong call rather than a partial one.
    """
    dependent_launch.prologue(PDL)

    source_rows = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    source_columns = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    rows = rows.to(tl.int64)
    columns = columns.to(tl.int64)
    inside = (source_rows[:, None] < rows) & (source_columns[None, :] < columns)
    tile = tl.load(
        input + source_rows[:, None] * columns + source_columns[None, :],
        mask=inside,
        other=0,
    )
    tl.store(
        out + source_columns[None, :] * rows + source_rows[:, None],
        tile,
        mask=inside,
    )
