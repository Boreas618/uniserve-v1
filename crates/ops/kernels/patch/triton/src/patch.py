"""Packed rows against the diffusion transformer's latent: the ``patch`` operator's Triton source.

The DiT reads packed rows — `[video | audio | prefix | padding]` — and one video row is one
``1×2×2`` patch of the latent's channels, so a row is ``channels · 4`` values wide. The two
directions are one index arithmetic:

```text
patchify    x [channels, frames, height, width] -> r [rows, channels · 4]
            row    = ((f · height/2) + i) · (width/2) + j      f, i, j in raster order
            column = ((c · 2) + a) · 2 + b                     c major, then the patch's (a, b)
            r[row, column] = x[c, f, 2·i + a, 2·j + b]
unpatchify  r [rows, channels · 4] -> x [channels, frames, height, width]   the exact inverse
```

A program covers one packed row's slice of ``BLOCK`` columns, which is the grid the operator
launches. Every destination element is written by exactly one program and its source is a single
element, so the mapping is a permutation: no arithmetic, no accumulation, no order between
programs. Columns and rows are decoded from the operator's own formulas, and the latent's
addresses are ``i64`` because a latent's element count outruns ``i32`` at production sizes.
"""

import triton
import triton.language as tl


@triton.jit
def _patch_kernel(
    x_ptr,
    out_ptr,
    frames,
    rows,
    channels,
    height,
    width,
    block,
    pdl,
    PATCHIFY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Move one packed row's ``BLOCK`` columns to or from the latent.

    ``PATCHIFY`` reads the latent in ``x`` and writes the packed rows in ``out``; without it the
    direction and the two views swap, as the operator's arguments say. ``rows``, ``block`` and
    ``pdl`` are the launch's words: the grid already names the row and the column slice, the
    block width is ``BLOCK``, and programmatic dependent launch is a launching decision.
    """
    row = tl.program_id(0).to(tl.int64)
    slice_index = tl.program_id(1).to(tl.int64)
    columns = slice_index * BLOCK + tl.arange(0, BLOCK).to(tl.int64)

    row_width = channels.to(tl.int64) * 4
    live = columns < row_width

    # The packed column decomposes into the channel and the patch's two inner coordinates.
    channel = columns // 4
    inner_row = (columns % 4) // 2
    inner_column = columns % 2

    # The packed row decomposes into the frame and the patch's raster position in it.
    patch_width = width.to(tl.int64) // 2
    patch_height = height.to(tl.int64) // 2
    per_frame = patch_height * patch_width
    frame = row // per_frame
    within = row % per_frame
    patch_row = within // patch_width
    patch_column = within % patch_width

    latent = ((channel * frames.to(tl.int64) + frame) * height.to(tl.int64)
              + (2 * patch_row + inner_row)) * width.to(tl.int64) + 2 * patch_column + inner_column
    packed = row * row_width + columns

    if PATCHIFY:
        value = tl.load(x_ptr + latent, mask=live)
        tl.store(out_ptr + packed, value, mask=live)
    else:
        value = tl.load(x_ptr + packed, mask=live)
        tl.store(out_ptr + latent, value, mask=live)
