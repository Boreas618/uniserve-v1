"""The seam between two decoded tiles: the ``cross_fade`` operator's Triton source.

The video decoder decodes overlapping tiles and blends their shared band so the seam does not
show. The overlap runs along the height (axis 2) or the width (axis 3) of a
``[channels, frames, height, width]`` tile, and the leading ``extent`` positions of that axis are
the band:

```text
for i < extent along the axis:
    out[..., i, ...] = previous[..., along - extent + i, ...] · (1 - i/extent)
                     + current [..., i,                ...] · (i/extent)
elsewhere along the axis:
    out[..., i, ...] = current[..., i, ...]
```

**The weights are the decoded dtype's, not FP32's.** The reference builds its ramp as
``torch.arange(extent, dtype=current.dtype) / extent``, so the position is rounded to the tile's
type, the divisor is rounded to it, the quotient and ``1 − that`` are that type's, and each product
and the sum round there too. That is why the operator's tolerance is ``exact``: a kernel that
computed the ramp in FP32 and rounded once would be a slightly different fade. This source rounds
each step into the element type, which is what the reference's own arithmetic does.

**One thread per output voxel, and all three views are addressed by their strides.** The call gives
the views' four extents and, for each of them, three outer strides, so a program decodes its flat
index into (channel, frame, row, column) and addresses every operand as
``channel · stride[0] + frame · stride[1] + row · stride[2] + column``. One row is contiguous and the
three outer strides are the call's, which is what lets **a source** — not only the destination — be a
view inside a larger raster: a tile the grid decoded sits at a stride of the raster's width, and the
seam between two such tiles is a band whose rows are that same stride apart.

**Both the tile and the seam are this one program.** Passed whole tiles, the leading ``extent``
positions of the band axis blend and the rest copy; passed a seam — the views' band axis exactly
``extent`` wide, which is what the decoder hands over — ``columns - extent`` is zero and every
position is in the band. The arithmetic is the same expression either way, so one module serves both
and the key does not move.

**``out`` may alias ``current``.** Every element is read from ``current`` at the position it is
written at, so one buffer serves as both; nothing here reads a neighbouring program's output.
"""

import triton
import triton.language as tl


@triton.jit
def _cross_fade_kernel(
    previous_ptr,
    current_ptr,
    out_ptr,
    extent,
    axis,
    channels,
    frames,
    rows,
    columns,
    previous_stride0,
    previous_stride1,
    previous_stride2,
    current_stride0,
    current_stride1,
    current_stride2,
    out_stride0,
    out_stride1,
    out_stride2,
    block,
    pdl,
    AXIS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Blend one block of the destination.

    Each of the three views is addressed by the three outer strides the call states and a
    unit-strided last axis, so a source may be a view inside the raster the tiles were decoded into.

    ``AXIS`` is the descriptor's, because the blend's position is a different coordinate of the
    index on each axis and both the read and the mask follow from it. ``extent``, ``channels``,
    ``frames``, ``rows``, ``columns``, each view's three outer strides, ``block`` and ``pdl`` are the
    launch's words.
    """
    dtype = out_ptr.dtype.element_ty
    columns = columns.to(tl.int64)
    rows = rows.to(tl.int64)
    frames = frames.to(tl.int64)
    channels = channels.to(tl.int64)
    extent = extent.to(tl.int64)

    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    total = channels * frames * rows * columns
    live = index < total

    # The flat index is the destination's own [channels, frames, rows, columns] order; one row is
    # contiguous and the outer three are the call's strides.
    column = index % columns
    plane = index // columns
    row = plane % rows
    frame_plane = plane // rows
    frame = frame_plane % frames
    channel = frame_plane // frames

    # Every view is addressed the same way: three outer strides the call states and a unit-strided
    # last axis. A source inside a larger raster differs from a contiguous one only in these numbers.
    plane_offset = (
        channel * previous_stride0.to(tl.int64) + frame * previous_stride1.to(tl.int64)
    )
    current_offset = (
        channel * current_stride0.to(tl.int64)
        + frame * current_stride1.to(tl.int64)
        + row * current_stride2.to(tl.int64)
        + column
    )
    if AXIS == 3:
        # The band is the leading columns of every row; the previous tile's band is its last ones.
        # A view whose band axis is exactly `extent` wide (the seam the decoder passes) makes the
        # first term zero and reads the band it was handed.
        previous_offset = (
            plane_offset
            + row * previous_stride2.to(tl.int64)
            + (columns - extent + column)
        )
        position = column
    else:
        # The band is the leading rows of every frame; the previous tile's band is its last rows.
        previous_offset = (
            plane_offset
            + (rows - extent + row) * previous_stride2.to(tl.int64)
            + column
        )
        position = row
    out_offset = (
        channel * out_stride0.to(tl.int64)
        + frame * out_stride1.to(tl.int64)
        + row * out_stride2.to(tl.int64)
        + column
    )

    inside = position < extent
    current_value = tl.load(current_ptr + current_offset, mask=live, other=0.0).to(dtype)
    previous_value = tl.load(
        previous_ptr + previous_offset, mask=live & inside, other=0.0
    ).to(dtype)

    # The reference's ramp, in the element type at every step: `arange(extent, dtype) / extent`,
    # then `1 - that`, then the two products and their sum. The two integers reach the element type
    # through FP32: every position and overlap a tile holds is far below 2**24, so the intermediate
    # is exact and the rounding into the element type is the reference's own.
    position_in_dtype = position.to(tl.float32).to(dtype).to(tl.float32)
    extent_in_dtype = extent.to(tl.float32).to(dtype).to(tl.float32)
    ratio = (position_in_dtype / extent_in_dtype).to(dtype)
    previous_weight = (1.0 - ratio.to(tl.float32)).to(dtype)
    blended = (
        (previous_value.to(tl.float32) * previous_weight.to(tl.float32)).to(dtype).to(tl.float32)
        + (current_value.to(tl.float32) * ratio.to(tl.float32)).to(dtype).to(tl.float32)
    ).to(dtype)

    value = tl.where(inside, blended, current_value)
    tl.store(out_ptr + out_offset, value, mask=live)
