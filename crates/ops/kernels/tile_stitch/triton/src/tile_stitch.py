"""A clip's decoded tiles straight from their tokens into its raster: the ``tile_stitch``
operator's Triton source.

One program writes ``BLOCK`` columns of one row of one frame of one channel. A pixel belongs to one
tile, and its value is that tile's token value, faded against the tile above it and then against the
tile to its left where it lies in a seam. The two neighbours are read as they were decoded, so a
corner is computed in the order an edge is, which is the reference's rule
(``uniserve/nn/vae/tiled.py``, ``stitch_tiles``).

The fade is the reference's arithmetic in the element type at every step
(``uniserve/nn/video.py``, ``blend_decoded_overlap``): the position and the extent reach the element
type through FP32, the ratio and its complement round there, each product rounds and the sum rounds.

The grid's geometry is two tables of ``[tile, position, seam]`` rows, so the program holds no tile
logic: the row's three words are scalars of the program and the columns' are a load per lane.
"""

import triton
import triton.language as tl


@triton.jit
def _fade(previous, current, position, extent, dtype: tl.constexpr):
    """``previous · (1 − i/n) + current · (i/n)``, rounded to ``dtype`` at every step."""
    position_in_dtype = position.to(tl.float32).to(dtype).to(tl.float32)
    extent_in_dtype = extent.to(tl.float32).to(dtype).to(tl.float32)
    ratio = (position_in_dtype / extent_in_dtype).to(dtype)
    complement = (1.0 - ratio.to(tl.float32)).to(dtype)
    return (
        (previous.to(tl.float32) * complement.to(tl.float32)).to(dtype).to(tl.float32)
        + (current.to(tl.float32) * ratio.to(tl.float32)).to(dtype).to(tl.float32)
    ).to(dtype)


@triton.jit
def _address(
    tile,
    row,
    column,
    channel,
    token_frame,
    inner_frame,
    token_rows,
    cells_high,
    cells_wide,
    CHANNELS: tl.constexpr,
    PATCH_FRAMES: tl.constexpr,
    PATCH_SIDE: tl.constexpr,
):
    """The element of ``tokens`` a tile's pixel lies in: the token, then the place inside its patch."""
    token = (token_frame * cells_high + row // PATCH_SIDE) * cells_wide + column // PATCH_SIDE
    inside = (
        (channel * PATCH_FRAMES + inner_frame) * PATCH_SIDE + row % PATCH_SIDE
    ) * PATCH_SIDE + column % PATCH_SIDE
    patch: tl.constexpr = CHANNELS * PATCH_FRAMES * PATCH_SIDE * PATCH_SIDE
    return (tile * token_rows + token).to(tl.int64) * patch + inside


@triton.jit
def _tile_stitch_kernel(
    tokens_ptr,
    rows_ptr,
    columns_ptr,
    out_ptr,
    frames,
    width,
    out_stride0,
    out_stride1,
    out_stride2,
    tile_columns,
    token_rows,
    tile_height,
    tile_width,
    CHANNELS: tl.constexpr,
    PATCH_FRAMES: tl.constexpr,
    PATCH_SIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write one block of one raster row."""
    dtype = out_ptr.dtype.element_ty
    # The launch's words are unsigned and the program ids are not, so the arithmetic takes one sign.
    frames = frames.to(tl.int32)
    width = width.to(tl.int32)
    tile_columns = tile_columns.to(tl.int32)
    token_rows = token_rows.to(tl.int32)
    tile_height = tile_height.to(tl.int32)
    tile_width = tile_width.to(tl.int32)
    y = tl.program_id(1)
    plane = tl.program_id(2)
    channel = plane // frames
    frame = plane % frames
    token_frame = frame // PATCH_FRAMES
    inner_frame = frame % PATCH_FRAMES
    cells_high = tile_height // PATCH_SIDE
    cells_wide = tile_width // PATCH_SIDE

    tile_row = tl.load(rows_ptr + y * 3)
    row = tl.load(rows_ptr + y * 3 + 1)
    row_seam = tl.load(rows_ptr + y * 3 + 2)

    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = x < width
    tile_column = tl.load(columns_ptr + x * 3, mask=live, other=0)
    column = tl.load(columns_ptr + x * 3 + 1, mask=live, other=0)
    column_seam = tl.load(columns_ptr + x * 3 + 2, mask=live, other=0)
    tile = tile_row * tile_columns + tile_column

    value = tl.load(
        tokens_ptr
        + _address(
            tile,
            row,
            column,
            channel,
            token_frame,
            inner_frame,
            token_rows,
            cells_high,
            cells_wide,
            CHANNELS,
            PATCH_FRAMES,
            PATCH_SIDE,
        ),
        mask=live,
        other=0.0,
    )
    if row_seam > 0:
        above = tl.load(
            tokens_ptr
            + _address(
                tile - tile_columns,
                tile_height - row_seam + row,
                column,
                channel,
                token_frame,
                inner_frame,
                token_rows,
                cells_high,
                cells_wide,
                CHANNELS,
                PATCH_FRAMES,
                PATCH_SIDE,
            ),
            mask=live,
            other=0.0,
        )
        value = _fade(above, value, row, row_seam, dtype)

    seamed = column_seam > 0
    left = tl.load(
        tokens_ptr
        + _address(
            tile - 1,
            row,
            tile_width - column_seam + column,
            channel,
            token_frame,
            inner_frame,
            token_rows,
            cells_high,
            cells_wide,
            CHANNELS,
            PATCH_FRAMES,
            PATCH_SIDE,
        ),
        mask=live & seamed,
        other=0.0,
    )
    faded = _fade(left, value, column, tl.where(seamed, column_seam, 1), dtype)
    value = tl.where(seamed, faded, value)

    destination = (
        channel.to(tl.int64) * out_stride0
        + frame.to(tl.int64) * out_stride1
        + y.to(tl.int64) * out_stride2
        + x
    )
    tl.store(out_ptr + destination, value, mask=live)
