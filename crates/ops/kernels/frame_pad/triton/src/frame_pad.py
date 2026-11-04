"""Causal frame padding: the Triton source of the ``frame_pad`` operator.

Programs write one row segment of the padded output through its strides, so
channel-first, channels-last and permuted sources all work, and the output equals
the two-step ``F.pad`` result element for element: every element is a copy of one
source element or zero. Output frames before ``pad_front`` are zero; every other
output pixel copies the reflected or replicated source pixel.

The kernel body is the one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/norm/frame.py``), unchanged except that
it imports nothing but Triton: the launch-time wrapper, its operand checks and
its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "stride_batch",
        "stride_channel",
        "stride_frame",
        "stride_height",
        "stride_width",
        "out_batch",
        "out_channel",
        "out_frame_stride",
        "out_height_stride",
        "out_width_stride",
        "height",
        "width",
        "out_frames",
        "out_height",
        "out_width",
        "channels",
        "pad_top",
        "pad_left",
        "pad_front",
    ]
)
def _frame_pad_kernel(
    values,
    out,
    stride_batch,
    stride_channel,
    stride_frame,
    stride_height,
    stride_width,
    out_batch,
    out_channel,
    out_frame_stride,
    out_height_stride,
    out_width_stride,
    height,
    width,
    out_frames,
    out_height,
    out_width,
    channels,
    pad_top,
    pad_left,
    pad_front,
    reflect: tl.constexpr,
    block_pixels: tl.constexpr,
    block_channels: tl.constexpr,
):
    """Write one row segment of the padded output through its strides.

    Grid: (samples * out_frames * out_height, width blocks, channel blocks).
    Output frames before ``pad_front`` are zero; every other output pixel copies
    the reflected or replicated source pixel.
    """
    row_id = tl.program_id(0)
    out_row = row_id % out_height
    out_frame = (row_id // out_height) % out_frames
    sample = row_id // (out_height * out_frames)
    columns = tl.program_id(1) * block_pixels + tl.arange(0, block_pixels)
    channel = tl.program_id(2) * block_channels + tl.arange(0, block_channels)
    valid = (columns < out_width)[:, None] & (channel < channels)[None, :]

    # The block's columns are adjacent in channel-first storage and its channels
    # in channels-last storage, so either layout's stores coalesce.
    destination = (
        out
        + sample.to(tl.int64) * out_batch
        + out_frame.to(tl.int64) * out_frame_stride
        + out_row.to(tl.int64) * out_height_stride
        + columns[:, None].to(tl.int64) * out_width_stride
        + channel[None, :].to(tl.int64) * out_channel
    )

    frame = out_frame - pad_front
    if frame < 0:
        tl.store(
            destination,
            tl.zeros((block_pixels, block_channels), dtype=tl.float32).to(
                out.dtype.element_ty
            ),
            mask=valid,
        )
    else:
        source_row = out_row - pad_top
        source_columns = columns - pad_left
        if reflect:
            source_row = tl.where(source_row < 0, -source_row, source_row)
            source_row = tl.where(
                source_row >= height,
                2 * height - 2 - source_row,
                source_row,
            )
            source_columns = tl.where(
                source_columns < 0, -source_columns, source_columns
            )
            source_columns = tl.where(
                source_columns >= width,
                2 * width - 2 - source_columns,
                source_columns,
            )
        else:
            source_row = tl.minimum(tl.maximum(source_row, 0), height - 1)
            source_columns = tl.minimum(
                tl.maximum(source_columns, 0), width - 1
            )
        copied = tl.load(
            values
            + sample.to(tl.int64) * stride_batch
            + frame.to(tl.int64) * stride_frame
            + source_row.to(tl.int64) * stride_height
            + source_columns[:, None].to(tl.int64) * stride_width
            + channel[None, :].to(tl.int64) * stride_channel,
            mask=valid,
        )
        tl.store(destination, copied, mask=valid)
