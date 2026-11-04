"""Packed latent tokens into a planar raster: the ``unpatchify_video`` operator's Triton source.

The video VAE's decoder reads a ``[channels, frames, height, width]`` raster, not the diffusion
transformer's packed rows, and one token's values are one patch of the raster, laid out
channel-major:

```text
tokens [(frames·height·width), channels · patch_volume]
out    [channels, frames·patch_frames, height·patch_height, width·patch_width]

token (t, y, x), element c·pf·ph·pw + pt·ph·pw + ph_i·pw + pw_i
  -> out[c, t·pf + pt, y·ph + ph_i, x·pw + pw_i]

with `affine`:  out = value · scale[c] + bias[c]
```

The program walks the *output*: one lane writes one voxel, so a token's voxels are written by the
lanes that own them, no lane waits for another, and a token buffer with spare rows is never read.
Every product and sum is FP32 and the store rounds once into the output's element type.
"""

import triton
import triton.language as tl


@triton.jit
def _token_offsets(offsets, channels, frames, height, width, patch_frames, patch_height, patch_width):
    """The token offset and the channel a raster offset reads, and whether it is inside the raster.

    The raster decomposes into channel, frame, row and column; each of those inverts the packing
    into the token and the position inside its patch.
    """
    patch_frames = patch_frames.to(tl.int64)
    patch_height = patch_height.to(tl.int64)
    patch_width = patch_width.to(tl.int64)
    channels = channels.to(tl.int64)
    frames = frames.to(tl.int64)
    height = height.to(tl.int64)
    width = width.to(tl.int64)

    raster_frames = frames * patch_frames
    raster_height = height * patch_height
    raster_width = width * patch_width
    frame_elements = raster_height * raster_width
    channel_elements = raster_frames * frame_elements
    volume = channels * channel_elements
    live = offsets < volume

    channel = offsets // channel_elements
    within = offsets % channel_elements
    frame = within // frame_elements
    within = within % frame_elements
    row = within // raster_width
    column = within % raster_width

    token_frame = frame // patch_frames
    inner_frame = frame % patch_frames
    token_row = row // patch_height
    inner_row = row % patch_height
    token_column = column // patch_width
    inner_column = column % patch_width
    token = (token_frame * height + token_row) * width + token_column

    patch_volume = patch_frames * patch_height * patch_width
    patch_channel = (
        ((channel * patch_frames + inner_frame) * patch_height + inner_row) * patch_width
        + inner_column
    )
    # A token row holds `channels · patch_volume` values.
    return token * (channels * patch_volume) + patch_channel, channel, live, volume


@triton.jit
def _unpatchify_video_kernel(
    tokens_ptr,
    out_ptr,
    channels,
    frames,
    height,
    width,
    patch_frames,
    patch_height,
    patch_width,
    block,
    pdl,
    BLOCK: tl.constexpr,
):
    """Write one block of the raster from the packed tokens, as they are.

    ``block`` and ``pdl`` are the launch's words: the block width is ``BLOCK`` and programmatic
    dependent launch is a launching decision.
    """
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    source, _channel, live, volume = _token_offsets(
        offsets, channels, frames, height, width, patch_frames, patch_height, patch_width
    )
    value = tl.load(tokens_ptr + source, mask=live).to(tl.float32)
    tl.store(
        out_ptr + offsets,
        value.to(out_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=offsets < volume,
    )


@triton.jit
def _unpatchify_video_affine_kernel(
    tokens_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    channels,
    frames,
    height,
    width,
    patch_frames,
    patch_height,
    patch_width,
    block,
    pdl,
    BLOCK: tl.constexpr,
):
    """Write one block of the raster, denormalized by the per-channel scale and bias.

    The affine form is its own program because its launch carries the two operands: the operator's
    descriptor says which form a compilation is, and a launch that disagrees is refused before it
    reaches a kernel.
    """
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    source, channel, live, volume = _token_offsets(
        offsets, channels, frames, height, width, patch_frames, patch_height, patch_width
    )
    # Every product and sum is FP32; the store rounds once into the raster's type.
    value = tl.load(tokens_ptr + source, mask=live).to(tl.float32)
    scale = tl.load(scale_ptr + channel, mask=live, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=live, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + offsets,
        (value * scale + bias).to(out_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=offsets < volume,
    )
