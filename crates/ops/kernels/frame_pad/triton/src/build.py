"""The frame_pad package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The tile (columns and channels per program), the padding mode and the element type are baked in;
every stride and every extent is an int32 launch word, as the eager launch casts them.
"""

import _compile
import frame_pad as kernels

#: The floating dtypes a video frame may have, as Triton spells them in an argument signature.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one frame_pad descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    signature = {"values": dtype, "out": dtype}
    # Five source strides, five output strides, then the extents and the padding:
    # the order `FramePad::launch` packs them in.
    for name in (
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
    ):
        signature[name] = "i32"
    signature["reflect"] = "constexpr"
    signature["block_pixels"] = "constexpr"
    signature["block_channels"] = "constexpr"
    return {
        "kernel": kernels._frame_pad_kernel,
        "signature": signature,
        "constexprs": {
            "reflect": fields["reflect"] == "true",
            "block_pixels": int(fields["block_pixels"]),
            "block_channels": int(fields["block_channels"]),
        },
        "options": {
            "num_warps": 4,
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
