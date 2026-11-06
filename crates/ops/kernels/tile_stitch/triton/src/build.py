"""The tile_stitch package's compile script: one descriptor -> one module.

The element type, the channels, the patch and the block are the key's; the raster's extents, its
strides and the grid's numbers are the launch's words, and the grid itself arrives as two tables.
"""

import _compile
import tile_stitch as kernels

#: The element types a decoded tile is stored in.
DTYPES = {"f16": "fp16", "bf16": "bf16", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one tile_stitch descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    return {
        "kernel": kernels._tile_stitch_kernel,
        "signature": {
            "tokens_ptr": dtype,
            "rows_ptr": "*i32",
            "columns_ptr": "*i32",
            "out_ptr": dtype,
            "frames": "u32",
            "width": "u32",
            "out_stride0": "i64",
            "out_stride1": "i64",
            "out_stride2": "i64",
            "tile_columns": "u32",
            "token_rows": "u32",
            "tile_height": "u32",
            "tile_width": "u32",
            "CHANNELS": "constexpr",
            "PATCH_FRAMES": "constexpr",
            "PATCH_SIDE": "constexpr",
            "BLOCK": "constexpr",
        },
        "constexprs": {
            "CHANNELS": int(fields["channels"]),
            "PATCH_FRAMES": int(fields["patch_frames"]),
            "PATCH_SIDE": int(fields["patch_side"]),
            "BLOCK": int(fields["block"]),
        },
        "options": {"num_warps": 8},
    }


if __name__ == "__main__":
    _compile.main(spec)
