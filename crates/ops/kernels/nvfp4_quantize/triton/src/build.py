"""The nvfp4_quantize package's compile script: one descriptor -> one module.

A descriptor is the activations' element type, the row width, the block scales'
layout, the elements one program encodes and the dependent-launch flag. The row
count comes from the grid, so one module serves every row count.
"""

import triton

import _compile
import nvfp4_quantize as kernels

#: The element types the operator encodes, as Triton spells them.
DTYPES = {"bf16": "*bf16", "f16": "*fp16"}

#: The row block and scale-column group the swizzled layout interleaves.
SWIZZLE_ROWS = 128
SWIZZLE_COLUMNS = 4


def warps_of(block):
    """The warp count one program's tile needs."""
    return 8 if block >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one nvfp4_quantize descriptor."""
    columns = int(fields["columns"])
    block = int(fields["block"])
    if columns % block != 0:
        raise SystemExit(f"{block} elements do not divide a row of {columns}")
    scale_columns = columns // 16
    swizzled = fields["layout"] == "swizzled_128x4"
    if fields["layout"] not in ("linear", "swizzled_128x4"):
        raise SystemExit(f"the {fields['layout']} scale layout is not served")
    padded_columns = (
        (scale_columns + SWIZZLE_COLUMNS - 1) // SWIZZLE_COLUMNS * SWIZZLE_COLUMNS
        if swizzled
        else scale_columns
    )
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._nvfp4_quantize_kernel,
        "signature": {
            "input_ptr": DTYPES[fields["input"]],
            "values_ptr": "*u8",
            "scales_ptr": "*u8",
            "rows": "i64",
            "global_scale": "fp32",
            "COLUMNS": "constexpr",
            "BLOCK": "constexpr",
            "SCALE_COLUMNS": "constexpr",
            "SWIZZLED": "constexpr",
            "PADDED_COLUMNS": "constexpr",
            "PADDED_TILE": "constexpr",
        },
        "constexprs": {
            "COLUMNS": columns,
            "BLOCK": block,
            "SCALE_COLUMNS": scale_columns,
            "SWIZZLED": swizzled,
            "PADDED_COLUMNS": padded_columns,
            # One program writes a whole padded row of zeros; the tile is the
            # padded width rounded up to the power of two arange needs.
            "PADDED_TILE": triton.next_power_of_2(max(padded_columns, 1)),
        },
        "options": {"num_warps": warps_of(block), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
