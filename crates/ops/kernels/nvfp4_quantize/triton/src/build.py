"""The nvfp4_quantize package's compile script: one descriptor -> one module.

A descriptor is the activations' element type, the row width, the block scales'
layout, the elements one program encodes and the dependent-launch flag. The row
count comes from the grid, so one module serves every row count.
"""

import _compile
import nvfp4_quantize as kernels

#: The element types the operator encodes, as Triton spells them.
DTYPES = {"bf16": "*bf16", "f16": "*fp16"}

#: The block-scale layouts a program writes. Only the linear one is served: the
#: swizzled layout pads rows to 128 and scale columns to 4, and those padding
#: elements are part of the scale tensor the operator hands over, so writing it
#: needs the padding in the same launch that writes the scales.
LAYOUTS = {"linear"}


def warps_of(block):
    """The warp count one program's tile needs."""
    return 8 if block >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one nvfp4_quantize descriptor."""
    columns = int(fields["columns"])
    block = int(fields["block"])
    if fields["layout"] not in LAYOUTS:
        raise SystemExit(f"the {fields['layout']} scale layout is not served")
    if columns % block != 0:
        raise SystemExit(f"{block} elements do not divide a row of {columns}")
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
        },
        "constexprs": {
            "COLUMNS": columns,
            "BLOCK": block,
            "SCALE_COLUMNS": columns // 16,
        },
        "options": {"num_warps": warps_of(block), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
