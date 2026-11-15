"""The fp8_quantize package's compile script: one descriptor -> one module.

A descriptor is the activation's element type and the elements of a row one
program encodes. The row's width and stride are launch words, and the grid is
one row per grid row and its blocks across the grid's columns.
"""

import fp8_quantize as kernels
from uniserve_build_server import triton as driver

#: The 16-bit types an activation may have, as Triton spells them.
DTYPES = {"bf16": "*bf16", "f16": "*fp16"}


def spec(fields, parameters, target):
    """The compilation of one fp8_quantize descriptor."""
    if driver.capability(target) < 89:
        raise SystemExit(f"E4M3 codes need sm_89 or newer, not {target}")
    block = int(fields["block"])
    return {
        "kernel": kernels._fp8_quantize_kernel,
        "signature": {
            "x_ptr": DTYPES[fields["dtype"]],
            "out_ptr": "*fp8e4nv",
            "k": "i64",
            "x_row_stride": "i64",
            "encode_ptr": "*fp32",
            "BLOCK": "constexpr",
        },
        "constexprs": {"BLOCK": block},
        "options": {"num_warps": 4 if block <= 1024 else 8},
        # The operator refuses a row width or a row stride that is not a multiple of sixteen, so
        # whole vectors of a row are loaded and stored.
        "aligned": ["x_ptr", "out_ptr", "k", "x_row_stride"],
    }
