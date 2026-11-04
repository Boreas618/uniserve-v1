"""The softmax package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The row width, the lane tile that covers it and whether the module issues the dependent launch are
baked in; the row count, the width and the flag stay launch words.
"""

import _compile
import softmax as kernels

#: The dtypes a row and its output may have.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(tile):
    """The warp count one row tile needs."""
    return 8 if tile >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one softmax descriptor."""
    tile = int(fields["tile"])
    dtype = "*" + DTYPES[fields["dtype"]]
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._softmax_kernel,
        "signature": {
            "x_ptr": dtype,
            "out_ptr": dtype,
            "columns": "i32",
            "pdl": "i32",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {"BLOCK": tile, "PDL": pdl},
        "options": {"num_warps": warps_of(tile), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
