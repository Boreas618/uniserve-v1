"""The snake package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The channel count is baked in because it is the parameter count, and the block because it is the
program's lane width; the row count, the channel count, the width and the flag stay launch words.
"""

import _compile
import snake as kernels

#: The dtypes the signal and the parameter may have.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(block):
    """The warp count one block of elements needs."""
    return 8 if block >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one snake descriptor."""
    block = int(fields["block"])
    dtype = "*" + DTYPES[fields["dtype"]]
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._snake_kernel,
        "signature": {
            "x_ptr": dtype,
            "alpha_ptr": dtype,
            "out_ptr": dtype,
            "rows": "i64",
            "channels": "i32",
            "width": "i32",
            "pdl": "i32",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {"BLOCK": block, "PDL": pdl},
        "options": {"num_warps": warps_of(block), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
