"""The transpose package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel
function, its argument signature in the order the operator launches them, the
compile-time values the descriptor dimensions give, and the compilation options.

The tile is compiled in because the launch grid is computed from it — one program per
tile — and the flag because a module compiled for programmatic dependent launch waits
and releases around its grid. Neither extent is: the frame count varies per request
while the arithmetic does not.
"""

import _compile
import transpose as kernels

#: Triton's element type for each element type the operator accepts.
POINTERS = {"f16": "*fp16", "bf16": "*bf16", "f32": "*fp32"}


def warps_of(block):
    """The warp count one tile needs: a block of columns is one load, and so is a row."""
    return max(1, min(8, block // 8))


def spec(fields, parameters, target):
    """The compilation of one transpose descriptor."""
    block = int(fields["block"])
    pdl = fields["pdl"] == "true"
    return {
        "kernel": kernels._transpose_kernel,
        "signature": {
            "input": POINTERS[fields["dtype"]],
            "out": POINTERS[fields["dtype"]],
            "rows": "i64",
            "columns": "i64",
            "block": "i32",
            "pdl": "i32",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {"BLOCK": block, "PDL": pdl},
        "options": {"num_warps": warps_of(block), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
