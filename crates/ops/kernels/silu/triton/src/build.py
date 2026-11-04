"""The silu package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The element type and the block width are compilations; the element count and the row width are
launch words. Programmatic dependent launch is a compilation option, so a descriptor that asks for
it compiles a module whose metadata declares it.
"""

import _compile
import silu as kernels

#: The element types the activation may have; the arithmetic is FP32 in each.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(block):
    """The warp count one block of elements needs."""
    return 8 if block >= 1024 else 4


def spec(fields, parameters, target):
    """The compilation of one silu descriptor."""
    block = int(fields["block"])
    dtype = "*" + DTYPES[fields["dtype"]]
    return {
        "kernel": kernels._silu_kernel,
        "signature": {
            "x_ptr": dtype,
            "out_ptr": dtype,
            "elements": "i64",
            "columns": "i32",
            "pdl": "i32",
            "BLOCK": "constexpr",
        },
        "constexprs": {"BLOCK": block},
        "options": {
            "num_warps": warps_of(block),
            "launch_pdl": fields["pdl"] == "true",
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
