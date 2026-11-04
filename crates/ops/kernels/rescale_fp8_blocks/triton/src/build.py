"""The rescale_fp8_blocks package's compile script: one descriptor -> one module.

A descriptor is the logical compute dtype the code round through, and the width
of one block's codes (one whole KV page). The rounded dtype is a compile-time
value of the kernel, so a cache whose logical dtype differs asks for its own
module.
"""

import triton.language as tl

import _compile
import rescale_fp8_blocks as kernels

#: The logical cache dtypes a decode rounds through before re-encoding.
COMPUTE = {"fp16": tl.float16, "bf16": tl.bfloat16, "fp32": tl.float32}


def spec(fields, parameters, target):
    """The compilation of one rescale_fp8_blocks descriptor."""
    width = int(fields["width"])
    return {
        "kernel": kernels._rescale_fp8_kernel,
        "signature": {
            "values": "*fp8e4nv",
            "old_scales": "*fp32",
            "new_scales": "*fp32",
            "initialized": "*i8",
            "width": "constexpr",
            "tile": "constexpr",
            "compute_dtype": "constexpr",
        },
        "constexprs": {
            "width": width,
            # A tile of at most 1024 elements per step keeps the block's walk
            # in registers.
            "tile": 1 << min(10, max(0, (width - 1).bit_length())),
            "compute_dtype": COMPUTE[fields["dtype"]],
        },
        "options": {"num_warps": 4},
    }


if __name__ == "__main__":
    _compile.main(spec)
