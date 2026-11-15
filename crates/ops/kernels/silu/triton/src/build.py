"""The silu package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Build Server's Triton source drives
``uniserve_build_server.triton`` with it.

The element type and the block width are compilations; the element count and the row width are
launch words. Programmatic dependent launch is a compilation option, so a descriptor that asks for
it compiles a module whose metadata declares it.
"""

import silu as kernels
from uniserve_build_server import triton as driver

#: The element types the activation may have; the arithmetic is FP32 in each.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(block):
    """The warp count one block of elements needs."""
    return 8 if block >= 1024 else 4


def spec(fields, parameters, target):
    """The compilation of one silu descriptor."""
    pdl = fields["pdl"] == "true" and driver.capability(target) >= 100
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
            "PDL": "constexpr",
        },
        "constexprs": {"BLOCK": block, "PDL": pdl},
        "options": {
            "num_warps": warps_of(block),
            "launch_pdl": pdl,
        },
    }
