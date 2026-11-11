"""The scaled_add package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Build Server's Triton source drives
``uniserve_build_server.triton`` with it.

The two element types are separate pointer arguments, so a descriptor whose target is BF16 and
whose source is FP32 compiles a kernel taking two different dtypes — which is the operator's point.
The coefficients are launch scalars and the element count is a launch word.
"""

import scaled_add as kernels
from uniserve_build_server import triton as driver

#: The element types either operand may have; the arithmetic is FP32 in each.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(block):
    """The warp count one block of elements needs."""
    return 8 if block >= 1024 else 4


def spec(fields, parameters, target):
    """The compilation of one scaled_add descriptor."""
    block = int(fields["block"])
    return {
        "kernel": kernels._scaled_add_kernel,
        "signature": {
            "target_ptr": "*" + DTYPES[fields["target"]],
            "source_ptr": "*" + DTYPES[fields["source"]],
            "alpha": "fp32",
            "beta": "fp32",
            "elements": "i64",
            "BLOCK": "constexpr",
        },
        "constexprs": {"BLOCK": block},
        "options": {"num_warps": warps_of(block)},
    }


if __name__ == "__main__":
    driver.main(spec)
