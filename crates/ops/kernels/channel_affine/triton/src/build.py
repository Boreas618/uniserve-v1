"""The channel_affine package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Build Server's Triton source drives
``uniserve_build_server.triton`` with it.

The channel count, the positions per channel and the element count are words, so one module serves
every geometry of an element type.
"""

import channel_affine as kernels
from uniserve_build_server import triton as driver

#: The dtypes a value may have, in the descriptor's vocabulary.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(block):
    """The warp count one block of elements needs."""
    return 8 if block >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one channel_affine descriptor."""
    block = int(fields["block"])
    dtype = "*" + DTYPES[fields["dtype"]]
    pdl = fields["pdl"] == "true" and driver.capability(target) >= 100
    return {
        "kernel": kernels._channel_affine_kernel,
        "signature": {
            "x": dtype,
            "scale": dtype,
            "shift": dtype,
            "out": dtype,
            "channels": "i64",
            "positions": "i64",
            "elements": "i64",
            "pdl": "i32",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {"BLOCK": block, "PDL": pdl},
        "options": {"num_warps": warps_of(block), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    driver.main(spec)
