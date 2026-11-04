"""The bias_add package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The two residual variants are compiled in: a call without a residual reads two operands and rounds
once, a call with one reads four and rounds two or three times, so they are separate programs. The
strides, the channel count and the sample and position counts are words.
"""

import _compile
import bias_add as kernels

#: The dtypes a value may have, in the descriptor's vocabulary.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32", "f8e4m3": "fp8e4nv"}


def warps_of(block):
    """The warp count one block of elements needs."""
    return 8 if block >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one bias_add descriptor."""
    block = int(fields["block"])
    dtype = "*" + DTYPES[fields["dtype"]]
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._bias_add_kernel,
        "signature": {
            "values": dtype,
            "bias": dtype,
            "residual": dtype,
            "residual_bias": dtype,
            "out": dtype,
            "channels": "i32",
            "batch": "i64",
            "positions": "i64",
            "stride_batch": "i64",
            "stride_channel": "i64",
            "residual_batch": "i64",
            "residual_channel": "i64",
            "pdl": "i32",
            "BLOCK": "constexpr",
            "HAS_RESIDUAL": "constexpr",
            "HAS_RESIDUAL_BIAS": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "BLOCK": block,
            "HAS_RESIDUAL": fields["residual"] == "true",
            "HAS_RESIDUAL_BIAS": fields["residual_bias"] == "true",
            "PDL": pdl,
        },
        "options": {"num_warps": warps_of(block), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
