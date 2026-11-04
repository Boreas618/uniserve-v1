"""The posterior package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options.

The block is compiled in as a constexpr. Nothing about the geometry is: the channel count, the eight
strides and the element count are launch values, so one module serves every latent shape and both
physical layouts a request can have. The element type is always f32 and is not in the key: the reference's chain is f32
throughout and the FP16 round trip is inside the kernel.
"""

import _compile
import posterior as kernels

#: Triton's element type for the one element type this operator computes in.
POINTER = "*fp32"


def warps_of(block):
    """The warp count one block of samples needs: four loads, five roundings and a store."""
    return 8 if block >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one posterior descriptor."""
    block = int(fields["block"])
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._posterior_kernel,
        "signature": {
            "mean_ptr": POINTER,
            "log_variance_ptr": POINTER,
            "noise_ptr": POINTER,
            "out_ptr": POINTER,
            "channels": "u32",
            "mean_row": "u32",
            "mean_channel": "u32",
            "log_variance_row": "u32",
            "log_variance_channel": "u32",
            "noise_row": "u32",
            "noise_channel": "u32",
            "out_row": "u32",
            "out_channel": "u32",
            "total": "u64",
            "pdl": "i32",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {"BLOCK": block, "PDL": pdl},
        "options": {"num_warps": warps_of(block), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
