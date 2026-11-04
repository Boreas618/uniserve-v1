"""The audio_activation package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The tap count, the ratio and the block width are constexprs because the kernel's two loops and its
lane count are unrolled from them; the channel count and both lengths are launch words, since an
audio length follows the request's duration.
"""

import _compile
import audio_activation as kernels

#: The element types the signal, the two filters and the output may have.
DTYPES = {"f32": "fp32", "bf16": "bf16", "f16": "fp16"}


def warps_of(block):
    """The warp count one block of samples needs."""
    return 8 if block >= 1024 else 4


def spec(fields, parameters, target):
    """The compilation of one audio_activation descriptor."""
    block = int(fields["block"])
    dtype = "*" + DTYPES[fields["dtype"]]
    return {
        "kernel": kernels._audio_activation_kernel,
        "signature": {
            "values_ptr": dtype,
            "alpha_ptr": "*fp32",
            "beta_ptr": "*fp32",
            "filter_up_ptr": dtype,
            "filter_down_ptr": dtype,
            "out_ptr": dtype,
            "channels": "i32",
            "kernel": "i32",
            "ratio": "i32",
            "length": "i32",
            "out_length": "i32",
            "block": "i32",
            "pdl": "i32",
            "KERNEL": "constexpr",
            "RATIO": "constexpr",
            "BLOCK": "constexpr",
        },
        "constexprs": {
            "KERNEL": int(fields["kernel"]),
            "RATIO": int(fields["ratio"]),
            "BLOCK": block,
        },
        "options": {"num_warps": warps_of(block)},
    }


if __name__ == "__main__":
    _compile.main(spec)
