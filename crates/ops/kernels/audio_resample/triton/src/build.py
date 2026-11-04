"""The audio_resample package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The direction, the filter length, the ratio, the channel count and the block are compiled in: the
mode selects the convolution (a strided one or its transpose), the filter length is the program's tap
loop, and the ratio is the stride. The row and sample extents, the two pads and the flag are launch
words.
"""

import _compile
import audio_resample as kernels

#: The dtypes the signal and the output may have: the convolution is FP32.
DTYPES = {"f32": "fp32", "bf16": "bf16", "f16": "fp16"}


def warps_of(block):
    """The warp count one block of output samples needs."""
    return 8 if block >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one audio_resample descriptor."""
    block = int(fields["block"])
    dtype = "*" + DTYPES[fields["dtype"]]
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._audio_resample_kernel,
        "signature": {
            "x_ptr": dtype,
            "filter_ptr": "*fp32",
            "out_ptr": dtype,
            "rows": "i64",
            "channels": "i32",
            "width": "i32",
            "out_width": "i32",
            "pad": "i32",
            "pad_left": "i32",
            "pad_right": "i32",
            "pdl": "i32",
            "CHANNELS": "constexpr",
            "KERNEL": "constexpr",
            "RATIO": "constexpr",
            "UPSAMPLE": "constexpr",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "CHANNELS": int(fields["channels"]),
            "KERNEL": int(fields["kernel"]),
            "RATIO": int(fields["ratio"]),
            "UPSAMPLE": fields["mode"] == "upsample",
            "BLOCK": block,
            "PDL": pdl,
        },
        "options": {"num_warps": warps_of(block), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
