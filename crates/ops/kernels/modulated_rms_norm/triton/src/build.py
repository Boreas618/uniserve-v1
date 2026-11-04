"""The modulated_rms_norm package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The launch passes the three modulation row strides as int32 words, so they are runtime arguments;
the row width, the lane block, the epsilon and the three flags are baked in, which is what makes
the descriptor a compilation.
"""

import struct

import _compile
import modulated_rms_norm as kernels

#: The floating dtypes an activation, its weight and its modulation vectors may have, as Triton
#: spells them in an argument signature.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}

#: The E4M3 output type, which carries one dequantization scale per row, as the descriptor's
#: ``output`` prints it, and as Triton spells it in an argument signature.
FP8 = "f8e4m3"
FP8_TRITON = "fp8e4nv"


def bits(hexadecimal):
    """The f32 a descriptor's ``epsilon`` prints as, ``"0x358637bd"`` -> 1e-06."""
    return struct.unpack("<f", struct.pack("<I", int(hexadecimal, 16)))[0]


def warps_of(width):
    """The warp count one activation row needs."""
    return 8 if width >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one modulated_rms_norm descriptor."""
    width = int(fields["width"])
    dtype = "*" + DTYPES[fields["dtype"]]
    output = fields["output"]
    fp8 = output == FP8
    stored = FP8_TRITON if fp8 else DTYPES[output]
    return {
        "kernel": kernels._modulated_rms_kernel,
        "signature": {
            "hidden_ptr": dtype,
            "update_ptr": dtype,
            "gate_ptr": dtype,
            "weight_ptr": dtype,
            "shift_ptr": dtype,
            "scale_ptr": dtype,
            "row_indices_ptr": "*" + fields["indices"],
            "output_ptr": "*" + stored,
            # The dequantization scale is FP32 whether or not the descriptor writes one; with a
            # non-E4M3 output the launch reuses another view's address, which the kernel never
            # dereferences.
            "output_scale_ptr": "*fp32",
            "retain_ptr": "*" + stored,
            "gate_row_stride": "i32",
            "shift_row_stride": "i32",
            "scale_row_stride": "i32",
            "WIDTH": "constexpr",
            "BLOCK": "constexpr",
            "EPS": "constexpr",
            "HAS_UPDATE": "constexpr",
            "FP8_OUTPUT": "constexpr",
            "RETAIN": "constexpr",
        },
        "constexprs": {
            "WIDTH": width,
            "BLOCK": int(fields["block"]),
            "EPS": bits(fields["epsilon"]),
            "HAS_UPDATE": fields["update"] == "true",
            "FP8_OUTPUT": fp8,
            "RETAIN": fields["retain"] == "true",
        },
        "options": {
            "num_warps": warps_of(width),
            # Every intermediate rounds to the activation dtype as the eager expression does, so a
            # contracted multiply-add would round the wrong count of times.
            "enable_fp_fusion": False,
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
