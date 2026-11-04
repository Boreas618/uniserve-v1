"""The weighted_rms_norm package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The row width and the lane tile that covers it are baked in, as is whether the program publishes its
row's largest magnitude — a program with a magnitude buffer takes one more pointer, so the two
variants are two kernels. The epsilon stays a launch word.
"""

import _compile
import weighted_rms_norm as kernels

#: The floating dtypes a descriptor prints, and the Triton spelling of each: the vocabulary is
#: `DType`'s, and Triton spells FP16 `fp16` and FP32 `fp32`.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(columns):
    """The warp count one activation row needs."""
    return 8 if columns >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one weighted_rms_norm descriptor."""
    columns = int(fields["columns"])
    dtype = "*" + DTYPES[fields["dtype"]]
    absmax = fields["absmax"] == "true"
    signature = {"hidden_ptr": dtype, "weight_ptr": dtype, "output_ptr": dtype}
    if absmax:
        signature["partials_ptr"] = "*fp32"
    signature["eps"] = "fp32"
    signature["WIDTH"] = "constexpr"
    signature["BLOCK"] = "constexpr"
    return {
        "kernel": (
            kernels._weighted_rms_norm_absmax_kernel
            if absmax
            else kernels._weighted_rms_norm_kernel
        ),
        "signature": signature,
        "constexprs": {"WIDTH": columns, "BLOCK": int(fields["tile"])},
        "options": {"num_warps": warps_of(columns)},
    }


if __name__ == "__main__":
    _compile.main(spec)
