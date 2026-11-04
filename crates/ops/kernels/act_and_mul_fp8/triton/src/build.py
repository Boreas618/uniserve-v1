"""The act_and_mul_fp8 package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

One program reduces a complete row to find its E4M3 scale, so the row width and the lane block that
covers it are baked in, as are the activation and the stepwise flag. The two row strides are int32
launch words.
"""

import _compile
import act_and_mul_fp8 as kernels

#: The floating dtypes the gates and the values may have, as Triton spells them in an argument
#: signature.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one act_and_mul_fp8 descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    return {
        "kernel": kernels._act_and_mul_fp8_kernel,
        "signature": {
            "gate_ptr": dtype,
            "value_ptr": dtype,
            # The output encodes in E4M3; Triton spells the format the hardware has.
            "out_ptr": "*fp8e4nv",
            # One FP32 dequantization scale per row.
            "scale_ptr": "*fp32",
            "gate_row_stride": "i32",
            "value_row_stride": "i32",
            "n_cols": "constexpr",
            "ACTIVATION": "constexpr",
            "block": "constexpr",
            "stepwise": "constexpr",
        },
        "constexprs": {
            "n_cols": int(fields["width"]),
            "ACTIVATION": fields["activation"],
            "block": int(fields["block"]),
            "stepwise": fields["stepwise"] == "true",
        },
        "options": {
            "num_warps": 8,
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
