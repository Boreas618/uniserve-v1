"""The swiglu package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The row width, the tile, both row strides and the three flags are baked in; the element count is an
int32 launch word, and the kernel checks its mask in int64.
"""

import _compile
import swiglu as kernels

#: The floating dtypes the halves, the biases and the output may have, as Triton spells them in an
#: argument signature.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one swiglu descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    return {
        "kernel": kernels._swiglu_kernel,
        "signature": {
            "value_ptr": dtype,
            "gate_ptr": dtype,
            "value_bias_ptr": dtype,
            "gate_bias_ptr": dtype,
            "output_ptr": dtype,
            # Each program's largest magnitude is FP32, whether or not one is written.
            "partials_ptr": "*fp32",
            "elements": "i32",
            "width": "constexpr",
            "value_row_stride": "constexpr",
            "gate_row_stride": "constexpr",
            "HAS_VALUE_BIAS": "constexpr",
            "HAS_GATE_BIAS": "constexpr",
            "RETURN_ABSMAX": "constexpr",
            "BLOCK": "constexpr",
        },
        "constexprs": {
            "width": int(fields["width"]),
            "value_row_stride": int(fields["value_stride"]),
            "gate_row_stride": int(fields["gate_stride"]),
            "HAS_VALUE_BIAS": fields["value_bias"] == "true",
            "HAS_GATE_BIAS": fields["gate_bias"] == "true",
            "RETURN_ABSMAX": fields["absmax"] == "true",
            "BLOCK": int(fields["block"]),
        },
        "options": {
            "num_warps": 8,
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
