"""The value_first_swiglu package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The output row width, the tile one program walks and the two flags are baked in — the packed input
row is ``width`` elements past the value half, and the magnitude buffer's length is the program
count the launch derives from the same tile. The element count and the packed row stride are int32
launch words, and the kernel indexes in int64 either way.
"""

import _compile
import value_first_swiglu as kernels

#: The floating dtypes the packed input, the bias and the output may have, as Triton spells them in
#: an argument signature.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one value_first_swiglu descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    return {
        "kernel": kernels._value_first_swiglu_kernel,
        "signature": {
            "value_gate_ptr": dtype,
            "bias_ptr": dtype,
            "output_ptr": dtype,
            # Each program's largest magnitude is FP32, whether or not one is written.
            "partials_ptr": "*fp32",
            "elements": "i32",
            "input_row_stride": "i32",
            "width": "constexpr",
            "HAS_BIAS": "constexpr",
            "RETURN_ABSMAX": "constexpr",
            "BLOCK": "constexpr",
        },
        "constexprs": {
            "width": int(fields["width"]),
            "HAS_BIAS": fields["bias"] == "true",
            "RETURN_ABSMAX": fields["absmax"] == "true",
            "BLOCK": int(fields["block"]),
        },
        "options": {
            "num_warps": 8,
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
