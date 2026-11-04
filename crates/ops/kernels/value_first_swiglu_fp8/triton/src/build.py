"""The value_first_swiglu_fp8 package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

One program reduces a complete packed row to find its E4M3 scale, so the output row width and the
lane block that covers it are baked in; the packed row stride is an int32 launch word.
"""

import _compile
import value_first_swiglu_fp8 as kernels

#: The floating dtypes the packed input may have, as Triton spells them in an argument signature.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one value_first_swiglu_fp8 descriptor."""
    return {
        "kernel": kernels._value_first_swiglu_fp8_kernel,
        "signature": {
            "value_gate_ptr": "*" + DTYPES[fields["dtype"]],
            # The output encodes in E4M3; Triton spells the format the hardware has.
            "output_ptr": "*fp8e4nv",
            # One FP32 dequantization scale per row.
            "output_scale_ptr": "*fp32",
            "input_row_stride": "i32",
            "width": "constexpr",
            "BLOCK": "constexpr",
        },
        "constexprs": {
            "width": int(fields["width"]),
            "BLOCK": int(fields["block"]),
        },
        "options": {
            "num_warps": 8,
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
