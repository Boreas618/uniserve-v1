"""The scaled_residual package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The row width, the element tile and whether the update carries a bias are baked in; the element
count is an int32 launch word.
"""

import _compile
import scaled_residual as kernels

#: The floating dtypes the rows, the parameters and the output may have, as Triton spells them in
#: an argument signature.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one scaled_residual descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    return {
        "kernel": kernels._scaled_residual_kernel,
        "signature": {
            "hidden_ptr": dtype,
            "update_ptr": dtype,
            "update_bias_ptr": dtype,
            "scale_ptr": dtype,
            "elements": "i32",
            "HAS_UPDATE_BIAS": "constexpr",
            "WIDTH": "constexpr",
            "BLOCK": "constexpr",
        },
        "constexprs": {
            "HAS_UPDATE_BIAS": fields["update_bias"] == "true",
            "WIDTH": int(fields["columns"]),
            "BLOCK": int(fields["block"]),
        },
        "options": {"num_warps": 4},
    }


if __name__ == "__main__":
    _compile.main(spec)
