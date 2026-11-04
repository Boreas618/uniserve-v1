"""The gated_residual package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The element count and the row width are baked in — the kernel masks its tile with the count and
recovers each element's row from the width — so a descriptor is a compilation. The gate row stride
is an int32 launch argument.
"""

import _compile
import gated_residual as kernels

#: The tile one launch walks, the operator's `GATED_BLOCK`.
BLOCK = 1024

#: The floating dtypes a residual, its update and its gate may have, as Triton spells them in an
#: argument signature.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one gated_residual descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    return {
        "kernel": kernels._gated_residual_kernel,
        "signature": {
            "hidden_ptr": dtype,
            "update_ptr": dtype,
            "gate_ptr": dtype,
            "row_indices_ptr": "*" + fields["indices"],
            "gate_row_stride": "i32",
            "ELEMENTS": "constexpr",
            "WIDTH": "constexpr",
            "BLOCK": "constexpr",
        },
        "constexprs": {
            "ELEMENTS": int(fields["elements"]),
            "WIDTH": int(fields["width"]),
            "BLOCK": BLOCK,
        },
        "options": {
            "num_warps": 4,
            # The gated update rounds to the activation dtype before the residual sum, so the two
            # would otherwise contract into one rounding.
            "enable_fp_fusion": False,
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
