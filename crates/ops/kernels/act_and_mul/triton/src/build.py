"""The act_and_mul package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The element count one launch walks, the gate and value row strides, the row width, the activation,
the 64-bit addressing and the two flags are all in the key, because each one changes what the module
computes: the strides and the element count are launch words whose addressing width the module bakes
in.
"""

import _compile
import act_and_mul as kernels

#: The floating dtypes the gates, the values and the output may have, as Triton spells them in an
#: argument signature.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one act_and_mul descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    wide = fields["wide"] == "true"
    # The descriptor's flag decides whether the module issues the dependent launch; a target older
    # than sm_100 cannot, so the module is compiled without it there.
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._act_and_mul_kernel,
        "signature": {
            "gate_ptr": dtype,
            "value_ptr": dtype,
            "out_ptr": dtype,
            # A 64-bit descriptor counts elements and rows in 64 bits; a 32-bit one in 32.
            "elements": "i64" if wide else "i32",
            "gate_row_stride": "i64" if wide else "i32",
            "value_row_stride": "i64" if wide else "i32",
            "n_cols": "constexpr",
            "ACTIVATION": "constexpr",
            "BLOCK": "constexpr",
            "WIDE": "constexpr",
            "PDL": "constexpr",
            "stepwise": "constexpr",
        },
        "constexprs": {
            "n_cols": int(fields["width"]),
            "ACTIVATION": fields["activation"],
            "BLOCK": int(fields["block"]),
            "WIDE": wide,
            "PDL": pdl,
            "stepwise": fields["stepwise"] == "true",
        },
        "options": {
            "num_warps": 8,
            "launch_pdl": pdl,
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
