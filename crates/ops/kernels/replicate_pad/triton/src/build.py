"""The replicate_pad package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel
function, its argument signature in the operator's launch order, the
compile-time values the descriptor dimension gives, and the compilation options.
The Program Server's Triton source drives ``_compile`` with it.
"""

import _compile
import replicate_pad as kernels

#: The element types a padded view may have, as Triton spells them.
DTYPES = {"u8": "u8", "i32": "i32", "i64": "i64", "f16": "fp16", "bf16": "bf16", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one replicate_pad descriptor."""
    block = int(fields["block"])
    axes = int(fields["axes"])
    dtype = "*" + DTYPES[fields["dtype"]]
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._replicate_pad_kernel,
        "signature": {
            "x_ptr": dtype,
            "out_ptr": dtype,
            "leading": "u64",
            "in0": "u64",
            "out0": "u64",
            "left0": "i32",
            "right0": "i32",
            "in1": "u64",
            "out1": "u64",
            "left1": "i32",
            "right1": "i32",
            "in2": "u64",
            "out2": "u64",
            "left2": "i32",
            "right2": "i32",
            "pdl": "i32",
            "BLOCK": "constexpr",
            "AXES": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {"BLOCK": block, "AXES": axes, "PDL": pdl},
        "options": {"num_warps": max(4, block // 256), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
