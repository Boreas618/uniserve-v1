"""The gelu package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The approximation is compiled in: the reference selects a formula, and the two are separate programs.
``pdl`` is a flag the launch repeats as a word the kernel does not read; the module's own
`launch_pdl` option is what issues the dependent launch, and only a Blackwell target has it.
"""

import _compile
import gelu as kernels

#: The dtypes a row may have, in the descriptor's vocabulary.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32", "f8e4m3": "fp8e4nv"}


def warps_of(block):
    """The warp count one block of elements needs."""
    return 8 if block >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one gelu descriptor."""
    block = int(fields["block"])
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._gelu_kernel,
        "signature": {
            "x_ptr": "*" + DTYPES[fields["dtype"]],
            "out_ptr": "*" + DTYPES[fields["dtype"]],
            "elements": "i64",
            "columns": "i32",
            "pdl": "i32",
            "APPROXIMATE": "constexpr",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "APPROXIMATE": fields["approximate"],
            "BLOCK": block,
            "PDL": pdl,
        },
        "options": {"num_warps": warps_of(block), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
