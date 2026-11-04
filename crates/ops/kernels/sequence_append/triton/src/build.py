"""The sequence_append package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The element type, the register count and the tail are the compilation — the register rows are the
model's and the tail is the bucket's — while the caller's row count is a launch word.
"""

import _compile
import sequence_append as kernels

#: The element types the buffer and the register rows may have.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(block):
    """The warp count one block of elements needs."""
    return 8 if block >= 1024 else 4


def spec(fields, parameters, target):
    """The compilation of one sequence_append descriptor."""
    block = int(fields["block"])
    dtype = "*" + DTYPES[fields["dtype"]]
    return {
        "kernel": kernels._sequence_append_kernel,
        "signature": {
            "out_ptr": dtype,
            "registers_ptr": dtype,
            "rows": "i32",
            "width": "i32",
            "registers": "i32",
            "tail": "i32",
            "block": "i32",
            "pdl": "i32",
            "BLOCK": "constexpr",
        },
        "constexprs": {"BLOCK": block},
        "options": {
            "num_warps": warps_of(block),
            "launch_pdl": fields["pdl"] == "true",
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
