"""The vsa_scores package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options.

`width` is the contraction and `block` the lanes it uses, so both are compiled; `heads` sizes no
tile — a program is one head and the grid carries the count — and it is in the key because the
scores' layout and the tolerance's reduction length are the operator's, not this package's choice.
"""

import _compile
import vsa_scores as kernels

#: The element types the pooled rows are stored in; the products accumulate in FP32.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(block):
    """The warp count the contraction's lanes need."""
    return 8 if block >= 1024 else 4


def spec(fields, parameters, target):
    """The compilation of one vsa_scores descriptor."""
    block = int(fields["block"])
    return {
        "kernel": kernels._vsa_scores_kernel,
        "signature": {
            "query_ptr": "*" + DTYPES[fields["dtype"]],
            "key_ptr": "*" + DTYPES[fields["dtype"]],
            "out_ptr": "*fp32",
            "query_tiles": "i32",
            "key_tiles": "i32",
            "pdl": "i32",
            "WIDTH": "constexpr",
            "BLOCK": "constexpr",
            "KEY_TILES": "constexpr",
        },
        "constexprs": {"WIDTH": int(fields["width"]), "BLOCK": block, "KEY_TILES": 16},
        "options": {
            "num_warps": warps_of(block),
            "launch_pdl": fields["pdl"] == "true",
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
