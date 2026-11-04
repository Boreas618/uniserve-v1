"""The vsa_scores package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options.

`width` is the contraction and `block` the lanes it uses, so both are compiled; `heads` sizes no
tile — a program is one head and the grid carries the count — and it is in the key because the
scores' layout and the tolerance's reduction length are the operator's, not this package's choice.
There is no element-type dimension: both operands are FP32, the type `vsa_prepare_tiles` pools in.
"""

import _compile
import vsa_scores as kernels

#: The pooled rows are FP32 by construction — `vsa_prepare_tiles` pools each tile's per-head means
#: as the accumulation its pooling runs in — so the operand type is the operator's, not a
#: descriptor field, and the products accumulate in FP32 over a `width`-term contraction.
POOLED = "fp32"


def warps_of(block):
    """The warp count the contraction's lanes need."""
    return 8 if block >= 1024 else 4


def spec(fields, parameters, target):
    """The compilation of one vsa_scores descriptor."""
    block = int(fields["block"])
    return {
        "kernel": kernels._vsa_scores_kernel,
        "signature": {
            "query_ptr": "*" + POOLED,
            "key_ptr": "*" + POOLED,
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
