"""The adaptive_pool package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel
function, its argument signature in the order the operator launches them, the
compile-time values the descriptor dimensions give, and the compilation options.

Everything the key names is compiled in, and all of it is a *shape*: the element
type, the head count, the input width and the pooled width. That is what makes the
pool a compile-time loop — the window's longest length is `ceil(width / output)` and
the head axis is unrolled — while the row count is a launch value, so one module
serves every frame count.
"""

import _compile
import adaptive_pool as kernels

#: Triton's element type for each element type the operator accepts, as the
#: signature spells a pointer to it.
POINTERS = {"f16": "*fp16", "bf16": "*bf16", "f32": "*fp32"}


def warps_of(block):
    """The warp count one block of output elements needs.

    Each output element reads `heads · window` values, so the block is wide enough to
    keep the loads in flight without running out of registers for the accumulator.
    """
    return max(1, min(8, block // 256))


def spec(fields, parameters, target):
    """The compilation of one adaptive_pool descriptor."""
    block = int(fields["block"])
    heads = int(fields["heads"])
    width = int(fields["width"])
    output = int(fields["output"])
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._adaptive_pool_kernel,
        "signature": {
            "x": POINTERS[fields["dtype"]],
            "out": POINTERS[fields["dtype"]],
            "rows": "i64",
            "heads": "i32",
            "width": "i32",
            "output": "i32",
            "pdl": "i32",
            "HEADS": "constexpr",
            "WIDTH": "constexpr",
            "OUTPUT": "constexpr",
            "WINDOW": "constexpr",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "HEADS": heads,
            "WIDTH": width,
            "OUTPUT": output,
            # The longest window the descriptor can give. `ceil(width / output)` is
            # one short when the width does not divide: the window is
            # `ceil((i+1)·width/output) - floor(i·width/output)`, whose two roundings
            # can add a position (24 pooled to 7 gives windows of four and five).
            "WINDOW": -(-width // output) + 1,
            "BLOCK": block,
            "PDL": pdl,
        },
        "options": {"num_warps": warps_of(block), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
