"""The copy_block package's compile script: one descriptor -> one module.

Every dimension of the geometry -- the four extents, both stride vectors -- is a launch word, so the
``spec`` below has no compile-time value but the element block. The descriptor names the two element
types, and they are equal at launch: a copy converts nothing.
"""

import _compile
import copy_block as kernels

#: The element types a block is stored in.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(block):
    """The warp count one block of elements needs."""
    return 8 if block >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one copy_block descriptor."""
    source = DTYPES[fields["source"]]
    out = DTYPES[fields["out"]]
    wide = fields["wide"] == "true"
    return {
        "kernel": kernels._copy_block_kernel,
        "signature": {
            "source_ptr": "*" + source,
            "out_ptr": "*" + out,
            "e0": "u64",
            "e1": "u64",
            "e2": "u64",
            "e3": "u64",
            "s0": "i64",
            "s1": "i64",
            "s2": "i64",
            "s3": "i64",
            "d0": "i64",
            "d1": "i64",
            "d2": "i64",
            "d3": "i64",
            "total": "u64",
            "block": "u32",
            "pdl": "i32",
            "BLOCK": "constexpr",
            "WIDE": "constexpr",
        },
        # The block is the operator's own constant; the descriptor does not carry it, so the key is
        # the two element types and a change of block is a change of this file.
        "constexprs": {"BLOCK": 2048 if wide else 1024, "WIDE": wide},
        "options": {"num_warps": 4, "launch_pdl": False},
        # A wide copy's launch refuses anything else: `Op::launch` checks the same words.
        "aligned": ["source_ptr", "out_ptr", "e3", "s0", "s1", "s2", "d0", "d1", "d2"]
        if wide
        else [],
    }


if __name__ == "__main__":
    _compile.main(spec)
