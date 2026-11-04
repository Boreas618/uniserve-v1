"""The cross_fade package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options.

The axis is a ``constexpr``: the blend's position is the column on axis 3 and the row on axis 2, and
the previous tile's band sits at the opposite end of that axis, so the read and the mask both follow
from it. `extent`, the four extents and the destination's three strides are the launch's words,
which is why the fixture's two extent axes are two cells rather than one program that branches.
"""

import _compile
import cross_fade as kernels

#: The element types a decoded tile is stored in.
DTYPES = {"f16": "fp16", "bf16": "bf16", "f32": "fp32"}


def warps_of(block):
    """The warp count one block of voxels needs."""
    return 8 if block >= 1024 else 4


def spec(fields, parameters, target):
    """The compilation of one cross_fade descriptor."""
    block = int(fields["block"])
    dtype = DTYPES[fields["dtype"]]
    return {
        "kernel": kernels._cross_fade_kernel,
        "signature": {
            "previous_ptr": "*" + dtype,
            "current_ptr": "*" + dtype,
            "out_ptr": "*" + dtype,
            "extent": "u32",
            "axis": "u32",
            "channels": "u32",
            "frames": "u32",
            "rows": "u32",
            "columns": "u32",
            "out_stride0": "u32",
            "out_stride1": "u32",
            "out_stride2": "u32",
            "block": "u32",
            "pdl": "i32",
            "AXIS": "constexpr",
            "BLOCK": "constexpr",
        },
        "constexprs": {"AXIS": int(fields["axis"]), "BLOCK": block},
        "options": {
            "num_warps": warps_of(block),
            "launch_pdl": fields["pdl"] == "true",
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
