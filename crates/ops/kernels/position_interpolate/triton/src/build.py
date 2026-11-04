"""The position_interpolate package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options.

`channels` is the table's feature width and `block` the lanes one program covers, so both are
compiled. `in_side` sizes no tile — the tap indices are the caller's — but it is in the key because
the operator's tolerance and its operand shapes are the descriptor's, and the fixture names it. The
output's point count is the launch's word, so one module serves every target grid.
"""

import _compile
import position_interpolate as kernels

#: The element types the table and the output are stored in; the taps accumulate in FP32.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(block):
    """The warp count one block of channels needs."""
    return 8 if block >= 1024 else 4


def spec(fields, parameters, target):
    """The compilation of one position_interpolate descriptor."""
    block = int(fields["block"])
    return {
        "kernel": kernels._position_interpolate_kernel,
        "signature": {
            "table_ptr": "*" + DTYPES[fields["dtype"]],
            "indices_ptr": "*i32",
            "weights_ptr": "*fp32",
            "out_ptr": "*" + DTYPES[fields["dtype"]],
            "points": "u32",
            "channels": "u32",
            "block": "u32",
            "pdl": "i32",
            "CHANNELS": "constexpr",
            "BLOCK": "constexpr",
        },
        "constexprs": {"CHANNELS": int(fields["channels"]), "BLOCK": block},
        "options": {
            "num_warps": warps_of(block),
            "launch_pdl": fields["pdl"] == "true",
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
