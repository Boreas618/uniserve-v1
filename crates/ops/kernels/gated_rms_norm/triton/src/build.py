"""The gated_rms_norm package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel
function, its argument signature in the order the operator launches them, the
compile-time values the descriptor gives, and the compilation options. The
Program Server's Triton source drives ``uniserve_program_server.triton`` with it.
"""

import gated_rms_norm as kernels
from uniserve_program_server import triton as driver

#: The floating dtypes a row, its gate, its weight and its output may have, as Triton
#: spells them in an argument signature.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(tile):
    """The warp count one row tile needs."""
    return 8 if tile >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one gated_rms_norm descriptor."""
    tile = int(fields["tile"])
    dtype = "*" + DTYPES[fields["dtype"]]
    return {
        # Every operand is an arena tensor or a weight, whose rows start on sixteen-byte bounds.
        "aligned": ["x_ptr", "gate_ptr", "w_ptr", "y_ptr"]
        + (["columns"] if int(fields["columns"]) % 16 == 0 else []),
        "kernel": kernels._gated_rms_norm_kernel,
        "signature": {
            "x_ptr": dtype,
            "gate_ptr": dtype,
            "w_ptr": dtype,
            "y_ptr": dtype,
            "columns": "i32",
            "eps": "fp32",
            "BLOCK": "constexpr",
        },
        "constexprs": {"BLOCK": tile},
        "options": {"num_warps": warps_of(tile)},
    }


if __name__ == "__main__":
    driver.main(spec)
