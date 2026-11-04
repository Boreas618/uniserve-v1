"""The add_rms_norm package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor; see
``rms_norm``'s build script for the shape of that statement.
"""

import _compile
import add_rms_norm as kernels

#: The floating dtypes an activation row, its residual and its outputs may have.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}


def warps_of(tile):
    """The warp count one row tile needs."""
    return 8 if tile >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one add_rms_norm descriptor."""
    tile = int(fields["tile"])
    dtype = "*" + DTYPES[fields["dtype"]]
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._add_rms_norm_kernel,
        "signature": {
            "x_ptr": dtype,
            "r_ptr": dtype,
            "w_ptr": dtype,
            "y_ptr": dtype,
            "c_ptr": dtype,
            "columns": "i32",
            "eps": "fp32",
            "pdl": "i32",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {"BLOCK": tile, "PDL": pdl},
        "options": {"num_warps": warps_of(tile), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
