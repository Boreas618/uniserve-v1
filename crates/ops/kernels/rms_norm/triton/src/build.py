"""The rms_norm package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel
function, its argument signature, the compile-time values the descriptor
dimension and the package parameters give, and the compilation options. The
Program Server's Triton source drives ``_compile`` with it.
"""

import _compile
import rms_norm as kernels

#: The floating dtypes a row, its weight and its output may have, as Triton
#: spells them in an argument signature.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32", "f32": "fp32"}


def warps_of(tile):
    """The warp count one row tile needs."""
    return 8 if tile >= 2048 else 4


def spec(fields, parameters, target):
    """The compilation of one rms_norm descriptor."""
    tile = int(fields["tile"])
    dtype = "*" + DTYPES[fields["dtype"]]
    # The descriptor's flag decides whether the module issues the dependent
    # launch; the launch repeats it as a word the kernel does not read.
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._rms_norm_kernel,
        "signature": {
            "x_ptr": dtype,
            "w_ptr": dtype,
            "y_ptr": dtype,
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
