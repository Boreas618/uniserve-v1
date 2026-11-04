"""The log_softmax package's compile script: one descriptor -> one module.

A descriptor is the input logits' dtype, the row width, the program's tile and
the dependent-launch flag. The output is FP32 in every case; the row count comes
from the grid.
"""

import _compile
import log_softmax as kernels

#: The largest column tile a program holds in registers in one step.
LANES = 4096

#: The input logits' dtype.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one log_softmax descriptor."""
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    tile = int(fields["tile"])
    return {
        "kernel": kernels._log_softmax_kernel,
        "signature": {
            "x_ptr": "*" + DTYPES[fields["dtype"]],
            "out_ptr": "*fp32",
            "columns": "i32",
            "pdl": "i32",
            "STEP": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {"STEP": min(tile, LANES), "PDL": pdl},
        "options": {"num_warps": 8 if min(tile, LANES) >= 2048 else 4, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
