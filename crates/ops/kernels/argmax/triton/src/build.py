"""The argmax package's compile script: one descriptor -> one module.

A descriptor is the candidates' dtype, the row width, the emitted id dtype and
the dependent-launch flag. The row count comes from the grid.
"""

import _compile
import argmax as kernels

#: The candidates' dtype.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32", "f32": "fp32"}

#: The emitted index dtype.
IDS = {"i32": "i32", "i64": "i64"}


def spec(fields, parameters, target):
    """The compilation of one argmax descriptor."""
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._argmax_kernel,
        "signature": {
            "values_ptr": "*" + DTYPES[fields["dtype"]],
            "out_ptr": "*" + IDS[fields["ids"]],
            "rows": "i64",
            "WIDTH": "constexpr",
        },
        "constexprs": {"WIDTH": int(fields["width"])},
        "options": {"num_warps": 8 if int(fields["width"]) >= 2048 else 4, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
