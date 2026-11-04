"""The logits_gather package's compile script: one descriptor -> one module.

A descriptor is the hidden state's dtype, its width, the program's block and the
dependent-launch flag. The row count stays a runtime value.
"""

import _compile
import logits_gather as kernels

#: The hidden state's dtype, and the gathered row's.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one logits_gather descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._logits_gather_kernel,
        "signature": {
            "hidden_ptr": dtype,
            "rows_ptr": "*i32",
            "out_ptr": dtype,
            "count": "i64",
            "columns": "i32",
            "BLOCK": "constexpr",
        },
        "constexprs": {"BLOCK": int(fields["block"])},
        "options": {"num_warps": 4, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
