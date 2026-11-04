"""The logits_shape package's compile script: one descriptor -> one module.

A descriptor is the input logits' dtype, the vocabulary width of a row, the
columns one program shapes, how many extra token columns a row carries, whether
the row is divided by its temperature and the dependent-launch flag. The row
count stays a runtime value.
"""

import _compile
import logits_shape as kernels

#: The input logits' dtype; the shaped logits are FP32 in every case.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}

#: The largest column tile a program holds in registers in one step.
LANES = 4096


def spec(fields, parameters, target):
    """The compilation of one logits_shape descriptor."""
    block = int(fields["block"])
    extras = int(fields["extras"])
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._logits_shape_kernel,
        "signature": {
            "logits_ptr": "*" + DTYPES[fields["dtype"]],
            "counts_ptr": "*i32",
            "slots_ptr": "*i32",
            "parameters_ptr": "*fp32",
            "extras_ptr": "*i32",
            "out_ptr": "*fp32",
            "rows": "i64",
            "vocabulary": "i32",
            "divide": "i32",
            "BLOCK": "constexpr",
            "STEP": "constexpr",
            "EXTRAS": "constexpr",
            "DIVIDE": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "BLOCK": block,
            "STEP": min(block, LANES),
            "EXTRAS": extras,
            "DIVIDE": fields["divide"] == "true",
            "PDL": pdl,
        },
        "options": {"num_warps": 4, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
