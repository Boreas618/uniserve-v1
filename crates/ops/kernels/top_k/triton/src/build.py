"""The top_k package's compile script: one descriptor -> one module.

A descriptor is the logits' dtype, the vocabulary width of a row, the candidates
a row keeps, the candidate id dtype and the dependent-launch flag. The row count
stays a runtime value and the vocabulary is a launch value, so one module serves
every batch.
"""

import _compile
import top_k as kernels

#: The logits' dtype.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "f32": "fp32", "fp32": "fp32"}

#: The candidate id dtype.
IDS = {"i32": "i32", "i64": "i64"}

#: The elements one radix or emission pass reads in one step.
STEP = 2048


def spec(fields, parameters, target):
    """The compilation of one top_k descriptor."""
    width = int(fields["width"])
    if width & (width - 1):
        raise SystemExit(f"a width of {width} is not a power of two, which the final sort needs")
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._top_k_kernel,
        "signature": {
            "logits_ptr": "*" + DTYPES[fields["dtype"]],
            "values_ptr": "*fp32",
            "ids_ptr": "*" + IDS[fields["ids"]],
            "rows": "i64",
            "vocabulary": "i32",
            "width": "i32",
            "WIDTH": "constexpr",
            "STEP": "constexpr",
        },
        "constexprs": {"WIDTH": width, "STEP": STEP},
        "options": {"num_warps": 8, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
