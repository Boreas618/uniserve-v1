"""The candidate_order package's compile script: one descriptor -> one module.

A descriptor is the candidates' element types and the dependent-launch flag.
The row count and the candidate count are launch words: a request's selection
depth changes without recompiling the sampler.
"""

import _compile
import candidate_order as kernels

#: The element types the operator's views carry.
DTYPES = {"f32": "fp32", "bf16": "bf16", "f16": "fp16", "i32": "i32", "i64": "i64"}


def spec(fields, parameters, target):
    """The compilation of one candidate_order descriptor."""
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._candidate_order_kernel,
        "signature": {
            "values": "*" + DTYPES[fields["values"]],
            "ids": "*" + DTYPES[fields["ids"]],
            "ordered_values": "*" + DTYPES[fields["values"]],
            "ordered_ids": "*" + DTYPES[fields["ids"]],
            "rows": "u64",
            "width": "u32",
            "TILE": "constexpr",
            "PDL": "constexpr",
        },
        # A tile of 128 candidates holds the reference's whole selection depth
        # (top-k gathers at most 128) in one pass; a wider row costs more passes
        # rather than another compilation.
        "constexprs": {"TILE": 128, "PDL": pdl},
        "options": {"num_warps": 4, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
