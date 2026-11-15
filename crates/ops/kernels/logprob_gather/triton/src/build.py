"""The logprob_gather package's compile script: one descriptor -> one module.

A descriptor is the row width, the token id dtype, the tile the rank's count
walks the row in and the dependent-launch flag. The row count stays a runtime
value.
"""

import logprob_gather as kernels
from uniserve_build_server import triton as driver

#: The largest tile a program holds in registers in one step.
LANES = 4096

#: Token id dtypes: a sampler writes int32 or int64 tokens.
IDS = {"i32": "i32", "i64": "i64"}


def spec(fields, parameters, target):
    """The compilation of one logprob_gather descriptor."""
    pdl = fields["pdl"] == "true" and driver.capability(target) >= 100
    step = min(int(fields["tile"]), LANES)
    return {
        "kernel": kernels._logprob_gather_kernel,
        "signature": {
            "logprobabilities_ptr": "*fp32",
            "tokens_ptr": "*" + IDS[fields["ids"]],
            "out_ptr": "*fp32",
            "ranks_ptr": "*i32",
            "rows": "i64",
            "columns": "i32",
            "STEP": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {"STEP": step, "PDL": pdl},
        "options": {"num_warps": 8 if step >= 2048 else 4, "launch_pdl": pdl},
    }
