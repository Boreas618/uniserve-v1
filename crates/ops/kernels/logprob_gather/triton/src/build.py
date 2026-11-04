"""The logprob_gather package's compile script: one descriptor -> one module.

A descriptor is the row width, the token id dtype and the dependent-launch flag.
The row count stays a runtime value.
"""

import _compile
import logprob_gather as kernels

#: Token id dtypes: a sampler writes int32 or int64 tokens.
IDS = {"i32": "i32", "i64": "i64"}


def spec(fields, parameters, target):
    """The compilation of one logprob_gather descriptor."""
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._logprob_gather_kernel,
        "signature": {
            "logprobabilities_ptr": "*fp32",
            "tokens_ptr": "*" + IDS[fields["ids"]],
            "out_ptr": "*fp32",
            "rows": "i64",
            "columns": "i32",
        },
        "constexprs": {},
        "options": {"num_warps": 1, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
