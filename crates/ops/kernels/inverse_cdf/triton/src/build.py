"""The inverse_cdf package's compile script: one descriptor -> one module.

A descriptor is the row width, the candidate id dtype, the emitted token dtype
and the dependent-launch flag. The row count stays a runtime value.
"""

import _compile
import inverse_cdf as kernels

#: Candidate id dtypes.
IDS = {"i32": "i32", "i64": "i64"}

#: Emitted token dtypes.
TOKENS = {"i32": "i32", "i64": "i64"}


def spec(fields, parameters, target):
    """The compilation of one inverse_cdf descriptor."""
    width = int(fields["width"])
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._inverse_cdf_kernel,
        "signature": {
            "probabilities_ptr": "*fp32",
            "ids_ptr": "*" + IDS[fields["ids"]],
            "draws_ptr": "*fp32",
            "tokens_ptr": "*" + TOKENS[fields["tokens"]],
            "rows": "i64",
            "WIDTH": "constexpr",
        },
        "constexprs": {"WIDTH": width},
        "options": {"num_warps": 8 if width >= 2048 else 4, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
