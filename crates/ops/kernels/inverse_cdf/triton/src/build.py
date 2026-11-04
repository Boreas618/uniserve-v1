"""The inverse_cdf package's compile script: one descriptor -> one module.

A descriptor is the row width, the tile the program walks it in, the candidate id
dtype, the emitted token dtype and the dependent-launch flag. The row count and the
width are runtime values: the program loops over the row's tiles, so no block covers
the whole row.
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
    tile = int(fields["tile"])
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._inverse_cdf_kernel,
        "signature": {
            "probabilities_ptr": "*fp32",
            "ids_ptr": "*" + IDS[fields["ids"]],
            "draws_ptr": "*fp32",
            "tokens_ptr": "*" + TOKENS[fields["tokens"]],
            "rows": "i64",
            "width": "i32",
            "BLOCK": "constexpr",
        },
        "constexprs": {"BLOCK": tile},
        "options": {"num_warps": 8 if tile >= 2048 else 4, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
