"""The filter_candidates package's compile script: one descriptor -> one module.

A descriptor is the row width, the candidate id dtype and the dependent-launch
flag. The row count stays a runtime value through the grid.
"""

import _compile
import filter_candidates as kernels

#: The candidate id dtype: read by the operators after this one.
IDS = {"i32": "i32", "i64": "i64"}


def spec(fields, parameters, target):
    """The compilation of one filter_candidates descriptor."""
    width = int(fields["width"])
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._filter_candidates_kernel,
        "signature": {
            "values_ptr": "*fp32",
            "ids_ptr": "*" + IDS[fields["ids"]],
            "parameters_ptr": "*fp32",
            "valid_ptr": "*i32",
            "rows": "i64",
            "WIDTH": "constexpr",
        },
        "constexprs": {"WIDTH": width},
        "options": {"num_warps": 8 if width >= 2048 else 4, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
