"""The filter_candidates package's compile script: one descriptor -> one module.

A descriptor is the row width, the tile the program walks it in, the candidate id dtype
and the dependent-launch flag. The row count and the width are runtime values: the
program loops over the row's tiles, so no block covers the whole row.
"""

import filter_candidates as kernels
from uniserve_build_server import triton as driver

#: The candidate id dtype: read by the operators after this one.
IDS = {"i32": "i32", "i64": "i64"}


def spec(fields, parameters, target):
    """The compilation of one filter_candidates descriptor."""
    tile = int(fields["tile"])
    pdl = fields["pdl"] == "true" and driver.capability(target) >= 100
    return {
        "kernel": kernels._filter_candidates_kernel,
        "signature": {
            "values_ptr": "*fp32",
            "ids_ptr": "*" + IDS[fields["ids"]],
            "parameters_ptr": "*fp32",
            "valid_ptr": "*i32",
            "rows": "i64",
            "width": "i32",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {"BLOCK": tile, "PDL": pdl},
        "options": {"num_warps": 8 if tile >= 2048 else 4, "launch_pdl": pdl},
    }
