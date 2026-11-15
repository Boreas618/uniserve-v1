"""The top_logprobs package's compile script: one descriptor -> one module.

A descriptor is the candidates a chunk keeps, the chunk's lanes, whether the
entries' ids are read beside them and the dependent-launch flag. The row count,
the row width and the chunk count are launch values.
"""

import top_logprobs as kernels
from uniserve_build_server import triton as driver

#: The widest chunk a program holds in registers with its ids.
MAX_TILE = 8192


def spec(fields, parameters, target):
    """The compilation of one top_logprobs descriptor."""
    tile = int(fields["tile"])
    if tile == 0 or tile & (tile - 1) or tile > MAX_TILE:
        raise SystemExit(f"a tile of {tile} is not a power of two up to {MAX_TILE}")
    candidates = int(fields["candidates"])
    if candidates == 0:
        raise SystemExit("a selection keeps at least one candidate")
    pdl = fields["pdl"] == "true" and driver.capability(target) >= 100
    return {
        "kernel": kernels._top_logprobs_kernel,
        "signature": {
            "values_ptr": "*fp32",
            "ids_ptr": "*i32",
            "out_values_ptr": "*fp32",
            "out_ids_ptr": "*i32",
            "width": "i32",
            "tiles": "i32",
            "CANDIDATES": "constexpr",
            "TILE": "constexpr",
            "INDIRECT": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "CANDIDATES": candidates,
            "TILE": tile,
            "INDIRECT": fields["indirect"] == "true",
            "PDL": pdl,
        },
        "options": {"num_warps": 8 if tile >= 2048 else 4, "launch_pdl": pdl},
    }
