"""The dynamic_scale package's compile script: one descriptor -> one module.

A descriptor is the partial magnitudes' element type, the tile the one program
reads at a time and the dependent-launch flag. The count of partials, the
format's range and the floor are launch words, so one module serves every
activation of one element type.
"""

import dynamic_scale as kernels
from uniserve_build_server import triton as driver

#: The element types the partials may have, as Triton spells them.
DTYPES = {"bf16": "*bf16", "f16": "*fp16", "f32": "*fp32"}


def spec(fields, parameters, target):
    """The compilation of one dynamic_scale descriptor."""
    tile = int(fields["tile"])
    pdl = fields["pdl"] == "true" and driver.capability(target) >= 100
    return {
        "kernel": kernels._dynamic_scale_kernel,
        "signature": {
            "partials": DTYPES[fields["dtype"]],
            "weight_amax": "*fp32",
            "encode": "*fp32",
            "alpha": "*fp32",
            "count": "i64",
            "range_": "fp32",
            "floor": "fp32",
            "TILE": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {"TILE": tile, "PDL": pdl},
        "options": {"num_warps": 4 if tile <= 1024 else 8, "launch_pdl": pdl},
    }
