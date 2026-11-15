"""The elementwise_where package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the
kernel function, its argument signature in the order the operator launches
them, the compile-time values the descriptor dimensions give, and the
compilation options. The Build Server's Triton source drives
``uniserve_build_server.triton`` with it.

The values' element type, the condition's and the number of axes are compiled
in. The shape and the three inputs' strides are words, so one module serves
every shape and every broadcast that traverses ``rank`` axes.
"""

import elementwise_where as kernels
import strided
from uniserve_build_server import triton as driver


def spec(fields, parameters, target):
    """The compilation of one elementwise_where descriptor."""
    block = int(fields["block"])
    pointer = strided.pointer(fields["dtype"])
    pdl = fields["pdl"] == "true" and driver.capability(target) >= 100
    return {
        "kernel": kernels._elementwise_where_kernel,
        "signature": {
            "condition": strided.pointer(fields["condition"]),
            "on_true": pointer,
            "on_false": pointer,
            "out": pointer,
            "elements": "i64",
            **strided.extents("d"),
            **strided.strides("c"),
            **strided.strides("t"),
            **strided.strides("f"),
            "pdl": "i32",
            "RANK": "constexpr",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "RANK": int(fields["rank"]),
            "BLOCK": block,
            "PDL": pdl,
        },
        "options": {"num_warps": strided.warps_of(block), "launch_pdl": pdl},
    }
