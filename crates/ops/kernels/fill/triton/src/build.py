"""The fill package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the
kernel function, its argument signature in the order the operator launches
them, the compile-time values the descriptor dimensions give, and the
compilation options. The Build Server's Triton source drives
``uniserve_build_server.triton`` with it.

The element type and the number of axes are compiled in; the shape, the
strides and the element's bits are words. The bits travel in the unsigned
integer as wide as the element, whose width the kernel is told.
"""

import fill as kernels
import strided
from uniserve_build_server import triton as driver

#: The width in bits of each unsigned word an element's bits travel in.
WIDTHS = {"u8": 8, "u16": 16, "u32": 32, "u64": 64}


def spec(fields, parameters, target):
    """The compilation of one fill descriptor."""
    block = int(fields["block"])
    word = strided.BITS[fields["dtype"]]
    pdl = fields["pdl"] == "true" and driver.capability(target) >= 100
    return {
        "kernel": kernels._fill_kernel,
        "signature": {
            "destination": strided.pointer(fields["dtype"]),
            "elements": "i64",
            **strided.extents("d"),
            **strided.strides("t"),
            "bits": word,
            "pdl": "i32",
            "WIDTH": "constexpr",
            "RANK": "constexpr",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "WIDTH": WIDTHS[word],
            "RANK": int(fields["rank"]),
            "BLOCK": block,
            "PDL": pdl,
        },
        "options": {"num_warps": strided.warps_of(block), "launch_pdl": pdl},
    }
