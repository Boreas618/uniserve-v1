"""The copy_strided package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the
kernel function, its argument signature in the order the operator launches
them, the compile-time values the descriptor dimensions give, and the
compilation options. The Build Server's Triton source drives
``uniserve_build_server.triton`` with it.

The element type and the number of axes are compiled in. The shape and both
sides' strides are words, so one module serves every copy that traverses
``rank`` axes: a strided source made contiguous, rows whose strides differ on
the two sides, a source written into a block of a larger tensor.
"""

import copy_strided as kernels
import strided
from uniserve_build_server import triton as driver


def spec(fields, parameters, target):
    """The compilation of one copy_strided descriptor."""
    block = int(fields["block"])
    pointer = strided.pointer(fields["dtype"])
    pdl = fields["pdl"] == "true" and driver.capability(target) >= 100
    return {
        "kernel": kernels._copy_strided_kernel,
        "signature": {
            "source": pointer,
            "destination": pointer,
            "elements": "i64",
            **strided.extents("d"),
            **strided.strides("s"),
            **strided.strides("t"),
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


if __name__ == "__main__":
    driver.main(spec)
