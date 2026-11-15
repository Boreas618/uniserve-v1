"""The index package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the
kernel function, its argument signature in the order the operator launches
them, the compile-time values the descriptor dimensions give, and the
compilation options. The Build Server's Triton source drives
``uniserve_build_server.triton`` with it.

A ``gather`` module is the kernel over destination elements; a
``scatter_set`` or ``scatter_add`` module is the kernel over columns, which
traverses the indexed axis in order. The element type, the indices' type and
the number of axes traversed are compiled in; extents, strides and the steps
along the indexed axis are words.
"""

import index as kernels
import strided
from uniserve_build_server import triton as driver

#: The index that names no element, by the indices' type: its largest value.
ABSENT = {"u8": 2**8 - 1, "u32": 2**32 - 1, "i64": 2**63 - 1}


def gather(fields, pdl):
    """The compilation of a gather: one program per block of elements."""
    block = int(fields["block"])
    pointer = strided.pointer(fields["dtype"])
    return {
        "kernel": kernels._gather_kernel,
        "signature": {
            "source": pointer,
            "indices": strided.pointer(fields["indices"]),
            "destination": pointer,
            "elements": "i64",
            **strided.extents("d"),
            **strided.strides("s"),
            **strided.strides("i"),
            "step": "i64",
            "pdl": "i32",
            "ABSENT": "constexpr",
            "RANK": "constexpr",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "ABSENT": ABSENT[fields["indices"]],
            "RANK": int(fields["rank"]),
            "BLOCK": block,
            "PDL": pdl,
        },
        "options": {"num_warps": strided.warps_of(block), "launch_pdl": pdl},
    }


def scatter(fields, pdl):
    """The compilation of a scatter: one program per block of columns."""
    block = int(fields["block"])
    pointer = strided.pointer(fields["dtype"])
    # Only an addition computes, and so only it names a computing type: a
    # `scatter_set` of an element that only moves stores the bits it loaded.
    adds = fields["function"] == "scatter_add"
    compute = strided.compute(fields["dtype"]) if adds else "int"
    return {
        "kernel": kernels._scatter_kernel,
        "signature": {
            "source": pointer,
            "indices": strided.pointer(fields["indices"]),
            "destination": pointer,
            "columns": "i64",
            "length": "i64",
            **strided.extents("c"),
            **strided.strides("s"),
            **strided.strides("i"),
            **strided.strides("t"),
            "source_step": "i64",
            "indices_step": "i64",
            "destination_step": "i64",
            "pdl": "i32",
            "ADDS": "constexpr",
            "COMPUTE": "constexpr",
            "ABSENT": "constexpr",
            "RANK": "constexpr",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "ADDS": adds,
            "COMPUTE": compute,
            "ABSENT": ABSENT[fields["indices"]],
            "RANK": int(fields["rank"]),
            "BLOCK": block,
            "PDL": pdl,
        },
        "options": {"num_warps": strided.warps_of(block), "launch_pdl": pdl},
    }


def spec(fields, parameters, target):
    """The compilation of one index descriptor."""
    pdl = fields["pdl"] == "true" and driver.capability(target) >= 100
    if fields["function"] == "gather":
        return gather(fields, pdl)
    return scatter(fields, pdl)
