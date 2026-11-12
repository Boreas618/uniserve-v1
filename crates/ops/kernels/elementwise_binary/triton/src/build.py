"""The elementwise_binary package's compile script: a descriptor -> a module.

``spec`` states what this package's kernel needs for one descriptor: the
kernel function, its argument signature in the order the operator launches
them, the compile-time values the descriptor dimensions give, and the
compilation options. The Build Server's Triton source drives
``uniserve_build_server.triton`` with it.

The function, the element type and the number of axes are compiled in. The shape
and both inputs' strides are words, so one module serves every shape and every
broadcast that traverses ``rank`` axes. A comparison's output is ``u8`` whatever
the inputs hold. The functions of the integer element types alone name no
module for a floating one.
"""

import elementwise_binary as kernels
import strided
from uniserve_build_server import triton as driver

#: The functions whose output is a ``u8`` truth value.
COMPARISONS = ("eq", "ne", "lt", "le", "gt", "ge")

#: The functions that exist on the integer element types alone.
INTEGRAL = (
    "remainder",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
)

#: The integer element types that hold a sign.
SIGNED = ("i16", "i32", "i64")


def spec(fields, parameters, target):
    """The compilation of one elementwise_binary descriptor."""
    block = int(fields["block"])
    compute = strided.compute(fields["dtype"])
    if fields["function"] in INTEGRAL and compute != "int":
        raise SystemExit(
            f"{fields['function']} exists on the integer element types alone, "
            f"not on {fields['dtype']}"
        )
    compares = fields["function"] in COMPARISONS
    pointer = strided.pointer(fields["dtype"])
    pdl = fields["pdl"] == "true" and driver.capability(target) >= 100
    return {
        "kernel": kernels._elementwise_binary_kernel,
        "signature": {
            "lhs": pointer,
            "rhs": pointer,
            "out": "*u8" if compares else pointer,
            "elements": "i64",
            **strided.extents("d"),
            **strided.strides("l"),
            **strided.strides("r"),
            "pdl": "i32",
            "FUNCTION": "constexpr",
            "COMPARES": "constexpr",
            "COMPUTE": "constexpr",
            "SIGNED": "constexpr",
            "RANK": "constexpr",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "FUNCTION": fields["function"],
            "COMPARES": compares,
            "COMPUTE": compute,
            "SIGNED": fields["dtype"] in SIGNED,
            "RANK": int(fields["rank"]),
            "BLOCK": block,
            "PDL": pdl,
        },
        "options": {"num_warps": strided.warps_of(block), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    driver.main(spec)
