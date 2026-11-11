"""The elementwise_unary package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the
kernel function, its argument signature in the order the operator launches
them, the compile-time values the descriptor dimensions give, and the
compilation options. The Program Server's Triton source drives
``uniserve_program_server.triton`` with it.

The function, the element type and the number of axes are compiled in. The
shape, the strides and the function's two scalars are words, so one module
serves every shape that traverses ``rank`` axes; the scalars travel in the type
the function evaluates in.
"""

import elementwise_unary as kernels
import strided
from uniserve_program_server import triton as driver

#: The type the function's scalars travel in, by the computing type.
SCALARS = {"fp32": "fp32", "fp64": "fp64", "int": "i64"}


def spec(fields, parameters, target):
    """The compilation of one elementwise_unary descriptor."""
    block = int(fields["block"])
    compute = strided.compute(fields["dtype"])
    pointer = strided.pointer(fields["dtype"])
    pdl = fields["pdl"] == "true" and driver.capability(target) >= 100
    return {
        "kernel": kernels._elementwise_unary_kernel,
        "signature": {
            "x": pointer,
            "out": pointer,
            "elements": "i64",
            **strided.extents("d"),
            **strided.strides("s"),
            "a": SCALARS[compute],
            "b": SCALARS[compute],
            "pdl": "i32",
            "FUNCTION": "constexpr",
            "COMPUTE": "constexpr",
            "RANK": "constexpr",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "FUNCTION": fields["function"],
            "COMPUTE": compute,
            "RANK": int(fields["rank"]),
            "BLOCK": block,
            "PDL": pdl,
        },
        "options": {"num_warps": strided.warps_of(block), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    driver.main(spec)
