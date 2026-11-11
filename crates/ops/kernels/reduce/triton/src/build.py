"""The reduce package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the
kernel function, its argument signature in the order the operator launches
them, the compile-time values the descriptor dimensions give, and the
compilation options. The Program Server's Triton source drives
``uniserve_program_server.triton`` with it.

The function, the element type, the number of kept and of reduced axes, the
tile and the block are compiled in; the extents, the strides and the length
of the reduction are words. A program folds ``block / tile`` outputs, a tile
of each at a time. An index (``argmin``, ``argmax``) is written as ``u32``.
"""

import reduce as kernels
import strided
from uniserve_program_server import triton as driver

#: The two functions that return an index along their one axis.
INDEXES = ("argmin", "argmax")

#: The largest and the smallest value a 64-bit accumulator holds.
I64_MAX = 2**63 - 1
I64_MIN = -(2**63)


def accumulate(dtype):
    """The type a reduction over ``dtype`` elements accumulates in."""
    if dtype == "f64":
        return "fp64"
    if dtype in ("f16", "bf16", "f32"):
        return "fp32"
    return "i64"


def neutral(function, accumulator):
    """The value an output starts at: what the function leaves unchanged."""
    if function == "sum":
        return 0
    if function in ("min", "argmin"):
        return I64_MAX if accumulator == "i64" else float("inf")
    return I64_MIN if accumulator == "i64" else float("-inf")


def spec(fields, parameters, target):
    """The compilation of one reduce descriptor."""
    block = int(fields["block"])
    tile = int(fields["tile"])
    if tile > block:
        raise SystemExit(f"a tile of {tile} elements exceeds its block {block}")

    function = fields["function"]
    accumulator = accumulate(fields["dtype"])
    pdl = fields["pdl"] == "true" and driver.capability(target) >= 100
    return {
        "kernel": kernels._reduce_kernel,
        "signature": {
            "x": strided.pointer(fields["dtype"]),
            "out": "*u32"
            if function in INDEXES
            else strided.pointer(fields["dtype"]),
            "outputs": "i64",
            "length": "i64",
            **strided.extents("k"),
            **strided.strides("ks"),
            **strided.extents("r"),
            **strided.strides("rs"),
            "pdl": "i32",
            "FUNCTION": "constexpr",
            "ACCUMULATE": "constexpr",
            "NEUTRAL": "constexpr",
            "KEPT": "constexpr",
            "REDUCED": "constexpr",
            "ROWS": "constexpr",
            "TILE": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "FUNCTION": function,
            "ACCUMULATE": accumulator,
            "NEUTRAL": neutral(function, accumulator),
            "KEPT": int(fields["kept"]),
            "REDUCED": int(fields["reduced"]),
            "ROWS": block // tile,
            "TILE": tile,
            "PDL": pdl,
        },
        "options": {"num_warps": strided.warps_of(block), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    driver.main(spec)
