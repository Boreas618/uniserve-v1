"""The commit package's compile script: one descriptor -> one module.

A descriptor is the vocabulary (the penalty table's width, a compile-time value
of the kernel), the width of the slot's future-token row and the batch size the
module is captured for. Graph capture drives the row count, so a captured
bucket of ``n`` rows commits with the module whose block is the next power of
two at least ``n``; the operator passes the real ``count`` at launch.
"""

import _compile
import commit as kernels


def spec(fields, parameters, target):
    """The compilation of one commit descriptor."""
    return {
        "kernel": kernels._commit_kernel,
        "signature": {
            "columns_ptr": "*i64",
            "tokens_ptr": "*i64",
            "valid_ptr": "*i8",
            "active_ptr": "*i8",
            "continuation_ptr": "*i8",
            "points_ptr": "*i64",
            "future_ptr": "*i64",
            "predicates_ptr": "*i8",
            "logical_ptr": "*i64",
            "sampling_ptr": "*i64",
            "cache_ptr": "*i32",
            "penalties_ptr": "*i32",
            "count": "i32",
            "stride": "constexpr",
            "future_stride": "constexpr",
            "vocab": "constexpr",
            "block": "constexpr",
        },
        "constexprs": {
            "stride": int(parameters["columns"]),
            "future_stride": int(fields["future"]),
            "vocab": int(fields["vocab"]),
            "block": int(fields["batch"]),
        },
        "options": {"num_warps": 4},
    }


if __name__ == "__main__":
    _compile.main(spec)
