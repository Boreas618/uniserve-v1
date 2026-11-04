"""The embedding package's compile script: one descriptor -> one module.

A descriptor is the token dtype, the table's dtype and the hidden width. The
shard's bounds are runtime values: they come from the rank's vocabulary
partition, which is a build-config value, not part of the kernel's shape.
"""

import _compile
import embedding as kernels

#: Token id dtypes.
IDS = {"i32": "i32", "i64": "i64"}

#: The embedding table's and the output rows' dtype.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one embedding descriptor."""
    return {
        "kernel": kernels._embedding_kernel,
        "signature": {
            "ids_ptr": "*" + IDS[fields["ids"]],
            "weight_ptr": "*" + DTYPES[fields["dtype"]],
            "out_ptr": "*" + DTYPES[fields["dtype"]],
            "shard_start": "i64",
            "shard_end": "i64",
            "width": "constexpr",
            "block": "constexpr",
        },
        "constexprs": {
            "width": int(fields["width"]),
            "block": int(parameters["block"]),
        },
        "options": {"num_warps": int(parameters["warps"])},
    }


if __name__ == "__main__":
    _compile.main(spec)
