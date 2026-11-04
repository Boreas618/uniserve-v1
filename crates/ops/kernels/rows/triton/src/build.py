"""The rows package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options.

`mode` is a compilation — `replace` and `add` are different arithmetic, and the operator's tolerance
differs between them — and `columns` is the row width the program addresses with. `count`, `columns`,
`block` and the PDL flag also travel at launch, so one module serves every row count.
"""

import _compile
import rows as kernels

#: The element types the rows are stored in.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(block):
    """The warp count one block of columns needs."""
    return 8 if block >= 1024 else 4


def spec(fields, parameters, target):
    """The compilation of one rows descriptor."""
    block = int(fields["block"])
    return {
        "kernel": kernels._rows_kernel,
        "signature": {
            "source_ptr": "*" + DTYPES[fields["dtype"]],
            "destination_ptr": "*" + DTYPES[fields["dtype"]],
            "indices_ptr": "*i32",
            "count": "u32",
            "columns": "u32",
            "block": "u32",
            "pdl": "i32",
            "COLUMNS": "constexpr",
            "REPLACE": "constexpr",
            "BLOCK": "constexpr",
        },
        "constexprs": {
            "COLUMNS": int(fields["columns"]),
            "REPLACE": fields["mode"] == "replace",
            "BLOCK": block,
        },
        "options": {
            "num_warps": warps_of(block),
            "launch_pdl": fields["pdl"] == "true",
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
