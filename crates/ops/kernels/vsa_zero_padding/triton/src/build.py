"""The vsa_zero_padding package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature, the order the operator launches them in, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.
"""

import _compile
import vsa_zero_padding as kernels

#: The floating dtypes a row may have: the descriptor prints `DType`'s names and Triton
#: spells FP16 `fp16` and FP32 `fp32`.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one vsa_zero_padding descriptor."""
    return {
        "kernel": kernels._zero_tile_padding_kernel,
        "signature": {
            "values": "*" + DTYPES[fields["dtype"]],
            "variable_block_sizes": "*i32",
            "stride_row": "i32",
            "stride_head": "i32",
            "heads": "constexpr",
            "tile_rows": "constexpr",
            "width": "constexpr",
            "block_rows": "constexpr",
        },
        "constexprs": {
            "heads": int(fields["heads"]),
            "tile_rows": int(fields["tile_rows"]),
            "width": int(fields["width"]),
            "block_rows": int(fields["block_rows"]),
        },
        "options": {"num_warps": 4},
    }

if __name__ == "__main__":
    _compile.main(spec)
