"""The vsa_compress_pooled package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options.

`max_tiles` is compiled as the block the softmax reduces in, rounded up to a power of two — Triton's
`tl.arange` needs a power-of-two extent, and the operator refuses a tile count past the descriptor's
bound, so one block covers every count the module accepts. `width` is the feature block and `block`
the lanes it uses; `heads` sizes the pooled values' layout (a program is one head and the grid
carries the count) and is in the key because the tolerance's reduction length and the layout are the
operator's, not this package's choice. `dtype` is the row type the store narrows into; the operands
are FP32 by construction and are not a dimension.
"""

import vsa_compress_pooled as kernels
from uniserve_build_server import triton as driver

#: The two operands and the tile sizes are FP32/I32 by construction — the pooling and the
#: reference's softmax run in FP32 — so only the compressed rows carry a descriptor type.
POOLED = "*fp32"
DTYPES = {"bf16": "*bf16", "f16": "*fp16", "f32": "*fp32"}


#: The query tiles a program holds and the key tiles one step reads. The first is the launch's grid
#: divisor as well, which `crates/ops/src/vsa.rs` states as `POOLED_QUERIES`.
QUERIES = 16
KEYS = 64


def spec(fields, parameters, target):
    """The compilation of one vsa_compress_pooled descriptor."""
    pdl = fields["pdl"] == "true" and driver.capability(target) >= 100
    block = int(fields["block"])
    return {
        "kernel": kernels._vsa_compress_pooled_kernel,
        "signature": {
            "scores_ptr": POOLED,
            "values_ptr": POOLED,
            "sizes_ptr": "*i32",
            "out_ptr": DTYPES[fields["dtype"]],
            "query_tiles": "i32",
            "tiles": "i32",
            "pdl": "i32",
            "HEADS": "constexpr",
            "WIDTH": "constexpr",
            "BLOCK": "constexpr",
            "QUERIES": "constexpr",
            "KEYS": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "HEADS": int(fields["heads"]),
            "WIDTH": int(fields["width"]),
            "BLOCK": block,
            "QUERIES": QUERIES,
            "KEYS": KEYS,
            "PDL": pdl,
        },
        "options": {
            "num_warps": 4,
            "launch_pdl": pdl,
        },
    }
