"""The FP8 paged_kv_write package's compile script: one descriptor -> one module.

A descriptor is the source dtype, the cache's per-token head geometry, the page
geometry and whether the caller tracks initialization. The page count and the
page size are compile-time values of the kernel (its device asserts bound a
slot by them), so one epoch's KV layout asks for one module.
"""

import _compile
import paged_kv_write as kernels

#: The floating source dtypes the kernel encodes from.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one FP8 paged_kv_write descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    head_dim = int(fields["head_dim"])
    heads = int(fields["heads"])
    block_size = int(fields["block_size"])
    row_width = heads * head_dim
    flags = fields["flags"] == "blocks"
    constexprs = {
        "num_pages": int(fields["pages"]),
        "block_size": block_size,
        "row_width": row_width,
        "head_dim": head_dim,
        "k_strides": (row_width, head_dim, 1),
        "v_strides": (row_width, head_dim, 1),
        "tile": int(parameters["tile"]),
    }
    if not flags:
        constexprs["k_initialized_ptr"] = None
        constexprs["v_initialized_ptr"] = None
    return {
        "kernel": kernels._paged_kv_write_fp8_kernel,
        "signature": {
            "k_cache_ptr": "*fp8e4nv",
            "v_cache_ptr": "*fp8e4nv",
            "slots_ptr": "*i32",
            "k_src_ptr": dtype,
            "v_src_ptr": dtype,
            "k_scale_ptr": "*fp32",
            "v_scale_ptr": "*fp32",
            "k_initialized_ptr": "*i8" if flags else "constexpr",
            "v_initialized_ptr": "*i8" if flags else "constexpr",
            "num_pages": "constexpr",
            "block_size": "constexpr",
            "row_width": "constexpr",
            "head_dim": "constexpr",
            "k_strides": "constexpr",
            "v_strides": "constexpr",
            "tile": "constexpr",
        },
        "constexprs": constexprs,
        # Device asserts bound a slot against the cache, and Triton compiles
        # them only in debug mode.
        "options": {"num_warps": 4, "debug": True},
    }


if __name__ == "__main__":
    _compile.main(spec)
