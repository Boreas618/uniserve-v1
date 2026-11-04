"""The paged_kv_write package's compile script: one descriptor -> one module.

A descriptor is the cache dtype, the cache's per-token head geometry
(``heads``, ``head_dim``), the page geometry (``block_size``, ``pages``), the
source layout and whether the caller tracks block initialization. ``pages`` and
``block_size`` are compile-time values of the kernel: its device asserts bound
a slot by them. Both follow the epoch's KV layout, which the build config
fixes, so one epoch asks for one module.
"""

import _compile
import paged_kv_write as kernels

#: The floating cache dtypes a scatter may store into.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}

#: Source layouts: one [rows, heads, dim] view whose head and dim strides are
#: the cache's, i.e. contiguous rows of the same geometry.
LAYOUTS = {"contiguous": True}


def spec(fields, parameters, target):
    """The compilation of one paged_kv_write descriptor."""
    dtypes = "*" + DTYPES[fields["dtype"]]
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
    # A `None` flag pointer is a compile-time constant: it selects the
    # specialization that stores no initialization state.
    if not flags:
        constexprs["k_initialized_ptr"] = None
        constexprs["v_initialized_ptr"] = None
    return {
        "kernel": kernels._paged_kv_write_kernel,
        "signature": {
            "k_cache_ptr": dtypes,
            "v_cache_ptr": dtypes,
            "slots_ptr": "*i32",
            "k_src_ptr": dtypes,
            "v_src_ptr": dtypes,
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
        # Device asserts are what bounds a slot against the cache, and Triton
        # compiles them only in debug mode.
        "options": {"num_warps": 4, "debug": True},
    }


if __name__ == "__main__":
    _compile.main(spec)
