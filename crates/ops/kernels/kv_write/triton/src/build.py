"""The kv_write package's compile script: one descriptor -> one module.

A descriptor is the keys' dtype, the cache's dtype, the page size, the heads and
head width this rank holds, the elements of a cache row one program writes and
the dependent-launch flag. The token count stays a runtime value.

The keys and the values share `dtype`, the two cache planes share `kv`, and the
two are any pair of BF16, FP16 and FP32: the kernel rounds each element into the
cache's type as it stores it. An FP8 cache is the `kv_write_scaled` package's,
because its pages are written through a scale this operator does not carry.
"""

import kv_write as kernels
from uniserve_build_server import triton as driver

#: The descriptor's element types, as Triton spells them.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one kv_write descriptor."""
    # The pointers' element types are the kernel's: the store's cast reads the
    # cache pointer's, so the conversion is compiled, not a launch argument.
    source = "*" + DTYPES[fields["dtype"]]
    cache = "*" + DTYPES[fields["kv"]]
    pdl = fields["pdl"] == "true" and driver.capability(target) >= 100
    return {
        "kernel": kernels._kv_write_kernel,
        "signature": {
            "key_ptr": source,
            "value_ptr": source,
            "key_cache_ptr": cache,
            "value_cache_ptr": cache,
            "slots_ptr": "*i32",
            "key_initialized_ptr": "*u8",
            "value_initialized_ptr": "*u8",
            "tokens": "i64",
            "page": "i32",
            "heads": "i32",
            "head_dim": "i32",
            "pdl": "i32",
            "TILE": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {"TILE": int(fields["tile"]), "PDL": pdl},
        "options": {"num_warps": 4, "launch_pdl": pdl},
    }
