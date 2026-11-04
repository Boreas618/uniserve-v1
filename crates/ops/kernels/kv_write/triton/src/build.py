"""The kv_write package's compile script: one descriptor -> one module.

A descriptor is the keys' dtype, the cache's dtype, the page size, the heads and
head width this rank holds, the elements of a cache row one program writes and
the dependent-launch flag. The token count stays a runtime value.

The cache's dtype is the same as the keys' in every descriptor this package
declares: an FP8 page is written through one scale per page, and the operator's
arguments carry no scale tensors, so an FP8 cache has no correct write here yet.
"""

import _compile
import kv_write as kernels

#: Element types the keys, the values and the cache share here.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one kv_write descriptor."""
    dtype = fields["dtype"]
    if fields["kv"] != dtype:
        raise SystemExit(
            f"a {fields['kv']} cache written from {dtype} keys needs a per-page scale, "
            "which the operator's arguments do not carry"
        )
    pointer = "*" + DTYPES[dtype]
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._kv_write_kernel,
        "signature": {
            "key_ptr": pointer,
            "value_ptr": pointer,
            "key_cache_ptr": pointer,
            "value_cache_ptr": pointer,
            "slots_ptr": "*i32",
            "key_initialized_ptr": "*i32",
            "value_initialized_ptr": "*i32",
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


if __name__ == "__main__":
    _compile.main(spec)
