"""The kv_write_scaled package's compile script: one descriptor -> one module.

A descriptor is the element type of the source, the page's scale type, the page
size, the heads and head width this rank holds, the elements of a row one pass
covers, and the dependent-launch flag. The token count and the page count stay
runtime values: the grid is the pages the cache has.
"""

import _compile
import kv_write_scaled as kernels

#: Element types the source may have. The cache is E4M3 in every descriptor, which
#: is what this operator is for; a full-precision cache is `kv_write`.
DTYPES = {"bf16": "bf16", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one kv_write_scaled descriptor."""
    dtype = fields["dtype"]
    if fields["scale"] != "f32":
        raise SystemExit(
            f"a {fields['scale']} page scale is not the rule: the page's maximum is compared and "
            "the rows are divided in f32, and a narrower scale would round the page twice"
        )
    pointer = "*" + DTYPES[dtype]
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._kv_write_scaled_kernel,
        "signature": {
            "source_ptr": pointer,
            "slots_ptr": "*i32",
            "scales_ptr": "*fp32",
            "cache_ptr": "*fp8e4nv",
            "initialized_ptr": "*i32",
            "tokens": "i64",
            "page": "i32",
            "heads": "i32",
            "head_dim": "i32",
            "pdl": "i32",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {"BLOCK": int(fields["block"]), "PDL": pdl},
        "options": {"num_warps": 4, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
