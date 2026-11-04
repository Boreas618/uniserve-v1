"""The kv_quantize package's compile script: one descriptor -> one module.

A descriptor is the source's element type, the page scale's element type, the
page size, the key/value heads and head width this rank holds, the elements of a
row one program writes and the dependent-launch flag. The cache's own element
type is E4M3 by the layout every reader agrees on, so it is not a field. The
token count is a runtime word.
"""

import _compile
import kv_quantize as kernels

#: The source's element type: a full-precision accumulation's output.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one kv_quantize descriptor."""
    dtype = fields["dtype"]
    if dtype not in ("bf16", "f32"):
        raise SystemExit(f"a {dtype} source is not an accumulation's output")
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._kv_quantize_kernel,
        "signature": {
            "source": "*" + DTYPES[dtype],
            "slots": "*i32",
            "scales": "*" + DTYPES[fields["scale"]],
            "out": "*fp8e4nv",
            "initialized": "*i32",
            "tokens": "u64",
            "page": "u32",
            "heads": "u32",
            "head_dim": "u32",
            "pdl": "i32",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "BLOCK": int(fields["block"]),
            "PDL": pdl,
        },
        "options": {"num_warps": 4, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
