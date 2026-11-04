"""The embedding package's compile script: one descriptor -> one module.

A descriptor is the weight dtype, the id dtype, the hidden width, the shard's
first id and row count, the tokens one program copies and the dependent-launch
flag. The token count stays a runtime value.
"""

import _compile
import embedding as kernels

#: The vocabulary weight's dtype, and the output row's.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32", "f32": "fp32"}

#: Token id dtypes: a fused embedding table arrives as int32 or int64 ids.
IDS = {"i32": "i32", "i64": "i64"}


def spec(fields, parameters, target):
    """The compilation of one embedding descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    tile = int(parameters["tile"])
    return {
        "kernel": kernels._embedding_kernel,
        "signature": {
            "ids_ptr": "*" + IDS[fields["ids"]],
            "weight_ptr": dtype,
            "out_ptr": dtype,
            "tokens": "i64",
            "hidden": "i32",
            "first": "i32",
            "rows": "i32",
            "BLOCK": "constexpr",
            "TILE": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "BLOCK": int(fields["block"]),
            "TILE": tile,
            "PDL": pdl,
        },
        "options": {"num_warps": 4, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
