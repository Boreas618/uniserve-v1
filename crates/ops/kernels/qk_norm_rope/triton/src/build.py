"""The qk_norm_rope package's compile script: one descriptor -> one module.

A descriptor is the queries' dtype, the output dtype, the head width, the query
and key head counts, the rotation, the epsilon's bit pattern, the tokens one
program normalizes, the program's block and the dependent-launch flag. The token
count stays a runtime value.
"""

import struct

import _compile
import qk_norm_rope as kernels

#: Element types of the queries, keys and their normalized outputs.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}

#: The two rotations a single rotary axis takes.
ROTATIONS = {"half": False, "interleaved": True}


def float_of(bits):
    """The float a descriptor's hexadecimal bit pattern names."""
    return struct.unpack("<f", struct.pack("<I", int(bits, 16)))[0]


def spec(fields, parameters, target):
    """The compilation of one qk_norm_rope descriptor."""
    head_dim = int(fields["head_dim"])
    if head_dim % 2:
        raise SystemExit(f"a {head_dim}-wide head has no rotary pairs")
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._qk_norm_rope_kernel,
        "signature": {
            "query_ptr": "*" + DTYPES[fields["dtype"]],
            "key_ptr": "*" + DTYPES[fields["dtype"]],
            "q_weight_ptr": "*" + DTYPES[fields["dtype"]],
            "k_weight_ptr": "*" + DTYPES[fields["dtype"]],
            "cosine_ptr": "*fp32",
            "sine_ptr": "*fp32",
            "q_out_ptr": "*" + DTYPES[fields["output"]],
            "k_out_ptr": "*" + DTYPES[fields["output"]],
            "tokens": "i64",
            "epsilon": "fp32",
            "pdl": "i32",
            "Q_HEADS": "constexpr",
            "K_HEADS": "constexpr",
            "DIM": "constexpr",
            "ROWS": "constexpr",
            "INTERLEAVED": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "Q_HEADS": int(fields["q_heads"]),
            "K_HEADS": int(fields["k_heads"]),
            "DIM": head_dim,
            "ROWS": int(fields["rows"]),
            "INTERLEAVED": ROTATIONS[fields["rotation"]],
            "PDL": pdl,
        },
        "options": {"num_warps": 4, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
