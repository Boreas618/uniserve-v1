"""The qk_norm_rope package's compile script: one descriptor -> one module.

A descriptor is the queries' and the output's element type, the head width, the
query and key head counts, the rotation's pairing, the epsilon's bit pattern,
the tokens one program normalizes, the program's block and the dependent-launch
flag — plus, only where they differ from the full-width weighted form, the width
of the rotation and the absence of a learned scale. The token count stays a
runtime value.
"""

import struct

import _compile
import qk_norm_rope as kernels

#: Element types of the queries, keys and their normalized outputs.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}

#: The two pairings a rotation takes.
ROTATIONS = {"half": False, "interleaved": True}


def float_of(bits):
    """The float a descriptor's hexadecimal bit pattern names."""
    return struct.unpack("<f", struct.pack("<I", int(bits, 16)))[0]


def spec(fields, parameters, target):
    """The compilation of one qk_norm_rope descriptor."""
    head_dim = int(fields["head_dim"])
    # A key names `rotated` only when the rotation is narrower than the head, and
    # `affine` only when the normalization has no learned scale.
    rotated = int(fields.get("rotated", head_dim))
    affine = fields.get("affine", "true") == "true"
    if rotated % 2 or rotated > head_dim or rotated == 0:
        raise SystemExit(f"a rotation of {rotated} channels of a {head_dim} channel head")
    # The turned pairs in tiles of a power of two: 96 channels turn 48 pairs.
    pair_lanes = 1 << (rotated // 2 - 1).bit_length()
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100

    signature = {
        "query_ptr": "*" + DTYPES[fields["dtype"]],
        "key_ptr": "*" + DTYPES[fields["dtype"]],
    }
    if affine:
        signature["q_weight_ptr"] = "*" + DTYPES[fields["dtype"]]
        signature["k_weight_ptr"] = "*" + DTYPES[fields["dtype"]]
    signature["cosine_ptr"] = "*fp32"
    signature["sine_ptr"] = "*fp32"
    signature["q_out_ptr"] = "*" + DTYPES[fields["output"]]
    signature["k_out_ptr"] = "*" + DTYPES[fields["output"]]
    signature["tokens"] = "i64"
    signature["epsilon"] = "fp32"
    signature["pdl"] = "i32"
    signature["Q_HEADS"] = "constexpr"
    signature["K_HEADS"] = "constexpr"
    signature["DIM"] = "constexpr"
    signature["ROTATED"] = "constexpr"
    signature["PAIRS"] = "constexpr"
    signature["ROWS"] = "constexpr"
    signature["INTERLEAVED"] = "constexpr"
    if affine:
        signature["AFFINE"] = "constexpr"
    signature["PDL"] = "constexpr"
    if rotated != head_dim:
        # A narrowed program's call names how many channels turn; the module
        # compiled its own width from the key, so the word is the call's.
        signature["rotated"] = "u32"

    constexprs = {
        "Q_HEADS": int(fields["q_heads"]),
        "K_HEADS": int(fields["k_heads"]),
        "DIM": head_dim,
        "ROTATED": rotated,
        "PAIRS": pair_lanes,
        "ROWS": int(fields["rows"]),
        "INTERLEAVED": ROTATIONS[fields["rotation"]],
        "PDL": pdl,
    }
    if affine:
        constexprs["AFFINE"] = True

    # The arguments decide the program: weights and a width word are ports.
    if affine:
        kernel = (
            kernels._qk_norm_rope_kernel
            if rotated == head_dim
            else kernels._qk_norm_rope_narrowed_kernel
        )
    else:
        kernel = (
            kernels._qk_norm_rope_unweighted_kernel
            if rotated == head_dim
            else kernels._qk_norm_rope_unweighted_narrowed_kernel
        )

    return {
        "kernel": kernel,
        "signature": signature,
        "constexprs": constexprs,
        "options": {"num_warps": 4, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
