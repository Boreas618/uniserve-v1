"""The rope_apply package's compile script: one descriptor -> one module.

A descriptor is the values' and the output's element types, the head width, the
head count, how the rotation pairs its channels, the channels that turn, the
tokens one program handles, the head tile and the dependent-launch flag. The
token count is a runtime word; the rotation's width is a field only when it is
narrower than the head.
"""

import triton

import _compile
import rope_apply as kernels

#: The element types the operator's values and outputs take.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def spec(fields, parameters, target):
    """The compilation of one rope_apply descriptor."""
    head_dim = int(fields["head_dim"])
    rotated = int(fields.get("rotated", head_dim))
    if rotated <= 0 or rotated > head_dim or rotated % 2:
        raise SystemExit(f"a rotation of {rotated} of {head_dim} channels is not an even count")
    heads = int(fields["heads"])
    block = int(fields["block"])
    if heads > block:
        raise SystemExit(f"{heads} heads do not fit a {block} lane head tile")
    pairs = rotated // 2
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    narrowed = rotated != head_dim
    signature = {
        "x": "*" + DTYPES[fields["dtype"]],
        "cosine": "*fp32",
        "sine": "*fp32",
        "out": "*" + DTYPES[fields["output"]],
        "tokens": "u64",
        "pdl": "i32",
    }
    if narrowed:
        signature["rotated"] = "u32"
    return {
        "kernel": (
            kernels._rope_apply_narrowed_kernel if narrowed else kernels._rope_apply_full_kernel
        ),
        "signature": signature,
        "constexprs": {
            "ROTATED": rotated,
            "PAIRS": pairs,
            "PAIRS_TILE": triton.next_power_of_2(pairs),
            "HEAD_DIM": head_dim,
            "HEADS": heads,
            "ROWS": int(fields["rows"]),
            # The head tile is the descriptor's block, which must cover the heads.
            "HEADS_TILE": block,
            "INTERLEAVED": fields["rotation"] == "interleaved",
            "PDL": pdl,
        },
        "options": {"num_warps": 4 if block <= 128 else 8, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
