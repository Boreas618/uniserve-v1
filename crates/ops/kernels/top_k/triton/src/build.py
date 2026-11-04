"""The top_k package's compile script: one descriptor -> one module.

A descriptor is the logits' dtype, the chunk one pass reads, the candidate id dtype
and the dependent-launch flag. The row count, the vocabulary and the kept count are
all launch values, so one module serves every batch, every vocabulary and every
candidate count: the emission walks the row in `TILE`-lane chunks and orders the
kept pairs with radix passes over the scratch, so no buffer or network of the kept
count is compiled in.
"""

import _compile
import top_k as kernels

#: The logits' dtype.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "f32": "fp32", "fp32": "fp32"}

#: The candidate id dtype.
IDS = {"i32": "i32", "i64": "i64"}

#: The widest chunk one pass reads: an ordering pass holds two `TILE`-lane planes and a nibble's
#: ranks, which is the last size that fits a block's registers at the event's warp count.
MAX_TILE = 8192


def spec(fields, parameters, target):
    """The compilation of one top_k descriptor."""
    # `tile` is the chunk one pass reads. Nothing else is compiled in: the vocabulary, the row count
    # and the kept count are launch scalars, so one module serves any of them - which is why the
    # descriptor names no candidate width at all.
    tile = int(fields["tile"])
    if tile == 0 or tile & (tile - 1):
        raise SystemExit(f"a tile of {tile} is not a power of two")
    if tile > MAX_TILE:
        raise SystemExit(
            f"a tile of {tile} is past the {MAX_TILE} lanes one pass holds at the event's warp count"
        )
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._top_k_kernel,
        "signature": {
            "logits_ptr": "*" + DTYPES[fields["dtype"]],
            "values_ptr": "*fp32",
            "ids_ptr": "*" + IDS[fields["ids"]],
            "workspace_ptr": "*i32",
            "rows": "i64",
            "vocabulary": "i32",
            "width": "i32",
            "TILE": "constexpr",
        },
        "constexprs": {"TILE": tile},
        "options": {"num_warps": 8, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
