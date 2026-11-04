"""The top_k package's compile script: one descriptor -> one module.

A descriptor is the logits' dtype, the candidates a row keeps, the candidate id
dtype and the dependent-launch flag. The row count and the vocabulary are launch
values, so one module serves every batch and every vocabulary; the candidate count
is the program's `WIDTH`, because the emission's buffers and its final sort are
that many lanes wide.
"""

import _compile
import top_k as kernels

#: The logits' dtype.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "f32": "fp32", "fp32": "fp32"}

#: The candidate id dtype.
IDS = {"i32": "i32", "i64": "i64"}

#: The widest network one program can hold: a bitonic sort of this many lanes is the last size that
#: fits a block's registers at the event's warp count.
NETWORK = 8192

#: The elements one radix or emission pass reads in one step.
STEP = 2048


def spec(fields, parameters, target):
    """The compilation of one top_k descriptor."""
    # The key's width is the **network's**: the live candidate count padded to a power of two. The
    # live count is a runtime argument, and it is the buffers' row stride.
    width = int(fields["width"])
    if width == 0 or width & (width - 1):
        raise SystemExit(f"a network width of {width} is not a power of two")
    if width > NETWORK:
        raise SystemExit(
            f"a network width of {width} is past the {NETWORK} lanes one program can sort: the "
            f"emission ends in a full-width bitonic sort, so a candidate count this wide needs an "
            f"emission that writes its slots in descending order itself (a radix pass over the kept "
            f"keys) instead of sorting them afterwards. The buffers need no change: the row stride "
            f"is the live width and every lane past it is masked."
        )
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._top_k_kernel,
        "signature": {
            "logits_ptr": "*" + DTYPES[fields["dtype"]],
            "values_ptr": "*fp32",
            "ids_ptr": "*" + IDS[fields["ids"]],
            "rows": "i64",
            "vocabulary": "i32",
            "width": "i32",
            "WIDTH": "constexpr",
            "STEP": "constexpr",
        },
        "constexprs": {"WIDTH": width, "STEP": STEP},
        "options": {"num_warps": 8, "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
