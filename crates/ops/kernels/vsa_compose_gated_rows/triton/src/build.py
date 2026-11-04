"""The vsa_compose_gated_rows package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

The row block, the tile, the head count, the width and the members' slots are baked in: the kernel
selects a member's payload slot from the compiled slot table, so the table is part of the module. The
six strides and the member interval are launch words.
"""

import _compile
import vsa_compose_gated_rows as kernels

#: The floating dtypes a row may have, in the descriptor's vocabulary.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def slots_of(fields):
    """The members' payload slots as the tuple the kernel compares against."""
    slots = tuple(int(slot) for slot in fields["slots"].split(","))
    members = int(fields["members"])
    if len(slots) != members:
        raise SystemExit(
            f"{members} members need {members} slots, the descriptor prints {slots}"
        )
    return slots


def spec(fields, parameters, target):
    """The compilation of one vsa_compose_gated_rows descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    return {
        "kernel": kernels._compose_gated_rows_kernel,
        "signature": {
            "output": dtype,
            "fine": dtype,
            "compressed": dtype,
            "gate": dtype,
            "fine_stride_row": "i32",
            "fine_stride_head": "i32",
            "gate_stride_row": "i32",
            "gate_stride_head": "i32",
            "compressed_stride_head": "i32",
            "compressed_stride_tile": "i32",
            "owner_rows": "i64",
            "start": "i64",
            "count": "i64",
            "heads": "constexpr",
            "tile_rows": "constexpr",
            "width": "constexpr",
            "block_rows": "constexpr",
            "slots": "constexpr",
        },
        "constexprs": {
            "heads": int(fields["heads"]),
            "tile_rows": int(fields["tile_rows"]),
            "width": int(fields["width"]),
            "block_rows": int(fields["block_rows"]),
            "slots": slots_of(fields),
        },
        "options": {"num_warps": 4},
    }


if __name__ == "__main__":
    _compile.main(spec)
