"""The slot_commit package's compile script: one descriptor -> one module.

A descriptor is the state's element types, the commit table's stride, the next
input matrix's stride, the penalty counts' width and the lanes one program
covers. The step's row count is a runtime word and the grid is one program, so
one module serves every step of that shape.
"""

import _compile
import slot_commit as kernels
import triton.language as tl

#: The element types a descriptor may name, as Triton spells them.
DTYPES = {
    "i64": tl.int64,
    "i32": tl.int32,
    "u8": tl.uint8,
}

#: The signature's spelling of a pointer to each element type.
POINTERS = {
    "i64": "i64",
    "i32": "i32",
    "u8": "u8",
}


def spec(fields, parameters, target):
    """The compilation of one slot_commit descriptor."""
    index = fields["index"]
    verified = fields["verified"]
    penalties = fields["penalties"]
    for name, value in (
        ("index", index),
        ("verified", verified),
        ("penalties", penalties),
    ):
        if value not in DTYPES:
            raise SystemExit(f"{name} is {value}, which no descriptor names")
    block = int(fields["block"])
    if block & (block - 1):
        raise SystemExit(f"the {block} lane block is not a power of two")
    return {
        "kernel": kernels._slot_commit_kernel,
        "signature": {
            "columns": "*" + POINTERS[index],
            "tokens": "*" + POINTERS[index],
            # Both flags are read as bytes: 0 and 1 are 0 and 1 in the low byte
            # of any integer element the caller may hold them in.
            "valid": "*u8",
            "active": "*u8",
            "continuation": "*u8",
            "points": "*" + POINTERS[index],
            "future": "*" + POINTERS[index],
            "predicates": "*u8",
            "logical": "*" + POINTERS[index],
            "sampling": "*" + POINTERS[index],
            "verified": "*" + POINTERS[verified],
            "penalties": "*" + POINTERS[penalties],
            "count": "i64",
            "INDEX": "constexpr",
            "VERIFIED": "constexpr",
            "PENALTY_DTYPE": "constexpr",
            "STRIDE": "constexpr",
            "FUTURE_STRIDE": "constexpr",
            "VOCAB": "constexpr",
            "BLOCK": "constexpr",
        },
        "constexprs": {
            "INDEX": DTYPES[index],
            "VERIFIED": DTYPES[verified],
            "PENALTY_DTYPE": DTYPES[penalties],
            "STRIDE": int(fields["stride"]),
            "FUTURE_STRIDE": int(fields["future_stride"]),
            "VOCAB": int(fields["vocab"]),
            "BLOCK": block,
        },
        # One program covers the step's rows with `BLOCK` lanes; the commit is
        # latency-bound state movement, not a throughput kernel.
        "options": {"num_warps": 4},
    }


if __name__ == "__main__":
    _compile.main(spec)
