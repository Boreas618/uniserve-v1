"""The vsa_prepare_tiles package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Program Server's Triton source drives
``_compile`` with it.

Everything the descriptor names is compiled in: the three row and head stride pairs, the row strides
the embedded stepwise routine reads, the head count, the width, the rotated feature count, the
epsilon, the tile and block rows, and whether the rows are normalized. The operand views are launch
words. The stepwise routine's bits are fixed by its two warps and by unfused arithmetic, so the
module compiles with `num_warps = 2` and `enable_fp_fusion = false`, as the operator launched it.
"""

import struct

import _compile
import vsa_prepare_tiles as kernels

#: The floating dtypes a row may have, in the descriptor's vocabulary.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def epsilon_of(text):
    """The epsilon a descriptor key prints, as a float.

    `Float` prints its bits in hexadecimal. A 32-bit pattern widens into the same
    storage, so a value below `2**32` is the bit pattern of a float and the rest is
    the bit pattern of a double; decoding a float's pattern as a double would yield
    a denormal, which no epsilon is.
    """
    bits = int(text, 16)
    if bits < 1 << 32:
        return struct.unpack("<f", struct.pack("<I", bits))[0]
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def spec(fields, parameters, target):
    """The compilation of one vsa_prepare_tiles descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    row = int(fields["query_row_stride"])
    key_row = int(fields["key_row_stride"])
    return {
        "kernel": kernels._prepare_tiles_kernel,
        "signature": {
            "query": dtype,
            "key": dtype,
            "value": dtype,
            "query_weight": dtype,
            "key_weight": dtype,
            "cosine": dtype,
            "sine": dtype,
            "variable_block_sizes": "*i32",
            "pooled_query": "*fp32",
            "pooled_key": "*fp32",
            "pooled_value": "*fp32",
            "QUERY_STRIDES": "constexpr",
            "KEY_STRIDES": "constexpr",
            "VALUE_STRIDES": "constexpr",
            "QUERY_ROWS": "constexpr",
            "KEY_ROWS": "constexpr",
            "FACTOR_STRIDE": "constexpr",
            "HEADS": "constexpr",
            "DIM": "constexpr",
            "ROTATED": "constexpr",
            "EPS": "constexpr",
            "TILE": "constexpr",
            "BLOCK": "constexpr",
            "NORM": "constexpr",
        },
        "constexprs": {
            "QUERY_STRIDES": (row, int(fields["query_head_stride"])),
            "KEY_STRIDES": (key_row, int(fields["key_head_stride"])),
            "VALUE_STRIDES": (int(fields["value_row_stride"]), int(fields["value_head_stride"])),
            # The stepwise routine reads a block's rows as one head's consecutive
            # tokens, so it takes the row stride alone.
            "QUERY_ROWS": (row, 0),
            "KEY_ROWS": (key_row, 0),
            "FACTOR_STRIDE": int(fields["factor_stride"]),
            "HEADS": int(fields["heads"]),
            "DIM": int(fields["width"]),
            "ROTATED": int(fields["rotated"]),
            "EPS": epsilon_of(fields["epsilon"]),
            "TILE": int(fields["tile_rows"]),
            "BLOCK": int(fields["block_rows"]),
            "NORM": fields["norm"] == "true",
        },
        "options": {"num_warps": 2, "enable_fp_fusion": False},
    }


if __name__ == "__main__":
    _compile.main(spec)
