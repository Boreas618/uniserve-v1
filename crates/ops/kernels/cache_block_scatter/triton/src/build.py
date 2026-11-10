"""The cache_block_scatter package's compile script: one descriptor -> one module.

A descriptor is the storage's element type, the bank's block-major shape and
its strides, the index vector's stride and the copy's tile. The extent of the
copy and the bank's block count are runtime words — the index vector's length
and the field's shape — so one module serves every copy of that geometry.
"""

import _compile
import cache_block_scatter as kernels

#: The storage's element type, as Triton spells it.
DTYPES = {"u8": "u8", "bf16": "bf16", "f16": "fp16", "fp32": "fp32", "i32": "i32"}


def extents(one):
    """A descriptor extent list, `"2048x64x1x128"` -> `(2048, 64, 1, 128)`."""
    return tuple(int(part) for part in one.split("x"))


def spec(fields, parameters, target):
    """The compilation of one cache_block_scatter descriptor."""
    storage = DTYPES[fields["dtype"]]
    shape = extents(fields["shape"])
    strides = extents(fields["strides"])
    if len(shape) != len(strides):
        raise SystemExit(
            f"shape {fields['shape']} and strides {fields['strides']} differ in rank"
        )
    return {
        "kernel": kernels._scatter_kernel,
        "signature": {
            "field": "*" + storage,
            "blocks": "*i64",
            "snapshot": "*" + storage,
            "index_stride": "constexpr",
            "shape": "constexpr",
            "strides": "constexpr",
            "width": "constexpr",
            "BLOCK": "constexpr",
        },
        "constexprs": {
            "index_stride": int(fields["index_stride"]),
            "shape": shape,
            "strides": strides,
            "width": int(fields["width"]),
            "BLOCK": int(fields["block"]),
        },
        "options": {"num_warps": 4},
    }


if __name__ == "__main__":
    _compile.main(spec)
