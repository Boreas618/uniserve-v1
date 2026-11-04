"""The cache_copy_region package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options.

The pool's geometry is compiled in because the key names it: one page's plane size, the row a token
occupies and the columns per unit are all fixed for a given pool geometry, and the kernel's
addressing is built from them. The span is not: the token count, the page offset, the unit ids and
the field's units per column are launch words.
"""

import _compile
import cache_copy_region as kernels

#: Triton's element type for each stored type the operator accepts, and its byte width. An FP8 pool
#: is addressed as bytes: Triton's signature table has no pointer to a float8, so the kernel bitcasts
#: the byte to `tl.float8e4nv` itself.
STORED = {"bf16": ("bf16", 2), "f16": ("fp16", 2), "f8e4m3": ("u8", 1)}


def warps_of(row_size):
    """The warp count one token row needs: a load, a conversion and a store per lane."""
    return 4 if row_size <= 1024 else 8


def spec(fields, parameters, target):
    """The compilation of one cache_copy_region descriptor."""
    stored, element_bytes = STORED[fields["element"]]
    scaled = fields["scaled"] == "true"
    if scaled != (fields["element"] == "f8e4m3"):
        raise SystemExit(
            f"{fields['scaled']} with {fields['element']}: an FP8 destination is the scaled one, "
            "and a dense destination stores the values themselves"
        )
    heads = int(fields["heads"])
    head_dim = int(fields["head_dim"])
    plane_bytes = int(fields["plane_bytes"])
    row_size = heads * head_dim
    if plane_bytes % element_bytes:
        raise SystemExit(
            f"a plane of {plane_bytes} bytes is not whole {element_bytes}-byte elements"
        )
    plane_elements = plane_bytes // element_bytes
    if plane_elements % row_size:
        raise SystemExit(
            f"a plane of {plane_elements} elements is not whole {row_size}-element token rows"
        )
    row_pow2 = 1
    while row_pow2 < row_size:
        row_pow2 *= 2
    pdl = _compile.capability(target) >= 100
    return {
        "kernel": kernels._cache_copy_region_kernel,
        "signature": {
            "values_ptr": "*" + stored,
            # The per-unit F32 scale. A dense pool passes no scale view and the operator's flag
            # slot then carries the initialized view's address; the value is never read as a scale
            # because `SCALED` is false, and the pointer is only ever dereferenced in that branch.
            "scale_ptr": "*fp32" if scaled else "*u8",
            "initialized_ptr": "*u8",
            "source_ptr": "*fp32",
            "units_ptr": "*i32",
            "offset": "i64",
            "count": "i64",
            "units_total": "i32",
            "ROW_SIZE": "constexpr",
            "ROW_POW2": "constexpr",
            "PLANE_ELEMENTS": "constexpr",
            "COLUMNS": "constexpr",
            "BF16": "constexpr",
            "SCALED": "constexpr",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {
            "ROW_SIZE": row_size,
            "ROW_POW2": row_pow2,
            "PLANE_ELEMENTS": plane_elements,
            "COLUMNS": int(fields["columns"]),
            "BF16": 1 if fields["element"] == "bf16" else 0,
            "SCALED": 1 if scaled else 0,
            "BLOCK": row_pow2,
            "PDL": pdl,
        },
        "options": {"num_warps": warps_of(row_size), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
