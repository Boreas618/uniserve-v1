"""The page_conversion package's compile script: one descriptor -> one module.

A descriptor is the field this launch converts, the stored element type, the
logical type a decoded value rounds through, the head width, the heads and
layers of this rank's K/V group, and whether the source carries per-page scales.
The span's extent, the source's page size and scale group, and the number of
pages are runtime words, so one module serves every span of that geometry.
"""

import _compile
import page_conversion as kernels
import triton.language as tl

#: The stored element type: what the raw areas hold.
SOURCES = {
    "f8e4m3": (tl.float8e4nv, 1),
    "bf16": (tl.bfloat16, 2),
    "f16": (tl.float16, 2),
    "f32": (tl.float32, 4),
}

#: The logical element type a decoded value rounds through.
COMPUTES = {"bf16": tl.bfloat16, "f16": tl.float16, "f32": tl.float32}

#: The fields of a K/V page, as the descriptor numbers them.
FIELDS = {"key": 0, "value": 1}


def spec(fields, parameters, target):
    """The compilation of one page_conversion descriptor."""
    source, source_bytes = SOURCES[fields["source"]]
    # The operator refuses a scaled source in any other element type before a
    # call reaches a module, so that key can never be launched; it compiles to
    # the dense path rather than refusing the build, because a manifest states
    # one value list per descriptor field and no manifest can say that `scaled`
    # holds only for E4M3.
    scaled = fields["scaled"] == "true" and fields["source"] == "f8e4m3"
    return {
        "kernel": kernels._page_conversion_kernel,
        "signature": {
            "scratch": "*u8",
            "elements": "u64",
            "scales": "u64",
            "count": "u64",
            "source_offset": "u64",
            "source_block_size": "u32",
            "scale_heads": "u32",
            "pages": "u64",
            "FIELD": "constexpr",
            "SOURCE": "constexpr",
            "COMPUTE": "constexpr",
            "HEAD_DIM": "constexpr",
            "HEADS": "constexpr",
            "LAYERS": "constexpr",
            "SCALED": "constexpr",
            "SOURCE_BYTES": "constexpr",
        },
        "constexprs": {
            "FIELD": FIELDS[fields["field"]],
            "SOURCE": source,
            "COMPUTE": COMPUTES[fields["compute"]],
            "HEAD_DIM": int(fields["head_dim"]),
            "HEADS": int(fields["heads"]),
            "LAYERS": int(fields["layers"]),
            "SCALED": scaled,
            "SOURCE_BYTES": source_bytes,
        },
        "options": {"num_warps": 4},
    }


if __name__ == "__main__":
    _compile.main(spec)
