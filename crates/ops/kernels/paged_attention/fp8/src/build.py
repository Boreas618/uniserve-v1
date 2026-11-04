"""The FP8 paged_attention package's compile script: one descriptor -> one module.

A descriptor is the query dtype, the head geometry (query and key/value heads,
head width), the page geometry and the number of pages the cache addresses.
The softmax scale is a package parameter (the model's `1 / sqrt(head_dim)`
scaled by any YaRN attention factor).
"""

import _compile
import paged_attention as kernels

#: The query and output dtype; accumulation is FP32 in every case.
DTYPES = {"bf16": "bf16", "fp16": "fp16"}


def spec(fields, parameters, target):
    """The compilation of one FP8 paged_attention descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    head_dim = int(fields["head_dim"])
    return {
        "kernel": kernels._paged_attention_fp8_kernel,
        "signature": {
            "q_ptr": dtype,
            "k_cache_ptr": "*fp8e4nv",
            "v_cache_ptr": "*fp8e4nv",
            "k_scale_ptr": "*fp32",
            "v_scale_ptr": "*fp32",
            "block_tables_ptr": "*i32",
            "seq_lens_ptr": "*i32",
            "out_ptr": dtype,
            "q_heads": "constexpr",
            "kv_heads": "constexpr",
            "head_dim": "constexpr",
            "page_size": "constexpr",
            "max_pages": "constexpr",
            "num_pages": "constexpr",
            "softmax_scale": "constexpr",
            "block": "constexpr",
        },
        "constexprs": {
            "q_heads": int(fields["q_heads"]),
            "kv_heads": int(fields["kv_heads"]),
            "head_dim": head_dim,
            "page_size": int(fields["page_size"]),
            "max_pages": int(fields["max_pages"]),
            "num_pages": int(fields["pages"]),
            "softmax_scale": float(parameters["softmax_scale"]),
            # One program holds one head's whole row of features.
            "block": 1 << max(5, (head_dim - 1).bit_length()),
        },
        "options": {"num_warps": 4},
    }


if __name__ == "__main__":
    _compile.main(spec)
