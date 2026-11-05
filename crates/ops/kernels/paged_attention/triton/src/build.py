"""The paged_attention package's compile script: one descriptor -> one module.

This package serves the cache the vendor libraries refuse — an `E4M3` page with one f32 scale per
page — so its decode row is the one program here. `phase` is compiled by choosing the kernel (only
`decode` is served), `page` and `head_dim` are the tiles, and `q_heads`/`kv_heads` give the group a
program divides its query head by.

`kv` decides the cache's element type and `quantized` whether the program multiplies a page's scale
back: both are compiled, because a program that read a scale it was not given, or read an E4M3 page as
if it were dense, would be right only by accident.
"""

import _compile
import paged_attention as kernels

#: The element types a cache plane may have, as Triton spells them.
PLANES = {"bf16": "bf16", "f16": "fp16", "f8e4m3": "fp8e4nv"}

#: The element types the query and the output may have, as Triton spells them.
ACTIVATIONS = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(page, head_dim):
    """The warp count one `[page, head_dim]` block needs."""
    return 8 if page * head_dim >= 8192 else 4


#: The strides and extents both kernels take, in the order the operator packs them.
STRIDES = (
    "q_stride_tokens",
    "q_stride_heads",
    "q_stride_dim",
    "k_stride_pages",
    "k_stride_tokens",
    "k_stride_heads",
    "k_stride_dim",
    "v_stride_pages",
    "v_stride_tokens",
    "v_stride_heads",
    "v_stride_dim",
    "o_stride_tokens",
    "o_stride_heads",
    "o_stride_dim",
)


def spec(fields, parameters, target):
    """The compilation of one paged_attention descriptor."""
    page = int(fields["page"])
    head_dim = int(fields["head_dim"])
    q_heads = int(fields["q_heads"])
    kv_heads = int(fields["kv_heads"])
    if q_heads % kv_heads != 0:
        raise SystemExit(
            f"q_heads={q_heads} is not a whole multiple of kv_heads={kv_heads}: a query head has no "
            "key-value head"
        )
    quantized = fields["kv"] == "f8e4m3"
    plane = PLANES[fields["kv"]]
    activation = ACTIVATIONS[fields["dtype"]]
    out = ACTIVATIONS[fields["out"]]
    prefill = fields["phase"] == "prefill"
    parameters_of = {
        "out_ptr": "*" + out,
        "query_ptr": "*" + activation,
        "key_ptr": "*" + plane,
        "value_ptr": "*" + plane,
        # The scales are read only when the cache is quantized; a dense call passes the null
        # pointer the operator states, and the program's `QUANTIZED` branch never touches it.
        "key_scale_ptr": "*fp32",
        "value_scale_ptr": "*fp32",
        "page_table_ptr": "*i32",
        "seq_lens_ptr": "*i32",
    }
    parameters_of.update({name: "i32" for name in STRIDES})
    parameters_of.update({"max_pages": "i32", "page_size": "i32", "softmax_scale": "fp32"})
    if prefill:
        # The prefill program reads a row's first query token and whether the batch is packed; the
        # decode program has one query token per table row and needs neither.
        parameters_of.update(
            {"cum_seq_lens_q_ptr": "*i32", "max_q_len": "i32", "packed": "i32"}
        )
    constexprs = {
        "HEAD_DIM": head_dim,
        "PAGE": page,
        "GROUP": q_heads // kv_heads,
        "QUANTIZED": quantized,
        "OUT_BF16": fields["out"] == "bf16",
    }
    return {
        "kernel": kernels._paged_attention_prefill_kernel
        if prefill
        else kernels._paged_attention_decode_kernel,
        "signature": parameters_of,
        "constexprs": constexprs,
        # No `launch_pdl` option: this descriptor states no `pdl` field — the operator's key for the
        # quantized cache ends at `len` — so a program that read one would be compiling a word the
        # key never carries.
        "options": {"num_warps": warps_of(page, head_dim)},
    }


if __name__ == "__main__":
    _compile.main(spec)
