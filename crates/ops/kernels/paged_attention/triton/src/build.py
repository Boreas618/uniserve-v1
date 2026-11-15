"""The paged_attention package's compile script: one descriptor -> one module.

This package serves the caches the vendor libraries do not read: an `E4M3` page
with one f32 scale per page, and a dense page (BF16, F16 or F32) of another type
than the queries. `phase` is compiled by choosing the kernel, `page` and
`head_dim` are the tiles, and `q_heads`/`kv_heads` give the group a program
divides its query head by, and at prefill the query tile's shape
(`prefill_tiles`).

`kv` decides the cache's element type and `quantized` whether the program
multiplies a page's scale back: both are compiled, because a program that read a
scale it was not given, or read an E4M3 page as if it were dense, would be right
only by accident.
"""

import paged_attention as kernels

#: The element types a cache plane may have, as Triton spells them.
PLANES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32", "f8e4m3": "fp8e4nv"}

#: The element types the query and the output may have, as Triton spells them.
ACTIVATIONS = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}


def warps_of(page, head_dim):
    """The warp count one decode program's `[page, head_dim]` block needs."""
    return 8 if page * head_dim >= 8192 else 4


#: The arguments the operator's `check` holds to a sixteen-byte base, the views
#: it checks by name, stated to the prefill program's compilation so that its
#: tile loads move whole vectors.
ALIGNED = (
    "out_ptr",
    "query_ptr",
    "key_ptr",
    "value_ptr",
    "page_table_ptr",
    "seq_lens_ptr",
)


#: The strides and extents both kernels take, in the order the operator packs
#: them.
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
            f"q_heads={q_heads} is not a whole multiple of "
            f"kv_heads={kv_heads}: a query head has no key-value head"
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
        # The scales are read only when the cache is quantized; a dense call
        # passes the null pointer the operator states, and the program's
        # `QUANTIZED` branch never touches it.
        "key_scale_ptr": "*fp32",
        "value_scale_ptr": "*fp32",
        "page_table_ptr": "*i32",
        "seq_lens_ptr": "*i32",
    }
    if prefill:
        # The prefill program reads a row's first query token and whether the
        # batch is packed; the decode program has one query token per table row
        # and needs neither. Both sit in the program's own parameter order, not
        # at the end: the module's metadata reports this dict's order, and the
        # operator's word list is held to it position by position.
        parameters_of["cum_seq_lens_q_ptr"] = "*i32"
    parameters_of.update(dict.fromkeys(STRIDES, "i32"))
    parameters_of.update(
        {"max_pages": "i32", "page_size": "i32", "softmax_scale": "fp32"}
    )
    if prefill:
        # The prefill program also reads the launch's token count, which bounds
        # the padding it writes, and its row count, which bounds the search for
        # a tile's row.
        parameters_of["max_q_len"] = "i32"
        parameters_of["packed"] = "i32"
        parameters_of["tokens"] = "i32"
        parameters_of["rows"] = "i32"
    # The program's parameter list is the authority: its order is what a caller
    # packs, and this dict's order is what the module's metadata reports. One
    # order, stated twice, is a launch that dereferences a stride as a pointer -
    # so a disagreement stops the build rather than a run.
    kernel = (
        kernels._paged_attention_prefill_kernel
        if prefill
        else kernels._paged_attention_decode_kernel
    )
    # The program's list continues with its constexprs, which are named
    # separately and are not packed: compare the prefix, and refuse a signature
    # that would land a constexpr among them.
    names = list(kernel.arg_names)
    if list(parameters_of) != names[: len(parameters_of)]:
        raise SystemExit(
            f"this signature's order {list(parameters_of)} is not the "
            f"program's {names[: len(parameters_of)]}"
        )
    constexprs = {
        "HEAD_DIM": head_dim,
        "PAGE": page,
        "GROUP": q_heads // kv_heads,
        "QUANTIZED": quantized,
        "OUT_BF16": fields["out"] == "bf16",
    }
    # No `launch_pdl` option: this descriptor states no `pdl` field — the
    # operator's key for the quantized cache ends at `len` — so a program that
    # read one would be compiling a word the key never carries.
    options = {"num_warps": warps_of(page, head_dim)}
    if prefill:
        # The query tile, and whether the query and the page multiply on the
        # tensor cores.
        constexprs.update(kernels.prefill_tiles(q_heads, kv_heads, head_dim))
        constexprs.update(kernels.prefill_products(activation, plane))
        options = dict(kernels.PREFILL_OPTIONS)
    return {
        "kernel": kernel,
        "signature": parameters_of,
        "constexprs": constexprs,
        "options": options,
        "aligned": list(ALIGNED) if prefill else [],
    }
