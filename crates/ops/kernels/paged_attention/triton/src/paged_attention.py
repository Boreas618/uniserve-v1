"""The paged attention's Triton program: one decode row, an E4M3 page decoded once.

The operator, its descriptor and its refusals are `ops::attention::PagedAttention`'s
(`crates/ops/src/attention/mod.rs`); what this program adds to an ordinary paged attention is the
cache's encoding: a page of `E4M3` bytes and **one f32 scale for the whole page**
(`notes/programs/paged_attention_fp8.md`).

The order is the contract's:

1. per page the query row's table names, load the page's scalar `s = key_scale[page]` (and the value
   plane's own);
2. **decode each element once**, `x = f32(byte) * s`, in f32, before any product or sum;
3. the attention itself unchanged — `Q · Kᵀ · softmax_scale` in f32, an online softmax in f32, the
   `P · V` sum in f32 — and one rounding into the output's type;
4. **nothing is written back**: the page stays E4M3 and the next step decodes it again.

Three things this program must never do, each of them a wrong number rather than a crash:

* re-quantize a decoded value (a read that changes the cache);
* round a decoded value through a narrower type before the product, which would add an error the
  operator's tolerance does not claim;
* accumulate in E4M3 — the accumulators are f32 whatever the page holds.

One program is one (query row, query head): it walks the pages once, carrying a running maximum, a
running normaliser and a `[HEAD_DIM]` accumulator. The key-value head is the query head divided by the
group, so a call whose `q_heads` is not a whole multiple of `kv_heads` is the launch's refusal, not
this program's rounding.
"""

import triton
import triton.language as tl


@triton.jit
def _paged_attention_decode_kernel(
    out_ptr,
    query_ptr,
    key_ptr,
    value_ptr,
    key_scale_ptr,
    value_scale_ptr,
    page_table_ptr,
    seq_lens_ptr,
    q_stride_tokens,
    q_stride_heads,
    q_stride_dim,
    k_stride_pages,
    k_stride_tokens,
    k_stride_heads,
    k_stride_dim,
    v_stride_pages,
    v_stride_tokens,
    v_stride_heads,
    v_stride_dim,
    o_stride_tokens,
    o_stride_heads,
    o_stride_dim,
    max_pages,
    page_size,
    softmax_scale,
    HEAD_DIM: tl.constexpr,
    PAGE: tl.constexpr,
    GROUP: tl.constexpr,
    QUANTIZED: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // GROUP
    dims = tl.arange(0, HEAD_DIM)
    lanes = tl.arange(0, PAGE)

    query = tl.load(
        query_ptr
        + row * q_stride_tokens
        + head * q_stride_heads
        + dims * q_stride_dim
    ).to(tl.float32)

    seq_len = tl.load(seq_lens_ptr + row)
    pages = tl.cdiv(seq_len, PAGE)

    running_max = tl.full((), float("-inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((HEAD_DIM,), tl.float32)

    for index in range(0, pages):
        page = tl.load(page_table_ptr + row * max_pages + index)
        key_offsets = (
            key_ptr
            + page * k_stride_pages
            + lanes[:, None] * k_stride_tokens
            + kv_head * k_stride_heads
            + dims[None, :] * k_stride_dim
        )
        value_offsets = (
            value_ptr
            + page * v_stride_pages
            + lanes[:, None] * v_stride_tokens
            + kv_head * v_stride_heads
            + dims[None, :] * v_stride_dim
        )
        keys = tl.load(key_offsets)
        values = tl.load(value_offsets)
        if QUANTIZED:
            # One multiply per element, in f32, and the decoded value is never written back.
            key_scale = tl.load(key_scale_ptr + page)
            value_scale = tl.load(value_scale_ptr + page)
            keys = keys.to(tl.float32) * key_scale
            values = values.to(tl.float32) * value_scale
        else:
            keys = keys.to(tl.float32)
            values = values.to(tl.float32)

        scores = tl.sum(keys * query[None, :], axis=1) * softmax_scale
        # The last page is partial: the keys past the row's length are not attended to, and a
        # masked score cannot win the maximum or the normaliser.
        live = lanes < seq_len - index * PAGE
        scores = tl.where(live, scores, float("-inf"))

        page_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, page_max)
        weights = tl.exp(scores - new_max)
        correction = tl.exp(running_max - new_max)
        running_sum = running_sum * correction + tl.sum(weights, axis=0)
        accumulator = accumulator * correction + tl.sum(weights[:, None] * values, axis=0)
        running_max = new_max

    result = accumulator / running_sum
    out_offsets = out_ptr + row * o_stride_tokens + head * o_stride_heads + dims * o_stride_dim
    if OUT_BF16:
        tl.store(out_offsets, result.to(tl.bfloat16))
    else:
        tl.store(out_offsets, result)
