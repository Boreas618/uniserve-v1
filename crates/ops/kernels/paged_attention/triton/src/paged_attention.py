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
    out_offsets = out_ptr + row * o_stride_tokens + head * o_stride_heads + dims * o_stride_dim
    # A row with no keys is a padded row of a bucket wider than its batch, and it is a **valid
    # token whose output is discarded** - not a row with no route: the capture runs the router over
    # every token the bucket holds, so this row's answer is read by the MoE block whether or not
    # anyone serves it. `0/0` would hand that block a NaN, and a NaN picks an expert from no number
    # and hands the sort an index nothing can serve, which faults one kernel later. The inert value
    # is written instead, so the route the padding produces is a route of the right arity.
    if seq_len == 0:
        padded = (
            tl.zeros((HEAD_DIM,), tl.bfloat16)
            if OUT_BF16
            else tl.zeros((HEAD_DIM,), tl.float32)
        )
        tl.store(out_offsets, padded)
        return

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
    if OUT_BF16:
        tl.store(out_offsets, result.to(tl.bfloat16))
    else:
        tl.store(out_offsets, result)


@triton.jit
def _paged_attention_prefill_kernel(
    out_ptr,
    query_ptr,
    key_ptr,
    value_ptr,
    key_scale_ptr,
    value_scale_ptr,
    page_table_ptr,
    seq_lens_ptr,
    cum_seq_lens_q_ptr,
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
    max_q_len,
    packed,
    HEAD_DIM: tl.constexpr,
    PAGE: tl.constexpr,
    GROUP: tl.constexpr,
    QUANTIZED: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    """One query row of one batch row, over the pages its table names, causally.

    The query row is a *position in its batch row*: the cache already holds that row's keys from
    `kv_write`, so key `j` is position `j` and the causal mask is `j <= position`. A packed batch
    states each row's first query token in `cum_seq_lens_q`; an unpacked one gives every row
    `max_q_len` tokens, so the row's first token is `row * max_q_len`. Either way the output row is
    `first + position`, which is what the operator's `[tokens, q_heads, head_dim]` view means.
    """
    position = tl.program_id(0)
    head = tl.program_id(1)
    row = tl.program_id(2)
    kv_head = head // GROUP
    dims = tl.arange(0, HEAD_DIM)
    lanes = tl.arange(0, PAGE)

    if packed != 0:
        first = tl.load(cum_seq_lens_q_ptr + row)
        length = tl.load(cum_seq_lens_q_ptr + row + 1) - first
    else:
        first = row * max_q_len
        length = max_q_len
    token = first + position
    out_offsets = out_ptr + token * o_stride_tokens + head * o_stride_heads + dims * o_stride_dim
    # A tile past the row's own length is padding, and it is a valid token whose output is
    # discarded: this bucket's graph runs the router over all of its tokens, so leaving the row
    # untouched hands it whatever the arena held - stale bytes from the previous replay - and the
    # route computed from them indexes a table nothing can serve (an illegal address inside the
    # MoE gather, one kernel later). The inert value is written instead.
    if position >= length:
        # ...and only a token the bucket has room for. Every padded row of a packed batch starts
        # where the live tokens end, so `first + position` runs past the buffer's token axis for all
        # but the first - and a store there lands in whatever the arena holds next, silently, which
        # would trade one illegal address for a corrupted tensor. `tl.num_programs(0)` is the
        # launch's token count: the host passes the same number as `tokens`, and at prefill
        # `max_q_len` is that number too ("the packed token count, which bounds every row"), so one
        # bound holds the row and the buffer at once.
        if token < tl.num_programs(0):
            padded = (
                tl.zeros((HEAD_DIM,), tl.bfloat16)
                if OUT_BF16
                else tl.zeros((HEAD_DIM,), tl.float32)
            )
            tl.store(out_offsets, padded)
        return

    seq_len = tl.load(seq_lens_ptr + row)
    query = tl.load(
        query_ptr + token * q_stride_tokens + head * q_stride_heads + dims * q_stride_dim
    ).to(tl.float32)

    running_max = tl.full((), float("-inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((HEAD_DIM,), tl.float32)

    pages = tl.cdiv(seq_len, PAGE)
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
            key_scale = tl.load(key_scale_ptr + page)
            value_scale = tl.load(value_scale_ptr + page)
            keys = keys.to(tl.float32) * key_scale
            values = values.to(tl.float32) * value_scale
        else:
            keys = keys.to(tl.float32)
            values = values.to(tl.float32)

        scores = tl.sum(keys * query[None, :], axis=1) * softmax_scale
        # Two masks, one where: a key past the row's length is not there, and a key after this query
        # row's position is in its future. The second is the causal mask and it is the whole
        # difference from the decode program.
        key_position = index * PAGE + lanes
        visible = (key_position <= position) & (key_position < seq_len)
        scores = tl.where(visible, scores, float("-inf"))

        page_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, page_max)
        weights = tl.exp(scores - new_max)
        correction = tl.exp(running_max - new_max)
        running_sum = running_sum * correction + tl.sum(weights, axis=0)
        accumulator = accumulator * correction + tl.sum(weights[:, None] * values, axis=0)
        running_max = new_max

    result = accumulator / running_sum
    if OUT_BF16:
        tl.store(out_offsets, result.to(tl.bfloat16))
    else:
        tl.store(out_offsets, result)
