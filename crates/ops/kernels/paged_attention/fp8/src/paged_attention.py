"""Paged FP8 decode attention: the Triton source of the ``paged_attention``
operator's ``fp8`` backend.

Each program serves one query head of one decode row: it walks the row's block
table, loads each page's key and value rows as E4M3 codes, decodes them with
that page's scale, and accumulates the softmax-weighted values in FP32 with the
online-softmax recurrence, so the row's cache length may be any value and no
page-sized buffer is materialized.

An FP8 page holds ``value = code * scale`` with one scale per page, written by
the ``fp8`` backend of ``paged_kv_write``. Query and value heads may differ
(grouped-query attention): a query head reads the key/value head it shares.

This kernel is new in the design's sense (design 9: "新写读写这种页的分页注意力
kernel program"): no existing kernel reads per-page E4M3 pages. It covers the
decode call; the prefill call over the same pages is the operator's other
phase and is not in this package yet.
"""

import triton
import triton.language as tl


@triton.jit
def _paged_attention_fp8_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    out_ptr,
    q_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    max_pages: tl.constexpr,
    num_pages: tl.constexpr,
    softmax_scale: tl.constexpr,
    block: tl.constexpr,
):
    """Attend one decode row's query head over its FP8 key/value pages.

    ``q`` is ``[rows, q_heads, head_dim]``, the caches
    ``[num_pages, page_size, kv_heads, head_dim]`` E4M3 with one FP32 scale per
    page, ``block_tables`` ``[rows, max_pages]`` int32 and ``seq_lens``
    ``[rows]`` int32 (the resident cache length). Grid: (rows, q_heads).
    """
    row = tl.program_id(0)
    head = tl.program_id(1)
    group = q_heads // kv_heads
    kv_head = head // group

    length = tl.load(seq_lens_ptr + row)
    dimensions = tl.arange(0, block)
    feature_mask = dimensions < head_dim
    query = tl.load(
        q_ptr + (row * q_heads + head) * head_dim + dimensions,
        mask=feature_mask,
        other=0.0,
    ).to(tl.float32)

    # The online-softmax recurrence: `running_max` of the scores seen, their
    # running sum and the accumulated value rows.
    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.zeros((), dtype=tl.float32)
    accumulated = tl.zeros((block,), dtype=tl.float32)

    tokens = tl.arange(0, page_size)
    pages = tl.cdiv(length, page_size)
    for page_index in range(0, pages):
        page = tl.load(block_tables_ptr + row * max_pages + page_index)
        positions = page_index * page_size + tokens
        live = positions < length
        # [page_size, kv_heads * head_dim] rows of one page, this head's slice.
        offsets = (
            (page * page_size + tokens)[:, None] * (kv_heads * head_dim)
            + kv_head * head_dim
            + dimensions[None, :]
        )
        mask = live[:, None] & feature_mask[None, :]

        key = tl.load(k_cache_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        value = tl.load(v_cache_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        key = key * tl.load(k_scale_ptr + page)
        value = value * tl.load(v_scale_ptr + page)

        scores = tl.sum(key * query[None, :], axis=1) * softmax_scale
        scores = tl.where(live, scores, -float("inf"))

        page_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, page_max)
        # A page whose scores are all masked leaves the accumulator unchanged.
        rescale = tl.exp(running_max - new_max)
        weights = tl.exp(scores - new_max)
        running_sum = running_sum * rescale + tl.sum(weights, axis=0)
        accumulated = accumulated * rescale + tl.sum(weights[:, None] * value, axis=0)
        running_max = new_max

    # A row with no cache yet stores zeros rather than a division by zero.
    result = tl.where(running_sum > 0.0, accumulated / running_sum, 0.0)
    tl.store(
        out_ptr + (row * q_heads + head) * head_dim + dimensions,
        result,
        mask=feature_mask,
    )
