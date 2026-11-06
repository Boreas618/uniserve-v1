"""The pooled tile similarities VSA's selection ranks: the ``vsa_scores`` operator's Triton source.

VSA pools each tile's per-head means, contracts them per head, and ranks the result to choose the
tiles the sparse attention reads:

```text
pooled_query [query tiles, heads, width] · pooled_key [key tiles, heads, width]
scores[head, i, j] = Σ_w pooled_query[i, head, w] · pooled_key[j, head, w] / sqrt(width)
```

The operands are in the layout ``vsa_prepare_tiles`` writes, a tile first. The scale is the
reference's: the selection only compares, but the compression's softmax reads these same scores.
There is no softmax here. Both operands are FP32 — the type ``vsa_prepare_tiles`` pools a tile's
per-head means in — and the products accumulate in FP32, which is the operator's tolerance: the
standard bound for a ``width``-term contraction.

A program is one block of ``QUERIES`` query tiles of one head: it holds their rows and walks the key
tiles ``KEY_TILES`` at a time as one FP32 matrix step, without TF32, because the reference's product
is FP32. A key row is read once per block of queries. The destination is the contiguous ``[heads, query tiles, key tiles]`` matrix
the selection reads, which the operator requires and nothing else may be.
"""

import triton
import triton.language as tl

#: Key tiles one program contracts per step. It is a register tile, not a descriptor field: the
#: contraction is the descriptor's `width`, and how many keys a step holds changes no result.
KEY_BLOCK = tl.constexpr(16)


@triton.jit
def _vsa_scores_kernel(
    query_ptr,
    key_ptr,
    out_ptr,
    query_tiles,
    key_tiles,
    pdl,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    KEY_TILES: tl.constexpr,
    QUERIES: tl.constexpr,
):
    """Write ``QUERIES`` (head, query tile) rows of scores.

    ``WIDTH`` is the contraction and ``BLOCK`` the lanes it uses, both the descriptor's; ``KEY_TILES``
    is the key tiles one step contracts and ``QUERIES`` the query tiles a program holds.
    ``query_tiles`` and ``key_tiles`` are the launch's words - the grid names the block of query tiles
    and the head - and ``pdl`` is a launching decision.
    """
    head = tl.program_id(1).to(tl.int64)
    heads = tl.num_programs(1).to(tl.int64)
    lanes = tl.arange(0, BLOCK)
    live = lanes < WIDTH

    query_tiles = query_tiles.to(tl.int64)
    key_tiles = key_tiles.to(tl.int64)
    rows = tl.program_id(0).to(tl.int64) * QUERIES + tl.arange(0, QUERIES).to(tl.int64)
    row_live = rows < query_tiles

    # The query rows stay in registers: [QUERIES, BLOCK] FP32.
    query = tl.load(
        query_ptr + (rows * heads + head)[:, None] * WIDTH + lanes[None, :],
        mask=row_live[:, None] & live[None, :],
        other=0.0,
    ).to(tl.float32)
    root = tl.sqrt(tl.full((QUERIES, KEY_TILES), WIDTH, tl.float32))

    # One step contracts KEY_TILES key tiles against the query rows.
    step = tl.arange(0, KEY_TILES).to(tl.int64)
    out_base = (head * query_tiles + rows) * key_tiles
    for start in range(0, key_tiles, KEY_TILES):
        keys = start + step
        key_mask = keys < key_tiles
        values = tl.load(
            key_ptr + (keys * heads + head)[None, :] * WIDTH + lanes[:, None],
            mask=live[:, None] & key_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.dot(query, values, input_precision="ieee") / root
        tl.store(
            out_ptr + out_base[:, None] + keys[None, :],
            scores,
            mask=row_live[:, None] & key_mask[None, :],
        )
