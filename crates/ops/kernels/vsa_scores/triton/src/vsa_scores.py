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

A program is one (head, query tile): it holds the query row in registers and walks the key tiles a
tile of them at a time, so every key row is read once per query tile and the scores of a row are
written contiguously. The destination is the contiguous ``[heads, query tiles, key tiles]`` matrix
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
):
    """Write one (head, query tile) row of scores.

    ``WIDTH`` is the contraction and ``BLOCK`` the lanes it uses, both the descriptor's; ``KEY_TILES``
    is the key tiles one step contracts. ``query_tiles`` and ``key_tiles`` are the launch's words —
    the grid names the query tile and the head — and ``pdl`` is a launching decision.
    """
    tile = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    lanes = tl.arange(0, BLOCK)
    live = lanes < WIDTH

    query_tiles = query_tiles.to(tl.int64)
    key_tiles = key_tiles.to(tl.int64)

    # The query row stays in registers: [BLOCK] FP32.
    heads = tl.num_programs(1).to(tl.int64)
    query = tl.load(
        query_ptr + (tile * heads + head) * WIDTH + lanes, mask=live, other=0.0
    ).to(tl.float32)
    root = tl.sqrt(tl.full((KEY_TILES,), WIDTH, tl.float32))

    # One step contracts KEY_TILES key tiles: [KEY_TILES, BLOCK] loads against the query row.
    step = tl.arange(0, KEY_TILES).to(tl.int64)
    out_base = (head * query_tiles + tile) * key_tiles
    for start in range(0, key_tiles, KEY_TILES):
        keys = start + step
        key_mask = keys < key_tiles
        values = tl.load(
            key_ptr + (keys * heads + head)[:, None] * WIDTH + lanes[None, :],
            mask=key_mask[:, None] & live[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(values * query[None, :], axis=1) / root
        tl.store(out_ptr + out_base + keys, scores, mask=key_mask)
