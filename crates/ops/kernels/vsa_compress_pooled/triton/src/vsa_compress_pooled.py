"""The compression VSA's gate adds: the ``vsa_compress_pooled`` operator's Triton source.

The sparse attention reads the tiles the selection chose and adds a compressed summary of *every*
tile to each query row, so the rows it did not read are still represented by their pooled mean. The
summary is one softmax over the key tiles, weighted into the pooled values:

```text
weights[head, i, j] = softmax_j(scores[head, i, j])      j over the live key tiles
out[head, i, w]     = Σ_j weights[head, i, j] · values[j, head, w]
```

`sizes[j]` is tile `j`'s live key count, so a partial last tile — and a tile that ended up empty —
weighs only the keys it has. A tile with no live key contributes nothing: its score is masked to
`-inf` before the softmax, which is what makes the last tile of a packed sequence harmless.

Both operands are FP32 (``vsa_prepare_tiles`` pools a tile's per-head means in FP32 and the
reference's softmax runs in FP32) and the sum accumulates in FP32; only the store narrows, to the
attention's own row type, which is the descriptor's `dtype`. That is the operator's tolerance: the
standard bound over the longer of the two reductions, the tile bound and the width.

A program is one (query tile, head). The grid names those two; the tiles are one block of
`MAX_TILES` — the descriptor's bound rounded up to a power of two, so one block covers every tile
count the module accepts — and the features one block of `BLOCK` lanes.
"""

import triton
import triton.language as tl


@triton.jit
def _vsa_compress_pooled_kernel(
    scores_ptr,
    values_ptr,
    sizes_ptr,
    out_ptr,
    query_tiles,
    tiles,
    pdl,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    MAX_TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write one (head, query tile) row of the compression.

    ``MAX_TILES`` is the descriptor's tile bound rounded up to a power of two and ``BLOCK`` the
    lanes one feature row uses, both compiled; ``HEADS`` and ``WIDTH`` size the pooled values' layout.
    ``query_tiles`` and ``tiles`` are the launch's words, and ``pdl`` is a launching decision.
    """
    query = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    query_tiles = query_tiles.to(tl.int64)
    tiles = tiles.to(tl.int64)

    # The softmax over the key tiles: [MAX_TILES] FP32, with the tiles past `tiles` masked out.
    step = tl.arange(0, MAX_TILES).to(tl.int64)
    live = step < tiles
    size = tl.load(sizes_ptr + step, mask=live, other=0)
    scores = tl.load(
        scores_ptr + (head * query_tiles + query) * tiles + step,
        mask=live,
        other=0.0,
    ).to(tl.float32)
    # A tile with no live key weighs nothing. `-inf` is the identity the softmax needs; an empty
    # whole row cannot happen (the operator refuses a compression with no tile) but the guard below
    # keeps it from turning the maximum into a NaN if it ever did.
    keep = live & (size > 0)
    scores = tl.where(keep, scores, -float("inf"))
    peak = tl.max(scores, axis=0)
    peak = tl.where(peak == -float("inf"), 0.0, peak)
    weights = tl.exp(scores - peak)
    total = tl.sum(weights, axis=0)
    total = tl.where(total == 0.0, 1.0, total)
    weights = weights / total

    # The weighted sum of the pooled values: [MAX_TILES, BLOCK] against the weights, accumulated in
    # FP32 and rounded once into the row's own type.
    lanes = tl.arange(0, BLOCK).to(tl.int64)
    lane_live = lanes < WIDTH
    pooled = tl.load(
        values_ptr + step[:, None] * (HEADS * WIDTH) + head * WIDTH + lanes[None, :],
        mask=live[:, None] & lane_live[None, :],
        other=0.0,
    ).to(tl.float32)
    compressed = tl.sum(weights[:, None] * pooled, axis=0)
    tl.store(
        out_ptr + (head * query_tiles + query) * WIDTH + lanes,
        compressed.to(out_ptr.dtype.element_ty),
        mask=lane_live,
    )
