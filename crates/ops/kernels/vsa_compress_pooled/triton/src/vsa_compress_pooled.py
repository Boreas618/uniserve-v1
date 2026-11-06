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

A program is one block of ``QUERIES`` query tiles of one head. It walks the key tiles ``KEYS`` at a
time with a running softmax, so every pooled value row is read once per block of queries rather
than once per query tile: reading them per query tile made this the largest kernel of a denoising
step. The running form is the exact softmax up to rounding: the maximum so far rescales what was
already summed, and the sum divides once at the end. The product is one FP32 matrix step, without
TF32, because the reference's product is FP32.
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
    BLOCK: tl.constexpr,
    QUERIES: tl.constexpr,
    KEYS: tl.constexpr,
):
    """Write ``QUERIES`` (head, query tile) rows of the compression.

    ``BLOCK`` is the lanes one feature row uses; ``HEADS`` and ``WIDTH`` size the pooled values'
    layout; ``QUERIES`` and ``KEYS`` are the block of query tiles a program holds and the key tiles
    one step reads. ``query_tiles`` and ``tiles`` are the launch's words, and ``pdl`` is a launching
    decision.
    """
    head = tl.program_id(1).to(tl.int64)
    query_tiles = query_tiles.to(tl.int64)
    tiles = tiles.to(tl.int64)
    rows = tl.program_id(0).to(tl.int64) * QUERIES + tl.arange(0, QUERIES).to(tl.int64)
    row_live = rows < query_tiles
    lanes = tl.arange(0, BLOCK).to(tl.int64)
    lane_live = lanes < WIDTH

    peak = tl.full((QUERIES,), -float("inf"), tl.float32)
    total = tl.zeros((QUERIES,), tl.float32)
    summed = tl.zeros((QUERIES, BLOCK), tl.float32)
    for start in range(0, tiles, KEYS):
        keys = start + tl.arange(0, KEYS).to(tl.int64)
        live = keys < tiles
        size = tl.load(sizes_ptr + keys, mask=live, other=0)
        scores = tl.load(
            scores_ptr + (head * query_tiles + rows)[:, None] * tiles + keys[None, :],
            mask=row_live[:, None] & live[None, :],
            other=0.0,
        ).to(tl.float32)
        # A tile with no live key weighs nothing: `-inf` is the identity the softmax needs. A block
        # that has seen no live tile yet keeps a peak of `-inf`, so the offset below stands at zero
        # to keep the exponent from reading `-inf - -inf`.
        scores = tl.where((live & (size > 0))[None, :], scores, -float("inf"))
        peak_now = tl.maximum(peak, tl.max(scores, axis=1))
        offset = tl.where(peak_now == -float("inf"), 0.0, peak_now)
        rescale = tl.exp(peak - offset)
        weights = tl.exp(scores - offset[:, None])
        total = total * rescale + tl.sum(weights, axis=1)
        values = tl.load(
            values_ptr + keys[:, None] * (HEADS * WIDTH) + head * WIDTH + lanes[None, :],
            mask=live[:, None] & lane_live[None, :],
            other=0.0,
        ).to(tl.float32)
        summed = summed * rescale[:, None] + tl.dot(weights, values, input_precision="ieee")
        peak = peak_now
    total = tl.where(total == 0.0, 1.0, total)
    compressed = summed / total[:, None]
    tl.store(
        out_ptr + (head * query_tiles + rows)[:, None] * WIDTH + lanes[None, :],
        compressed.to(out_ptr.dtype.element_ty),
        mask=row_live[:, None] & lane_live[None, :],
    )
