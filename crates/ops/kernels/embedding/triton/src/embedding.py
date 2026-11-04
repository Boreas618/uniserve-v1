"""Vocabulary embedding: the Triton source of the ``embedding`` operator.

One program gathers one block of one token's row from this rank's vocabulary
shard. A token outside the shard stores zeros, so the tensor-parallel
all-reduce that follows sums the shard's rows into the full embedding row. The
kernel replaces the masked ``F.embedding`` UniServe evaluated
(``uniserve/nn/linear.py:793-803``), with the mask, the gather and the zeroing
in one launch.
"""

import triton
import triton.language as tl


@triton.jit
def _embedding_kernel(
    ids_ptr,
    weight_ptr,
    out_ptr,
    shard_start,
    shard_end,
    width: tl.constexpr,
    block: tl.constexpr,
):
    """Gather this shard's row for each token, zeros outside it.

    ``ids`` are the request tokens' vocabulary ids, ``weight`` the shard's
    ``[shard, width]`` table and ``out`` the ``[tokens, width]`` rows the
    all-reduce reduces. Grid: (tokens, ceil(width / block)).
    """
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * block + tl.arange(0, block)
    mask = columns < width

    # One vocabulary id per token; a token outside this rank's shard keeps
    # zeros so the shard contributes nothing to the sum.
    token = tl.load(ids_ptr + row)
    local = (token - shard_start).to(tl.int64)
    inside = (token >= shard_start) & (token < shard_end)
    value = tl.load(
        weight_ptr + local * width + columns, mask=mask & inside, other=0.0
    )
    tl.store(out_ptr + row * width + columns, value, mask=mask)
