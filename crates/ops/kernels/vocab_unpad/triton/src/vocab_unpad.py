"""Vocabulary unpadding: the Triton source of the ``vocab_unpad`` operator.

A tensor-parallel vocabulary head computes a padded shard of the logits; the
operator drops the padding so the sampler sees the model's vocabulary. One
program copies a block of the unpadded matrix.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "padded", "vocabulary", "ranks"])
def _vocab_unpad_kernel(
    gathered_ptr,
    out_ptr,
    rows,
    padded,
    vocabulary,
    ranks,
    BLOCK: tl.constexpr,
):
    """Copy every row's vocabulary out of the ranks' gathered shards.

    A collective gather lays the ranks' shards one after another: ``gathered`` holds ``ranks``
    blocks of ``[rows, padded / ranks]``, rank by rank, and shard ``r`` holds the ids from
    ``r · ceil(vocabulary / ranks)`` on. Column ``c`` of a row is column ``c mod shard`` of that row
    in shard ``c / shard``. One rank is one block, which is the row-major ``[rows, padded]``.
    ``out`` is ``[rows, vocabulary]``. Grid: ``ceil(rows · vocabulary / BLOCK)``.
    """
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < rows * vocabulary
    row = offsets // vocabulary
    column = offsets % vocabulary
    width = padded // ranks
    shard = (vocabulary + ranks - 1) // ranks
    rank = column // shard
    value = tl.load(
        gathered_ptr + rank * rows * width + row * width + column % shard, mask=mask
    )
    tl.store(out_ptr + offsets, value, mask=mask)
