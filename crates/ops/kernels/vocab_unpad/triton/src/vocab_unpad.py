"""Vocabulary unpadding: the Triton source of the ``vocab_unpad`` operator.

A tensor-parallel vocabulary head computes a padded shard of the logits; the
operator drops the padding so the sampler sees the model's vocabulary. One
program copies a block of the unpadded matrix.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "padded", "vocabulary"])
def _vocab_unpad_kernel(
    gathered_ptr,
    out_ptr,
    rows,
    padded,
    vocabulary,
    BLOCK: tl.constexpr,
):
    """Copy the unpadded prefix of every row of ``gathered``.

    ``gathered`` is ``[rows, padded]`` and ``out`` ``[rows, vocabulary]``. Grid:
    ``ceil(rows · vocabulary / BLOCK)``.
    """
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < rows * vocabulary
    row = offsets // vocabulary
    column = offsets % vocabulary
    value = tl.load(gathered_ptr + row * padded + column, mask=mask)
    tl.store(out_ptr + offsets, value, mask=mask)
