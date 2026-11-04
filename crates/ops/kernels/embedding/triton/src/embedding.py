"""Vocabulary embedding: the Triton source of the ``embedding`` operator.

One program copies the hidden rows of a block of tokens out of this shard's
vocabulary weight, or writes zeros when a token belongs to another shard of a
tensor-parallel split. The token count, the shard's first id and its row count
arrive with the launch.

The kernel body is the one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/embedding.py``), unchanged except that
it imports nothing but Triton.
"""

import triton
import triton.language as tl


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(do_not_specialize=["tokens", "hidden", "first", "rows"])
def _embedding_kernel(
    ids_ptr,
    weight_ptr,
    out_ptr,
    tokens,
    hidden,
    first,
    rows,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Copy the embedding rows of ``BLOCK`` tokens.

    ``ids`` is ``[tokens]``, ``weight`` ``[rows, hidden]`` (this shard) and
    ``out`` ``[tokens, hidden]``. Grid: ``ceil(tokens / BLOCK)``; each program
    walks its rows' ``TILE`` wide column blocks. A token outside
    ``[first, first + rows)`` writes zeros.
    """
    _pdl_prologue(PDL)

    token = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    token_mask = token < tokens
    identity = tl.load(ids_ptr + token, mask=token_mask, other=-1)
    local = identity.to(tl.int64) - first
    inside = token_mask & (local >= 0) & (local < rows)

    for start in range(0, hidden, TILE):
        columns = start + tl.arange(0, TILE)
        column_mask = columns < hidden
        value = tl.load(
            weight_ptr + local[:, None] * hidden + columns[None, :],
            mask=inside[:, None] & column_mask[None, :],
            other=0.0,
        )
        tl.store(
            out_ptr + token[:, None] * hidden + columns[None, :],
            value,
            mask=token_mask[:, None] & column_mask[None, :],
        )
