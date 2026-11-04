"""One token per row drawn from its candidates: the Triton source of the
``inverse_cdf`` operator.

One program draws one row's token from its candidate probabilities with the
row's uniform draw. The candidates arrive **in ascending vocabulary-id order** -
the caller's obligation, stated on `InverseCdfArgs`: the layer sorts them by
token id before this walk (`uniserve/sampling/top_k.py:307-314`), so a fixed draw
names the same token however the selection kernel ordered equally likely
candidates, which is the contract UniServe's composition defined.

The row is walked in tiles of ``BLOCK`` lanes and never held whole, so its width
is unbounded: the alternative, one block covering the row, asks for a register per
candidate. Two passes:

1. the row's total mass, and the highest id that holds any mass (the fallback);
2. a running inclusive cumulative, keeping the first column whose cumulative
   strictly exceeds ``draw × total``. Because the row is in id order, that column
   is the lowest id whose mass the draw reaches.
"""

import triton
import triton.language as tl

#: The sentinel a column that no crossing reached keeps: above every candidate width a deployment
#: can state (a vocabulary is orders of magnitude below it) and inside int32, which is the type the
#: row's offsets carry.
MAX_COLUMN = tl.constexpr(1 << 30)


@triton.jit(do_not_specialize=["rows", "width"])
def _inverse_cdf_kernel(
    probabilities_ptr,
    ids_ptr,
    draws_ptr,
    tokens_ptr,
    rows,
    width,
    BLOCK: tl.constexpr,
):
    """Draw one token per program from one row's candidates.

    ``probabilities`` is ``[rows, width]`` FP32 in ascending vocabulary-id order,
    ``ids`` the candidates' vocabulary ids, ``draws`` ``[rows]`` uniform values
    in ``[0, 1)`` and ``tokens`` ``[rows]`` the drawn ids in the descriptor's
    token dtype. Grid: (rows,). ``BLOCK`` is the width of one tile of the row.
    """
    row = tl.program_id(0).to(tl.int64)
    width = width.to(tl.int64)
    lanes = tl.arange(0, BLOCK)
    start = row * width
    tiles = tl.cdiv(width, BLOCK)
    draw = tl.load(draws_ptr + row).to(tl.float32)

    # Pass one: the row's total mass, and the highest id that holds any of it. The
    # fallback is the last candidate with mass, which is where a draw at the very top of
    # the mass - or a row whose candidates were all dropped - lands.
    total = 0.0
    # The candidate ids are int64, so the running maximum is int64 from the start: a loop-carried
    # variable cannot change type inside the loop.
    fallback = tl.full([], 0, tl.int64)
    for tile in range(0, tiles):
        offsets = tile * BLOCK + lanes
        live = offsets < width
        probabilities = tl.load(
            probabilities_ptr + start + offsets, mask=live, other=0.0
        ).to(tl.float32)
        ids = tl.load(ids_ptr + start + offsets, mask=live, other=0).to(tl.int64)
        total += tl.sum(probabilities, axis=0)
        fallback = tl.maximum(
            fallback, tl.max(tl.where(probabilities > 0.0, ids, 0).to(tl.int64), axis=0)
        )

    # Pass two: the first column whose inclusive cumulative strictly exceeds the draw's
    # share of the mass. A candidate with zero mass leaves the cumulative where the previous
    # one put it, so `>` never selects it, and a draw of zero lands on the first candidate
    # with mass rather than on the lowest id. The last candidate stays reachable because its
    # cumulative is the total and every draw is below one.
    target = draw * total
    # The row's offsets are int64, so the running first-crossing column is int64 from the start.
    chosen = tl.full([], MAX_COLUMN, tl.int64)
    running = 0.0
    for tile in range(0, tiles):
        offsets = tile * BLOCK + lanes
        live = offsets < width
        probabilities = tl.load(
            probabilities_ptr + start + offsets, mask=live, other=0.0
        ).to(tl.float32)
        inclusive = running + tl.cumsum(probabilities, axis=0)
        crossed = inclusive > target
        chosen = tl.minimum(
            chosen, tl.min(tl.where(crossed, offsets, MAX_COLUMN), axis=0)
        )
        running += tl.sum(probabilities, axis=0)

    # A row no crossing reached keeps the fallback: a draw at the very top of the mass, or a
    # row whose candidates are all dropped. `chosen` is a column of the row, not of the buffer.
    token = tl.load(
        ids_ptr + start + chosen, mask=(chosen != MAX_COLUMN), other=0
    ).to(tokens_ptr.dtype.element_ty)
    tl.store(
        tokens_ptr + row,
        tl.where(chosen == MAX_COLUMN, fallback.to(tokens_ptr.dtype.element_ty), token),
    )
