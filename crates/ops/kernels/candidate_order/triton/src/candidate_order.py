"""Candidate ordering: the Triton source of the ``candidate_order`` operator.

One program orders one row's candidates into ascending token id, which is the order the reference's
inverse-CDF draw needs: it draws deterministically for a fixed random value only if the candidate
order does not depend on the order the top-k program produced
(``uniserve/sampling/top_k.py:307-314``).

The permutation is by **id**, never by value, and entry ``k`` of a row's output is the input
candidate with the ``k``-th smallest id. Ties in an id cannot occur within a row (a row holds distinct
tokens), so the rule is a total order on each row.

A row's candidates may be the whole vocabulary: a deployment with top-k disabled draws over every
token, so the ordering must be linear in the vocabulary and not quadratic in it. The rank of a
candidate is the number of candidates with a smaller id, so the program **counts** rather than
compares: it zeroes a `[vocabulary]` count row, adds one at each candidate's id, turns that row into
an exclusive prefix sum, and reads each candidate's rank back out of it. A tile of candidates is
touched twice and the id space once, and no comparison between candidates ever happens.

Neither the row count, the candidate count nor the vocabulary is compiled in: the grid names the
rows and the launch hands the other two, so one module serves a request whatever top-k produced.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "width", "vocabulary"])
def _candidate_order_kernel(
    values_ptr,
    ids_ptr,
    ordered_values_ptr,
    ordered_ids_ptr,
    counts_ptr,
    rows,
    width,
    vocabulary,
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Order one row's candidates by token id.

    ``values`` and ``ids`` are ``[rows, width]`` row-major, ``ordered_values`` and ``ordered_ids`` the
    same shapes to write, and ``counts`` is ``[rows, vocabulary]`` i32 scratch that this program owns
    for the duration of the launch: it holds one count per token id, then that id's rank. ``rows`` is
    unused beyond naming the grid's extent and is kept because the operator packs it. Grid: (rows,).
    """
    row = tl.program_id(0).to(tl.int64)
    base = row * width
    counts = counts_ptr + row * vocabulary

    # 1. The id space starts with no candidates in it.
    for tile in range(tl.cdiv(vocabulary, BLOCK)):
        columns = tile * BLOCK + tl.arange(0, BLOCK)
        tl.store(
            counts + columns,
            tl.zeros([BLOCK], tl.int32),
            mask=columns < vocabulary,
        )

    # 2. One candidate per id: a row holds distinct tokens, so this count is 0 or 1 and the
    #    prefix sum below is an exact rank rather than a bucket's start.
    for tile in range(tl.cdiv(width, BLOCK)):
        columns = tile * BLOCK + tl.arange(0, BLOCK)
        inside = columns < width
        ids = tl.load(ids_ptr + base + columns, mask=inside, other=0)
        tl.atomic_add(counts + ids, 1, mask=inside)

    # 3. The exclusive prefix sum over the id space: an id's value is how many ids precede it.
    running = 0
    for tile in range(tl.cdiv(vocabulary, BLOCK)):
        columns = tile * BLOCK + tl.arange(0, BLOCK)
        inside = columns < vocabulary
        seeded = tl.load(counts + columns, mask=inside, other=0)
        prefix = tl.cumsum(seeded, axis=0) - seeded + running
        tl.store(counts + columns, prefix, mask=inside)
        running += tl.sum(seeded, axis=0)

    # 4. Each candidate's rank is the prefix at its id, and writing at the rank is the permutation.
    for tile in range(tl.cdiv(width, BLOCK)):
        columns = tile * BLOCK + tl.arange(0, BLOCK)
        inside = columns < width
        values = tl.load(values_ptr + base + columns, mask=inside, other=0.0)
        ids = tl.load(ids_ptr + base + columns, mask=inside, other=0)
        rank = tl.load(counts + ids, mask=inside, other=0)
        tl.store(ordered_values_ptr + base + rank, values, mask=inside)
        tl.store(ordered_ids_ptr + base + rank, ids, mask=inside)
