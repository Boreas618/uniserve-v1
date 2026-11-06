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

The row this program writes is a whole permutation of `width` slots, every time: the candidates it
counts take the slots their rank gives them, and the remaining slots are filled with id `-1` and
value `-inf`. A slot nothing wrote is therefore not a reading anyone has to interpret - it cannot be
mistaken for the legitimate token `0`, and the draw ignores it because it holds no mass.
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
    #
    #    An id outside ``[0, vocabulary)`` is not a token, and the row's count plane has no slot for
    #    it: counting it would write outside this row's counts, into the next row's or past the
    #    tensor, and the prefix sum would then rank every candidate after it wrongly. The planning
    #    operator's ids are masked to the vocabulary, so such an id is a defect upstream - the
    #    column is skipped here rather than counted at an invented id, and step 5 leaves it out of
    #    the permutation, where the caller's own id check can name it.
    for tile in range(tl.cdiv(width, BLOCK)):
        columns = tile * BLOCK + tl.arange(0, BLOCK)
        inside = columns < width
        ids = tl.load(ids_ptr + base + columns, mask=inside, other=0)
        token = (ids >= 0) & (ids < vocabulary)
        tl.atomic_add(counts + ids, 1, mask=inside & token)

    # 3. The exclusive prefix sum over the id space: an id's value is how many ids precede it.
    running = 0
    for tile in range(tl.cdiv(vocabulary, BLOCK)):
        columns = tile * BLOCK + tl.arange(0, BLOCK)
        inside = columns < vocabulary
        seeded = tl.load(counts + columns, mask=inside, other=0)
        prefix = tl.cumsum(seeded, axis=0) - seeded + running
        tl.store(counts + columns, prefix, mask=inside)
        running += tl.sum(seeded, axis=0)

    # The number of candidates this row holds, which is the number of slots step 4 fills: it is the
    # total the prefix sum accumulated, so step 5 needs no second reduction.
    filled = running

    # 4. Each candidate's rank is the prefix at its id, and writing at the rank is the permutation.
    #    The prefix is *taken* rather than read: `atomic_add` returns the value the slot held, so two
    #    candidates with the same id receive consecutive ranks instead of colliding on one, and every
    #    counted candidate lands in a slot of its own. The row's counts are consumed here, which is
    #    why nothing after this step reads them.
    #
    #    The same guard as step 2: an id that is not a token has no rank, and a rank read from the
    #    wrong slot would carry the *store* outside the row.
    for tile in range(tl.cdiv(width, BLOCK)):
        columns = tile * BLOCK + tl.arange(0, BLOCK)
        inside = columns < width
        values = tl.load(values_ptr + base + columns, mask=inside, other=0.0)
        ids = tl.load(ids_ptr + base + columns, mask=inside, other=0)
        token = (ids >= 0) & (ids < vocabulary)
        rank = tl.atomic_add(counts + ids, 1, mask=inside & token)
        tl.store(ordered_values_ptr + base + rank, values, mask=inside & token)
        tl.store(ordered_ids_ptr + base + rank, ids, mask=inside & token)

    # 5. Every slot the permutation did not reach is filled, so the row is a whole permutation and
    #    not a prefix over memory nothing wrote. `-1` is outside every vocabulary and `-inf` holds
    #    no mass, so the draw ignores such a slot (`inverse_cdf`'s fallback takes the highest id
    #    that holds mass) and a reader sees a value no candidate can be. `filled` is the count of
    #    counted ids, which is the number of slots step 4 filled.
    tail = tl.full([BLOCK], -1, tl.int32)
    empty = tl.full([BLOCK], -float("inf"), tl.float32)
    for tile in range(tl.cdiv(width, BLOCK)):
        columns = tile * BLOCK + tl.arange(0, BLOCK)
        gap = (columns >= filled) & (columns < width)
        tl.store(ordered_ids_ptr + base + columns, tail, mask=gap)
        tl.store(ordered_values_ptr + base + columns, empty, mask=gap)
