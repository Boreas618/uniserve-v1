"""Candidate ordering: the Triton source of the ``candidate_order`` operator.

One program orders one row's candidates into ascending token id, which is the order the reference's
inverse-CDF draw needs: it draws deterministically for a fixed random value only if the candidate
order does not depend on the order the top-k program produced
(``uniserve/sampling/top_k.py:307-314``).

The permutation is by **id**, never by value, and entry ``k`` of a row's output is the input
candidate with the ``k``-th smallest id. Ties in an id cannot occur within a row (a row holds distinct
tokens), so the rule is a total order on each row.

Neither the row count nor the candidate count is compiled in. The grid names the rows and the launch
hands the width, so one module serves a request whatever top-k produced; the program walks the row in
tiles of ``BLOCK`` and, for each element, counts the elements that precede it — the count is its rank,
and writing at the rank is the permutation.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "width"])
def _candidate_order_kernel(
    values_ptr,
    ids_ptr,
    ordered_values_ptr,
    ordered_ids_ptr,
    rows,
    width,
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Order one row's candidates by token id.

    ``values`` and ``ids`` are ``[rows, width]`` row-major, ``ordered_values`` and ``ordered_ids`` the
    same shapes to write. ``rows`` is unused beyond naming the grid's extent and is kept because the
    operator packs it; ``width`` is this row's candidate count. Grid: (rows,).
    """
    row = tl.program_id(0).to(tl.int64)
    base = row * width
    tiles = tl.cdiv(width, BLOCK)
    for tile in range(tiles):
        columns = tile * BLOCK + tl.arange(0, BLOCK)
        inside = columns < width
        values = tl.load(values_ptr + base + columns, mask=inside, other=0.0)
        ids = tl.load(ids_ptr + base + columns, mask=inside, other=0)
        rank = tl.zeros([BLOCK], tl.int32)
        for other in range(tiles):
            other_columns = other * BLOCK + tl.arange(0, BLOCK)
            other_inside = other_columns < width
            other_ids = tl.load(
                ids_ptr + base + other_columns, mask=other_inside, other=0
            )
            earlier_id = other_ids[None, :] < ids[:, None]
            same_id_earlier = (other_ids[None, :] == ids[:, None]) & (
                other_columns[None, :] < columns[:, None]
            )
            counted = (earlier_id | same_id_earlier) & (
                other_inside[None, :] & inside[:, None]
            )
            rank += tl.sum(counted.to(tl.int32), axis=1)
        tl.store(ordered_values_ptr + base + rank, values, mask=inside)
        tl.store(ordered_ids_ptr + base + rank, ids, mask=inside)
