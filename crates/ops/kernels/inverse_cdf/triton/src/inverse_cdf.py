"""One token per row drawn from its candidates: the Triton source of the
``inverse_cdf`` operator.

One program draws one row's token from its candidate probabilities with the
row's uniform draw. The candidates are ordered by vocabulary id before the walk,
so a fixed draw names the same token however the selection kernel ordered
equally likely candidates, which is the contract UniServe's composition defined
(``uniserve/sampling/top_k.py::sample_top_k``).
"""

import triton
import triton.language as tl

#: The sentinel a candidate that the draw never reaches keeps.
MAX_ID = tl.constexpr((1 << 63) - 1)


@triton.jit(do_not_specialize=["rows"])
def _inverse_cdf_kernel(
    probabilities_ptr,
    ids_ptr,
    draws_ptr,
    tokens_ptr,
    rows,
    WIDTH: tl.constexpr,
):
    """Draw one token per program from one row's candidates.

    ``probabilities`` is ``[rows, WIDTH]`` FP32 in descending logit order,
    ``ids`` the candidates' vocabulary ids, ``draws`` ``[rows]`` uniform values
    in ``[0, 1)`` and ``tokens`` ``[rows]`` the drawn ids in the descriptor's
    token dtype. Grid: (rows,).
    """
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, WIDTH)
    probabilities = tl.load(probabilities_ptr + row * WIDTH + columns)
    ids = tl.load(ids_ptr + row * WIDTH + columns).to(tl.int64)
    draw = tl.load(draws_ptr + row)

    # Order the candidates by vocabulary id: the id is packed above the column
    # so one integer sort yields the id order and carries the column with it.
    packed = (ids << 32) | columns.to(tl.int64)
    ordered = tl.sort(packed, dim=0)
    order = (ordered & 0xFFFFFFFF).to(tl.int32)
    ordered_ids = ordered >> 32
    ordered_probabilities = tl.gather(probabilities, order, axis=0)

    # Walk the candidates by vocabulary id until the draw's share of the mass is
    # spent. A candidate with zero mass never crosses, so a dropped candidate is
    # never drawn.
    total = tl.sum(ordered_probabilities, axis=0)
    cumulative = tl.cumsum(ordered_probabilities, axis=0)
    crossed = cumulative >= draw * total
    # The candidates are in ascending id order, so the first crossing holds the
    # lowest of the ids whose mass the draw reaches.
    token = tl.min(tl.where(crossed, ordered_ids, MAX_ID), axis=0)
    # A draw at the very top of the mass, or a row whose candidates are all
    # dropped, keeps the largest candidate with positive mass.
    fallback = tl.max(tl.where(ordered_probabilities > 0.0, ordered_ids, 0), axis=0)
    tl.store(tokens_ptr + row, tl.where(token == MAX_ID, fallback, token))
