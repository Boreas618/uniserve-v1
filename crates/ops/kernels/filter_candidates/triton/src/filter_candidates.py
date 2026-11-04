"""Candidate filtering: the Triton source of the ``filter_candidates`` operator.

One program filters one row of candidates, in place and in the operator's order:
the row's top-k width drops the candidates past it, the nucleus threshold drops
every candidate at or after the first cumulative probability above ``top_p``
(keeping that candidate), and min-p drops the candidates below the largest one's
logits plus ``log(min_p)``. A dropped candidate becomes ``-inf``; the program
also reports whether the row's remaining candidates form a usable distribution.

The numbers and the order are the ones UniServe's tensor composition defined
(``uniserve/sampling/top_k.py::top_k_candidates`` and ``sample_top_k``).
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _filter_candidates_kernel(
    values_ptr,
    ids_ptr,
    parameters_ptr,
    valid_ptr,
    rows,
    WIDTH: tl.constexpr,
):
    """Filter one row of candidates in place and report its validity.

    ``values`` is ``[rows, WIDTH]`` FP32 in descending logit order, ``ids`` the
    same shape (read only by the operators that follow), ``parameters``
    ``[rows, 8]`` FP32 with the top-k width at 1, the nucleus threshold at 2 and
    the min-p threshold at 3, and ``valid`` ``[rows]`` i32. Grid: (rows,).
    """
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, WIDTH)
    values = tl.load(values_ptr + row * WIDTH + columns)

    parameters = parameters_ptr + row * 8
    top_k = tl.load(parameters + 1)
    top_p = tl.load(parameters + 2)
    min_p = tl.load(parameters + 3)

    # 1. The row's top-k width: a width that is not positive keeps every
    # candidate, which only a zero-temperature row asks for.
    limit = tl.where(
        top_k > 0.0,
        tl.minimum(top_k, float(WIDTH)).to(tl.int32),
        WIDTH,
    )
    kept = tl.where(columns < limit, values, -float("inf"))

    # 2. The nucleus: the first candidate whose inclusive cumulative
    # probability crosses `top_p` is kept, the ones after it are dropped. The
    # inclusive cumulative minus the candidate's own probability is the
    # exclusive one, so nothing has to be gathered.
    maximum = tl.max(kept, axis=0)
    exponentials = tl.exp(kept - maximum)
    probabilities = exponentials / tl.sum(exponentials, axis=0)
    cumulative = tl.cumsum(probabilities, axis=0)
    dropped = (cumulative - probabilities) > top_p
    kept = tl.where(dropped, -float("inf"), kept)

    # 3. Min-p: measured against the largest surviving candidate's logits.
    threshold = tl.max(kept, axis=0) + tl.log(min_p)
    kept = tl.where((min_p <= 0.0) | (kept >= threshold), kept, -float("inf"))
    tl.store(values_ptr + row * WIDTH + columns, kept)

    # 4. A row whose candidates are not all real numbers, or hold no finite
    # candidate at all, has no well-defined distribution.
    not_a_number = kept != kept
    infinite = kept == float("inf")
    finite = ~not_a_number & ~infinite & (kept != -float("inf"))
    usable = (
        (tl.sum(not_a_number.to(tl.int32), axis=0) == 0)
        & (tl.sum(infinite.to(tl.int32), axis=0) == 0)
        & (tl.sum(finite.to(tl.int32), axis=0) > 0)
    )
    tl.store(valid_ptr + row, tl.where(usable, 1, 0))
