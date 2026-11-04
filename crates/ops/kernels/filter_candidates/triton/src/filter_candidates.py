"""Candidate filtering: the Triton source of the ``filter_candidates`` operator.

One program filters one row of candidates, in place and in the operator's order:
the row's top-k width drops the candidates past it, the nucleus threshold drops
every candidate at or after the first cumulative probability above ``top_p``
(keeping that candidate), and min-p drops the candidates below the largest one's
logits plus ``log(min_p)``. A dropped candidate becomes ``-inf``; the program
also reports whether the row's remaining candidates form a usable distribution.

The numbers and the order are the ones UniServe's tensor composition defined
(``uniserve/sampling/top_k.py::top_k_candidates`` and ``sample_top_k``).

The row is walked in tiles of ``BLOCK`` lanes and never held whole, so its width is
unbounded: one block covering the row asks for a register per candidate, which is
what a vocabulary-sized row cannot give. Four passes, because every stage needs a
row-wide answer the one before it produces:

1. the largest candidate the top-k width keeps;
2. the total of its exponentials, which is the row's normalising constant;
3. the inclusive cumulative, carried tile to tile, which drops the nucleus -- and the
   largest survivor, which min-p measures against;
4. min-p itself, the store, and the validity reduction.

Only the third pass carries a value between tiles (the cumulative), so the arithmetic is
the row-wide form's; a cumulative is a tree scan in either shape, and the seam between
two tiles is one more addition in a different order, not a different decision rule.
"""

import triton
import triton.language as tl

#: How many columns of the parameter row this operator reads: the top-k width, the nucleus
#: threshold and the min-p threshold, at 1, 2 and 3, inside a row of this many columns.
PARAMETER_COLUMNS = tl.constexpr(8)


@triton.jit(do_not_specialize=["rows", "width"])
def _filter_candidates_kernel(
    values_ptr,
    ids_ptr,
    parameters_ptr,
    valid_ptr,
    rows,
    width,
    BLOCK: tl.constexpr,
):
    """Filter one row of candidates in place and report its validity.

    ``values`` is ``[rows, width]`` FP32 in descending logit order, ``ids`` the same shape
    (read only by the operators that follow), ``parameters`` ``[rows, 8]`` FP32 with the
    top-k width at 1, the nucleus threshold at 2 and the min-p threshold at 3, and
    ``valid`` ``[rows]`` i32. Grid: (rows,). ``BLOCK`` is the width of one tile of the row.
    """
    row = tl.program_id(0).to(tl.int64)
    width = width.to(tl.int64)
    lanes = tl.arange(0, BLOCK)
    base = row * width
    tiles = tl.cdiv(width, BLOCK)

    parameters = parameters_ptr + row * PARAMETER_COLUMNS
    top_k = tl.load(parameters + 1)
    top_p = tl.load(parameters + 2)
    min_p = tl.load(parameters + 3)

    # The row's top-k width: a width that is not positive keeps every candidate, which only a
    # zero-temperature row asks for.
    limit = tl.where(
        top_k > 0.0,
        tl.minimum(top_k, width.to(tl.float32)).to(tl.int64),
        width,
    )

    # Pass one: the largest candidate the top-k width keeps. It is the shift of the
    # exponentials, so the row has to agree on it before any of them is taken.
    maximum = tl.full([], -float("inf"), tl.float32)
    for tile in range(tiles):
        columns = tile * BLOCK + lanes
        values = tl.load(values_ptr + base + columns, mask=columns < width,
                         other=-float("inf"))
        kept = tl.where(columns < limit, values, -float("inf"))
        maximum = tl.maximum(maximum, tl.max(kept, axis=0))

    # Pass two: the normalising constant, over the same kept set.
    total = 0.0
    for tile in range(tiles):
        columns = tile * BLOCK + lanes
        values = tl.load(values_ptr + base + columns, mask=columns < width,
                         other=-float("inf"))
        kept = tl.where(columns < limit, values, -float("inf"))
        total += tl.sum(tl.exp(kept - maximum), axis=0)

    # Pass three: the nucleus, in a cumulative carried across tiles, and the largest survivor.
    # The first candidate whose inclusive cumulative crosses `top_p` is kept and the ones after
    # it are dropped; the inclusive cumulative minus the candidate's own probability is the
    # exclusive one, so nothing has to be gathered.
    running = tl.full([], 0.0, tl.float32)
    largest = tl.full([], -float("inf"), tl.float32)
    for tile in range(tiles):
        columns = tile * BLOCK + lanes
        values = tl.load(values_ptr + base + columns, mask=columns < width,
                         other=-float("inf"))
        kept = tl.where(columns < limit, values, -float("inf"))
        probabilities = tl.exp(kept - maximum) / total
        cumulative = running + tl.cumsum(probabilities, axis=0)
        # The next tile's cumulative starts where this one's mass ends. A sum of the tile's own
        # probabilities, not the last lane of `cumulative`: the tile's last lanes are padding when
        # the width is not a whole number of tiles.
        running += tl.sum(probabilities, axis=0)
        dropped = (cumulative - probabilities) > top_p
        kept = tl.where(dropped, -float("inf"), kept)
        largest = tl.maximum(largest, tl.max(kept, axis=0))
        tl.store(values_ptr + base + columns, kept, mask=columns < width)

    # Pass four: min-p, measured against the largest surviving candidate's logits, then the
    # validity reduction. A row whose candidates are not all real numbers, or hold no finite
    # candidate at all, has no well-defined distribution.
    threshold = largest + tl.log(min_p)
    not_a_number = tl.full([], 0, tl.int32)
    infinite = tl.full([], 0, tl.int32)
    finite = tl.full([], 0, tl.int32)
    for tile in range(tiles):
        columns = tile * BLOCK + lanes
        kept = tl.load(values_ptr + base + columns, mask=columns < width,
                       other=-float("inf"))
        kept = tl.where((min_p <= 0.0) | (kept >= threshold), kept, -float("inf"))
        tl.store(values_ptr + base + columns, kept, mask=columns < width)
        not_a_number += tl.sum((kept != kept).to(tl.int32), axis=0)
        infinite += tl.sum((kept == float("inf")).to(tl.int32), axis=0)
        finite += tl.sum(
            ((kept == kept) & (kept != float("inf")) & (kept != -float("inf"))).to(tl.int32),
            axis=0,
        )

    usable = (not_a_number == 0) & (infinite == 0) & (finite > 0)
    tl.store(valid_ptr + row, tl.where(usable, 1, 0))
