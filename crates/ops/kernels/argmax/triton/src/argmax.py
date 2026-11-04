"""Largest value of a row: the Triton source of the ``argmax`` operator.

One program reads one row of candidates and stores the index of its largest
value, the lowest index among equals. A row without a positive temperature
selects this way.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _argmax_kernel(
    values_ptr,
    out_ptr,
    rows,
    WIDTH: tl.constexpr,
):
    """Store the index of one row's largest value.

    ``values`` is ``[rows, WIDTH]`` and ``out`` ``[rows]`` in the descriptor's id
    dtype. Grid: (rows,).
    """
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, WIDTH)
    values = tl.load(values_ptr + row * WIDTH + columns)
    largest = tl.max(values, axis=0)
    # Ties keep the lowest index: the smallest column that holds the largest
    # value wins.
    index = tl.min(tl.where(values == largest, columns, WIDTH).to(tl.int32), axis=0)
    tl.store(out_ptr + row, index)
