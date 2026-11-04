"""Largest value of a row: the Triton source of the ``argmax`` operator.

One program reads one row and stores the index of its largest value, the lowest index among equals. A
row without a positive temperature selects this way.

The width is a descriptor value but not a program's vector length: a vocabulary (151936) is not a
power of two, and ``tl.arange`` builds power-of-two ranges only. The row is walked in tiles of
``BLOCK``, with the running answer kept across tiles -- a later tile replaces it only when its largest
value is strictly greater, which is what keeps the lowest index among equals.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _argmax_kernel(
    values_ptr,
    out_ptr,
    rows,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Store the index of one row's largest value.

    ``values`` is ``[rows, WIDTH]`` and ``out`` ``[rows]`` in the descriptor's id dtype. Grid: (rows,).
    """
    row = tl.program_id(0).to(tl.int64)
    base = row * WIDTH
    best_value = tl.full([1], float("-inf"), tl.float32)
    best_index = tl.full([1], WIDTH, tl.int32)
    for tile in range(tl.cdiv(WIDTH, BLOCK)):
        columns = tile * BLOCK + tl.arange(0, BLOCK)
        inside = columns < WIDTH
        values = tl.load(values_ptr + base + columns, mask=inside, other=float("-inf"))
        largest = tl.max(values, axis=0).to(tl.float32)
        index = tl.min(tl.where(values == largest, columns, WIDTH).to(tl.int32), axis=0)
        take = largest > best_value
        best_value = tl.where(take, largest, best_value)
        best_index = tl.where(take, index, best_index)
    tl.store(out_ptr + row + tl.arange(0, 1), best_index.to(out_ptr.dtype.element_ty))
